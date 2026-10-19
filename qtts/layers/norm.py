"""RMS normalization.

There are 112 RMSNorm calls in one talker decode step (28 layers x input, post-
attention, q and k) and 300 more in the code predictor's walk across the
codebook axis. Written eagerly each one is six kernels -- cast up, square, mean,
rsqrt, scale, cast back -- so the normalizations alone account for roughly half
the kernels in a step, on tensors of a few thousand elements. That is why the
fused kernel below exists: not for the arithmetic, which is trivial, but because
a decode step is a launch-count problem and this is the cheapest large reduction
in launch count available.

With CUDA graphs the launches are free but each kernel still costs GPU-side
dispatch latency, so fusing keeps paying: with graphs on, fusing the norms takes
a decode step from 19.1 ms to 11.8 ms.

``backend="eager"`` is not dead code. The fused reduction differs from torch's
by about two ULP of fp32, and while that is nothing as arithmetic, greedy
decoding takes an argmax over ~2000 near-tied logits sixteen times per frame, so
a perturbation at that scale flips one of those ties every few hundred
decisions. The bit-exact parity suite therefore runs on the eager path, and
``tests/test_norm.py`` pins the fused path against it directly.
"""

from __future__ import annotations

import torch
from torch import nn

# Module scope, not lazy: Triton resolves a kernel's annotations against the
# module globals of the function it compiles, so a kernel defined inside a
# function that imports ``triton.language as tl`` locally fails on
# ``BLOCK: tl.constexpr`` with a bare NameError.
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the install
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

#: Above this the row does not fit one block and the eager path is used instead.
#: Every hidden size in this model (2048 talker, 1024 predictor, 512 codec) fits.
_MAX_FUSED_WIDTH = 8192


if _HAS_TRITON:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        stride_row,
        width,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < width
        x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        # Reducing in f32 is not an optimisation detail. The talker stacks 28
        # residual blocks in bf16 and the codec decoder puts LayerScale on top
        # of its residual; a bf16 variance drifts far enough to break parity.
        variance = tl.sum(x * x, axis=0) / width
        # 1/sqrt rather than tl.rsqrt: rsqrt lowers to the approximate hardware
        # instruction, which is ~1e-7 off the correctly rounded value. That is
        # harmless as arithmetic and not harmless here -- greedy decoding takes
        # an argmax over 2048 near-tied logits 16 times per frame, and a
        # perturbation at that scale flips one of those ties every few hundred
        # decisions, which reads as a parity failure rather than as rounding.
        inv_rms = 1.0 / tl.sqrt(variance + eps)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride_row + offs, (x * inv_rms * w).to(out_ptr.dtype.element_ty), mask=mask)


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension, reducing in fp32."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        backend: str = "fused",
    ) -> None:
        super().__init__()
        if backend not in ("fused", "eager"):
            raise ValueError(f"rmsnorm backend must be fused|eager, got {backend!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype), requires_grad=False)
        self.eps = eps
        self.hidden_size = hidden_size
        self.backend = backend
        self.fused = backend == "fused" and _HAS_TRITON and hidden_size <= _MAX_FUSED_WIDTH
        self._block = triton.next_power_of_2(hidden_size) if self.fused else 0
        # The eager path would otherwise cast the weight on every call.
        self.register_buffer("weight_f32", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and x.is_cuda and not torch.is_grad_enabled():
            return self._fused_forward(x)
        return self._eager_forward(x)

    def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.hidden_size)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out = torch.empty_like(flat)
        _rms_norm_kernel[(flat.shape[0],)](
            flat,
            self.weight,
            out,
            flat.stride(0),
            self.hidden_size,
            self.eps,
            BLOCK=self._block,
            num_warps=4,
        )
        return out.reshape(x.shape)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Operation order follows transformers' Qwen3RMSNorm exactly, including
        # the separate multiply by the weight. Folding it into one expression
        # re-associates the multiplies, and fp32 multiplication is not
        # associative: it moves the result by an ULP, which is enough to flip a
        # greedy argmax and break the bit-exact parity suite this path exists
        # to serve.
        if self.weight_f32 is None:
            self.weight_f32 = self.weight.detach().to(torch.float32)
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return (self.weight_f32 * x32).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}, backend={'fused' if self.fused else 'eager'}"


class LayerScale(nn.Module):
    """Per-channel learnt scale on a residual branch (Touvron et al. 2021)."""

    def __init__(self, channels: int, initial_scale: float = 0.01, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, dtype=dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x
