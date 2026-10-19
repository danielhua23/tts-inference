"""SnakeBeta and the gated MLP.

SnakeBeta is ``x + 1/beta * sin^2(alpha * x)``, applied per channel. It appears
~30 times in the codec decoder, always on a tensor that has just been upsampled,
so by the last decoder block it runs on ``[B, 96, frames * 1920]``. Written
eagerly that is four kernels (multiply, sin, square, fused-multiply-add) each
reading and writing the full tensor, which is why vllm-omni fuses it into one
Triton kernel; ``--snake-backend eager`` exists to measure how much that fusion
is worth rather than to be used.

Both paths read ``exp(alpha)`` and ``1/(exp(beta)+eps)`` from frozen buffers.
The checkpoint stores alpha and beta in log space, so without the precompute
every call would also pay two exponentials over the channel axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .linear import LinearBase, MergedLinear

_EPS = 1e-9
_MAX_BLOCK_T = 4096

# Imported at module scope, not lazily inside a loader function, because Triton
# resolves the names in a kernel's annotations against the module globals of the
# function it is compiling. A kernel defined inside a function that does
# ``import triton.language as tl`` locally compiles its body and then fails on
# ``block_t: tl.constexpr`` with a bare NameError -- and because the fallback
# here is silent, the eager path would just quietly take over.
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the install
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _snake_beta_kernel(
        x_ptr,
        exp_alpha_ptr,
        inv_beta_ptr,
        out_ptr,
        stride_b,
        stride_c,
        t_len,
        block_t: tl.constexpr,
    ):
        bid = tl.program_id(0)
        cid = tl.program_id(1)
        t_off = tl.program_id(2) * block_t + tl.arange(0, block_t)
        mask = t_off < t_len

        offset = bid * stride_b + cid * stride_c + t_off
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        exp_alpha = tl.load(exp_alpha_ptr + cid)
        inv_beta = tl.load(inv_beta_ptr + cid)
        sin_val = tl.sin(x.to(tl.float32) * exp_alpha)
        tl.store(out_ptr + offset, x + (inv_beta * sin_val * sin_val).to(x.dtype), mask=mask)


class SnakeBeta(nn.Module):
    """Input and output are ``[B, C, T]``; alpha and beta are per channel."""

    def __init__(self, channels: int, backend: str = "triton", dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.backend = backend
        # Log-scale, matching the checkpoint. Zeros, not ones: exp(0) = 1.
        self.alpha = nn.Parameter(torch.zeros(channels, dtype=dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(channels, dtype=dtype), requires_grad=False)
        self.register_buffer("exp_alpha", None, persistent=False)
        self.register_buffer("inv_beta", None, persistent=False)

    def precompute_cache(self) -> None:
        with torch.no_grad():
            self.exp_alpha = torch.exp(self.alpha).contiguous()
            self.inv_beta = (1.0 / (torch.exp(self.beta) + _EPS)).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.exp_alpha is None:
            self.precompute_cache()
        if self.backend == "triton" and _HAS_TRITON and x.is_cuda and not torch.is_grad_enabled():
            return self._triton_forward(x)
        return self._eager_forward(x)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_alpha = self.exp_alpha.view(1, -1, 1)
        inv_beta = self.inv_beta.view(1, -1, 1)
        return x + inv_beta * torch.sin(x * exp_alpha) ** 2

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch, channels, frames = x.shape
        out = torch.empty_like(x)
        block_t = min(triton.next_power_of_2(frames), _MAX_BLOCK_T)
        _snake_beta_kernel[(batch, channels, triton.cdiv(frames, block_t))](
            x,
            self.exp_alpha,
            self.inv_beta,
            out,
            x.stride(0),
            x.stride(1),
            t_len=frames,
            block_t=block_t,
        )
        return out

    def extra_repr(self) -> str:
        return f"channels={self.channels}, backend={self.backend}"


class GatedMLP(nn.Module):
    """SwiGLU feed-forward. ``fuse`` merges gate and up into one GEMM."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        fuse: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.fuse = fuse
        if fuse:
            self.gate_up_proj = MergedLinear(
                hidden_size, [intermediate_size, intermediate_size], bias=False, dtype=dtype
            )
        else:
            self.gate_proj = LinearBase(hidden_size, intermediate_size, bias=False, dtype=dtype)
            self.up_proj = LinearBase(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = LinearBase(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse:
            gate, up = self.gate_up_proj.split(self.gate_up_proj(x))
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)
