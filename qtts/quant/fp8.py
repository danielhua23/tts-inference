"""fp8 (e4m3) quantization methods.

:class:`Fp8Method` is W8A8 on the hardware fp8 path: per-token activation
scales, per-channel weight scales, ``torch._scaled_mm``. Unlike int8 W8A8 it has
no minimum row count, so it runs at batch 1, and unlike the Triton kernel it is
a vendor GEMM, so it does not lose occupancy on narrow shapes.

It is still slower than bf16 at every talker shape. Measured on an H100 inside a
CUDA graph, M=4:

    layer     shape        bf16     fp8 W8A8   ratio
    qkv       4096x2048     5.2 us   26.2 us   0.20x
    o_proj    2048x2048     5.0 us   25.2 us   0.20x
    gate_up  12288x2048    19.7 us   27.9 us   0.71x
    down      2048x6144     7.8 us   37.8 us   0.21x

The reason is not the GEMM. Quantizing the activations costs an amax reduction,
a divide and a cast -- three passes over a tensor that the bf16 path never
touches -- and at these durations three extra kernels are the whole budget. W8A8
needs the GEMM to be compute-bound before any of it pays, and a batch-1-to-64
decode step is bandwidth-bound. The relative error is also ~3.5%, which for a
model whose output is sampled from logits is not free.

So this class exists to be measured, and the measurement says no for this
model at these batch sizes. :class:`Fp8WeightOnlyMethod` is the variant that
could pay, and it is deliberately *not* implemented on top of a dequantize-then-
matmul: see the note in its docstring.
"""

from __future__ import annotations

import torch

from ..layers.linear import LinearBase
from ..utils import get_logger
from .base import QuantMethodBase
from .kernels import kernel_available

logger = get_logger("quant.fp8")

#: Largest finite magnitude in e4m3.
_FP8_MAX = 448.0


def quantize_weight_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel e4m3. Returns ``(fp8 weight, fp32 scale)``."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / _FP8_MAX
    return (w / scale[:, None]).to(torch.float8_e4m3fn), scale


class Fp8Method(QuantMethodBase):
    """W8A8 e4m3 through ``torch._scaled_mm``."""

    name = "fp8"

    def process_weights(self, layer: LinearBase) -> None:
        q, scale = quantize_weight_fp8(layer.weight)
        # _scaled_mm wants the second operand column-major. A row-major [N, K]
        # transposed is exactly that, so the transpose is free -- but it has to
        # be kept as a view, and calling .contiguous() on it would silently undo
        # the layout and make the op reject the call.
        layer.register_buffer("fp8_weight", q.t(), persistent=False)
        layer.register_buffer("fp8_scale", scale.reshape(1, -1), persistent=False)
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device), requires_grad=False
        )

    def applies_to(self, layer: LinearBase) -> bool:
        return getattr(layer, "fp8_weight", None) is not None

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        scale_x = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-6).float() / _FP8_MAX
        q = (flat.float() / scale_x).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            q,
            layer.fp8_weight,  # type: ignore[attr-defined]
            scale_a=scale_x,
            scale_b=layer.fp8_scale,  # type: ignore[attr-defined]
            bias=None,
            out_dtype=x.dtype,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*shape[:-1], layer.out_features)


class Fp8WeightOnlyMethod(QuantMethodBase):
    """fp8 weights, bf16 activations.

    Not implemented, on purpose. The only implementation available here would be
    ``F.linear(x, weight_fp8.to(x.dtype) * scale)``, which writes a bf16 copy of
    the weight to global memory on every call and so moves *more* bytes than
    leaving the layer alone -- 163 us against 21 us at the gate_up shape. It
    would halve the resident weight footprint while making every step slower,
    and reporting that as "fp8 weight-only" invites the number to be read as a
    speedup.

    The fused kernel in ``kernels.py`` is the right home for this, but it would
    need e4m3 loads: Triton cannot do pointer arithmetic on an
    ``torch.float8_e4m3fn`` tensor, so the weight has to be stored as int8,
    loaded as int8 and bitcast to ``tl.float8e4nv`` inside the kernel. That is a
    contained piece of work and it is not done yet.

    Use ``--quant-method int8_weight_only``, which has the same footprint and a
    kernel that reads the quantized weight directly.
    """

    name = "fp8_weight_only"

    def __init__(self) -> None:
        available, reason = kernel_available()
        detail = "" if available else f" (triton also unavailable: {reason})"
        raise NotImplementedError(
            "fp8 weight-only is not implemented: the only available path dequantizes the weight "
            "into global memory per call, which is slower than not quantizing at all. Use "
            "--quant-method int8_weight_only for the same footprint, or --quant-method fp8 for "
            "the W8A8 hardware path." + detail
        )
