"""int8 quantization methods.

Two variants, and they answer different questions.

:class:`Int8WeightOnlyMethod` stores the weight as int8 with one scale per
output channel and keeps activations in bf16. This is the variant that suits a
talker decode step, which is bound by weight bandwidth rather than by
arithmetic. Per-channel scales are not a refinement here -- they are what makes
the scale factor out of the dot product, so the dequantization is one multiply
per output instead of one per weight.

:class:`Int8Method` is W8A8: activations are quantized per token at run time and
the matmul runs on int8 tensor cores via ``torch._int_mm``. It needs the GEMM to
be compute-bound to pay for the two extra elementwise passes over the
activations, which at batch 1-64 it is not. It is here because "try W8A8" is the
first thing anyone suggests, and having it measurable is more useful than
arguing about it. ``torch._int_mm`` also requires more than 16 rows, so at
decode batch sizes it needs padding -- another cost that has nothing to do with
the arithmetic.

Quantization is symmetric (no zero point). Asymmetric would track the weight
distribution slightly better, but it adds a correction term
``sum(x) * zero_point`` per output channel, which is a second reduction over the
activations in the hot path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..layers.linear import LinearBase
from ..utils import get_logger
from .base import QuantMethodBase
from .calibrate import worth_quantizing
from .kernels import kernel_available, w8_gemm

logger = get_logger("quant.int8")

_INT8_MAX = 127.0
#: torch._int_mm rejects fewer rows than this.
_INT_MM_MIN_ROWS = 17


def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8. Returns ``(int8 weight, scale)``.

    The scale is computed and the rounding done in fp32 even when the weight is
    bf16: bf16 has 8 mantissa bits, so ``weight / scale`` rounded in bf16 cannot
    represent all 255 int8 levels and quietly throws away a bit or two of the
    range the quantizer just went to the trouble of measuring.
    """
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / _INT8_MAX
    q = (w / scale[:, None]).round().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale


class Int8WeightOnlyMethod(QuantMethodBase):
    """int8 weights, bf16 activations, dequantized inside the GEMM.

    ``calibrate`` decides per layer shape whether to use the fused kernel at
    all; a layer that measures slower keeps its original weight and its original
    GEMM. That means this method genuinely does nothing to some layers, which is
    the point -- see the table in ``kernels.py``.
    """

    name = "int8_weight_only"

    def __init__(self, calibrate: bool = True, calibration_batch: int = 16) -> None:
        available, reason = kernel_available()
        if not available:
            raise RuntimeError(f"int8 weight-only needs the fused Triton kernel: {reason}")
        self.calibrate = calibrate
        self.calibration_batch = calibration_batch
        self.quantized = 0
        self.declined = 0

    def _profitable(self, layer: LinearBase) -> bool:
        if not self.calibrate:
            return True
        if not torch.cuda.is_available() or layer.weight.device.type != "cuda":
            return False
        weight = layer.weight
        rows = max(1, self.calibration_batch)
        x = torch.zeros(rows, layer.in_features, device=weight.device, dtype=weight.dtype)
        q, scale = quantize_weight_per_channel(weight)
        scale = scale.to(weight.dtype)
        return worth_quantizing(
            lambda: w8_gemm(x, q, scale, None),
            lambda: F.linear(x, weight, None),
            key=("int8_wo", layer.out_features, layer.in_features, rows, str(weight.dtype)),
        )

    def process_weights(self, layer: LinearBase) -> None:
        if not self._profitable(layer):
            self.declined += 1
            # Leaving quant_method unset is what keeps the layer on F.linear.
            layer.register_buffer("int8_weight", None, persistent=False)
            return
        q, scale = quantize_weight_per_channel(layer.weight)
        # Buffers rather than plain attributes so they follow module.to() and
        # are counted by the byte accounting that reports what this method saved.
        layer.register_buffer("int8_weight", q, persistent=False)
        layer.register_buffer("int8_scale", scale.to(layer.weight.dtype), persistent=False)
        # Drop the full-precision weight: keeping both would have the run report
        # a memory saving it did not make.
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device), requires_grad=False
        )
        self.quantized += 1

    def applies_to(self, layer: LinearBase) -> bool:
        return getattr(layer, "int8_weight", None) is not None

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return w8_gemm(x, layer.int8_weight, layer.int8_scale, bias)  # type: ignore[attr-defined]


class Int8Method(QuantMethodBase):
    """W8A8: per-token activation scales, per-channel weight scales, int8 GEMM."""

    name = "int8"

    def process_weights(self, layer: LinearBase) -> None:
        q, scale = quantize_weight_per_channel(layer.weight)
        # _int_mm takes B as [K, N], so the transpose is materialised once here
        # rather than per call.
        layer.register_buffer("int8_weight", q.t().contiguous(), persistent=False)
        layer.register_buffer("int8_scale", scale, persistent=False)
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device), requires_grad=False
        )

    def applies_to(self, layer: LinearBase) -> bool:
        return getattr(layer, "int8_weight", None) is not None

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scale_x = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / _INT8_MAX
        q = (flat / scale_x).round().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)

        rows = q.shape[0]
        padded = rows < _INT_MM_MIN_ROWS
        if padded:
            q = torch.cat([q, q.new_zeros(_INT_MM_MIN_ROWS - rows, q.shape[1])], dim=0)
        acc = torch._int_mm(q, layer.int8_weight)  # type: ignore[attr-defined]
        if padded:
            acc = acc[:rows]

        out = acc.float() * scale_x.float() * layer.int8_scale.reshape(1, -1)  # type: ignore[attr-defined]
        if bias is not None:
            out += bias.float()
        return out.to(x.dtype).reshape(*shape[:-1], layer.out_features)
