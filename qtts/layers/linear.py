"""Linear layers with a single quantization hook.

Every matmul in the talker, the code predictor and the codec decoder goes
through :class:`LinearBase`, so ``qtts.quant`` can swap the GEMM for a
quantized one by assigning ``layer.quant_method`` after the checkpoint is
loaded. Nothing in the models knows about quantization.

:class:`MergedLinear` exists for the code predictor. It re-prefills a
17-token sequence once per residual codebook, 15 times per audio frame, so the
per-call launch overhead dominates and folding q/k/v into one GEMM (and
gate/up into another) removes four launches per layer per step. The partition
list is what lets the loader write separate checkpoint tensors into slices of
one weight.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


class QuantMethod(Protocol):
    name: str

    def process_weights(self, layer: LinearBase) -> None: ...

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor: ...


class LinearBase(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)
        self.quant_method: QuantMethod | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, self.bias)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        method = self.quant_method.name if self.quant_method is not None else "none"
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, quant={method}"


class MergedLinear(LinearBase):
    """One weight holding several output partitions concatenated along dim 0."""

    def __init__(
        self,
        in_features: int,
        partitions: list[int],
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(in_features, sum(partitions), bias=bias, dtype=dtype, device=device)
        self.partitions = list(partitions)
        offsets = [0]
        for size in partitions:
            offsets.append(offsets[-1] + size)
        self.offsets = offsets

    def load_partition(self, index: int, tensor: torch.Tensor) -> None:
        start, end = self.offsets[index], self.offsets[index + 1]
        if tuple(tensor.shape) != (end - start, self.in_features):
            raise ValueError(
                f"partition {index} expects shape {(end - start, self.in_features)}, got {tuple(tensor.shape)}"
            )
        self.weight.data[start:end].copy_(tensor)

    def load_bias_partition(self, index: int, tensor: torch.Tensor) -> None:
        if self.bias is None:
            raise ValueError("layer has no bias")
        start, end = self.offsets[index], self.offsets[index + 1]
        self.bias.data[start:end].copy_(tensor)

    def split(self, out: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(out.split(self.partitions, dim=-1))


def iter_linears(module: nn.Module) -> list[tuple[str, LinearBase]]:
    return [(name, m) for name, m in module.named_modules() if isinstance(m, LinearBase)]
