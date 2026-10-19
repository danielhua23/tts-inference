"""Causal convolutions for the codec decoder.

"Causal" here means left-padded: a frame's output may not depend on frames
after it, because the decoder is run incrementally on chunks and a frame
already emitted must not change when the next chunk arrives. Every conv in
this file pads on the left by ``(kernel-1)*dilation - stride + 1`` and trims
transposed-conv output on the right for the same reason.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            dtype=dtype,
        )
        self.conv.weight.requires_grad_(False)
        if self.conv.bias is not None:
            self.conv.bias.requires_grad_(False)
        self.stride = stride
        self.effective_kernel = (kernel_size - 1) * dilation + 1
        self.left_pad = self.effective_kernel - stride
        self._extra_pad: dict[int, int] = {}

    def _right_pad_for(self, length: int) -> int:
        """Pad so a strided conv consumes the tail instead of dropping it.

        Memoized by input length: the set of lengths a decode loop sees is
        small and fixed, so after the first few chunks this replaces a
        ``math.ceil`` per call, which matters under CUDA graph replay where the
        Python side is the cost being removed.
        """
        cached = self._extra_pad.get(length)
        if cached is not None:
            return cached
        n_frames = (length - self.effective_kernel + self.left_pad) / self.stride + 1
        ideal = (math.ceil(n_frames) - 1) * self.stride + (self.effective_kernel - self.left_pad)
        extra = ideal - length
        self._extra_pad[length] = extra
        return extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, self._right_pad_for(x.shape[-1])), mode="constant", value=0.0)
        return self.conv(x).contiguous()


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dtype=dtype)
        self.conv.weight.requires_grad_(False)
        if self.conv.bias is not None:
            self.conv.bias.requires_grad_(False)
        self.right_trim = int(kernel_size - stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.right_trim > 0:
            x = x[..., : x.shape[-1] - self.right_trim]
        return x.contiguous()


class ConvNeXtBlock(nn.Module):
    """Depthwise 7-tap conv, LayerNorm, pointwise expansion, learnt residual gain."""

    def __init__(self, dim: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim, dtype=dtype)
        self.norm = nn.LayerNorm(dim, eps=1e-6, dtype=dtype)
        self.pwconv1 = nn.Linear(dim, 4 * dim, dtype=dtype)
        self.pwconv2 = nn.Linear(4 * dim, dim, dtype=dtype)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim, dtype=dtype), requires_grad=False)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv2(F.gelu(self.pwconv1(x)))
        x = self.gamma * x
        return residual + x.permute(0, 2, 1)
