"""Quantization: one hook on ``LinearBase``, applied after the weights load."""

from .base import apply_quantization, build_method, fp8_supported, quantize_module

__all__ = ["apply_quantization", "build_method", "fp8_supported", "quantize_module"]
