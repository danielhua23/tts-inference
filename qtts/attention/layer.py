"""Backend dispatch for attention."""

from __future__ import annotations

import torch

from ..utils import get_logger
from .backends.base import AttentionBackend
from .backends.flash_attn import FlashAttentionBackend
from .backends.torch_sdpa import TorchSDPABackend, build_mask

logger = get_logger("attention")

_BACKENDS: dict[str, type[AttentionBackend]] = {
    "torch_sdpa": TorchSDPABackend,
    "flash_attn": FlashAttentionBackend,
}
_INSTANCES: dict[str, AttentionBackend] = {}
_WARNED: set[str] = set()


def available_backends() -> list[str]:
    return [name for name, cls in _BACKENDS.items() if cls.is_available()]


def _instance(name: str) -> AttentionBackend:
    if name not in _INSTANCES:
        _INSTANCES[name] = _BACKENDS[name]()
    return _INSTANCES[name]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    backend: str = "auto",
    causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """``q [B, Hq, S, D]``, ``k/v [B, Hkv, T, D]`` -> ``[B, Hq, S, D]``."""
    requested = backend
    if requested == "auto":
        requested = "flash_attn" if FlashAttentionBackend.is_available() else "torch_sdpa"

    cls = _BACKENDS.get(requested)
    if cls is None:
        raise ValueError(f"unknown attention backend {requested!r}, pick one of {sorted(_BACKENDS)}")

    if not cls.is_available() or not cls.supports(
        causal=causal,
        has_key_padding=attn_mask is not None,
        sliding_window=sliding_window,
        dtype=q.dtype,
    ):
        if backend != "auto" and requested not in _WARNED:
            _WARNED.add(requested)
            logger.info(
                "attention backend %s does not apply here (mask=%s, window=%s, dtype=%s); "
                "using torch_sdpa for these calls",
                requested,
                attn_mask is not None,
                sliding_window,
                q.dtype,
            )
        requested = "torch_sdpa"
        cls = TorchSDPABackend

    if requested == "flash_attn":
        return _instance(requested).forward(
            q, k, v, scale=scale, causal=causal, attn_mask=None, sliding_window=sliding_window
        )
    return _instance(requested).forward(q, k, v, scale=scale, causal=causal, attn_mask=attn_mask)


__all__ = ["attention", "build_mask", "available_backends"]
