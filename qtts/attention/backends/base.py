"""Attention backend interface.

A backend takes ``q [B, Hq, S, D]`` and ``k/v [B, Hkv, T, D]`` and returns
``[B, Hq, S, D]``. Grouped-query attention is the backend's problem, not the
caller's.

``supports`` is the part that matters. A backend that cannot express a request
must say so instead of silently computing something else: FlashAttention has no
dense-mask entry point, so an arbitrary key-padding mask or a mid-sequence
sliding window has to go to SDPA. Declining is how a configuration avoids
looking like it applied when it did not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    name = "base"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def supports(
        cls,
        *,
        causal: bool,
        has_key_padding: bool,
        sliding_window: int | None,
        dtype: torch.dtype | None = None,
    ) -> bool:
        return True

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
        causal: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``attn_mask`` is boolean, ``True`` meaning "this key participates"."""


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq, dim = x.shape
    return x[:, :, None].expand(batch, heads, n_rep, seq, dim).reshape(batch, heads * n_rep, seq, dim)
