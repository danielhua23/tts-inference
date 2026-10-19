"""FlashAttention backend.

Restricted on purpose. ``flash_attn_func`` takes causality and a symmetric
window as flags, but it has no dense-mask argument, so a batch with per-request
key padding cannot be expressed here without switching to the varlen entry
point and packing the batch. Rather than pack, this backend declines those
shapes and the caller falls back to SDPA. That keeps the padded-batch talker
decode honest: it is either flash on a uniform batch or SDPA on a masked one,
never flash on a mask it ignored.
"""

from __future__ import annotations

import torch

from .base import AttentionBackend

_HAVE_FLASH: bool | None = None
_flash_attn_func = None


def _load() -> bool:
    global _HAVE_FLASH, _flash_attn_func
    if _HAVE_FLASH is not None:
        return _HAVE_FLASH
    try:
        from flash_attn import flash_attn_func

        _flash_attn_func = flash_attn_func
        _HAVE_FLASH = True
    except ImportError:
        _HAVE_FLASH = False
    return _HAVE_FLASH


class FlashAttentionBackend(AttentionBackend):
    name = "flash_attn"

    @classmethod
    def is_available(cls) -> bool:
        return _load()

    #: flash_attn's kernels are half-precision only. The codec decoder runs in
    #: float32 and parity tests run in float64, so this is not a rare path --
    #: without the check those callers get a kernel-level type error instead of
    #: the SDPA fallback they should have received.
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    @classmethod
    def supports(
        cls,
        *,
        causal: bool,
        has_key_padding: bool,
        sliding_window: int | None,
        dtype: torch.dtype | None = None,
    ) -> bool:
        if dtype is not None and dtype not in cls.SUPPORTED_DTYPES:
            return False
        return _load() and not has_key_padding

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
        causal: bool = False,
        attn_mask: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError("flash_attn cannot take a dense mask; the dispatcher should have used SDPA")
        # flash wants [B, S, H, D]; the rest of the codebase uses [B, H, S, D].
        out = _flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            softmax_scale=scale,
            causal=causal,
            window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
        )
        return out.transpose(1, 2)
