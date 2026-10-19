"""PyTorch scaled_dot_product_attention backend.

The fallback for everything, and the only backend that can take a dense mask.
GQA goes through ``enable_gqa`` rather than an explicit ``repeat_kv`` so the
key/value tensors are not materialized at query-head width.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionBackend


class TorchSDPABackend(AttentionBackend):
    name = "torch_sdpa"

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
        # is_causal and an explicit mask are mutually exclusive in SDPA; when a
        # mask is supplied the caller has already folded causality into it.
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )


def build_mask(
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_valid: torch.Tensor | None = None,
    sliding_window: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Boolean ``[B, 1, S, T]`` mask, ``True`` where the key participates.

    Positions are absolute, which is what makes this work for incremental
    decode: a chunk starting at frame 300 gets ``query_positions = 300..324``
    and keys ``0..324``, and the same comparison expresses both causality and
    the sliding window without the caller tracking offsets.
    """
    q_pos = query_positions.unsqueeze(-1)  # [B, S, 1]
    k_pos = key_positions.unsqueeze(1)  # [B, 1, T]
    mask = (k_pos <= q_pos) if causal else (k_pos == k_pos)
    mask = mask.expand(q_pos.shape[0], q_pos.shape[1], k_pos.shape[2])
    if sliding_window is not None and sliding_window > 0:
        mask = mask & (k_pos > q_pos - sliding_window)
    if key_valid is not None:
        mask = mask & key_valid.unsqueeze(1)
    # A fully masked row makes softmax produce NaN, not zeros, and the NaN then
    # spreads through the residual stream where its origin is unrecoverable.
    # Padded query rows are discarded by the caller, so point them at key 0.
    empty = ~mask.any(dim=-1, keepdim=True)
    return (mask | (empty & (k_pos == k_pos.amin(dim=-1, keepdim=True)))).unsqueeze(1)
