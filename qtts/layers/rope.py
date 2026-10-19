"""Rotary position embeddings.

All three transformers in Qwen3-TTS use the same rotate-half (GPT-NeoX) layout
and differ only in ``head_dim`` and ``theta``, so one table serves all of them.

A note on the talker, because its config looks like it needs more than this:
``talker_config.rope_scaling`` declares ``mrope_section = [24, 20, 20]`` and
``interleaved = true``. mrope assigns a different position row to the temporal,
height and width axes, but Qwen3-TTS sets ``image_token_id`` and
``video_token_id`` to ``-1`` precisely so that nothing in a TTS prompt ever
matches a vision token. With no vision spans, all three rows hold the same
value at every position, and mrope reduces exactly to the single-axis rotation
below. ``tests/test_parity_talker.py`` checks this against the reference
implementation rather than leaving it as an argument.
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin table, indexed by absolute position.

    The table is built in fp32 and cast on read: at theta = 1e6 and 2048
    positions the angles are large enough that computing them in bf16 loses
    bits that matter for the later positions.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        positions = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype | None = None):
        """``position_ids`` is ``[batch, seq]``; returns cos/sin of that shape."""
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin
