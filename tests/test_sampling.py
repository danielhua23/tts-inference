"""Sampling helpers, tested without a checkpoint.

Both autoregressive axes share these, so a fault here is a fault everywhere.
The Gumbel test is not academic: an operator-precedence slip made the noise NaN
for every element, which turns ``argmax`` into "always return index 0" -- a
plausible token id, no exception, and audio that is silence or noise depending
on what index 0 means. Distributional tests catch that; shape tests do not.
"""

from __future__ import annotations

import math

import pytest
import torch

from qtts.engine.talker_runner import apply_repetition_penalty
from qtts.models.code_predictor import apply_top_k, sample_gumbel


def test_gumbel_noise_is_finite() -> None:
    logits = torch.zeros(4, 128)
    for seed in range(8):
        generator = torch.Generator().manual_seed(seed)
        drawn = sample_gumbel(logits, generator)
        assert drawn.shape == (4, 1)
        assert drawn.min() >= 0 and drawn.max() < 128


def test_gumbel_respects_masked_entries() -> None:
    """``-inf`` logits must never be selected, however the noise falls."""
    logits = torch.full((16, 64), float("-inf"))
    logits[:, 7] = 0.0
    logits[:, 30] = -1.0
    drawn = sample_gumbel(logits, torch.Generator().manual_seed(0))
    assert set(drawn.flatten().tolist()) <= {7, 30}


def test_gumbel_matches_softmax_distribution() -> None:
    """Gumbel-max is only a valid substitute if the distribution matches."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    expected = torch.softmax(logits, dim=-1)[0]
    generator = torch.Generator().manual_seed(1234)
    trials = 40_000
    draws = sample_gumbel(logits.expand(trials, -1), generator)
    observed = torch.bincount(draws.flatten(), minlength=4).float() / trials
    # Three standard errors of a binomial at this sample size.
    tolerance = 3 * math.sqrt(0.25 / trials) + 0.005
    assert torch.allclose(observed, expected, atol=tolerance), f"{observed} vs {expected}"


def test_top_k_keeps_exactly_k() -> None:
    logits = torch.randn(3, 100, generator=torch.Generator().manual_seed(0))
    filtered = apply_top_k(logits, 10)
    assert (torch.isfinite(filtered).sum(dim=-1) == 10).all()
    top = logits.topk(10, dim=-1).indices
    assert torch.isfinite(filtered.gather(1, top)).all()


@pytest.mark.parametrize("k", [0, -1, 1000])
def test_top_k_passthrough(k: int) -> None:
    logits = torch.randn(2, 50)
    assert torch.equal(apply_top_k(logits, k), logits)


def test_repetition_penalty_only_touches_seen_ids() -> None:
    logits = torch.tensor([[1.0, 2.0, -3.0, 4.0]])
    seen = torch.tensor([[1, 2]])
    valid = torch.tensor([[True, True]])
    out = apply_repetition_penalty(logits, seen, valid, 2.0)
    # Positive scores divide, negative ones multiply: both move toward -inf.
    assert out[0, 1].item() == pytest.approx(1.0)
    assert out[0, 2].item() == pytest.approx(-6.0)
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[0, 3].item() == pytest.approx(4.0)


def test_repetition_penalty_ignores_padded_slots() -> None:
    """An empty or partly filled window must leave the logits alone.

    Early frames have fewer past ids than the batch's widest row, and those
    padded slots all point at id 0. Letting them through would penalise id 0 on
    every step of every request.
    """
    logits = torch.tensor([[5.0, 2.0, 3.0]])
    seen = torch.tensor([[0, 0]])
    valid = torch.tensor([[False, False]])
    assert torch.equal(apply_repetition_penalty(logits, seen, valid, 1.05), logits)
