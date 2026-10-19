"""The fused RMSNorm against the arithmetic it replaces.

The fused kernel is worth a third of a graph-replayed decode step (17.6 ms to
11.9 ms), so it is worth knowing exactly how far it moves the numbers. In bf16
it lands within one representable step of the eager path,
because rounding to bf16 on the way out swamps the difference in the reduction.
In fp32 it is about two ULP away, which is what the parity suite has to work
around -- see ``test_parity.test_fused_norm_diverges_only_at_a_tie``.
"""

from __future__ import annotations

import pytest
import torch

from qtts.layers.norm import _HAS_TRITON, RMSNorm

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _HAS_TRITON, reason="needs triton"),
]

#: Every width the model actually uses: head_dim, codec pre-transformer,
#: code predictor hidden, talker hidden.
WIDTHS = [128, 512, 1024, 2048]


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """float64, so it is the answer rather than another implementation."""
    x64 = x.double()
    return x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * weight.double()


def _norm(width: int, dtype: torch.dtype, backend: str) -> RMSNorm:
    module = RMSNorm(width, eps=1e-6, dtype=dtype, backend=backend).cuda()
    generator = torch.Generator(device="cuda").manual_seed(width)
    module.weight.data.normal_(1.0, 0.1, generator=generator)
    return module


def _float_order(values: torch.Tensor) -> torch.Tensor:
    """bf16 values as integers that increase in float order.

    IEEE bit patterns are already monotonic within a sign, and negatives run
    backwards, so flipping them gives a total order in which adjacent
    representable values are adjacent integers. Counting *steps* rather than
    applying a relative tolerance is what makes this test meaningful near zero,
    where a cancelling sum leaves a result small enough that one step of
    rounding is a large fraction of it.
    """
    bits = values.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


@pytest.mark.parametrize("width", WIDTHS)
def test_fused_is_within_one_step_in_bf16(width: int) -> None:
    fused, eager = _norm(width, torch.bfloat16, "fused"), _norm(width, torch.bfloat16, "eager")
    eager.weight.data.copy_(fused.weight.data)
    generator = torch.Generator(device="cuda").manual_seed(width)
    x = torch.randn(16, width, device="cuda", dtype=torch.bfloat16, generator=generator) * 3
    with torch.inference_mode():
        ours, theirs = fused(x), eager(x)

    steps = (_float_order(ours) - _float_order(theirs)).abs()
    over = int((steps > 1).sum())
    assert over == 0, (
        f"{over}/{steps.numel()} outputs are more than one representable bf16 value apart "
        f"(worst {int(steps.max())} steps); that is a different answer, not a different rounding"
    )


@pytest.mark.parametrize("width", WIDTHS)
def test_fused_is_within_a_few_ulp_in_fp32(width: int) -> None:
    fused, eager = _norm(width, torch.float32, "fused"), _norm(width, torch.float32, "eager")
    eager.weight.data.copy_(fused.weight.data)
    generator = torch.Generator(device="cuda").manual_seed(width)
    x = torch.randn(16, width, device="cuda", dtype=torch.float32, generator=generator) * 3
    with torch.inference_mode():
        ours, theirs = fused(x), eager(x)
    truth = _reference(x, fused.weight, fused.eps)

    scale = truth.abs().max().item()
    fused_error = (ours.double() - truth).abs().max().item() / scale
    eager_error = (theirs.double() - truth).abs().max().item() / scale
    # Neither is the more accurate one; they round differently. The claim worth
    # asserting is that the fused kernel is no further from the f64 answer than
    # the implementation it replaces, to within a factor of two.
    assert fused_error < 1e-6, f"fused is {fused_error:.2e} from the f64 answer"
    assert fused_error < 2 * max(eager_error, 1e-9), (
        f"fused {fused_error:.2e} vs eager {eager_error:.2e}: the fused kernel should not be "
        "meaningfully less accurate, only differently rounded"
    )


def test_backend_is_validated() -> None:
    with pytest.raises(ValueError, match="fused|eager"):
        RMSNorm(128, backend="triton")


def test_eager_backend_does_not_use_the_kernel() -> None:
    """``--rmsnorm-backend eager`` has to actually take the eager path.

    It exists so the parity suite can compare against torch's own reduction
    order; a flag that silently kept using the kernel would make those tests
    pass for the wrong reason.
    """
    module = RMSNorm(128, backend="eager").cuda()
    assert not module.fused
    x = torch.randn(4, 128, device="cuda")
    with torch.inference_mode():
        assert torch.equal(module(x), module._eager_forward(x))
