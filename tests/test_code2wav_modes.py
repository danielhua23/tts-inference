"""What the three Code2Wav decode paths cost in accuracy.

These numbers are the reason the streaming knobs are worth benchmarking at
all, so they are pinned here rather than left to be rediscovered. Run with a
checkpoint:

    QTTS_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice pytest tests/test_code2wav_modes.py -v

The interesting result is the gap between ``decode_window`` and
``chunked_decode``. Both are "streaming Code2Wav"; one is exact and one is off
by nearly full scale.
"""

from __future__ import annotations

import os

import pytest
import torch

from qtts.models.code2wav import Qwen3TTSCode2Wav
from qtts.models.hf_config import Qwen3TTSHFConfig
from qtts.models.loader import load_code2wav_weights

MODEL = os.environ.get("QTTS_MODEL", "")
pytestmark = [
    pytest.mark.skipif(not MODEL, reason="set QTTS_MODEL to a Qwen3-TTS checkpoint"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

_FRAMES = 300


def _build(dtype: torch.dtype) -> Qwen3TTSCode2Wav:
    config = Qwen3TTSHFConfig.from_pretrained(MODEL)
    model = Qwen3TTSCode2Wav(config.code2wav, attention_backend="torch_sdpa", snake_backend="eager")
    load_code2wav_weights(model, MODEL, dtype)
    model = model.to(device="cuda", dtype=dtype).eval()
    model.precompute_caches()
    return model


@pytest.fixture(scope="module")
def fp32_model() -> Qwen3TTSCode2Wav:
    return _build(torch.float32)


def _codes(model: Qwen3TTSCode2Wav, frames: int = _FRAMES) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(
        1, model.config.codebook_size, (1, model.config.num_quantizers, frames),
        device="cuda", generator=generator,
    )


def _stream(model: Qwen3TTSCode2Wav, codes: torch.Tensor, chunk: int) -> torch.Tensor:
    state = model.new_state()
    total = int(codes.shape[-1])
    return torch.cat([model.decode_window(codes[..., s : s + chunk], state) for s in range(0, total, chunk)])


def test_frame_length_matches_upsample(fp32_model: Qwen3TTSCode2Wav) -> None:
    codes = _codes(fp32_model, 40)
    audio = fp32_model.decode_exact(codes)
    assert audio.shape == (1, 40 * fp32_model.total_upsample)
    assert audio.abs().max() <= 1.0, "decoder output must be clamped to [-1, 1]"


@pytest.mark.parametrize("chunk", [5, 25, 50])
def test_decode_window_is_exact_in_fp32(fp32_model: Qwen3TTSCode2Wav, chunk: int) -> None:
    """Streaming with the K/V cache matches whole-utterance decoding.

    The tolerance is floating-point headroom, not slack for a wrong answer. In
    float64 the first chunks agree to 1e-16 and later ones to ~5e-8, the
    difference being that once the cache saturates the attention reduction runs
    over a fixed 72 keys instead of a growing prefix. The downstream
    convolution stack then amplifies that by about 70x.
    """
    codes = _codes(fp32_model)
    reference = fp32_model.decode_exact(codes)[0]
    streamed = _stream(fp32_model, codes, chunk)
    assert streamed.shape == reference.shape
    assert (streamed - reference).abs().max() < 5e-2


def test_chunked_decode_is_not_exact(fp32_model: Qwen3TTSCode2Wav) -> None:
    """The stateless path trades audio fidelity, not just memory.

    ``sliding_window = 72`` invites the reading that 71 frames of left context
    suffice. It does not: the window bounds each of the 8 layers, so the true
    reach is 568 frames. Error at the usual ``left_context = 25`` is close to
    full scale on a signal bounded by 1.0, and more context helps only slowly.
    """
    codes = _codes(fp32_model)
    reference = fp32_model.decode_exact(codes)[0]
    errors = {
        lc: float((fp32_model.chunked_decode(codes, 25, lc)[0] - reference).abs().max())
        for lc in (25, 71, 120)
    }
    assert errors[25] > 0.5, f"expected a large error, got {errors}"
    assert errors[120] < errors[71] < errors[25], f"more context should help monotonically: {errors}"

    # The cached path stays inside fp32 headroom while the stateless path, even
    # at nearly five times the usual context, does not. The comparison is
    # against the band rather than as a ratio between the two: the streaming
    # error comes from a handful of ULP at the bottom of an amplifying
    # convolution stack, so its exact magnitude moves by several times when any
    # kernel underneath changes, while which side of the band it falls on does
    # not.
    exact = float((_stream(fp32_model, codes, 25) - reference).abs().max())
    assert exact < 5e-2, f"the cached path should be exact to fp32 headroom, got {exact:.2e}"
    assert errors[120] > 5e-2, (
        f"stateless decoding at 120 frames of context ({errors[120]:.2e}) is inside fp32 headroom, "
        "which would mean the approximation is free after all"
    )


def test_receptive_field_uses_stacked_window(fp32_model: Qwen3TTSCode2Wav) -> None:
    config = fp32_model.config
    per_layer = config.sliding_window - 1
    assert fp32_model.receptive_field_frames > config.num_hidden_layers * per_layer, (
        "receptive field must account for every layer's window plus the convolution context; "
        "using the per-layer figure alone is the mistake this guards"
    )


def test_state_size_is_reported_accurately(fp32_model: Qwen3TTSCode2Wav) -> None:
    """The scheduler sizes concurrency from the estimate, so it must not lie."""
    codes = _codes(fp32_model, 200)
    state = fp32_model.new_state()
    for start in range(0, 200, 25):
        fp32_model.decode_window(codes[..., start : start + 25], state)
    assert state.nbytes() == pytest.approx(fp32_model.state_bytes_estimate(), rel=0.05)
