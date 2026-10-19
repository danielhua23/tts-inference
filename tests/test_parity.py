"""Compare this runtime against the official ``qwen-tts`` implementation.

Two steps, because the reference needs transformers 4.57 and the container has
5.x:

    pip install --target .ref-env --no-deps qwen-tts==0.1.1 transformers==4.57.3 ...
    PYTHONPATH=.ref-env python scripts/dump_reference.py --model $QTTS_MODEL --out /tmp/reference.pt
    QTTS_MODEL=... QTTS_REFERENCE=/tmp/reference.pt pytest tests/test_parity.py -v

The tests go in dependency order, so the first failure points at the layer
that broke rather than at the audio:

1. ``prompt_embeds`` -- prompt assembly.
2. ``first_logits`` -- talker forward, no sampling involved.
3. ``codes`` -- the greedy decode loop, including the code predictor.
4. ``audio`` -- the codec decoder.

Everything is greedy. Under sampling, two correct implementations diverge on
the first frame and stay diverged, which tells you nothing.
"""

from __future__ import annotations

import os

import pytest
import torch

from qtts.config import EngineConfig, SamplingConfig
from qtts.engine.request import RequestTracker, TTSRequest
from qtts.engine.talker_runner import TalkerRunner
from qtts.models.code2wav import Qwen3TTSCode2Wav
from qtts.models.code_predictor import CodePredictor
from qtts.models.hf_config import Qwen3TTSHFConfig
from qtts.models.loader import load_code2wav_weights, load_talker_weights
from qtts.models.prompt import PromptBuilder, load_tokenizer
from qtts.models.talker import Qwen3TTSTalker

MODEL = os.environ.get("QTTS_MODEL", "")
REFERENCE = os.environ.get("QTTS_REFERENCE", "")

pytestmark = [
    pytest.mark.skipif(not MODEL, reason="set QTTS_MODEL to a Qwen3-TTS checkpoint"),
    pytest.mark.skipif(
        not REFERENCE or not os.path.exists(REFERENCE),
        reason="set QTTS_REFERENCE to the output of scripts/dump_reference.py",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

#: The reference dump runs in float32, so parity runs in float32 too. bfloat16
#: is the serving default and is checked separately, against this runtime's own
#: float32 output rather than against the reference.
_DTYPE = torch.float32

#: Parity runs on the eager RMSNorm, which reproduces torch's reduction order
#: bit for bit. The fused kernel differs by about two ULP of fp32, and greedy
#: decoding takes an argmax over ~2000 near-tied logits sixteen times per frame,
#: so a perturbation at that scale eventually flips a tie and looks like a
#: divergence. That is a property of comparing argmaxes, not of the kernel:
#: ``tests/test_norm.py`` pins the two paths against each other directly, and
#: ``test_fused_norm_changes_only_near_ties`` below shows what the difference
#: costs end to end.
_NORM_BACKEND = "eager"


@pytest.fixture(scope="module")
def reference() -> dict:
    return torch.load(REFERENCE, map_location="cpu", weights_only=False)


def _build_runtime(reference: dict, norm_backend: str) -> dict:
    config = Qwen3TTSHFConfig.from_pretrained(MODEL)
    device = torch.device("cuda")
    engine = EngineConfig(model=MODEL)
    engine.scheduler.max_model_len = 1024
    engine.talker.rmsnorm_backend = norm_backend

    talker = Qwen3TTSTalker(
        config, runtime=engine.talker, dtype=_DTYPE, device=device, max_num_seqs=1, max_model_len=1024
    ).to(device).eval()
    predictor = CodePredictor(
        config.talker.code_predictor, config.talker.hidden_size,
        runtime=engine.talker, dtype=_DTYPE, device=device,
    ).to(device).eval()
    load_talker_weights(talker, predictor, MODEL, _DTYPE)
    talker.init_runtime_buffers()

    builder = PromptBuilder(config, talker, load_tokenizer(MODEL))
    prompt = builder.build_prompt(
        text=reference["text"],
        speaker=reference["speaker"],
        language=reference["language"],
        instruct=reference["instruct"],
    )
    return {"config": config, "talker": talker, "predictor": predictor, "prompt": prompt, "engine": engine}


@pytest.fixture(scope="module")
def runtime(reference: dict) -> dict:
    return _build_runtime(reference, _NORM_BACKEND)


def _greedy_decode(runtime: dict, reference: dict) -> torch.Tensor:
    """Greedy decode for as many frames as the reference produced."""
    engine: EngineConfig = runtime["engine"]
    sampling = SamplingConfig(
        do_sample=False, subtalker_do_sample=False, repetition_penalty=1.0, temperature=0.0
    )
    runner = TalkerRunner(runtime["talker"], runtime["predictor"], engine, sampling)
    prompt = runtime["prompt"]
    request = TTSRequest(text=reference["text"], speaker=reference["speaker"])
    tracker = RequestTracker(request, slot=0, prompt_len=prompt.embeds.shape[0])
    runner.prefill(tracker, prompt)
    budget = int(reference["codes"].shape[0])
    while tracker.state.value == "decoding" and tracker.frames_generated < budget:
        runner.decode_step([tracker])
    assert tracker.frames, "greedy decode produced no frames"
    return torch.cat(tracker.frames, dim=0).cpu()


@pytest.fixture(scope="module")
def greedy_codes(runtime: dict, reference: dict) -> torch.Tensor:
    return _greedy_decode(runtime, reference)


def test_prompt_embeds_match(runtime: dict, reference: dict) -> None:
    """Prompt assembly: token order, the dual-stream sum, the codec shift."""
    ours = runtime["prompt"].embeds.float().cpu()
    theirs = reference["prompt_embeds"]
    assert ours.shape == theirs.shape, (
        f"prompt length differs: {tuple(ours.shape)} vs {tuple(theirs.shape)}. A length mismatch is a "
        "layout bug (a missing control tag or an off-by-one in the codec shift), not a numerical one"
    )
    difference = (ours - theirs).abs().max().item()
    assert difference < 1e-3, f"max |diff| {difference:.3e}; per-position: {(ours - theirs).abs().amax(dim=1)}"


def test_first_logits_match(runtime: dict, reference: dict) -> None:
    """Talker forward pass, isolated from sampling."""
    hidden = runtime["talker"].prefill(runtime["prompt"].embeds, 0)
    ours = runtime["talker"].codec_head(hidden).float().cpu()
    theirs = reference["first_logits"]
    assert ours.shape == theirs.shape
    assert ours.argmax().item() == theirs.argmax().item(), "the greedy first frame already disagrees"
    assert (ours - theirs).abs().max().item() < 0.05


def test_greedy_codes_match(greedy_codes: torch.Tensor, reference: dict) -> None:
    """The decode loop and the code predictor, frame by frame.

    Compared over the shorter of the two sequences: a run that stops early on
    EOS is itself a divergence, reported as such rather than as a shape error.
    """
    theirs = reference["codes"].cpu()
    ours = greedy_codes
    common = min(ours.shape[0], theirs.shape[0])
    assert common > 0
    matching = (ours[:common] == theirs[:common]).all(dim=1)
    first_bad = int((~matching).nonzero()[0]) if not matching.all() else common

    assert first_bad >= common, (
        f"frames 0..{first_bad - 1} match, frame {first_bad} does not:\n"
        f"  ours   {ours[first_bad].tolist()}\n"
        f"  theirs {theirs[first_bad].tolist()}"
    )
    assert ours.shape[0] == theirs.shape[0], (
        f"all {common} shared frames match but the lengths differ "
        f"({ours.shape[0]} vs {theirs.shape[0]}): the stop condition disagrees"
    )


def test_fused_norm_diverges_only_at_a_tie(reference: dict, greedy_codes: torch.Tensor) -> None:
    """What the fused RMSNorm costs against the bit-exact path, in codes.

    The fused kernel is ~40% of a decode step and about two ULP of fp32 away
    from torch's reduction order. Under greedy decoding that eventually changes
    the codes, and it cannot not: each frame takes 16 argmaxes over ~2000
    logits, some of which are near-tied, and once one flips the next frame is
    conditioned on a different code, so everything after it is a different
    (equally valid) sample. Counting how many codes differ in total therefore
    measures the length of the sequence, not the kernel.

    What does distinguish rounding from a bug is *how* the first divergence
    looks. Rounding flips one codebook at a near-tie, after a stretch of
    bit-identical frames. A bug -- a rotary position off by one, a mis-indexed
    cache -- is wrong at the first frame it touches and wrong in most of that
    frame's codebooks. This pins that shape.
    """
    fused = _greedy_decode(_build_runtime(reference, "fused"), reference)
    exact = greedy_codes
    common = min(fused.shape[0], exact.shape[0])
    assert common > 0

    differing = fused[:common] != exact[:common]
    if not bool(differing.any()):
        return  # the ties happened to fall the same way; nothing to assert

    first = int(differing.any(dim=1).nonzero()[0])
    codebooks = int(differing[first].sum())
    assert first >= 8, (
        f"the fused norm changes frame {first} already, after only {first} matching frames. "
        "Two ULP should survive longer than that; this looks like a wrong result rather than "
        "a different rounding"
    )
    assert codebooks <= 2, (
        f"the first divergence changes {codebooks} of {fused.shape[1]} codebooks in frame {first}:\n"
        f"  fused {fused[first].tolist()}\n"
        f"  exact {exact[first].tolist()}\n"
        "A tie flips one code; a whole frame changing means the frame was conditioned differently"
    )


def test_audio_matches(reference: dict, greedy_codes: torch.Tensor) -> None:
    """The codec decoder, fed the reference's own codes.

    Deliberately uses ``reference["codes"]`` rather than our own, so a talker
    divergence cannot be mistaken for a decoder one.
    """
    config = Qwen3TTSHFConfig.from_pretrained(MODEL)
    model = Qwen3TTSCode2Wav(config.code2wav, attention_backend="torch_sdpa", snake_backend="eager")
    load_code2wav_weights(model, MODEL, torch.float32)
    model = model.to(device="cuda", dtype=torch.float32).eval()
    model.precompute_caches()

    codes = reference["codes"].t().unsqueeze(0).cuda()  # [F, Q] -> [1, Q, F]
    ours = model.decode_exact(codes)[0].float().cpu()
    theirs = reference["audio"]
    assert ours.shape == theirs.shape, f"{tuple(ours.shape)} vs {tuple(theirs.shape)}"
    difference = (ours - theirs).abs().max().item()
    assert difference < 1e-2, f"max sample error {difference:.3e} on a signal bounded by 1.0"
