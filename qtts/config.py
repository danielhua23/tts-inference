"""Configuration objects for the qtts minimal Qwen3-TTS inference runtime.

The split mirrors the two stages of the model rather than the two stages of a
diffusion pipeline: :class:`TalkerConfig` covers the autoregressive stage that
emits codebook-0 per audio frame, :class:`Code2WavConfig` covers the codec
decoder that turns RVQ frames into a waveform, and :class:`SchedulerConfig`
covers what happens between them. Field names follow vllm-omni's so that a
measurement taken here lines up with a vllm-omni number without a translation
table.

Every field has a command-line flag named after it (``codec_chunk_frames``
becomes ``--codec-chunk-frames``); see :func:`build_arg_parser`.

Every field is also read by something. A knob that is accepted and ignored is
worse here than a missing one: it produces two benchmark rows that differ only
in their label, and the conclusion drawn from them is that the optimization
does not help.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}, pick one of {sorted(_DTYPES)}") from exc


@dataclass
class QuantConfig:
    """Weight/activation quantization of the linear layers.

    Unlike a diffusion DiT, the talker is a batch-1-to-batch-64 LLM decode loop
    and is memory-bound, so weight-only quantization is the variant expected to
    pay off here; W8A8 mainly helps once the batch is large enough for the GEMMs
    to be compute-bound. ``targets`` exists because the three submodels have
    very different profiles: the talker is 1.7B of decode-bound GEMMs, the code
    predictor re-prefills a 16-token sequence 15 times per frame, and Code2Wav
    is convolutional and runs in fp32.
    """

    method: Literal["none", "fp8", "fp8_weight_only", "int8", "int8_weight_only"] = "none"
    #: Which submodels to quantize. Code2Wav is deliberately not in the default:
    #: its weights load in fp32 (see Code2WavConfig.dtype) and quantizing a
    #: convolutional vocoder changes the waveform, not just the latency.
    targets: tuple[str, ...] = ("talker",)
    #: Layers whose name contains any of these keep full precision. The codec
    #: head maps to 3072 classes that are then masked down to ~2049 valid ids,
    #: so a quantization error there moves the sampled code, not just a logit.
    exclude_substrings: tuple[str, ...] = (
        "lm_head",
        "codec_head",
        "text_projection",
        "small_to_mtp_projection",
    )
    #: Time each distinct linear shape both ways at load and only replace the
    #: GEMM where the quantized one measures faster. On by default because
    #: applying a kernel to shapes it is slow on is how weight-only
    #: quantization ends up reported as "no speedup": for this model the same
    #: int8 kernel is 1.9x on the MLP up-projection and 0.6x on the down
    #: projection. Turn it off to measure the uniform version, or when the
    #: goal is the memory saving rather than the latency.
    calibrate: bool = True

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    @property
    def weight_only(self) -> bool:
        return self.method.endswith("_weight_only")

    def targets_submodel(self, name: str) -> bool:
        return self.enabled and name in self.targets

    def excludes(self, layer_name: str) -> bool:
        return any(s in layer_name for s in self.exclude_substrings)


@dataclass
class TalkerConfig:
    """Stage 0: the autoregressive talker plus its residual code predictor."""

    attention_backend: str = "auto"  # auto | torch_sdpa | flash_attn

    #: Fused Triton RMSNorm: the talker does 112 normalizations per step and the
    #: code predictor 300, each six kernels when written eagerly, on tensors of
    #: a few thousand elements. Worth a third of a graph-replayed step (17.6 ms
    #: to 11.9 ms) and 2% without graphs, because until the launch overhead is
    #: gone it is hiding underneath it -- which is the general shape of every
    #: fusion in this model. ``eager`` reproduces torch's reduction order bit
    #: for bit, which the parity suite needs and normal runs do not.
    rmsnorm_backend: str = "fused"  # fused | eager

    #: Replay one captured graph for the whole talker decode step. The largest
    #: single optimization in stage 0 by a wide margin: 67.0 ms per frame to
    #: 11.9 ms at batch 1, because a step launches ~2000 kernels for a few ms of
    #: arithmetic and the CPU cannot keep up. See qtts/compilation.py.
    cuda_graph: bool = False
    #: Batch sizes to capture. Requests are padded up to the next captured size.
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    #: Key lengths are rounded up to a multiple of this before choosing a graph.
    #: The cache grows one position per frame, so without bucketing every frame
    #: is a new shape and nothing is ever replayed. The cost is attending over
    #: up to bucket-1 masked positions.
    cuda_graph_key_len_bucket: int = 256

    compile_mode: str = "none"  # none | code_predictor | full

    #: The code predictor re-prefills a (Q+1)-token sequence once per residual
    #: codebook, so it has no KV cache to reuse; these are the two fusions that
    #: matter at that shape.
    fuse_qkv: bool = True
    fuse_gate_up: bool = True
    #: Runaway-generation guards (vllm-omni #6158). The talker can miss codec
    #: EOS entirely and run to max_tokens, which at 12.5 Hz is minutes of audio.
    max_frames: int = 2048
    #: Stop when the last N frames repeat the same codebook-0 id. 0 disables.
    repetition_guard_frames: int = 0

    def __post_init__(self) -> None:
        if self.compile_mode not in ("none", "code_predictor", "full"):
            raise ValueError(f"compile_mode must be none|code_predictor|full, got {self.compile_mode!r}")
        if self.rmsnorm_backend not in ("fused", "eager"):
            raise ValueError(f"rmsnorm_backend must be fused|eager, got {self.rmsnorm_backend!r}")


@dataclass
class Code2WavConfig:
    """Stage 1: the codec decoder.

    Two decode modes, and the difference between them is the whole point of the
    chunk knobs below:

    * ``stateless`` keeps no per-request state. Each chunk is re-decoded with
      ``codec_left_context_frames`` of history in front of it and that
      context's audio is discarded. The only mode that batches across
      requests, and therefore the throughput mode -- but it is *approximate*,
      by more than it looks. See ``models/code2wav.py``: the decoder's real
      reach is 568 frames, not the 71 that ``sliding_window`` suggests, so at
      the default 25 frames of context the peak sample error is close to full
      scale on a signal bounded by 1.0. It is a quality trade, not a free
      speedup.
    * ``stateful`` keeps a per-request K/V cache plus two small convolution
      caches, so a chunk costs work proportional to its own frames and the
      output matches whole-utterance decoding. Costs a few MB per request and
      cannot batch across requests.

    ``stateful`` is the default because a mode that changes the audio should be
    opted into.
    """

    mode: Literal["stateless", "stateful"] = "stateful"
    #: Code2Wav weights load in fp32 in vllm-omni. bf16 is offered because it
    #: is the obvious thing to try, not because it is known to be safe: the
    #: decoder runs ~20 serial convolution stages per frame and the accumulated
    #: rounding shows up as audible banding, so compare waveforms before
    #: trusting a speedup here.
    dtype: str = "float32"

    #: Frames per decode call, i.e. how much audio the consumer waits for. This
    #: is the connector-level chunk, not the decoder-internal one below.
    codec_chunk_frames: int = 25
    #: Frames of history prepended in stateless mode and discarded afterwards.
    codec_left_context_frames: int = 25
    #: A smaller first chunk trades steady-state throughput for time-to-first-
    #: audio; the ramp grows it back over the next few chunks.
    initial_codec_chunk_frames: int = 5
    codec_chunk_ramp: tuple[int, ...] = ()
    #: Pick the first chunk size from current load instead of using the fixed
    #: value: small under light load where latency is free, larger under load
    #: where a small first chunk only adds decoder calls everyone waits behind.
    #: Ignored when a ramp is set, since the ramp already states every size.
    dynamic_initial_chunk: bool = False

    snake_backend: str = "triton"  # triton | eager
    rmsnorm_backend: str = "fused"  # fused | eager
    #: Fold exp(alpha) and 1/(exp(beta)+eps) into frozen buffers once at load.
    snake_precompute_cache: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("stateless", "stateful"):
            raise ValueError(f"mode must be stateless|stateful, got {self.mode!r}")
        if self.snake_backend not in ("triton", "eager"):
            raise ValueError(f"snake_backend must be triton|eager, got {self.snake_backend!r}")
        if self.rmsnorm_backend not in ("fused", "eager"):
            raise ValueError(f"rmsnorm_backend must be fused|eager, got {self.rmsnorm_backend!r}")

    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)


@dataclass
class SchedulerConfig:
    """What happens between the two stages.

    The doc's throughput section is mostly about this object: total throughput
    and streaming continuity are different goals, and a scheduler that
    maximizes the first can starve individual streams badly enough to break
    playback (vllm-omni #2568).
    """

    max_num_seqs: int = 16
    #: Upper bound on talker KV per slot. At 12.5 Hz, 2048 frames is ~164 s of
    #: audio, and the prompt adds the text tokens in front of that.
    max_model_len: int = 4096

    #: ``sync`` runs the talker to completion and then decodes everything;
    #: ``async_chunk`` decodes each chunk as soon as its frames exist, which is
    #: what makes time-to-first-audio a separate number from total latency
    #: (vllm-omni #5011).
    pipeline_mode: Literal["sync", "async_chunk"] = "async_chunk"

    #: Which pending chunk stage 1 picks next, when more are queued than one
    #: call can take.
    #:
    #: * ``fifo`` — oldest chunk first. Fair, and the baseline.
    #: * ``deadline`` — the request with the least buffered audio first. A
    #:   listener only notices a gap once the audio already delivered runs
    #:   out, so this targets the metric that matters rather than the one that
    #:   is easy to compute (vllm-omni #2568).
    #: * ``round_robin`` — one chunk per request per turn, so no stream waits
    #:   behind another stream's backlog.
    chunk_policy: Literal["fifo", "deadline", "round_robin"] = "fifo"
    #: Cap on chunks batched into one stage-1 call. 0 means no cap.
    max_chunk_batch_size: int = 0

    #: Credit-based backpressure: a request stops generating frames once this
    #: many of its chunks are queued but not decoded. Without it, a fast talker
    #: builds an unbounded queue of undecoded codes and the memory shows up as
    #: stage-1 backlog rather than as anything the scheduler can see. 0
    #: disables the limit.
    max_pending_chunks_per_request: int = 4

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.pipeline_mode not in ("sync", "async_chunk"):
            raise ValueError(f"pipeline_mode must be sync|async_chunk, got {self.pipeline_mode!r}")
        if self.chunk_policy not in ("fifo", "deadline", "round_robin"):
            raise ValueError(f"chunk_policy must be fifo|deadline|round_robin, got {self.chunk_policy!r}")


@dataclass
class SamplingConfig:
    """Sampling for both autoregressive axes.

    The two axes are sampled separately in the reference implementation and the
    checkpoint ships different defaults for each, so they are separate fields
    rather than one shared set. ``subtalker_*`` is the code predictor, which
    walks the codebook axis inside one frame.
    """

    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0

    seed: int | None = None


@dataclass
class EngineConfig:
    """Everything fixed for the lifetime of the loaded model."""

    model: str = ""
    dtype: str = "bfloat16"
    device: str = "cuda"

    talker: TalkerConfig = field(default_factory=TalkerConfig)
    code2wav: Code2WavConfig = field(default_factory=Code2WavConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)

    #: Per-stage wall clock. Costs a synchronize per stage per step, so it is
    #: on by default for experiments and worth turning off for a latency run.
    collect_stage_timings: bool = True

    def __post_init__(self) -> None:
        if self.quant.enabled and "code2wav" in self.quant.targets and self.code2wav.dtype == "float32":
            # fp32 weights and an int8/fp8 GEMM cannot both describe the same
            # layer; quantizing while the decoder is pinned to fp32 would
            # silently keep the fp32 path and report a speedup that is noise.
            raise ValueError(
                "quantizing code2wav while --code2wav-dtype is float32 does nothing: the "
                "decoder weights stay fp32. Set --code2wav-dtype bfloat16 first, and compare "
                "the waveform before trusting the result."
            )

    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)

    def resolved_model_path(self) -> str:
        path = os.path.expanduser(self.model)
        if os.path.isdir(path):
            return path
        raise FileNotFoundError(
            f"model path {self.model!r} does not exist. Point --model at a local "
            "Qwen3-TTS checkpoint directory (the one containing config.json and "
            "speech_tokenizer/)."
        )


@dataclass
class RunConfig:
    """One synthesis request."""

    text: str = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
    speaker: str = "vivian"
    language: str = "Auto"
    instruct: str = ""
    #: CustomVoice and VoiceDesign default to non-streaming text conditioning:
    #: the whole text sits in the prefill prompt and every decode step consumes
    #: a pad embedding. Streaming instead feeds one text token per decode step.
    non_streaming_mode: bool | None = None
    output_path: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


# ---------------------------------------------------------------------------
# Command line plumbing
# ---------------------------------------------------------------------------

_NESTED: dict[str, tuple[type, str]] = {
    "talker": (TalkerConfig, "talker-"),
    "code2wav": (Code2WavConfig, "code2wav-"),
    "scheduler": (SchedulerConfig, ""),
    "quant": (QuantConfig, "quant-"),
    "sampling": (SamplingConfig, ""),
}


def _field_type_name(f: dataclasses.Field) -> str:
    # `from __future__ import annotations` makes f.type a string, and a field
    # like `int | None = None` has no default to infer a type from.
    return f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "str")


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: type, prefix: str = "") -> None:
    for f in dataclasses.fields(cls):
        if f.name in _NESTED:
            continue
        ann = _field_type_name(f)
        default = f.default if f.default is not dataclasses.MISSING else None
        flag = f"--{prefix}{f.name.replace('_', '-')}"
        if ann.startswith("tuple"):
            # Comma-separated, parsed in _coerce. argparse cannot express
            # "tuple[int, ...] with a dataclass default" on its own.
            parser.add_argument(flag, type=str, default=None)
        elif ann.startswith("bool") or isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif ann.startswith("int"):
            parser.add_argument(flag, type=int, default=None)
        elif ann.startswith("float"):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)


def build_arg_parser(description: str = "qtts") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    _add_dataclass_args(p, EngineConfig)
    _add_dataclass_args(p, RunConfig)
    for cls, prefix in _NESTED.values():
        _add_dataclass_args(p, cls, prefix)
    p.add_argument("--quantization", type=str, default=None, help="alias for --quant-method")
    return p


def _coerce(cls: type, args: argparse.Namespace, prefix: str = "") -> dict[str, Any]:
    """Collect the flags that were actually passed, converting tuple fields."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in _NESTED:
            continue
        attr = f"{prefix}{f.name}"
        value = getattr(args, attr, None)
        if value is None:
            continue
        ann = _field_type_name(f)
        if ann.startswith("tuple"):
            parts = [s.strip() for s in str(value).split(",") if s.strip()]
            inner = int if "int" in ann else str
            out[f.name] = tuple(inner(s) for s in parts)
        else:
            out[f.name] = value
    return out


def configs_from_args(args: argparse.Namespace) -> tuple[EngineConfig, RunConfig]:
    quant_kwargs = _coerce(QuantConfig, args, "quant_")
    if getattr(args, "quantization", None):
        quant_kwargs["method"] = args.quantization
    engine = EngineConfig(
        **_coerce(EngineConfig, args),
        talker=TalkerConfig(**_coerce(TalkerConfig, args, "talker_")),
        code2wav=Code2WavConfig(**_coerce(Code2WavConfig, args, "code2wav_")),
        scheduler=SchedulerConfig(**_coerce(SchedulerConfig, args)),
        quant=QuantConfig(**quant_kwargs),
    )
    run = RunConfig(
        **_coerce(RunConfig, args),
        sampling=SamplingConfig(**_coerce(SamplingConfig, args)),
    )
    return engine, run


def config_summary(engine: EngineConfig) -> dict[str, Any]:
    """Flat record of the knobs a benchmark row needs to be reproducible."""
    return {
        "dtype": engine.dtype,
        "quantization": engine.quant.method,
        "quant_targets": ",".join(engine.quant.targets) if engine.quant.enabled else "",
        "talker_attention_backend": engine.talker.attention_backend,
        "talker_cuda_graph": engine.talker.cuda_graph,
        "talker_compile_mode": engine.talker.compile_mode,
        "code2wav_mode": engine.code2wav.mode,
        "code2wav_dtype": engine.code2wav.dtype,
        "snake_backend": engine.code2wav.snake_backend,
        "rmsnorm_backend": engine.talker.rmsnorm_backend,
        "codec_chunk_frames": engine.code2wav.codec_chunk_frames,
        "initial_codec_chunk_frames": engine.code2wav.initial_codec_chunk_frames,
        "pipeline_mode": engine.scheduler.pipeline_mode,
        "chunk_policy": engine.scheduler.chunk_policy,
        "max_num_seqs": engine.scheduler.max_num_seqs,
        "max_pending_chunks": engine.scheduler.max_pending_chunks_per_request,
    }


def dump_config(engine: EngineConfig) -> str:
    return json.dumps(dataclasses.asdict(engine), indent=2, default=str)
