"""qtts: a minimal Qwen3-TTS inference runtime.

Two stages, each with its own optimisation surface:

* stage 0, ``models/talker.py`` plus ``models/code_predictor.py`` -- an
  autoregressive LLM emitting one RVQ frame of 16 codes per 12.5 Hz audio frame;
* stage 1, ``models/code2wav.py`` -- a convolutional codec decoder turning
  those frames into a waveform.

``engine/`` is what sits between them: a continuous-batching scheduler for
stage 0 and a chunk dispatcher for stage 1. Import :class:`~qtts.engine.
TTSEngine` and :class:`~qtts.engine.TTSRequest` to run it.
"""

from .config import (
    Code2WavConfig,
    EngineConfig,
    QuantConfig,
    RunConfig,
    SamplingConfig,
    SchedulerConfig,
    TalkerConfig,
    build_arg_parser,
    configs_from_args,
)
from .engine import TTSEngine, TTSOutput, TTSRequest

__all__ = [
    "Code2WavConfig",
    "EngineConfig",
    "QuantConfig",
    "RunConfig",
    "SamplingConfig",
    "SchedulerConfig",
    "TalkerConfig",
    "TTSEngine",
    "TTSOutput",
    "TTSRequest",
    "build_arg_parser",
    "configs_from_args",
]
