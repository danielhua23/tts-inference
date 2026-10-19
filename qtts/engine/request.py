"""Request, chunk and output types.

Field names follow vllm-omni's ``additional_information`` entries so a request
written for one runs on the other: ``task_type``, ``text``, ``language``,
``speaker``, ``instruct``, ``max_new_tokens``, ``non_streaming_mode``. There
they arrive as one-element lists after msgspec transport; here they are plain
values, and :meth:`TTSRequest.from_additional_information` unwraps the list
form so the reference's example inputs can be fed in directly.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

_REQUEST_COUNTER = itertools.count()


class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    #: Talker paused because too many of its chunks are queued for stage 1.
    #: This is credit-based backpressure, not an error: without it a fast
    #: talker builds an unbounded queue of undecoded frames.
    THROTTLED = "throttled"
    #: Talker done; chunks may still be in flight.
    DRAINING = "draining"
    FINISHED = "finished"
    FAILED = "failed"


class FinishReason(Enum):
    EOS = "eos"
    MAX_FRAMES = "max_frames"
    REPETITION = "repetition"
    ABORTED = "aborted"


def _unwrap(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


@dataclass
class TTSRequest:
    text: str
    speaker: str = ""
    language: str = "Auto"
    instruct: str = ""
    task_type: str = ""
    max_new_tokens: int | None = None
    non_streaming_mode: bool | None = None
    #: Per-request override of the first chunk size, as in the reference.
    initial_codec_chunk_frames: int | None = None
    seed: int | None = None
    request_id: str = field(default_factory=lambda: str(next(_REQUEST_COUNTER)))

    @classmethod
    def from_additional_information(cls, info: dict[str, Any], request_id: str | None = None) -> TTSRequest:
        kwargs: dict[str, Any] = {
            "text": str(_unwrap(info.get("text"), "")),
            "speaker": str(_unwrap(info.get("speaker"), "") or ""),
            "language": str(_unwrap(info.get("language"), "Auto") or "Auto"),
            "instruct": str(_unwrap(info.get("instruct"), "") or ""),
            "task_type": str(_unwrap(info.get("task_type"), "") or ""),
        }
        max_new = _unwrap(info.get("max_new_tokens"))
        if max_new is not None:
            kwargs["max_new_tokens"] = int(max_new)
        non_streaming = _unwrap(info.get("non_streaming_mode"))
        if isinstance(non_streaming, bool):
            kwargs["non_streaming_mode"] = non_streaming
        initial = _unwrap(info.get("initial_codec_chunk_frames"))
        if initial is not None:
            kwargs["initial_codec_chunk_frames"] = int(initial)
        if request_id is not None:
            kwargs["request_id"] = request_id
        return cls(**kwargs)


@dataclass
class AudioChunk:
    """One delivery to the consumer."""

    request_id: str
    index: int
    #: Mono float waveform in [-1, 1].
    audio: torch.Tensor
    sample_rate: int
    frames: int
    finished: bool
    #: Seconds since the request was submitted. For ``index == 0`` this is
    #: time-to-first-audio, which is the number a listener perceives; for later
    #: chunks the gap to the previous one is what determines whether playback
    #: underruns.
    elapsed: float = 0.0


@dataclass
class TTSOutput:
    request_id: str
    audio: torch.Tensor
    sample_rate: int
    frames: int
    finish_reason: FinishReason
    #: Wall clock from submission to the first and last chunk.
    ttfa: float = 0.0
    total_time: float = 0.0
    #: Gaps between consecutive chunks. The maximum is the one that matters:
    #: a single gap longer than the audio already buffered is an underrun,
    #: however good the average looks.
    inter_chunk_gaps: list[float] = field(default_factory=list)
    num_chunks: int = 0

    @property
    def duration(self) -> float:
        return self.audio.shape[-1] / self.sample_rate if self.sample_rate else 0.0

    @property
    def real_time_factor(self) -> float:
        """Generation time over audio duration. Below 1 is faster than real time."""
        return self.total_time / self.duration if self.duration > 0 else float("inf")

    @property
    def max_inter_chunk_gap(self) -> float:
        return max(self.inter_chunk_gaps) if self.inter_chunk_gaps else 0.0


class RequestTracker:
    """Per-request engine bookkeeping.

    Holds the talker's slot, the accumulated codec frames, the chunk policy and
    the stage-1 decoder state. Frames stay on the GPU by default: at 12.5 Hz
    with Q=16 a frame is 16 int64 values, so moving each one to host memory as
    it is produced costs a synchronize per frame for 128 bytes.
    """

    def __init__(self, request: TTSRequest, slot: int, prompt_len: int) -> None:
        self.request = request
        self.slot = slot
        self.prompt_len = prompt_len
        self.state = RequestState.WAITING
        self.finish_reason: FinishReason | None = None
        #: TalkerPrompt, built at submission and consumed by prefill.
        self.prompt: Any = None

        #: One ``[1, Q]`` tensor per generated frame, concatenated on emission.
        self.frames: list[torch.Tensor] = []
        #: Talker hidden state of the last step, which the code predictor
        #: conditions on. Produced by one step and consumed by the next.
        self.last_hidden: torch.Tensor | None = None
        #: Embedding for the next decode step: codec embeddings of the frame
        #: just produced, summed with this step's text embedding.
        self.next_embed: torch.Tensor | None = None
        #: Index into the trailing text queue, for streaming text conditioning.
        self.text_offset = 0
        self.trailing_text: torch.Tensor | None = None

        self.seq_len = prompt_len
        self.frames_generated = 0
        self.frames_sent = 0
        self.chunks_emitted = 0
        self.pending_chunks = 0

        self.policy = None  # ChunkPolicy, assigned by the engine
        self.decoder_state = None  # Code2WavRequestState in stateful mode
        self.generator: torch.Generator | None = None

        self.submitted_at = time.perf_counter()
        self.first_audio_at: float | None = None
        self.last_chunk_at: float | None = None
        self.inter_chunk_gaps: list[float] = []
        self.audio_pieces: list[torch.Tensor] = []

        #: Rolling window of codebook-0 ids, for the repetition guard.
        self.recent_layer0: list[int] = []

    @property
    def is_active(self) -> bool:
        return self.state in (RequestState.DECODING, RequestState.THROTTLED)

    def record_chunk(self, now: float) -> float:
        """Note a delivery and return seconds since submission."""
        elapsed = now - self.submitted_at
        if self.first_audio_at is None:
            self.first_audio_at = now
        else:
            self.inter_chunk_gaps.append(now - (self.last_chunk_at or now))
        self.last_chunk_at = now
        self.chunks_emitted += 1
        return elapsed

    def build_output(self, sample_rate: int) -> TTSOutput:
        audio = (
            torch.cat(self.audio_pieces, dim=-1)
            if self.audio_pieces
            else torch.zeros(0, dtype=torch.float32)
        )
        finished_at = self.last_chunk_at or time.perf_counter()
        return TTSOutput(
            request_id=self.request.request_id,
            audio=audio.reshape(-1),
            sample_rate=sample_rate,
            frames=self.frames_generated,
            finish_reason=self.finish_reason or FinishReason.EOS,
            ttfa=(self.first_audio_at - self.submitted_at) if self.first_audio_at else 0.0,
            total_time=finished_at - self.submitted_at,
            inter_chunk_gaps=list(self.inter_chunk_gaps),
            num_chunks=self.chunks_emitted,
        )
