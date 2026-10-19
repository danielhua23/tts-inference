"""When to hand accumulated codec frames to stage 1.

This is the decision vllm-omni makes in
``stage_input_processors/qwen3_tts.talker2code2wav_async_chunk``, and it is the
whole of what "streaming TTS" means at the engine level: the talker appends one
frame per decode step, and something has to decide when enough frames exist to
be worth a decoder call.

The tension is that a chunk is both a latency unit and a batching unit. Small
chunks reach the listener sooner but run the decoder more often on less work;
large chunks are efficient but the first one sets time-to-first-audio. The three
policies here are three answers:

* **fixed** -- emit at ``initial``, then every ``chunk`` frames. One small first
  chunk for latency, then steady state for throughput.
* **ramp** -- an explicit schedule, e.g. ``5,10,25``, growing the chunk over the
  first few emissions so there is no single cliff where cost jumps.
* **dynamic_initial** -- pick the first chunk size from current load. Under low
  load, latency is free, so go small; under high load a small first chunk just
  adds decoder calls that everyone else waits behind.

Emission boundaries are cumulative, and only the *new* frames are shipped: the
stateful decoder already holds the history. In stateless mode the caller asks
for left context on top, which is why :class:`ChunkPlan` reports the two counts
separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..utils import get_logger

logger = get_logger("chunking")


@dataclass
class ChunkPlan:
    """One emission decision."""

    emit: bool
    #: Frames not yet sent to stage 1. Zero with ``emit=True`` means "the
    #: request finished and there is nothing left", which still has to be
    #: delivered so the consumer can close the stream.
    new_frames: int = 0
    #: Extra already-sent frames to prepend, for the stateless decoder. Their
    #: audio is discarded after decoding.
    left_context: int = 0
    final: bool = False


def max_initial_for_chunk(chunk_size: int) -> int:
    """Largest power of two strictly below ``chunk_size``."""
    if chunk_size <= 2:
        return 1
    return 1 << ((chunk_size - 1).bit_length() - 1)


def dynamic_initial_chunk(active_requests: int, max_num_seqs: int, max_initial: int) -> int:
    """Choose the first chunk size from the power-of-two steps by load factor."""
    steps: list[int] = []
    step = 2
    while step <= max_initial:
        steps.append(step)
        step <<= 1
    if not steps:
        return max(1, max_initial)
    if max_num_seqs <= 0:
        return steps[0]
    load = min(active_requests / max_num_seqs, 1.0)
    return steps[int(round(load * (len(steps) - 1)))]


def scheduled_size(index: int, ramp: tuple[int, ...], initial: int, steady: int) -> int:
    """Size of chunk number ``index``, from whichever schedule is in force."""
    if ramp:
        return ramp[index] if index < len(ramp) else steady
    if index == 0 and 0 < initial < steady:
        return initial
    return steady


@dataclass
class ChunkPolicy:
    """Per-request emission state.

    One instance per request: the boundaries depend on how many chunks that
    request has already emitted, and a shared instance would make one stream's
    schedule depend on another's.
    """

    chunk_frames: int
    left_context_frames: int
    initial_chunk_frames: int
    ramp: tuple[int, ...] = ()
    needs_left_context: bool = False

    emitted_chunks: int = field(default=0, init=False)
    sent_frames: int = field(default=0, init=False)
    #: Set once a plan with ``final=True`` has been handed out. The engine polls
    #: every request every step, so without this latch a request that finished
    #: with nothing left to send would be handed the same empty final chunk on
    #: every subsequent step.
    final_emitted: bool = field(default=False, init=False)

    @property
    def done(self) -> bool:
        return self.final_emitted

    def __post_init__(self) -> None:
        if self.chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {self.chunk_frames}")
        if self.left_context_frames < 0 or self.initial_chunk_frames < 0:
            raise ValueError("chunk frame counts must be non-negative")
        if self.ramp:
            if len(self.ramp) < 2:
                raise ValueError("codec_chunk_ramp needs at least two entries")
            if any(v <= 0 for v in self.ramp):
                raise ValueError("codec_chunk_ramp entries must be positive")
            if self.ramp[-1] != self.chunk_frames:
                logger.warning(
                    "codec_chunk_ramp ends at %d but codec_chunk_frames is %d, which puts the "
                    "cost cliff the ramp exists to remove back at chunk %d",
                    self.ramp[-1],
                    self.chunk_frames,
                    len(self.ramp) - 1,
                )
        if self.initial_chunk_frames > self.chunk_frames:
            logger.warning(
                "initial_codec_chunk_frames=%d exceeds codec_chunk_frames=%d; clamping",
                self.initial_chunk_frames,
                self.chunk_frames,
            )
            self.initial_chunk_frames = self.chunk_frames

    def plan(self, available_frames: int, finished: bool) -> ChunkPlan:
        """Decide what to emit given the frames the talker has produced so far."""
        if self.final_emitted:
            return ChunkPlan(emit=False)
        if available_frames <= 0:
            if not finished:
                return ChunkPlan(emit=False)
            self.final_emitted = True
            return ChunkPlan(emit=True, final=True)

        new_frames = self._next_chunk(available_frames, finished)
        if new_frames is None:
            return ChunkPlan(emit=False)

        if new_frames <= 0:
            if not finished:
                return ChunkPlan(emit=False)
            self.final_emitted = True
            return ChunkPlan(emit=True, final=True)

        left_context = 0
        if self.needs_left_context:
            left_context = min(self.left_context_frames, self.sent_frames)

        self.emitted_chunks += 1
        self.sent_frames += new_frames
        final = finished and self.sent_frames >= available_frames
        self.final_emitted = final
        return ChunkPlan(
            emit=True,
            new_frames=new_frames,
            left_context=left_context,
            final=final,
        )

    def _next_chunk(self, available_frames: int, finished: bool) -> int | None:
        """Frames to emit now, or ``None`` for "not yet".

        Measured against what this policy has already *sent*, not against a
        boundary derived from ``available_frames``. The distinction only shows
        up off the happy path, and it showed up as two bugs: a request whose
        frame count already sat on a boundary was handed the same chunk again
        on the next poll, and one whose length landed exactly on a boundary was
        handed a whole extra chunk past the end. Both then sliced an empty
        range out of the frame list.
        """
        pending = available_frames - self.sent_frames
        if finished:
            # Everything left goes out in one chunk, whatever the schedule
            # says. A request that ends before its first boundary would
            # otherwise wait for a boundary it never reaches, and one that ends
            # just past a boundary would trail a one-frame chunk.
            return max(0, pending)
        size = scheduled_size(self.emitted_chunks, self.ramp, self.initial_chunk_frames, self.chunk_frames)
        return size if pending >= size else None


def build_policy(
    *,
    chunk_frames: int,
    left_context_frames: int,
    initial_chunk_frames: int,
    ramp: tuple[int, ...],
    needs_left_context: bool,
    active_requests: int = 0,
    max_num_seqs: int = 1,
    dynamic_initial: bool = False,
) -> ChunkPolicy:
    """Construct a per-request policy, resolving the first chunk size.

    ``dynamic_initial`` is ignored when a ramp is configured: the ramp already
    specifies every chunk size including the first, and letting load override
    its first entry would silently change a schedule the caller wrote down.
    """
    initial = initial_chunk_frames
    if dynamic_initial and not ramp:
        initial = dynamic_initial_chunk(active_requests, max_num_seqs, max_initial_for_chunk(chunk_frames))
    return ChunkPolicy(
        chunk_frames=chunk_frames,
        left_context_frames=left_context_frames,
        initial_chunk_frames=initial,
        ramp=ramp,
        needs_left_context=needs_left_context,
    )
