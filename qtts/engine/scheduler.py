"""Admission, slot allocation, and which chunks stage 1 runs next.

Three decisions live here, and they are the ones worth experimenting with.

**Admission.** A request needs a talker KV slot and, in stateful mode, a
decoder state. Both are preallocated, so ``max_num_seqs`` is a hard cap and
admission is just "is a slot free".

**Slot allocation, low index first.** Not arbitrary. The talker reads its KV
cache as the contiguous row prefix ``[0, n)`` because that is a view; gathering
scattered slots would copy the whole cache every layer of every step. So the
active set must stay packed at the bottom, and a finishing request in the
middle has to be backfilled. :meth:`Scheduler.active_rows` returns the prefix
and reports the waste when it is not dense.

**Chunk dispatch.** Once several requests have frames ready, stage 1 cannot run
them all at once, and the choice of which to run is the scheduling question
this framework exists to measure:

* ``fifo`` -- oldest pending chunk first. Fair, and the baseline.
* ``deadline`` -- the request with the least buffered audio first. A listener
  only notices a gap when the audio already delivered runs out, so this
  optimises the metric that matters instead of the one that is easy.
* ``round_robin`` -- one chunk per request per pass. Keeps a long request from
  monopolising the decoder.

**Backpressure.** A talker that outruns stage 1 builds an unbounded queue of
undecoded frames -- memory grows and latency grows with it, while the audio
already generated sits undelivered. Past ``max_pending_chunks_per_request`` a
request's talker is paused. That is the coupling between the two stages, and
turning it off is a good way to see why it is there.
"""

from __future__ import annotations

import itertools
import time

import torch

from ..config import EngineConfig
from ..utils import get_logger
from .chunking import build_policy
from .code2wav_runner import ChunkJob, Code2WavRunner
from .request import RequestState, RequestTracker, TTSRequest

logger = get_logger("scheduler")


class SlotPool:
    """Talker KV slots, handed out lowest-index-first."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._free = list(range(capacity))

    def acquire(self) -> int | None:
        if not self._free:
            return None
        self._free.sort()
        return self._free.pop(0)

    def release(self, slot: int) -> None:
        if slot in self._free:
            raise AssertionError(f"slot {slot} released twice")
        self._free.append(slot)

    @property
    def in_use(self) -> int:
        return self.capacity - len(self._free)


class Scheduler:
    def __init__(self, config: EngineConfig, code2wav: Code2WavRunner) -> None:
        self.config = config
        self.code2wav = code2wav
        self.slots = SlotPool(config.scheduler.max_num_seqs)
        self.waiting: list[RequestTracker] = []
        self.running: list[RequestTracker] = []
        self.pending: list[ChunkJob] = []
        self._sequence = itertools.count()
        self._round_robin = 0
        #: Peak observed padding waste, for the run summary.
        self.max_idle_rows = 0

    # -- admission ----------------------------------------------------------

    def add_request(self, request: TTSRequest, prompt_len: int) -> RequestTracker:
        tracker = RequestTracker(request, slot=-1, prompt_len=prompt_len)
        self.waiting.append(tracker)
        return tracker

    def admit(self) -> list[RequestTracker]:
        """Give slots to as many waiting requests as fit. Returns the admitted."""
        admitted = []
        while self.waiting:
            slot = self.slots.acquire()
            if slot is None:
                break
            tracker = self.waiting.pop(0)
            tracker.slot = slot
            tracker.state = RequestState.PREFILL
            tracker.policy = self._build_policy(tracker)
            self.code2wav.prepare(tracker)
            self.running.append(tracker)
            admitted.append(tracker)
        return admitted

    def _build_policy(self, tracker: RequestTracker):
        scheduler = self.config.scheduler
        code2wav = self.config.code2wav
        initial = tracker.request.initial_codec_chunk_frames
        if initial is None:
            initial = code2wav.initial_codec_chunk_frames
        return build_policy(
            chunk_frames=code2wav.codec_chunk_frames,
            left_context_frames=code2wav.codec_left_context_frames,
            initial_chunk_frames=initial,
            ramp=code2wav.codec_chunk_ramp,
            # Only the stateless decoder needs history re-sent; the stateful
            # one already holds it and would decode those frames twice.
            needs_left_context=code2wav.mode == "stateless",
            active_requests=len(self.running),
            max_num_seqs=scheduler.max_num_seqs,
            dynamic_initial=code2wav.dynamic_initial_chunk,
        )

    def finish(self, tracker: RequestTracker) -> None:
        tracker.state = RequestState.FINISHED
        if tracker in self.running:
            self.running.remove(tracker)
        if tracker.slot >= 0:
            self.slots.release(tracker.slot)
            tracker.slot = -1
        self.code2wav.release(tracker)

    # -- stage 0 ------------------------------------------------------------

    def active_rows(self) -> list[RequestTracker]:
        """The trackers the talker should step, as a dense low-slot prefix.

        Rows inside the prefix that are not returned would still be computed by
        the talker and thrown away, so the gap is counted rather than hidden.
        """
        runnable = [t for t in self.running if t.state == RequestState.DECODING]
        if not runnable:
            return []
        runnable.sort(key=lambda t: t.slot)
        highest = runnable[-1].slot
        idle = highest + 1 - len(runnable)
        if idle > self.max_idle_rows:
            self.max_idle_rows = idle
            if idle > 0:
                logger.debug(
                    "talker step computes %d idle rows: slots %s are active inside prefix [0, %d)",
                    idle,
                    [t.slot for t in runnable],
                    highest + 1,
                )
        return runnable

    def apply_backpressure(self) -> None:
        """Pause talkers whose stage-1 queue is too deep, resume the rest."""
        limit = self.config.scheduler.max_pending_chunks_per_request
        if limit <= 0:
            return
        for tracker in self.running:
            if tracker.state == RequestState.DECODING and tracker.pending_chunks >= limit:
                tracker.state = RequestState.THROTTLED
            elif tracker.state == RequestState.THROTTLED and tracker.pending_chunks < limit:
                tracker.state = RequestState.DECODING

    # -- stage 1 ------------------------------------------------------------

    def collect_chunks(self) -> None:
        """Ask every request's policy whether it has a chunk to emit.

        In ``sync`` pipeline mode a request is skipped until its own talker has
        stopped, so it emits exactly one whole-utterance chunk. The gate is per
        request rather than global: making a short utterance wait for a long
        one would put queueing delay into the mode that exists as the no-
        chunking baseline.
        """
        sync = self.config.scheduler.pipeline_mode == "sync"
        for tracker in list(self.running):
            if tracker.policy is None or tracker.state == RequestState.PREFILL:
                continue
            talker_done = tracker.state in (RequestState.DRAINING, RequestState.FAILED)
            if sync and not talker_done:
                continue
            plan = tracker.policy.plan(tracker.frames_generated, talker_done)
            if not plan.emit:
                continue
            if plan.new_frames == 0:
                # Nothing left to decode but the consumer still needs to be
                # told the stream ended.
                self.pending.append(
                    ChunkJob(tracker, torch.empty(0), 0, 0, True, next(self._sequence))
                )
                tracker.pending_chunks += 1
                continue
            start = tracker.frames_sent - plan.left_context
            end = tracker.frames_sent + plan.new_frames
            codes = torch.cat(tracker.frames[start:end], dim=0).t().unsqueeze(0).contiguous()
            tracker.frames_sent += plan.new_frames
            tracker.pending_chunks += 1
            self.pending.append(
                ChunkJob(tracker, codes, plan.new_frames, plan.left_context, plan.final, next(self._sequence))
            )

    def next_batch(self) -> list[ChunkJob]:
        """Pick which pending chunks stage 1 runs now, per ``chunk_policy``."""
        if not self.pending:
            return []
        budget = self.config.scheduler.max_chunk_batch_size
        if self.config.code2wav.mode == "stateful":
            # decode_window is inherently one request at a time.
            budget = min(budget, 1) if budget > 0 else 1

        policy = self.config.scheduler.chunk_policy
        if policy == "fifo":
            ordered = sorted(self.pending, key=lambda job: job.sequence)
        elif policy == "deadline":
            ordered = sorted(self.pending, key=self._slack)
        elif policy == "round_robin":
            ordered = self._round_robin_order()
        else:
            raise ValueError(f"unknown chunk_policy {policy!r}")

        chosen = ordered[:budget] if budget > 0 else ordered
        for job in chosen:
            self.pending.remove(job)
        return chosen

    def _slack(self, job: ChunkJob) -> float:
        """Seconds of audio already delivered but not yet played, lower first.

        Approximated as delivered duration minus elapsed wall clock, which is
        exactly the buffer a listener starting at submission would have left.
        A request that has not had its first chunk yet gets the lowest possible
        slack, because nothing is worse for the listener than silence at the
        start.
        """
        tracker = job.tracker
        if tracker.first_audio_at is None:
            return float("-inf")
        delivered = sum(piece.shape[-1] for piece in tracker.audio_pieces) / self.code2wav.sample_rate
        return delivered - (time.perf_counter() - tracker.submitted_at)

    def _round_robin_order(self) -> list[ChunkJob]:
        """One chunk per request, cycling which request leads each pass."""
        by_request: dict[str, list[ChunkJob]] = {}
        for job in sorted(self.pending, key=lambda j: j.sequence):
            by_request.setdefault(job.tracker.request.request_id, []).append(job)
        keys = sorted(by_request)
        if not keys:
            return []
        offset = self._round_robin % len(keys)
        self._round_robin += 1
        keys = keys[offset:] + keys[:offset]
        ordered: list[ChunkJob] = []
        while any(by_request[key] for key in keys):
            for key in keys:
                if by_request[key]:
                    ordered.append(by_request[key].pop(0))
        return ordered

    # -- reporting ----------------------------------------------------------

    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.pending)

    def stats(self) -> dict[str, int]:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "pending_chunks": len(self.pending),
            "slots_in_use": self.slots.in_use,
            "max_idle_rows": self.max_idle_rows,
        }
