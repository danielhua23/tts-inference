"""Stage 1 execution: codec frames to waveform chunks.

The two modes here are not two implementations of the same thing, and the
difference is the main thing this framework exists to let you measure.

**stateful** calls :meth:`Qwen3TTSCode2Wav.decode_window` per request. Each
request owns a K/V cache and the two convolution caches, so a chunk costs work
proportional to its own frames and the result matches whole-utterance decoding.
The price is one decoder call per request per chunk -- batch size 1, always --
plus a few megabytes of resident state per request.

**stateless** calls :meth:`Qwen3TTSCode2Wav.chunked_decode` semantics: re-read
``left_context`` frames ahead of the chunk, decode, throw the context's samples
away. Nothing is retained between chunks, so chunks from *different* requests
batch together into one call. That is a real throughput win and it is why the
mode exists. It is also approximate, by a lot -- see ``models/code2wav.py`` for
measurements. Treating it as a free speedup is the mistake.

Batching in stateless mode groups by total frame count, because a single call
needs one ``F``. Padding short chunks up to a common length would work too but
would decode frames nobody asked for; with the engine emitting fixed-size
chunks, the groups line up naturally after the first chunk anyway.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from ..config import Code2WavConfig
from ..models.code2wav import Qwen3TTSCode2Wav
from ..utils import get_logger
from .request import RequestTracker

logger = get_logger("code2wav_runner")


class ChunkJob:
    """One request's pending chunk, as handed to stage 1."""

    __slots__ = ("tracker", "codes", "new_frames", "left_context", "final", "sequence")

    def __init__(
        self,
        tracker: RequestTracker,
        codes: torch.Tensor,
        new_frames: int,
        left_context: int,
        final: bool,
        sequence: int,
    ) -> None:
        #: ``[1, Q, left_context + new_frames]``
        self.codes = codes
        self.tracker = tracker
        self.new_frames = new_frames
        self.left_context = left_context
        self.final = final
        #: Global emission order, for the fifo policy.
        self.sequence = sequence

    @property
    def total_frames(self) -> int:
        return self.new_frames + self.left_context


class Code2WavRunner:
    def __init__(self, model: Qwen3TTSCode2Wav, config: Code2WavConfig) -> None:
        self.model = model
        self.config = config
        self.stateful = config.mode == "stateful"
        self.sample_rate = model.sample_rate
        self.total_upsample = model.total_upsample

    def state_bytes_per_request(self) -> int:
        return self.model.state_bytes_estimate() if self.stateful else 0

    def prepare(self, tracker: RequestTracker) -> None:
        if self.stateful and tracker.decoder_state is None:
            tracker.decoder_state = self.model.new_state()

    def release(self, tracker: RequestTracker) -> None:
        tracker.decoder_state = None

    def run(self, jobs: list[ChunkJob]) -> list[tuple[ChunkJob, torch.Tensor]]:
        """Decode ``jobs`` and return each one's audio, in the order given."""
        if not jobs:
            return []
        return self._run_stateful(jobs) if self.stateful else self._run_stateless(jobs)

    def _run_stateful(self, jobs: list[ChunkJob]) -> list[tuple[ChunkJob, torch.Tensor]]:
        results = []
        for job in jobs:
            if job.left_context:
                raise AssertionError(
                    "stateful decoding must not be given left context: the cache already holds "
                    "the history, so re-sending it would decode those frames twice"
                )
            state = job.tracker.decoder_state
            if state is None:
                state = self.model.new_state()
                job.tracker.decoder_state = state
            results.append((job, self.model.decode_window(job.codes, state)))
        return results

    def _run_stateless(self, jobs: list[ChunkJob]) -> list[tuple[ChunkJob, torch.Tensor]]:
        groups: dict[int, list[ChunkJob]] = defaultdict(list)
        for job in jobs:
            groups[job.total_frames].append(job)

        audio: dict[int, torch.Tensor] = {}
        for frames, group in groups.items():
            batch = torch.cat([job.codes for job in group], dim=0)
            decoded = self.model.decode_exact(batch)
            for row, job in enumerate(group):
                samples = decoded[row]
                if job.left_context:
                    samples = samples[job.left_context * self.total_upsample :]
                audio[id(job)] = samples
            del decoded, batch, frames
        return [(job, audio[id(job)]) for job in jobs]
