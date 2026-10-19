"""The engine: load the model, run both stages, hand back audio.

The public surface mirrors vllm-omni's so requests transfer between the two:
:meth:`TTSEngine.generate` takes a list and blocks, like ``Omni.generate``, and
:meth:`TTSEngine.generate_stream` yields chunks as they are produced, like
``AsyncOmni.generate``'s ``async for``. Chunks carry what the reference's
``multimodal_output`` carries -- audio, sample rate, a finished flag -- plus the
timings its example script computes by hand.

The loop is synchronous and single-threaded, deliberately. The two stages could
overlap on separate streams, and a production server should overlap them, but
then every measurement includes an overlap-efficiency term and attributing a
regression means unpicking it first. Here one :meth:`TTSEngine.step` is:
advance the talker one frame for every active request, ask each request's
policy whether it owes a chunk, decode the chunks the scheduler picks, deliver.
The stage timings add up to the step, so a change that makes stage 1 faster
shows up as stage 1 being faster.

``pipeline_mode`` selects whether stage 1 runs interleaved:

* ``async_chunk`` -- the streaming path, and the only one where
  time-to-first-audio is a separate number from total latency.
* ``sync`` -- run the talker to completion, then decode once. No chunk
  boundaries at all, so it is the baseline for "what does chunking cost".
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch

from ..config import EngineConfig, SamplingConfig
from ..models.code2wav import Qwen3TTSCode2Wav
from ..models.code_predictor import CodePredictor
from ..models.hf_config import Qwen3TTSHFConfig
from ..models.loader import load_code2wav_weights, load_talker_weights
from ..models.prompt import PromptBuilder, load_tokenizer
from ..models.talker import Qwen3TTSTalker
from ..quant import apply_quantization
from ..utils import StageTimer, get_logger, human_bytes, peak_memory_mb, reset_peak_memory
from .code2wav_runner import ChunkJob, Code2WavRunner
from .request import AudioChunk, RequestState, RequestTracker, TTSOutput, TTSRequest
from .scheduler import Scheduler
from .talker_runner import TalkerRunner

logger = get_logger("engine")


class TTSEngine:
    def __init__(self, config: EngineConfig, sampling: SamplingConfig | None = None) -> None:
        path = config.resolved_model_path()
        self.config = config
        self.device = torch.device(config.device)
        self.hf_config = Qwen3TTSHFConfig.from_pretrained(path)
        self.sampling = sampling or self._sampling_from_checkpoint()

        talker_dtype = config.torch_dtype()
        code2wav_dtype = config.code2wav.torch_dtype()
        sched = config.scheduler

        reset_peak_memory()
        self.talker = (
            Qwen3TTSTalker(
                self.hf_config,
                runtime=config.talker,
                dtype=talker_dtype,
                device=self.device,
                max_num_seqs=sched.max_num_seqs,
                max_model_len=sched.max_model_len,
            )
            .to(self.device)
            .eval()
        )
        self.code_predictor = (
            CodePredictor(
                self.hf_config.talker.code_predictor,
                self.hf_config.talker.hidden_size,
                runtime=config.talker,
                dtype=talker_dtype,
                device=self.device,
            )
            .to(self.device)
            .eval()
        )
        load_talker_weights(self.talker, self.code_predictor, path, talker_dtype)
        # Both of these read loaded weights, so they cannot run in __init__ of
        # the modules themselves.
        self.talker.init_runtime_buffers()
        self.code_predictor.maybe_compile()

        self.code2wav = Qwen3TTSCode2Wav(
            self.hf_config.code2wav,
            attention_backend=config.talker.attention_backend,
            snake_backend=config.code2wav.snake_backend,
            norm_backend=config.code2wav.rmsnorm_backend,
        )
        load_code2wav_weights(self.code2wav, path, code2wav_dtype)
        self.code2wav = self.code2wav.to(device=self.device, dtype=code2wav_dtype).eval()
        if config.code2wav.snake_precompute_cache:
            self.code2wav.precompute_caches()
        else:
            # The codebooks are stored as EMA sums and are unusable until
            # divided by their usage counts, so that part is not optional even
            # when the activation caches are skipped.
            self.code2wav.quantizer.precompute()

        self.quant_report = apply_quantization(
            config.quant,
            talker=self.talker,
            code_predictor=self.code_predictor,
            code2wav=self.code2wav,
            calibration_batch=sched.max_num_seqs,
        )

        self.tokenizer = load_tokenizer(path)
        self.prompt_builder = PromptBuilder(self.hf_config, self.talker, self.tokenizer)
        self.talker_runner = TalkerRunner(self.talker, self.code_predictor, config, self.sampling)
        self.code2wav_runner = Code2WavRunner(self.code2wav, config.code2wav)
        self.scheduler = Scheduler(config, self.code2wav_runner)

        self.sample_rate = self.code2wav.sample_rate
        self.timer = StageTimer(enabled=config.collect_stage_timings)
        self.steps = 0
        self.load_peak_mb = peak_memory_mb()
        logger.info(
            "engine ready: talker %s, code2wav %s, %d slots x %d positions, %s/%s, weights peak %s",
            talker_dtype,
            code2wav_dtype,
            sched.max_num_seqs,
            sched.max_model_len,
            sched.pipeline_mode,
            config.code2wav.mode,
            human_bytes(int(self.load_peak_mb * 1024 * 1024)),
        )

    def _sampling_from_checkpoint(self) -> SamplingConfig:
        """Read sampling defaults from ``generation_config.json``.

        The checkpoint states the sampling it was tuned with, including the
        repetition penalty that keeps the talker from latching onto a periodic
        codebook-0 pattern. Defaulting to this file's own values instead would
        make every measurement describe a model the authors did not ship.
        """
        gen = self.hf_config.generation
        if not gen:
            return SamplingConfig()
        return SamplingConfig(
            do_sample=bool(gen.get("do_sample", True)),
            temperature=float(gen.get("temperature", 0.9)),
            top_k=int(gen.get("top_k", 50)),
            top_p=float(gen.get("top_p", 1.0)),
            repetition_penalty=float(gen.get("repetition_penalty", 1.05)),
            subtalker_do_sample=bool(gen.get("subtalker_dosample", gen.get("subtalker_do_sample", True))),
            subtalker_temperature=float(gen.get("subtalker_temperature", 0.9)),
            subtalker_top_k=int(gen.get("subtalker_top_k", 50)),
            subtalker_top_p=float(gen.get("subtalker_top_p", 1.0)),
        )

    # -- submission ---------------------------------------------------------

    def submit(self, request: TTSRequest) -> RequestTracker:
        """Build the prompt and queue the request. Returns its tracker."""
        with self.timer.measure("prompt"):
            prompt = self.prompt_builder.build_prompt(
                text=request.text,
                speaker=request.speaker,
                language=request.language,
                instruct=request.instruct,
                non_streaming_mode=request.non_streaming_mode,
            )
        tracker = self.scheduler.add_request(request, prompt.embeds.shape[0])
        tracker.prompt = prompt
        return tracker

    # -- the loop -----------------------------------------------------------

    def step(self) -> list[AudioChunk]:
        """One iteration: admit, one talker frame, chunk, decode, deliver."""
        self.steps += 1
        for tracker in self.scheduler.admit():
            with self.timer.measure("prefill"):
                self.talker_runner.prefill(tracker, tracker.prompt)

        rows = self.scheduler.active_rows()
        if rows:
            with self.timer.measure("talker"):
                self.talker_runner.decode_step(rows)

        # In sync mode the scheduler itself holds a request's frames back until
        # its talker has stopped, so there is nothing to special-case here.
        self.scheduler.collect_chunks()
        self.scheduler.apply_backpressure()

        jobs = self.scheduler.next_batch()
        chunks: list[AudioChunk] = []
        if jobs:
            decodable = [job for job in jobs if job.new_frames > 0]
            with self.timer.measure("code2wav"):
                decoded = self.code2wav_runner.run(decodable)
            audio_by_job = {id(job): audio for job, audio in decoded}
            for job in jobs:
                chunks.append(self._deliver(job, audio_by_job.get(id(job))))

        self._retire_drained()
        return chunks

    def _retire_drained(self) -> None:
        """Finish requests whose talker stopped and whose chunks have all landed."""
        for tracker in list(self.scheduler.running):
            if tracker.state != RequestState.DRAINING or tracker.pending_chunks:
                continue
            if tracker.policy is not None and not tracker.policy.done:
                continue
            self.scheduler.finish(tracker)

    def _deliver(self, job: ChunkJob, audio: torch.Tensor | None) -> AudioChunk:
        tracker = job.tracker
        tracker.pending_chunks -= 1
        samples = (
            audio.detach().reshape(-1).to("cpu", torch.float32)
            if audio is not None
            else torch.zeros(0, dtype=torch.float32)
        )
        if samples.numel():
            tracker.audio_pieces.append(samples)
        elapsed = tracker.record_chunk(time.perf_counter())
        return AudioChunk(
            request_id=tracker.request.request_id,
            index=tracker.chunks_emitted - 1,
            audio=samples,
            sample_rate=self.sample_rate,
            frames=job.new_frames,
            finished=job.final,
            elapsed=elapsed,
        )

    # -- public API ---------------------------------------------------------

    def generate(self, requests: list[TTSRequest]) -> list[TTSOutput]:
        """Run ``requests`` to completion. Result order matches the input."""
        trackers = [self.submit(request) for request in requests]
        while self.scheduler.has_work():
            self.step()
        return [tracker.build_output(self.sample_rate) for tracker in trackers]

    def generate_stream(self, request: TTSRequest) -> Iterator[AudioChunk]:
        """Yield this request's chunks as they are decoded.

        Requests already in flight keep progressing, so several of these can be
        interleaved -- but the engine is not thread-safe, so whatever drives it
        has to serialise :meth:`step`.
        """
        tracker = self.submit(request)
        while tracker.state != RequestState.FINISHED:
            produced = self.step()
            for chunk in produced:
                if chunk.request_id == request.request_id:
                    yield chunk
            if not self.scheduler.has_work():
                break

    # -- reporting ----------------------------------------------------------

    def timing_summary(self) -> dict[str, float]:
        """Per-stage totals in milliseconds, plus the per-call mean."""
        summary: dict[str, float] = {"steps": float(self.steps)}
        for stage, total in self.timer.totals.items():
            calls = self.timer.counts[stage]
            summary[f"{stage}_ms"] = total * 1000
            summary[f"{stage}_calls"] = float(calls)
            summary[f"{stage}_mean_ms"] = total / calls * 1000
        summary["peak_memory_mb"] = peak_memory_mb()
        summary.update({f"sched_{k}": float(v) for k, v in self.scheduler.stats().items()})
        return summary

    def reset_timings(self) -> None:
        self.timer.reset()
        self.steps = 0
        reset_peak_memory()
