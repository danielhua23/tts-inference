"""CLI: measure one configuration carefully.

    python -m qtts.entrypoints.benchmark \\
        --model /home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --concurrency 8 --repeats 3 --talker-cuda-graph

One process, one configuration, one row. Sweeping is left to
``examples/benchmark/sweep.py``, which runs this repeatedly: a sweep inside one
process would leave the previous configuration's captured graphs, calibration
verdicts, autotuning cache and allocator blocks in place and quietly bias
whatever ran second.

TTS needs different numbers from a text LLM, and reporting the wrong ones hides
exactly the failures streaming introduces:

* **Real-time factor**, not tokens per second. Audio has a duration, so the
  question is whether generation keeps ahead of playback. Below 1.0 is
  faster than real time; at 8 concurrent streams the aggregate matters more
  than any single one.
* **Time to first audio**, separately from total latency. A listener hears the
  first chunk, waits through nothing else, and judges the system on it.
* **The worst inter-chunk gap**, not the mean. Playback survives a bad average
  and dies on one gap longer than the audio already buffered, so the maximum is
  the number that decides whether a stream is listenable. Underruns are counted
  explicitly: a gap is an underrun when it exceeds the duration of the audio
  delivered before it.

Field names follow vllm-omni's serving benchmark (``throughput_qps``,
``latency_p50/p95/p99``, ``peak_memory_mb_*``) so a row from here can sit next
to a row from there without renaming anything.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import torch

from ..config import build_arg_parser, config_summary, configs_from_args
from ..engine import TTSEngine, TTSRequest
from ..utils import get_logger, init_logging, peak_memory_mb, percentile, reset_peak_memory

logger = get_logger("benchmark")


def underruns(output) -> int:
    """Gaps longer than the audio buffered when they started.

    The chunk at index ``i`` arrives after the audio of chunks ``0..i-1`` has
    been delivered, so a listener who started playing immediately has that much
    buffered. A gap longer than the buffer is silence they hear.
    """
    if not output.inter_chunk_gaps or output.num_chunks < 2:
        return 0
    per_chunk = output.duration / output.num_chunks
    return sum(1 for index, gap in enumerate(output.inter_chunk_gaps) if gap > per_chunk * (index + 1))


def run_once(engine: TTSEngine, requests: list[TTSRequest]) -> tuple[float, list]:
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    started = time.perf_counter()
    outputs = engine.generate(requests)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - started, outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("qtts benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="measured iterations (median is reported)")
    parser.add_argument("--concurrency", type=int, default=1, help="requests submitted together")
    parser.add_argument("--label", type=str, default="", help="name for this row in a sweep")
    parser.add_argument("--json-path", type=str, default=None, help="append one JSON record here")
    args = parser.parse_args(argv)

    engine_config, run = configs_from_args(args)
    init_logging()
    if args.concurrency > engine_config.scheduler.max_num_seqs:
        parser.error(
            f"--concurrency {args.concurrency} exceeds --max-num-seqs "
            f"{engine_config.scheduler.max_num_seqs}; the extra requests would queue and the row "
            "would measure queueing rather than concurrency"
        )

    engine = TTSEngine(engine_config, sampling=run.sampling)

    def build_requests(tag: str) -> list[TTSRequest]:
        return [
            TTSRequest(
                text=run.text,
                speaker=run.speaker,
                language=run.language,
                instruct=run.instruct,
                non_streaming_mode=run.non_streaming_mode,
                # Seeding makes a row reproducible and makes it measure a
                # different code path: a seeded request cannot use the captured
                # code-predictor graph. Off unless asked for, and the engine
                # warns when both are on.
                seed=run.sampling.seed,
                request_id=f"{tag}-{i}",
            )
            for i in range(max(1, args.concurrency))
        ]

    # Always warm up, whatever the caller asked for. The first iteration pays
    # for graph capture, quantization calibration, Triton JIT and allocator
    # growth, none of which recur, so including it measures startup.
    logger.info("warmup")
    run_once(engine, build_requests("warmup"))
    engine.reset_timings()

    latencies: list[float] = []
    peaks: list[float] = []
    ttfas: list[float] = []
    rtfs: list[float] = []
    gaps: list[float] = []
    durations: list[float] = []
    frames: list[int] = []
    total_underruns = 0
    for index in range(max(args.repeats, 1)):
        reset_peak_memory()
        elapsed, outputs = run_once(engine, build_requests(f"r{index}"))
        latencies.append(elapsed)
        peaks.append(peak_memory_mb())
        for output in outputs:
            ttfas.append(output.ttfa)
            rtfs.append(output.real_time_factor)
            durations.append(output.duration)
            frames.append(output.frames)
            gaps.append(output.max_inter_chunk_gap)
            total_underruns += underruns(output)
        logger.info(
            "iteration %d/%d: %.3f s wall, %.2f s audio x %d, TTFA p50 %.3f s",
            index + 1,
            args.repeats,
            elapsed,
            statistics.median([o.duration for o in outputs]),
            len(outputs),
            statistics.median([o.ttfa for o in outputs]),
        )

    streams = len(rtfs)
    audio_seconds = sum(durations)
    wall = sum(latencies)
    record = {
        "label": args.label or f"c{args.concurrency}",
        "model": engine_config.model,
        "concurrency": args.concurrency,
        "repeats": len(latencies),
        "streams": streams,
        **config_summary(engine_config),
        # Audio-side numbers: what the listener gets.
        "audio_seconds_total": round(audio_seconds, 3),
        "frames_median": statistics.median(frames) if frames else 0,
        # Aggregate real-time factor across every concurrent stream. Per-stream
        # RTF above 1 is fine as long as this is below it: the streams share one
        # GPU and the question is whether the GPU keeps up with all of them.
        "rtf_aggregate": round(wall / audio_seconds, 4) if audio_seconds else float("inf"),
        "rtf_per_stream_p50": round(percentile(rtfs, 50), 4),
        "rtf_per_stream_p95": round(percentile(rtfs, 95), 4),
        "ttfa_p50": round(percentile(ttfas, 50), 4),
        "ttfa_p95": round(percentile(ttfas, 95), 4),
        "ttfa_max": round(max(ttfas), 4) if ttfas else 0.0,
        "max_inter_chunk_gap_p50": round(percentile(gaps, 50), 4),
        "max_inter_chunk_gap_max": round(max(gaps), 4) if gaps else 0.0,
        "underruns": total_underruns,
        # vllm-omni field names, for side-by-side comparison.
        "throughput_qps": round(streams / wall, 6) if wall else 0.0,
        "latency_mean": round(statistics.fmean(latencies), 4),
        "latency_p50": round(percentile(latencies, 50), 4),
        "latency_p95": round(percentile(latencies, 95), 4),
        "latency_p99": round(percentile(latencies, 99), 4),
        "peak_memory_mb_max": round(max(peaks), 1),
        "peak_memory_mb_median": round(statistics.median(peaks), 1),
        "stage_timings": {k: round(v, 4) for k, v in engine.timing_summary().items()},
        "latency_samples": [round(s, 4) for s in latencies],
    }
    if engine.talker_runner.graphs is not None:
        record["talker_graphs"] = engine.talker_runner.graphs.stats()
    if engine.talker_runner.predictor_graphs is not None:
        record["code_predictor_graphs"] = engine.talker_runner.predictor_graphs.stats()
    if engine.quant_report:
        record["quant"] = engine.quant_report

    print(json.dumps(record, indent=2, default=str))
    print(
        f"\n{record['label']}: RTF {record['rtf_aggregate']:.3f} aggregate, "
        f"TTFA p50 {record['ttfa_p50']:.3f}s / p95 {record['ttfa_p95']:.3f}s, "
        f"worst gap {record['max_inter_chunk_gap_max']:.3f}s, "
        f"{record['underruns']} underruns, peak {record['peak_memory_mb_max']:.0f} MiB"
    )
    if args.json_path:
        with open(args.json_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
