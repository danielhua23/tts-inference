"""CLI: synthesize one utterance, with every optimization reachable from a flag.

    python -m qtts.entrypoints.generate \\
        --model /home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \\
        --speaker vivian --output-path out.wav --report

Streaming, printing each chunk as it lands, which is where the chunk knobs show
up as a number instead of a setting:

    python -m qtts.entrypoints.generate --model ... --stream \\
        --codec-chunk-frames 25 --initial-codec-chunk-frames 5
"""

from __future__ import annotations

import json
import sys

from ..config import build_arg_parser, config_summary, configs_from_args
from ..engine import TTSEngine, TTSRequest
from ..utils import get_logger, init_logging, write_wav

logger = get_logger("generate")


def build_request(run, index: int = 0) -> TTSRequest:
    return TTSRequest(
        text=run.text,
        speaker=run.speaker,
        language=run.language,
        instruct=run.instruct,
        non_streaming_mode=run.non_streaming_mode,
        seed=run.sampling.seed,
        request_id=str(index),
    )


def report_output(output, path: str | None) -> None:
    logger.info(
        "%d frames, %.2f s of audio in %.2f s (RTF %.3f), TTFA %.3f s, %d chunks, finish=%s",
        output.frames,
        output.duration,
        output.total_time,
        output.real_time_factor,
        output.ttfa,
        output.num_chunks,
        output.finish_reason.value,
    )
    if output.inter_chunk_gaps:
        # The mean gap is not the interesting one. Playback survives a long
        # average and dies on a single gap longer than the buffer, so the
        # maximum is what decides whether a stream is listenable.
        logger.info(
            "inter-chunk gaps: max %.3f s, mean %.3f s over %d gaps",
            output.max_inter_chunk_gap,
            sum(output.inter_chunk_gaps) / len(output.inter_chunk_gaps),
            len(output.inter_chunk_gaps),
        )
    if path:
        write_wav(path, output.audio, output.sample_rate)
        logger.info("wrote %s", path)


def run_streaming(engine: TTSEngine, request: TTSRequest, path: str | None) -> None:
    """Consume the chunk iterator, reporting each delivery as it happens.

    The point of running this way rather than calling ``generate`` is that the
    per-chunk timings are observed at the boundary a consumer would see them,
    not reconstructed afterwards from an engine counter.
    """
    import torch

    pieces = []
    previous = 0.0
    for chunk in engine.generate_stream(request):
        label = "first" if chunk.index == 0 else f"+{chunk.elapsed - previous:.3f}s"
        logger.info(
            "chunk %d: %d frames, %.3f s audio, at %.3f s (%s)%s",
            chunk.index,
            chunk.frames,
            chunk.audio.numel() / chunk.sample_rate,
            chunk.elapsed,
            label,
            " [final]" if chunk.finished else "",
        )
        previous = chunk.elapsed
        if chunk.audio.numel():
            pieces.append(chunk.audio)
    if path and pieces:
        write_wav(path, torch.cat(pieces), engine.sample_rate)
        logger.info("wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("qtts generate")
    parser.add_argument("--stream", action="store_true", help="consume chunks as they are produced")
    parser.add_argument("--report", action="store_true", help="print the stage timing table")
    parser.add_argument("--repeat", type=int, default=1, help="submit the same text N times in one batch")
    parser.add_argument("--config-json", action="store_true", help="print the resolved config and exit")
    args = parser.parse_args(argv)

    engine_config, run = configs_from_args(args)
    init_logging()
    if args.config_json:
        print(json.dumps(config_summary(engine_config), indent=2))
        return 0

    engine = TTSEngine(engine_config, sampling=run.sampling)

    if args.stream:
        if args.repeat != 1:
            parser.error("--stream follows one request; use --repeat with the batch path")
        run_streaming(engine, build_request(run), run.output_path)
    else:
        requests = [build_request(run, i) for i in range(max(1, args.repeat))]
        outputs = engine.generate(requests)
        for index, output in enumerate(outputs):
            path = run.output_path
            if path and len(outputs) > 1:
                root, dot, ext = path.rpartition(".")
                path = f"{root}_{index}{dot}{ext}" if dot else f"{path}_{index}"
            report_output(output, path)

    if args.report:
        print(json.dumps(engine.timing_summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
