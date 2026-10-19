"""CLI: run the HTTP server.

    python -m qtts.entrypoints.serve \\
        --model /home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --max-num-seqs 8 --talker-cuda-graph --port 8000

Then:

    curl -s localhost:8000/generate -H 'content-type: application/json' \\
        -d '{"text": "你好", "speaker": "vivian"}' -o out.wav

    curl -sN localhost:8000/generate_stream -H 'content-type: application/json' \\
        -d '{"text": "你好", "speaker": "vivian"}' | \\
        ffplay -f s16le -ar 24000 -ac 1 -
"""

from __future__ import annotations

import sys

from ..config import build_arg_parser, configs_from_args
from ..serving import serve
from ..utils import init_logging


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("qtts serve")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    engine_config, run = configs_from_args(args)
    init_logging()
    serve(engine_config, run.sampling, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
