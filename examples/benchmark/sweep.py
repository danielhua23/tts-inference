"""Run several configurations and print them as one comparison table.

    python examples/benchmark/sweep.py --model /path/to/checkpoint --suite optimizations
    python examples/benchmark/sweep.py --model /path/to/checkpoint --suite concurrency

Each row is a fresh ``python -m qtts.entrypoints.benchmark`` process. That is
the point of the file rather than an implementation detail: captured CUDA
graphs, quantization calibration verdicts, Triton's autotune cache and the
allocator's block pool all persist for the life of a process, so a second
configuration measured in the same process is measuring the first one's
leftovers. One process per row costs ~20 s of model loading and buys numbers
that mean something.

Suites are deliberately one-knob-at-a-time against a common baseline, since the
interesting result is usually that two optimizations do not add up -- fused
RMSNorm is worth much less once graphs remove the launch overhead it was
mostly hiding.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: label -> extra flags on top of the common baseline.
SUITES: dict[str, dict[str, list[str]]] = {
    "optimizations": {
        "baseline": ["--talker-rmsnorm-backend", "eager", "--code2wav-snake-backend", "eager"],
        "fused-norm": ["--code2wav-snake-backend", "eager"],
        "fused-norm+snake": [],
        "graphs": ["--talker-cuda-graph"],
        "graphs+int8": ["--talker-cuda-graph", "--quantization", "int8_weight_only"],
        # Uniform int8, no per-shape calibration: the same kernel is 1.9x on the
        # MLP up-projection and 0.6x on the down-projection, so this row is
        # expected to be *slower* than the calibrated one above.
        "graphs+int8-uncalibrated": [
            "--talker-cuda-graph",
            "--quantization",
            "int8_weight_only",
            "--no-quant-calibrate",
        ],
    },
    "concurrency": {
        f"c{n}": ["--talker-cuda-graph", "--concurrency", str(n), "--max-num-seqs", str(max(n, 8))]
        for n in (1, 2, 4, 8)
    },
    "chunking": {
        "sync": ["--talker-cuda-graph", "--pipeline-mode", "sync"],
        "chunk-5": ["--talker-cuda-graph", "--code2wav-initial-codec-chunk-frames", "5"],
        "chunk-25": ["--talker-cuda-graph", "--code2wav-initial-codec-chunk-frames", "25"],
        "ramp": ["--talker-cuda-graph", "--code2wav-codec-chunk-ramp", "5,10,25"],
        # Approximate: the decoder's real reach is 568 frames, so 25 frames of
        # re-read context leaves a large sample error. Here to show what the
        # throughput mode costs, not as a recommendation.
        "stateless": ["--talker-cuda-graph", "--code2wav-mode", "stateless"],
    },
    "scheduling": {
        policy: [
            "--talker-cuda-graph",
            "--concurrency",
            "8",
            "--max-num-seqs",
            "8",
            "--chunk-policy",
            policy,
        ]
        for policy in ("fifo", "deadline", "round_robin")
    },
}

COLUMNS = [
    ("label", "config", "{}", 26),
    ("rtf_aggregate", "RTF", "{:.3f}", 7),
    ("ttfa_p50", "TTFA p50", "{:.3f}", 9),
    ("ttfa_p95", "TTFA p95", "{:.3f}", 9),
    ("max_inter_chunk_gap_max", "worst gap", "{:.3f}", 10),
    ("underruns", "under", "{}", 6),
    ("peak_memory_mb_max", "peak MiB", "{:.0f}", 9),
]


def run_row(label: str, flags: list[str], common: list[str], json_path: Path) -> dict | None:
    command = [sys.executable, "-m", "qtts.entrypoints.benchmark", *common, *flags, "--label", label]
    print(f"\n=== {label}: {' '.join(flags) or '(baseline)'}", flush=True)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        # A row that cannot run is a result too (unsupported dtype, fp8 on the
        # wrong architecture, concurrency above max_num_seqs), so the sweep
        # reports it and keeps going.
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        print(f"    failed: {' | '.join(tail)}", flush=True)
        return None
    print("    " + result.stdout.strip().splitlines()[-1], flush=True)
    with open(json_path) as fh:
        return json.loads(fh.read().strip().splitlines()[-1])


def print_table(records: list[dict]) -> None:
    header = "".join(title.ljust(width) for _, title, _, width in COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for record in records:
        row = ""
        for key, _, fmt, width in COLUMNS:
            value = record.get(key)
            row += (fmt.format(value) if value is not None else "-").ljust(width)
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", default="optimizations", choices=sorted(SUITES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--json-path", default="sweep.jsonl")
    parser.add_argument("--only", default="", help="comma-separated labels to run")
    args = parser.parse_args(argv)

    json_path = Path(args.json_path)
    json_path.unlink(missing_ok=True)
    common = [
        "--model",
        args.model,
        "--repeats",
        str(args.repeats),
        "--max-model-len",
        str(args.max_model_len),
        # No per-stage synchronization: the timing hooks cost a synchronize per
        # stage per step, which is ~3% of the number being compared.
        #
        # Deliberately unseeded, even though that leaves sampling variance in
        # the audio length between rows. A seeded request cannot use the
        # captured code-predictor graph, so the rows that exist to measure
        # graphs would measure the eager fallback instead. RTF divides by audio
        # duration and the rest is a median over --repeats.
        "--no-collect-stage-timings",
        "--json-path",
        str(json_path),
    ]

    rows = SUITES[args.suite]
    if args.only:
        wanted = {label.strip() for label in args.only.split(",")}
        rows = {label: flags for label, flags in rows.items() if label in wanted}
        if not rows:
            parser.error(f"--only matched nothing; {args.suite} has {sorted(SUITES[args.suite])}")

    records = [r for label, flags in rows.items() if (r := run_row(label, flags, common, json_path))]
    if records:
        print_table(records)
        print(f"\n{len(records)} rows in {json_path}")
    return 0 if len(records) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
