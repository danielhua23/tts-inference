"""Re-derive the tile shape and the break-even table for the int8 GEMM.

    python examples/kernels/tune_w8_gemm.py            # the comparison table
    python examples/kernels/tune_w8_gemm.py --tune     # sweep tile shapes

The constants in ``qtts/quant/kernels.py`` (``BLOCK_N``, ``BLOCK_K``,
``NUM_WARPS``, ``NUM_STAGES``) came from this sweep on an H100, and the
per-shape verdicts in ``qtts/quant/int8.py`` came from the comparison. Both are
hardware-dependent, so this script exists to be re-run rather than trusted:
a GPU with a different SM count or a different HBM ratio moves the break-even,
and the answer for one shape is not the answer for the next.

Everything is timed inside a CUDA graph. Eagerly, every row loses by roughly the
~20 us it costs Python to launch a kernel that runs in 5-20 us, which measures
the launch path rather than the kernel.

The third column is the reason the fused kernel exists: ``F.linear(x,
weight.to(x.dtype) * scale)`` is the natural way to write weight-only
quantization and is ~8x slower than not quantizing at all, because it
materializes the dequantized weight in global memory before the GEMM reads it.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

import qtts.quant.kernels as kernels

#: The talker's four distinct linear shapes, as (label, N, K).
SHAPES = [
    ("qkv", 4096, 2048),
    ("o_proj", 2048, 2048),
    ("gate_up", 12288, 2048),
    ("down_proj", 2048, 6144),
]


def graph_time(fn, inner: int = 10, iters: int = 200) -> float:
    """Microseconds per call, replayed from a graph.

    ``inner`` calls are captured per replay so that the replay's own overhead
    is divided away as well.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters / inner * 1e6


def naive_dequant_linear(x, packed, scale, dtype) -> torch.Tensor:
    """Weight-only quantization written the obvious way, for the third column.

    Correct, and slower than not quantizing: the dequantized weight goes to
    global memory and the GEMM reads it back.
    """
    return F.linear(x, packed.to(dtype) * scale[:, None])


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (weight.abs().amax(1) / 127).to(weight.dtype)
    packed = (weight / scale[:, None].float()).round().clamp(-127, 127).to(torch.int8)
    return packed, scale


def run_tile(x, packed, scale, block_n, block_k, warps, stages) -> torch.Tensor:
    """Call the kernel with an explicit tile shape instead of the module's."""
    rows, k = x.shape
    n = packed.shape[0]
    block_m = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    grid = ((rows + block_m - 1) // block_m, (n + block_n - 1) // block_n)
    kernels._w8_gemm_kernel[grid](
        x, packed, scale, x, out,
        rows, n, k,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def compare(rows: int, dtype: torch.dtype) -> None:
    print(f"\nM={rows}, {dtype}, {torch.cuda.get_device_name(0)}")
    print(f"{'layer':>10} {'shape':>12} {'bf16':>9} {'fused w8':>9} {'naive w8':>9} {'ratio':>7} {'err':>9}")
    for label, n, k in SHAPES:
        weight = torch.randn(n, k, device="cuda", dtype=dtype) / (k**0.5)
        packed, scale = quantize(weight)
        x = torch.randn(rows, k, device="cuda", dtype=dtype)

        reference = F.linear(x, weight)
        fused = kernels.w8_gemm(x, packed, scale)
        error = (fused - reference).abs().max().item() / reference.abs().max().item()

        base_us = graph_time(partial(F.linear, x, weight))
        fused_us = graph_time(partial(kernels.w8_gemm, x, packed, scale))
        naive_us = graph_time(partial(naive_dequant_linear, x, packed, scale, dtype))
        print(
            f"{label:>10} {f'{n}x{k}':>12} {base_us:8.1f}u {fused_us:8.1f}u {naive_us:8.1f}u "
            f"{base_us / fused_us:6.2f}x {error:9.2e}"
        )


def tune(rows: int, dtype: torch.dtype) -> None:
    for label, n, k in SHAPES:
        weight = torch.randn(n, k, device="cuda", dtype=dtype) / (k**0.5)
        packed, scale = quantize(weight)
        x = torch.randn(rows, k, device="cuda", dtype=dtype)
        base_us = graph_time(partial(F.linear, x, weight))

        results = []
        for block_n, block_k, warps, stages in itertools.product(
            (32, 64, 128, 256), (32, 64, 128), (4, 8), (2, 3, 4)
        ):
            call = partial(run_tile, x, packed, scale, block_n, block_k, warps, stages)
            try:
                us = graph_time(call, iters=60)
            except Exception:
                # Out of shared memory or an unsupported combination; a tile
                # that will not compile is simply not a candidate.
                continue
            results.append((us, block_n, block_k, warps, stages))
        results.sort()
        print(f"\n{label} N={n} K={k}, cuBLAS bf16 {base_us:.1f}u")
        for us, block_n, block_k, warps, stages in results[:5]:
            print(
                f"    {us:6.1f}u {base_us / us:5.2f}x  BLOCK_N={block_n} BLOCK_K={block_k} "
                f"warps={warps} stages={stages}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true", help="sweep tile shapes instead of comparing")
    parser.add_argument("--rows", type=int, default=4, help="batch size M (talker decode is 1-64)")
    args = parser.parse_args(argv)

    available, reason = kernels.kernel_available()
    if not available or not torch.cuda.is_available():
        print(reason or "no CUDA device")
        return 1

    torch.manual_seed(0)
    with torch.inference_mode():
        if args.tune:
            tune(args.rows, torch.bfloat16)
        else:
            compare(args.rows, torch.bfloat16)
    return 0


if __name__ == "__main__":
    sys.exit(main())
