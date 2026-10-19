"""Per-shape measurement: is the quantized GEMM actually faster here?

A quantization method that is applied everywhere reports one number for a
mixture of layers where it wins big and layers where it loses, and the mixture
usually averages to nothing. ``kernels.py`` has the measurements: the same int8
kernel is 1.94x on the talker's gate_up projection and 0.59x on its down
projection, purely because of how wide the output is.

So the choice is made per shape, by measuring, at load time. There are about
six distinct linear shapes in the talker and the code predictor, each costs a
few milliseconds to time, and the result is cached by
``(shape, batch, dtype)``.

Timing is done inside a CUDA graph. That is not an optimisation of the
measurement, it is the only way to make it mean anything: eagerly, launching a
Triton kernel costs ~20 us against a 5-20 us GEMM, so an eager comparison
measures Python and answers "never use Triton" for every shape. The engine
replays the decode step from a graph when ``--talker-cuda-graph`` is set, so the
graph measurement is also the one that matches how the layer will really run.

The batch size used for the measurement is the engine's ``max_num_seqs``, not
1. It is an upper bound rather than the average, but the ranking barely moves
across M=1..32 (see the table in ``kernels.py``) while a per-batch-size decision
would mean re-deciding every time the scheduler's occupancy changed.
"""

from __future__ import annotations

import time

import torch

from ..utils import get_logger

logger = get_logger("quant.calibrate")

_CACHE: dict[tuple, bool] = {}


def graph_time(fn, inner: int = 10, iters: int = 50) -> float:
    """Seconds per call of ``fn``, measured from a captured graph.

    ``inner`` calls are captured into one graph so that replay overhead is
    amortised the same way the engine amortises it, and the graph is replayed
    ``iters`` times.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / (iters * inner)


def eager_time(fn, iters: int = 200) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters


def worth_quantizing(
    quantized_call,
    reference_call,
    *,
    key: tuple,
    margin: float = 1.05,
    use_graph: bool = True,
) -> bool:
    """Whether ``quantized_call`` beats ``reference_call`` by ``margin``.

    ``margin`` above 1 keeps a shape unquantized unless the win is real: a 2%
    difference at these durations is measurement noise, and swapping a GEMM for
    noise costs accuracy for nothing.
    """
    if key in _CACHE:
        return _CACHE[key]
    timer = graph_time if use_graph else eager_time
    try:
        reference = timer(reference_call)
        quantized = timer(quantized_call)
    except Exception as exc:
        # Capture can fail for reasons that have nothing to do with the layer
        # (an allocator in a bad state, a stream already capturing). Declining
        # to quantize is the safe answer.
        logger.warning("calibration failed for %s (%s); leaving it unquantized", key, exc)
        _CACHE[key] = False
        return False
    verdict = quantized * margin < reference
    _CACHE[key] = verdict
    logger.debug(
        "%s: reference %.1f us, quantized %.1f us (%.2fx) -> %s",
        key,
        reference * 1e6,
        quantized * 1e6,
        reference / quantized,
        "quantize" if verdict else "keep",
    )
    return verdict


def clear_cache() -> None:
    _CACHE.clear()
