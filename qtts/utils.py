"""Logging, memory accounting and timing helpers."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

import torch

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def init_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or os.environ.get("QTTS_LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"qtts.{name}")


def human_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def module_param_bytes(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class StageTimer:
    """Wall-clock accumulator keyed by stage name.

    Every ``measure`` synchronizes, because the two stages hand tensors to each
    other and an unsynchronized split would charge stage 0 for work the stream
    had not finished. Set ``enabled=False`` to take the syncs out of the hot
    loop once the numbers are no longer being read.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def measure(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in self.totals.items()}

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method.

    Written out rather than imported so that a handful of repeats does not
    silently become "the max": with 3 samples, p95 and p99 are interpolations
    between the top two and should be read as such.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def write_wav(path: str, wav: torch.Tensor | object, sample_rate: int) -> None:
    """Write a mono float waveform. Uses soundfile when available, else stdlib."""
    import numpy as np

    if isinstance(wav, torch.Tensor):
        data = wav.detach().to(torch.float32).cpu().numpy()
    else:
        data = np.asarray(wav, dtype=np.float32)
    data = np.reshape(data, (-1,))
    try:
        import soundfile as sf

        sf.write(path, data, sample_rate)
        return
    except ImportError:
        pass
    import wave

    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
