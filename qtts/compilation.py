"""CUDA graph capture for the talker decode step.

This is the single largest optimization in stage 0, and the reason is not
arithmetic. Profiling one decode step of the 28-layer talker at batch 1:

    2071 CUDA kernel launches
    6.3 ms of GPU time
    19.2 ms of wall clock

Two thirds of the step is the CPU failing to keep up with launching work that
small. Every individual kernel is fine; there are just far too many of them,
and 28 layers x (4 GEMMs + 4 RMSNorms + rotary + attention + 2 residual adds) is
what an autoregressive model at batch 1 looks like. Nothing about the
arithmetic can fix that -- the fix is to stop asking the CPU to launch it.

A captured graph replays the whole step as one submission. What it costs is that
every shape in the step has to be fixed at capture time, which for a decode step
means three things:

* **Batch size.** Requests arrive and finish, so the row count changes. Graphs
  are captured for the sizes in ``cuda_graph_batch_sizes`` and a step pads up to
  the next captured size. Padded rows are marked inactive, so their keys go to
  the cache's scratch slot rather than over a live request's.
* **Key length.** The KV cache grows by one position per frame, so a graph
  captured at one length is wrong at the next. Lengths are bucketed to
  ``key_len_bucket``, which turns "a new shape every frame" into "a new shape
  every 256 frames" -- 16 graphs for a 4096-position cache instead of 4096. The
  cost is attending over up to 255 masked-out positions, which is a few percent
  of a step that was never compute-bound.
* **Addresses.** A replay writes to the same memory it captured. The input
  buffers below are therefore allocated once and copied into, and the returned
  hidden state is the graph's own output buffer: it is valid until the next
  replay, and the caller has to be done with it by then. The engine is, because
  it stacks the hidden states into a fresh tensor at the top of the next step.

The capture itself allocates from a shared memory pool, so N graphs do not cost
N copies of the activation memory.

The code predictor gets the same treatment, and needs it more: filling in the
15 residual codebooks of one frame is 15 sequential prefixes of a 5-layer model,
~1500 launches for a few hundred microseconds of arithmetic, and it costs ~49 ms
against the talker's 19. Unlike the talker it has no dynamic shape at all -- the
loop runs a fixed number of times over a fixed-width frame -- so one graph per
batch size covers it. The only thing that disqualifies it is a per-request
``torch.Generator``, which a captured graph cannot advance; seeded requests fall
back to the eager loop.
"""

from __future__ import annotations

import torch

from .models.code_predictor import CodePredictor
from .models.talker import Qwen3TTSTalker
from .utils import get_logger, human_bytes

logger = get_logger("compilation")


class TalkerDecodeGraphs:
    """Captured talker decode steps, keyed by ``(rows, key_len bucket)``.

    Capture is lazy. Capturing every (batch, length) pair up front would mean
    16 x 7 = 112 graphs for a run that usually touches four of them, and each
    capture has to actually execute the step.
    """

    def __init__(
        self,
        talker: Qwen3TTSTalker,
        batch_sizes: tuple[int, ...],
        key_len_bucket: int = 256,
    ) -> None:
        self.talker = talker
        self.device = talker.device
        self.max_model_len = talker.max_model_len
        self.key_len_bucket = max(1, key_len_bucket)
        capacity = talker.cache.max_seqs
        self.batch_sizes = tuple(sorted({min(b, capacity) for b in batch_sizes if b >= 1}))
        if not self.batch_sizes:
            self.batch_sizes = (capacity,)

        hidden_size = talker.tc.hidden_size
        largest = self.batch_sizes[-1]
        # One set of input buffers for the largest batch; smaller graphs capture
        # views of them, so every graph reads from the same addresses and the
        # caller only ever writes to one place.
        self.embeds = torch.zeros(largest, hidden_size, dtype=talker.dtype, device=self.device)
        self.seq_lens = torch.zeros(largest, dtype=torch.long, device=self.device)
        self.active = torch.zeros(largest, dtype=torch.bool, device=self.device)
        self.host_seq_lens = torch.zeros(largest, dtype=torch.long).pin_memory()
        self.host_active = torch.zeros(largest, dtype=torch.bool).pin_memory()

        self.pool = None
        self.graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.replays = 0
        self.captures = 0

    # -- shape selection ----------------------------------------------------

    def _rows_for(self, rows: int) -> int:
        for size in self.batch_sizes:
            if size >= rows:
                return size
        raise ValueError(f"{rows} rows exceeds the largest captured batch size {self.batch_sizes[-1]}")

    def _key_len_for(self, key_len: int) -> int:
        bucketed = ((key_len + self.key_len_bucket - 1) // self.key_len_bucket) * self.key_len_bucket
        return min(max(bucketed, self.key_len_bucket), self.max_model_len)

    # -- capture ------------------------------------------------------------

    def _capture(self, rows: int, key_len: int) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        embeds = self.embeds[:rows]
        seq_lens = self.seq_lens[:rows]
        active = self.active[:rows]

        # Warm up on a side stream first. The first call through a code path
        # allocates workspaces, loads kernels and may autotune, none of which
        # can happen inside a capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.talker.decode(embeds, seq_lens, key_len, active)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            output = self.talker.decode(embeds, seq_lens, key_len, active)
        if self.pool is None:
            self.pool = graph.pool()
        self.captures += 1
        logger.info(
            "captured talker decode graph: %d rows x %d keys (%d graphs, pool %s)",
            rows,
            key_len,
            self.captures,
            human_bytes(torch.cuda.memory_reserved()),
        )
        return graph, output

    # -- replay -------------------------------------------------------------

    def run(
        self,
        embeds: torch.Tensor,
        seq_lens: list[int],
        active: list[bool],
        key_len: int,
    ) -> torch.Tensor:
        """Replay a step, padding up to a captured shape.

        Returns the graph's output buffer, valid until the next replay.
        """
        rows = embeds.shape[0]
        padded_rows = self._rows_for(rows)
        padded_key_len = self._key_len_for(key_len)

        self.embeds[:rows].copy_(embeds)
        if padded_rows > rows:
            # Zero rather than leave the previous step's values: a padded row is
            # marked inactive so its key goes to scratch, but it still runs
            # through the network, and stale activations there would show up in
            # a profile as work nobody asked for.
            self.embeds[rows:padded_rows].zero_()
        self.host_seq_lens[:rows] = torch.tensor(seq_lens, dtype=torch.long)
        self.host_active[:rows] = torch.tensor(active, dtype=torch.bool)
        self.host_seq_lens[rows:padded_rows] = 0
        self.host_active[rows:padded_rows] = False
        self.seq_lens[:padded_rows].copy_(self.host_seq_lens[:padded_rows], non_blocking=True)
        self.active[:padded_rows].copy_(self.host_active[:padded_rows], non_blocking=True)

        key = (padded_rows, padded_key_len)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(padded_rows, padded_key_len)
            self.graphs[key] = entry
        graph, output = entry
        graph.replay()
        self.replays += 1
        return output[:rows]

    def stats(self) -> dict[str, int]:
        return {"graphs": len(self.graphs), "replays": self.replays}


class CodePredictorGraphs:
    """Captured residual-codebook loops, keyed by batch size.

    Everything the loop does is a fixed-shape tensor op, including the sampling,
    so the whole 15-step walk over the codebook axis is one graph. The sampling
    is why capture needs care: ``torch.rand`` inside a graph works because
    PyTorch captures the default generator's philox offset and advances it per
    replay, but a *user-supplied* generator has no such handling, so a seeded
    request has to take the eager path instead. Silently ignoring the generator
    would make ``--seed`` stop being reproducible while still appearing to work.
    """

    def __init__(self, predictor: CodePredictor, batch_sizes: tuple[int, ...], capacity: int) -> None:
        self.predictor = predictor
        self.device = predictor.device
        self.batch_sizes = tuple(sorted({min(b, capacity) for b in batch_sizes if b >= 1})) or (capacity,)
        largest = self.batch_sizes[-1]
        talker_hidden = predictor.small_to_mtp_projection.in_features

        self.layer0_code = torch.zeros(largest, dtype=torch.long, device=self.device)
        self.layer0_embed = torch.zeros(largest, talker_hidden, dtype=predictor.dtype, device=self.device)
        self.talker_hidden = torch.zeros(largest, talker_hidden, dtype=predictor.dtype, device=self.device)

        self.pool = None
        self.graphs: dict[tuple[int, bool, float, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.replays = 0

    def _rows_for(self, rows: int) -> int:
        for size in self.batch_sizes:
            if size >= rows:
                return size
        raise ValueError(f"{rows} rows exceeds the largest captured batch size {self.batch_sizes[-1]}")

    def _capture(self, rows: int, do_sample: bool, temperature: float, top_k: int):
        args = (self.layer0_code[:rows], self.layer0_embed[:rows], self.talker_hidden[:rows])
        kwargs = {"do_sample": do_sample, "temperature": temperature, "top_k": top_k}

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.predictor(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            codes = self.predictor(*args, **kwargs)
        if self.pool is None:
            self.pool = graph.pool()
        logger.info("captured code predictor graph: %d rows (sample=%s)", rows, do_sample)
        return graph, codes

    def run(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        talker_hidden: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        rows = layer0_code.shape[0]
        padded = self._rows_for(rows)
        self.layer0_code[:rows].copy_(layer0_code)
        self.layer0_embed[:rows].copy_(layer0_embed)
        self.talker_hidden[:rows].copy_(talker_hidden)
        if padded > rows:
            self.layer0_code[rows:padded].zero_()
            self.layer0_embed[rows:padded].zero_()
            self.talker_hidden[rows:padded].zero_()

        key = (padded, do_sample, temperature, top_k)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(padded, do_sample, temperature, top_k)
            self.graphs[key] = entry
        graph, codes = entry
        graph.replay()
        self.replays += 1
        return codes[:rows]

    def stats(self) -> dict[str, int]:
        return {"graphs": len(self.graphs), "replays": self.replays}


def build_talker_graphs(talker: Qwen3TTSTalker, runtime) -> TalkerDecodeGraphs | None:
    if not _graphs_possible(runtime):
        return None
    return TalkerDecodeGraphs(
        talker,
        batch_sizes=runtime.cuda_graph_batch_sizes,
        key_len_bucket=runtime.cuda_graph_key_len_bucket,
    )


def build_code_predictor_graphs(
    predictor: CodePredictor, runtime, capacity: int
) -> CodePredictorGraphs | None:
    if not _graphs_possible(runtime):
        return None
    return CodePredictorGraphs(predictor, runtime.cuda_graph_batch_sizes, capacity)


def _graphs_possible(runtime) -> bool:
    if not runtime.cuda_graph:
        return False
    if not torch.cuda.is_available():
        logger.warning("--talker-cuda-graph requested without CUDA; ignoring")
        return False
    if runtime.compile_mode != "none":
        # torch.compile manages its own cudagraph trees, and letting both
        # capture the same module produces a replay of a replay.
        logger.warning(
            "--talker-cuda-graph and --talker-compile-mode %s both capture graphs; "
            "using torch.compile only",
            runtime.compile_mode,
        )
        return False
    return True
