# Measurements

All numbers from one H100 80GB in the `danie_torch_2504` container (torch
2.7.0a0+nv25.04, triton 3.2.0, CUDA 12.9), checkpoint
`Qwen3-TTS-12Hz-1.7B-CustomVoice`, speaker `vivian`, the 26-character Chinese
sentence in `RunConfig.text`, which comes out as ~59 frames of ~4.7 s of audio.

Reproduce a table with:

```bash
python examples/benchmark/sweep.py --model $MODEL --suite optimizations   # and concurrency, chunking, scheduling
```

Each row is a separate process: three measured iterations after a discarded
warmup, `--no-collect-stage-timings`, no fixed seed. Sampling is stochastic, so
audio length varies by a few percent between rows; RTF divides it out, and
TTFA does not depend on it.

Not seeding is deliberate. A captured CUDA graph cannot draw from a
caller-supplied generator, so a seeded request falls back to the eager code
predictor and the same configuration measures 0.656 RTF instead of 0.162 — a
reproducibility flag that quadruples the number it is there to make
reproducible. The engine warns when both are on.

**Reading the columns.** RTF is wall clock over audio duration, aggregated
across concurrent streams — below 1.0 is faster than real time. TTFA is
time-to-first-audio. "Worst gap" is the largest interval between consecutive
chunks, and an underrun is a gap longer than the audio delivered before it,
i.e. a gap a listener actually hears.

## Where a frame goes

One frame of audio = one talker step + fifteen code-predictor prefixes. The
split below is measured directly (`TalkerRunner.decode_step` versus the talker
forward alone), batch 1 and batch 4, in milliseconds per frame:

| | step | talker forward | code predictor |
| --- | --- | --- | --- |
| batch 1, eager norm, no graphs | 66.97 | 18.51 | 48.47 |
| batch 1, fused norm, no graphs | 64.98 | 17.56 | 47.42 |
| batch 1, eager norm, graphs | 17.57 | 5.74 | 11.83 |
| batch 1, fused norm, graphs | **11.85** | 4.16 | 7.69 |
| batch 4, fused norm, no graphs | 67.85 | 18.76 | 49.08 |
| batch 4, fused norm, graphs | **13.16** | 4.93 | 8.23 |

Three things worth taking from this table.

**The code predictor, not the talker, is most of a frame.** It is a 5-layer,
1024-wide model — 6% of the parameters — and it costs 2.6x the 28-layer talker,
because it runs fifteen times per frame with no K/V cache to reuse. Optimizing
the talker first is the obvious move and the wrong one.

**Both halves are launch-bound, not compute-bound.** Capturing the step into
CUDA graphs takes it from 67.0 ms to 11.9 ms, a 5.6x improvement that changes
no arithmetic whatsoever. Everything else on this page is small compared to
this, and several optimizations only become visible after it.

**Batch 4 costs the same as batch 1.** 13.16 ms versus 11.85 ms for four times
the audio: 3.3 ms per stream per frame. Stage 0 decode is entirely
bandwidth-bound at these sizes, so concurrency is close to free until the batch
is large enough for the GEMMs to become compute-bound.

## Optimizations, end to end

`--suite optimizations`, one stream:

| config | RTF | TTFA p50 | worst gap | underruns | peak MiB |
| --- | --- | --- | --- | --- | --- |
| baseline (eager norm, eager snake) | 0.849 | 0.357 | 1.670 | 3 | 6365 |
| + fused RMSNorm | 0.834 | 0.349 | 1.639 | 3 | 6365 |
| + fused SnakeBeta | 0.834 | 0.352 | 1.639 | 3 | 6365 |
| + CUDA graphs | **0.162** | **0.082** | 0.312 | 0 | 6461 |
| + int8 weight-only, calibrated | 0.159 | 0.083 | 0.297 | 0 | 6046 |
| + int8 weight-only, uniform | 0.167 | 0.086 | 0.334 | 0 | 5040 |

The baseline underruns three times per utterance at a single stream: 67 ms per
frame against 80 ms of audio per frame leaves no headroom for a decoder call,
so the first chunk boundary arrives late and stays late. Graphs remove the
underruns and cut TTFA by 4.3x.

The last two rows are the argument for per-shape calibration. Quantizing only
the shapes where the int8 kernel measured faster gives 0.159 RTF and 415 MiB
back; quantizing every eligible shape gives 1.4 GiB back and is *slower than
not quantizing at all* (0.167 vs 0.162). Which one is right depends on whether
you are short of memory or of time — but the uniform row is the one that gets
reported as "weight-only quantization didn't help", and it is a statement about
four particular matrix shapes rather than about quantization.

### Why the fusions look small here

Fused RMSNorm is worth 2% end-to-end in the table above and 32% of a
graph-replayed step (17.57 → 11.85 ms). Both numbers are real: without graphs
the step is waiting on kernel launches, and removing five of the six kernels in
a normalization removes launches that were overlapping with other launches.
Measure a fusion before you fix the launch overhead and you will conclude it
does nothing.

### The int8 kernel, per shape

`python examples/kernels/tune_w8_gemm.py`, M=4, bf16, inside a CUDA graph:

| layer | shape | cuBLAS bf16 | fused int8 | naive dequant | ratio |
| --- | --- | --- | --- | --- | --- |
| qkv | 4096x2048 | 5.2 µs | 5.8 µs | 49.7 µs | 0.90x |
| o_proj | 2048x2048 | 5.1 µs | 5.6 µs | 25.3 µs | 0.90x |
| gate_up | 12288x2048 | 19.3 µs | 9.9 µs | 157.9 µs | **1.94x** |
| down_proj | 2048x6144 | 8.4 µs | 13.1 µs | 78.6 µs | 0.64x |

The "naive dequant" column is `F.linear(x, w_int8.to(dtype) * scale)` — the
obvious way to write weight-only quantization, 2.5x to 8x *slower* than not
quantizing, because it writes a full-precision copy of the weight to global
memory for the GEMM to read back. The fused kernel wins only where the output
is wide enough to fill the machine: at M=4 the only parallelism is over N, so
N=12288 reaches 2.5 TB/s while N=2048 reaches 0.9 TB/s against cuBLAS's 3.3.
Reasoning about this in detail is in `qtts/quant/kernels.py`.

## Concurrency

`--suite concurrency`, CUDA graphs on, `max_num_seqs=8`:

| streams | RTF aggregate | TTFA p50 | TTFA p95 | worst gap | underruns | peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.165 | 0.082 | 0.082 | 0.301 | 0 | 5565 |
| 2 | 0.102 | 0.111 | 0.120 | 0.448 | 0 | 5629 |
| 4 | 0.060 | 0.165 | 0.191 | 0.486 | 0 | 5675 |
| 8 | 0.039 | 0.276 | 0.338 | 0.806 | 0 | 5726 |

4.2x the throughput from 1 to 8 streams for 3.4x the time-to-first-audio and
161 MiB. The memory is flat because the K/V cache for all 16 slots is allocated
at startup (`max_num_seqs` x `max_model_len`), so concurrency costs nothing at
run time; what grows is the per-request Code2Wav state, a few MB each.

TTFA degrades because stage 1 is serial in stateful mode: eight first chunks
queue behind one decoder. That queue, not the talker, is what
`--chunk-policy` and `--code2wav-dynamic-initial-chunk` exist to manage.

## Chunking

`--suite chunking`, one stream, CUDA graphs on:

| config | RTF | TTFA p50 | worst gap | peak MiB |
| --- | --- | --- | --- | --- |
| `--pipeline-mode sync` | 0.158 | 0.911 | 0.000 | 6645 |
| first chunk 5 frames | 0.161 | 0.083 | 0.302 | 6461 |
| first chunk 25 frames | 0.160 | 0.319 | 0.325 | 6462 |
| ramp 5,10,25 | 0.161 | 0.083 | 0.323 | 6461 |
| stateless decoder | 0.161 | 0.082 | 0.302 | 6521 |

Chunking is close to free in throughput (0.158 to 0.161 RTF, 2%) and worth 11x
in time-to-first-audio. The first chunk size is the entire lever: 5 frames
gives 83 ms, 25 frames gives 319 ms, and the difference is 20 frames of audio
the listener is waiting through. `sync` has no gaps at all because it delivers
once — the right baseline, and unusable for streaming.

The `stateless` row looks like a free win and is not: it matches the stateful
row on every column here while changing the audio, because the decoder's true
receptive field is ~568 frames and the mode re-reads 25. From
`tests/test_code2wav_modes.py`, peak sample error against whole-utterance
decoding on a signal bounded by 1.0:

| left context | 25 | 71 | 120 | stateful (any chunk size) |
| --- | --- | --- | --- | --- |
| peak error | ~0.9 | lower | lower still | ~1e-8 (fp32 headroom) |

It buys the ability to batch chunks from different requests into one decoder
call, which is the only way stage 1 stops being serial. That is a real
throughput argument on a loaded server, and it is a quality trade, not a
speedup.

## Scheduling policy

`--suite scheduling`, 8 identical streams:

| policy | RTF | TTFA p50 | TTFA p95 | worst gap |
| --- | --- | --- | --- | --- |
| fifo | 0.038 | 0.275 | 0.337 | 0.484 |
| deadline | 0.037 | 0.276 | 0.338 | 0.605 |
| round_robin | 0.037 | 0.275 | 0.342 | 0.518 |

Indistinguishable, and that is the expected result for this workload rather
than a bug: eight copies of the same sentence submitted at the same instant
reach their chunk boundaries in lockstep, so every policy sees a queue of
equivalent chunks and any order is as good as any other. A policy can only pay
when the requests differ — mixed text lengths, staggered arrivals, a slow
consumer on one stream. Making that workload measurable is the obvious next
thing to build here; until then, `deadline`'s advantage is a claim about a
workload this suite does not generate, which is why `fifo` is the default.

## What has not been measured

- Anything above 8 concurrent streams, or `max_model_len` beyond 1024 frames.
- fp8 W8A8 (`--quantization fp8`) on this model. The path exists and runs; no
  row here uses it because weight-only is the variant that suits a
  bandwidth-bound decode loop, and the fp8 comparison needs a compute-bound
  batch size to be interesting.
- Code2Wav in bf16 (`--code2wav-dtype bfloat16`). The decoder runs ~20 serial
  convolution stages per frame in fp32; the speedup is easy to measure and the
  accumulated rounding needs a waveform comparison, not an RTF number.
