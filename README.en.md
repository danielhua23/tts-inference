# qtts — a minimal Qwen3-TTS inference runtime

A from-scratch implementation of Qwen3-TTS inference: the talker, the code
predictor, the codec decoder, the prompt assembly, a continuous-batching
scheduler and an HTTP server, in ~8k lines with no inference framework
underneath. It exists to be *modified* — every layer is a plain module you can
replace with a quantized or fused one, and every optimization is a flag with a
measured before-and-after.

It is checked against the official `qwen-tts` package bit for bit: same prompt
embeddings, same first-frame logits, same greedy code sequence. That is the
part that makes the rest trustworthy, because a TTS bug does not look like a
bug — it sounds like a fluent voice saying something slightly wrong.

```
text ──> prompt ──> ┌─────────────── stage 0 ───────────────┐ ──> ┌─ stage 1 ─┐ ──> 24 kHz audio
                    │ talker (28L, 1.7B)  →  code predictor │     │  Code2Wav │
                    │ AR over time, 12.5 Hz  AR over 16 RVQ │     │ 16 codes  │
                    │ one codebook-0 id      codebooks      │     │ → 1920    │
                    │ per frame              within a frame │     │  samples  │
                    └───────────────────────────────────────┘     └───────────┘
```

## Quick start

Everything runs inside the `danie_torch_2504` container, where torch and triton
come from the NGC image (see `requirements.txt` for why they must stay that
way).

```bash
docker exec -it -w /home/qwen3-tts-min danie_torch_2504 bash

# One utterance to a file.
python -m qtts.entrypoints.generate \
    --model /home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
    --speaker vivian --output-path out.wav

# The same, streamed, with per-chunk arrival times.
python -m qtts.entrypoints.generate --model ... --stream

# HTTP server.
python -m qtts.entrypoints.serve --model ... --max-num-seqs 8 --talker-cuda-graph --port 8000
curl -s localhost:8000/generate -H 'content-type: application/json' \
    -d '{"text": "你好", "speaker": "vivian"}' -o out.wav
curl -sN localhost:8000/generate_stream -H 'content-type: application/json' \
    -d '{"text": "你好", "speaker": "vivian"}' | ffplay -f s16le -ar 24000 -ac 1 -

# One configuration, measured carefully.
python -m qtts.entrypoints.benchmark --model ... --concurrency 8 --talker-cuda-graph

# Several configurations, one table (one process per row; see the script).
python examples/benchmark/sweep.py --model ... --suite optimizations
```

In Python:

```python
from qtts import EngineConfig, SchedulerConfig, TalkerConfig, TTSEngine, TTSRequest

engine = TTSEngine(EngineConfig(
    model="/home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    talker=TalkerConfig(cuda_graph=True),
    scheduler=SchedulerConfig(max_num_seqs=8),
))

for chunk in engine.generate_stream(TTSRequest(text="你好", speaker="vivian")):
    play(chunk.audio)            # float32 mono in [-1, 1], on the host
```

## The model, in the terms the code uses

**Two autoregressive axes, not one.** The talker generates along *time*: one
step per 12.5 Hz audio frame, emitting the codebook-0 id for that frame. The
code predictor then generates along the *codebook* axis inside that single
frame, walking codebooks 1..15 conditioned on the talker's hidden state. So one
frame of audio costs one talker step plus fifteen code-predictor prefixes, and
the second number is why `qtts/compilation.py` exists.

**Dual-stream input embedding.** Every talker position is the sum of a
projected text embedding and a codec embedding — the text stream and the audio
stream are added, not concatenated. Getting the relative offset of the two
streams wrong produces fluent speech with the wrong words, which is the failure
mode `scripts/dump_reference.py` was written to catch.

**The codec head is wider than the vocabulary.** 3072 logits, of which only
`[1, 2048)` and the EOS id `2150` are legal. The mask is not cosmetic: without
it the model samples ids the decoder has no codebook entry for.

**Code2Wav is not a vocoder you can chunk freely.** It is an RVQ dequantizer, a
pre-convolution, an 8-layer transformer with `sliding_window=72`, and four
upsampling decoder blocks. The window bounds *each layer*, so the true
receptive field is ~568 frames, about 45 seconds — not the 71 the window
suggests. This one number decides whether streaming Code2Wav is exact or
approximate, and both modes are implemented so the difference can be measured
(`tests/test_code2wav_modes.py`).

## Layout

| Path | What lives there |
| --- | --- |
| `qtts/models/talker.py` | Stage 0: 28 layers, GQA, per-head RMSNorm, slot-indexed KV cache with a scratch slot for padded rows |
| `qtts/models/code_predictor.py` | The codebook-axis AR: 15 embedding tables, 15 heads, one exact prefix walk per codebook |
| `qtts/models/code2wav.py` | Stage 1, and the three decode paths: `decode_exact`, `decode_window` (stateful, exact), `chunked_decode` (stateless, approximate) |
| `qtts/models/prompt.py` | CustomVoice prompt assembly: control tags, the dual-stream sum, streaming vs non-streaming text conditioning |
| `qtts/models/loader.py` | safetensors loading, including the q/k/v and gate/up merges into fused linears |
| `qtts/layers/` | The replaceable parts: `LinearBase` with a quantization hook, RMSNorm (fused Triton / bit-exact eager), SnakeBeta, RoPE, causal conv |
| `qtts/engine/scheduler.py` | Admission, slot allocation, backpressure, and which pending chunk stage 1 runs next |
| `qtts/engine/chunking.py` | When to hand frames to stage 1: fixed, ramp, or load-dependent first chunk |
| `qtts/engine/talker_runner.py` | The decode step: sampling, the code predictor, stop conditions, cache advance |
| `qtts/engine/engine.py` | The loop: admit → one talker frame → collect chunks → decode → deliver |
| `qtts/quant/` | Quantization methods plus the per-shape calibration that decides where they are used |
| `qtts/compilation.py` | CUDA graph capture for the talker step and the code predictor |
| `qtts/config.py` | Every knob, with the reasoning for its default; each one has a flag and each one is read by something |
| `qtts/serving/` | aiohttp server; one thread owns the engine, handlers only touch queues |
| `qtts/entrypoints/` | The three CLIs: `generate`, `serve`, `benchmark` |
| `examples/` | The sweep driver and the kernel tuning script that produced the constants in `qtts/quant/kernels.py` |
| `.ref/` | vllm-omni's Qwen3-TTS source, kept for line-by-line comparison |

## Optimizations

Each is a flag, each is off-by-default when it changes the audio, and each has
a measured effect in [PERFORMANCE.md](PERFORMANCE.md). The short version: on
this model, at these batch sizes, *launch overhead is the bottleneck* and
almost everything else is a rounding error until it is dealt with.

| Flag | What it does | Why it matters |
| --- | --- | --- |
| `--talker-cuda-graph` | Replays one captured graph per (batch, key-length bucket) for the whole talker decode step | A step launches ~2000 kernels for a few ms of arithmetic. This is the single largest win and a precondition for most of the others |
| `--talker-rmsnorm-backend fused` | One Triton kernel per normalization instead of six eager ops | 112 norms per talker step, 300 per code-predictor frame, all on small tensors. Costs ~2 fp32 ULP, which is why parity runs use `eager` |
| `--code2wav-snake-backend triton` | Fused `x + sin²(αx)/β` | The decoder's activation, applied over every upsampled sample |
| `--quantization int8_weight_only` | Per-channel int8 weights, dequantized in registers inside the GEMM | Decode is weight-bandwidth-bound, so halving the weight bytes should halve the time — and does, on the shapes where the kernel reaches HBM peak |
| `--quant-calibrate` (default on) | Times each linear shape both ways at load and keeps the faster one | The same int8 kernel is 1.94x on the MLP up-projection and 0.64x on the down-projection. Applied uniformly the net effect is nil |
| `--pipeline-mode async_chunk` | Decode each chunk as soon as its frames exist | The only mode where time-to-first-audio is a separate number from total latency |
| `--code2wav-initial-codec-chunk-frames`, `--code2wav-codec-chunk-ramp` | Size of the first chunk, and how it grows | Time-to-first-audio is set entirely by the first chunk; the ramp buys it back without a cost cliff |
| `--code2wav-dynamic-initial-chunk` | Pick the first chunk size from current load | Under light load latency is free; under load a tiny first chunk is a decoder call every other stream waits behind |
| `--chunk-policy deadline` | Serve the stream with the least buffered audio first | A listener notices a gap only when the audio already delivered runs out |
| `--code2wav-mode stateless` | No per-request decoder state, so chunks from different requests batch | Real throughput win, **approximate audio** — see the receptive field note above |
| `--max-pending-chunks-per-request` | Pauses a talker whose chunks are piling up | Without it a fast talker builds an unbounded queue of undecoded frames |

## Verifying a change

Optimizing TTS without a parity check is how you ship a fast model that says
the wrong thing. The order these run in is the order they catch things.

1. **Bit-exact parity against the official package.** The reference runs in a
   separate interpreter (it pins `transformers` 4.57, the container ships 5.x),
   dumps tensors, and the suite compares them:

   ```bash
   PYTHONPATH=/path/to/ref-env python scripts/dump_reference.py --model $MODEL --out /tmp/reference.pt
   QTTS_MODEL=$MODEL QTTS_REFERENCE=/tmp/reference.pt python -m pytest tests -q
   ```

   Greedy throughout, so the first divergent frame localises the bug. Parity
   runs use `--talker-rmsnorm-backend eager`; the fused kernel's 2 ULP is
   enough to flip an occasional near-tied argmax, and
   `test_fused_norm_diverges_only_at_a_tie` pins the difference between that
   and a real bug.

2. **The engine-level suite, which needs no checkpoint.**
   `tests/test_scheduler.py` covers the parts whose bugs are inaudible:
   every frame emitted exactly once, slots reused lowest-index-first, the final
   chunk handed out once, left context only where the decoder needs it.

3. **Listen, then look at the gaps.** `--stream` prints each chunk's arrival
   time. The number that decides whether a stream is listenable is the *worst*
   inter-chunk gap, not the mean: playback survives a bad average and dies on
   one gap longer than the audio already buffered. The benchmark counts those
   as underruns.

4. **Compare waveforms, not just metrics,** for anything that touches Code2Wav
   — `--code2wav-mode stateless`, `--code2wav-dtype bfloat16`, quantizing the
   decoder. These change the audio by design or by accident, and RTF will
   happily improve while the output degrades.

## Pitfalls this code already stepped in

Kept because each one cost hours and none of them announced itself.

- **`seq_len` is both a count and a rotary position.** Incrementing it before
  the decode call leaves the current position zeroed and shifts every query's
  RoPE by one. The audio stays fluent and stops matching the text.
- **A dequantized weight written to global memory is slower than no
  quantization at all.** `F.linear(x, w_int8.to(dtype) * scale)` moves *more*
  bytes than the unquantized GEMM: 158 µs versus 19 µs on the talker's largest
  shape. Weight-only quantization only pays when the dequantization happens in
  registers inside the GEMM.
- **Weight-only quantization is not a uniform win, and measuring it eagerly
  measures the launch path.** Both facts are in `qtts/quant/kernels.py` with
  numbers; the consequence is `qtts/quant/calibrate.py`.
- **`sliding_window` bounds a layer, not the model.** Eight layers of window 72
  reach 568 frames. Sizing streaming context from the per-layer figure gives an
  error near full scale on a signal bounded by 1.0.
- **A Triton kernel defined inside a function that locally imports
  `triton.language as tl` fails with a bare `NameError` on its own
  annotations.** Triton resolves them against the module globals of the
  function it compiles. All kernels here are defined at module scope.
- **Folding the weight multiply into the RMSNorm expression changes the
  floating-point association order** and breaks bit-exactness against torch.
  `x32 = x32 * rsqrt(...)` then `weight * x32`, in that order.
- **Chunk boundaries must be measured against what was already sent,** not
  against a boundary derived from the frame count. Deriving them from the count
  re-emitted a chunk when a request was polled twice at the same frame, and
  emitted a chunk past the end when a request's length landed exactly on a
  boundary.
- **Asking for a fixed seed changes which code runs.** A captured graph cannot
  draw from a caller-supplied generator, so a seeded request quietly falls back
  to the eager code predictor and gets 4x slower. The engine warns; a benchmark
  that seeds for reproducibility otherwise reports the wrong path's numbers.
- **Gumbel sampling needs `-log(-log u)`.** Clamping `log u` — always negative
  — makes sampling uniform. Greedy decoding keeps working, so nothing fails
  until the audio quietly gets worse.

## Known gaps

- **No cancellation.** A client that abandons a stream mid-request still has it
  generated to completion. Doing this properly needs an abort path through the
  scheduler for a request that may be mid-step.
- **fp8 weight-only is not implemented** (W8A8 fp8 is). `qtts/quant/fp8.py`
  documents why: `torch._scaled_mm` wants both operands scaled, and an fp8
  weight with an unscaled activation has no fast path to call.
- **Stage 0 and stage 1 do not overlap.** One `step()` runs them in sequence.
  Overlapping them on separate streams would help throughput and would put an
  overlap-efficiency term into every measurement, so it is deliberately absent.
- **One GPU, one process.** No tensor or pipeline parallelism.
