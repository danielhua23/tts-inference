# qtts — 一个最小化的 Qwen3-TTS 推理框架

英文版：[README.en.md](README.en.md)

从零实现的 Qwen3-TTS 推理：talker、code predictor、codec decoder、prompt 构造、
连续批处理 scheduler 和 HTTP server，约 8k 行，底下没有任何推理框架。它是为"改"
而写的——每一层都是可以直接换成量化版或 fused 版的普通 module，每一项优化都是一个
开关，并且都有改动前后的实测数据。

它与官方 `qwen-tts` 包做过 bit 级对齐：相同的 prompt embedding、相同的首帧
logits、相同的 greedy code 序列。这件事是其余一切可信的前提——因为 TTS 的 bug 不长
得像 bug，它听起来是一个流畅的声音在说略微不对的内容。

```
text ──> prompt ──> ┌─────────────── stage 0 ───────────────┐ ──> ┌─ stage 1 ─┐ ──> 24 kHz audio
                    │ talker (28L, 1.7B)  →  code predictor │     │  Code2Wav │
                    │ AR over time, 12.5 Hz  AR over 16 RVQ │     │ 16 codes  │
                    │ one codebook-0 id      codebooks      │     │ → 1920    │
                    │ per frame              within a frame │     │  samples  │
                    └───────────────────────────────────────┘     └───────────┘
```

## 快速开始

所有命令都在 `danie_torch_2504` 容器里跑，其中 torch 和 triton 用的是 NGC 镜像自带
的版本（为什么不能换，见 `requirements.txt`）。

```bash
docker exec -it -w /home/qwen3-tts-min danie_torch_2504 bash

# 合成一句话，写成文件。
python -m qtts.entrypoints.generate \
    --model /home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
    --speaker vivian --output-path out.wav

# 同上，流式，并打印每个 chunk 的到达时刻。
python -m qtts.entrypoints.generate --model ... --stream

# HTTP server。
python -m qtts.entrypoints.serve --model ... --max-num-seqs 8 --talker-cuda-graph --port 8000
curl -s localhost:8000/generate -H 'content-type: application/json' \
    -d '{"text": "你好", "speaker": "vivian"}' -o out.wav
curl -sN localhost:8000/generate_stream -H 'content-type: application/json' \
    -d '{"text": "你好", "speaker": "vivian"}' | ffplay -f s16le -ar 24000 -ac 1 -

# 认真测一组配置。
python -m qtts.entrypoints.benchmark --model ... --concurrency 8 --talker-cuda-graph

# 测多组配置并汇总成一张表（每行一个独立进程，原因见脚本注释）。
python examples/benchmark/sweep.py --model ... --suite optimizations
```

在 Python 里用：

```python
from qtts import EngineConfig, SchedulerConfig, TalkerConfig, TTSEngine, TTSRequest

engine = TTSEngine(EngineConfig(
    model="/home/models/qtts/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    talker=TalkerConfig(cuda_graph=True),
    scheduler=SchedulerConfig(max_num_seqs=8),
))

for chunk in engine.generate_stream(TTSRequest(text="你好", speaker="vivian")):
    play(chunk.audio)            # float32 单声道，范围 [-1, 1]，已在 host 上
```

## 这个模型，按代码里的说法

**两条自回归轴，不是一条。** talker 沿*时间*生成：每个 12.5 Hz 音频帧走一步，吐出
该帧的 codebook-0 id。随后 code predictor 在这一帧*内部*沿 *codebook* 轴生成，以
talker 的 hidden state 为条件走完 codebook 1..15。所以一帧音频的代价是一次 talker
step 加十五次 code predictor prefill，后一个数字就是 `qtts/compilation.py` 存在的
理由。

**双流输入 embedding。** talker 的每个位置都是投影后的 text embedding 与 codec
embedding 之*和*——两条流是相加，不是拼接。把两条流的相对偏移搞错，产出的是语音流
畅但字不对的音频，这正是 `scripts/dump_reference.py` 要抓的故障模式。

**codec head 比词表宽。** 3072 个 logits，其中只有 `[1, 2048)` 和 EOS id `2150` 是
合法的。这个 mask 不是装饰：没有它，模型会采样到 decoder 根本没有对应 codebook 条
目的 id。

**Code2Wav 不是一个可以随便切块的 vocoder。** 它由 RVQ dequantizer、一层前置卷积、
一个 `sliding_window=72` 的 8 层 transformer 和四个上采样 decoder block 组成。这个
window 约束的是*每一层*，所以真实感受野是约 568 帧、约 45 秒——不是 window 字面暗
示的 71 帧。这一个数字决定了流式 Code2Wav 是精确还是近似，两种模式都实现了，就是为
了能把差距量出来（`tests/test_code2wav_modes.py`）。

## 目录结构

| 路径 | 里面是什么 |
| --- | --- |
| `qtts/models/talker.py` | stage 0：28 层、GQA、per-head RMSNorm，按 slot 索引的 KV cache（含一个吸收 padding 行写入的 scratch slot） |
| `qtts/models/code_predictor.py` | codebook 轴上的 AR：15 张 embedding 表、15 个 head，每个 codebook 走一次精确 prefix |
| `qtts/models/code2wav.py` | stage 1，以及三条解码路径：`decode_exact`、`decode_window`（stateful，精确）、`chunked_decode`（stateless，近似） |
| `qtts/models/prompt.py` | CustomVoice prompt 构造：控制 tag、双流相加、streaming 与 non-streaming 两种文本条件方式 |
| `qtts/models/loader.py` | safetensors 加载，包括把 q/k/v 与 gate/up 合并进 fused linear |
| `qtts/layers/` | 可替换的部分：带量化钩子的 `LinearBase`、RMSNorm（fused Triton / bit 级精确的 eager）、SnakeBeta、RoPE、causal conv |
| `qtts/engine/scheduler.py` | 准入、slot 分配、backpressure，以及 stage 1 下一步跑哪个 chunk |
| `qtts/engine/chunking.py` | 什么时候把帧交给 stage 1：固定、ramp，或首块大小随负载变化 |
| `qtts/engine/talker_runner.py` | decode step：采样、code predictor、停止条件、cache 推进 |
| `qtts/engine/engine.py` | 主循环：准入 → talker 走一帧 → 收 chunk → 解码 → 交付 |
| `qtts/quant/` | 量化方法，以及决定在哪些层启用它们的 per-shape calibration |
| `qtts/compilation.py` | talker step 与 code predictor 的 CUDA graph 捕获 |
| `qtts/config.py` | 所有开关，附带每个默认值的理由；每个开关都有对应 flag，也都真的被某处读取 |
| `qtts/serving/` | aiohttp server；由一个线程独占 engine，handler 只碰队列 |
| `qtts/entrypoints/` | 三个 CLI：`generate`、`serve`、`benchmark` |
| `examples/` | sweep 驱动脚本，以及产出 `qtts/quant/kernels.py` 里那些常量的 kernel 调优脚本 |
| `.ref/` | vllm-omni 的 Qwen3-TTS 源码，留着逐行对照 |

## 优化项

每一项都是一个开关；凡是会改变音频的，默认关闭；每一项的实测效果见
[PERFORMANCE.md](PERFORMANCE.md)。一句话版本：在这个模型、这些 batch size 下，
*launch overhead 就是瓶颈*，在它被解决之前其余一切都只是零头。

| 开关 | 做什么 | 为什么重要 |
| --- | --- | --- |
| `--talker-cuda-graph` | 按 (batch, key-length bucket) 各捕获一张 graph，整个 talker decode step 直接重放 | 一个 step 为几毫秒的算术发射约 2000 个 kernel。这是单项收益最大的优化，也是其余多数优化生效的前提 |
| `--talker-rmsnorm-backend fused` | 每次 normalization 用一个 Triton kernel，替掉六个 eager 算子 | talker 每 step 做 112 次 norm，code predictor 每帧 300 次，张量都只有几千个元素。代价是约 2 个 fp32 ULP，所以 parity 跑的是 `eager` |
| `--code2wav-snake-backend triton` | fused `x + sin²(αx)/β` | decoder 的激活函数，作用在每一个上采样后的采样点上 |
| `--quantization int8_weight_only` | per-channel int8 权重，在 GEMM 内部于寄存器里反量化 | decode 受权重带宽约束，权重字节数减半理应让时间减半——在 kernel 能打满 HBM 的那些 shape 上确实如此 |
| `--quant-calibrate`（默认开） | 加载时把每种 linear shape 两种走法都计一遍时，只留快的那个 | 同一个 int8 kernel 在 MLP up-projection 上是 1.94x，在 down-projection 上是 0.64x。一视同仁地上，净效果是零 |
| `--pipeline-mode async_chunk` | 一个 chunk 的帧一凑够就解码 | 只有这种模式下 TTFA 才是独立于总延迟的一个数 |
| `--code2wav-initial-codec-chunk-frames`、`--code2wav-codec-chunk-ramp` | 首块多大，以及之后怎么涨 | TTFA 完全由首块决定；ramp 把吞吐再挣回来，且不留下一个成本断崖 |
| `--code2wav-dynamic-initial-chunk` | 按当前负载决定首块大小 | 轻载时延迟是免费的；重载时一个很小的首块只是多一次 decoder 调用，而所有其他流都在后面排队 |
| `--chunk-policy deadline` | 优先服务已缓冲音频最少的那条流 | 听众只有在已交付的音频播完时才会察觉到 gap |
| `--code2wav-mode stateless` | 不保留 per-request decoder 状态，于是不同请求的 chunk 可以 batch | 吞吐上是实打实的收益，但**音频是近似的**——见前面关于感受野的那段 |
| `--max-pending-chunks-per-request` | 暂停那些 chunk 堆积过多的 talker | 没有它，跑得快的 talker 会堆出一个无上界的未解码帧队列 |

## 如何验证一处改动

不做 parity 检查就优化 TTS，结果就是上线一个说错话的快模型。下面的顺序就是它们各自
能抓到问题的顺序。

1. **与官方包做 bit 级 parity。** 参考实现必须在另一个解释器里跑（它锁 `transformers`
   4.57，容器里是 5.x），先 dump 张量，再由测试比对：

   ```bash
   PYTHONPATH=/path/to/ref-env python scripts/dump_reference.py --model $MODEL --out /tmp/reference.pt
   QTTS_MODEL=$MODEL QTTS_REFERENCE=/tmp/reference.pt python -m pytest tests -q
   ```

   全程 greedy，所以第一个分叉的帧就定位了 bug。parity 跑的是
   `--talker-rmsnorm-backend eager`：fused kernel 那 2 个 ULP 足以让某个接近平局的
   argmax 翻面，而 `test_fused_norm_diverges_only_at_a_tie` 钉住的正是"这只是舍入"
   与"这是个 bug"之间的界线。

2. **engine 层的测试，不需要权重。** `tests/test_scheduler.py` 覆盖的是那些 bug 听不
   出来的部分：每一帧恰好发出一次、slot 优先复用低编号、final chunk 只发一次、只在
   decoder 真需要时才带 left context。

3. **先听，再看 gap。** `--stream` 会打印每个 chunk 的到达时刻。决定一条流能不能听的
   是*最差*的那个 inter-chunk gap，不是均值：播放能扛住糟糕的平均值，但会死在某一个
   超过已缓冲音频长度的 gap 上。benchmark 会把这类 gap 记为 underrun。

4. **凡是碰到 Code2Wav 的改动，要比波形，不能只看指标**——`--code2wav-mode stateless`、
   `--code2wav-dtype bfloat16`、量化 decoder 都属于此类。它们会按设计或按意外改变音
   频，而 RTF 会一边好看一边让输出变差。

## 这份代码已经踩过的坑

留在这里，是因为每一个都花了几小时，而且没有一个会自己报出来。

- **`seq_len` 既是计数也是 rotary position。** 在 decode 调用之前就自增，会让当前位置
  留着零值，并把每个 query 的 RoPE 整体错开一位。音频依旧流畅，但不再对得上文本。
- **把反量化后的权重写回 global memory，比不量化还慢。**
  `F.linear(x, w_int8.to(dtype) * scale)` 搬的字节数*多于*未量化的 GEMM：在 talker
  最大的那个 shape 上是 158 µs 对 19 µs。weight-only 量化只有在反量化发生在 GEMM 内
  部的寄存器里时才划得来。
- **weight-only 量化不是一视同仁的收益，而且 eager 地测它测的是 launch 路径。**
  两件事都在 `qtts/quant/kernels.py` 里带数字写清了；由此才有了
  `qtts/quant/calibrate.py`。
- **`sliding_window` 约束的是一层，不是整个模型。** 8 层 window 72 叠起来能摸到 568
  帧。用单层的数字去定流式的上下文长度，在幅度被限制在 1.0 的信号上会得到接近满量程
  的误差。
- **在一个局部 `import triton.language as tl` 的函数里定义 Triton kernel，会在它自己
  的 annotation 上抛一个赤裸的 `NameError`。** Triton 是按被编译函数所在 module 的
  globals 去解析这些名字的。所以这里所有 kernel 都定义在 module 层。
- **把权重乘法折进 RMSNorm 的同一个表达式里，会改变浮点结合顺序**，从而破坏与 torch
  的 bit 级一致。顺序必须是先 `x32 = x32 * rsqrt(...)`，再 `weight * x32`。
- **chunk 边界必须相对"已经发送了多少帧"来算**，不能从当前帧数反推。从帧数反推的写法
  导致：同一帧上被 poll 两次时会重发一个 chunk，请求长度刚好落在边界上时会多发一个
  越过末尾的 chunk。
- **要一个固定 seed，会换掉实际运行的代码。** 捕获好的 graph 无法从调用方传入的
  generator 取随机数，于是带 seed 的请求会悄悄退回 eager code predictor，慢 4 倍。
  engine 会告警；否则一个为了可复现而设 seed 的 benchmark 报的是另一条路径的数字。
- **Gumbel 采样要的是 `-log(-log u)`。** 去 clamp `log u`（它恒为负）会让采样变成均
  匀分布。greedy 解码照样能跑，所以在音频悄悄变差之前什么都不会失败。

## 已知缺口

- **没有取消机制。** 客户端中途放弃的请求仍会被生成完。要做对，需要在 scheduler 里为
  一个可能正处于 step 中途的请求开一条 abort 路径。
- **fp8 weight-only 没有实现**（fp8 的 W8A8 有）。`qtts/quant/fp8.py` 里写了原因：
  `torch._scaled_mm` 要求两个操作数都带 scale，而"fp8 权重 + 未 scale 的激活"没有可
  调用的快路径。
- **stage 0 与 stage 1 不重叠。** 一次 `step()` 里两者是顺序执行的。让它们在不同 stream
  上重叠能提升吞吐，也会给每一次测量都塞进一个"重叠效率"项，所以这里刻意不做。
- **单卡单进程。** 没有 tensor parallelism，也没有 pipeline parallelism。

## 对比 `qwen3_tts_sortout.md` 的缺口

对照笔记第五～八节与 vllm-omni 参考实现，qtts 已对齐的部分包括：SnakeBeta fused kernel、code predictor 的 fused QKV/gate-up 与 stacked residual embedding、stateful 增量 Code2Wav + sliding window、decoder fp32 权重、`async_chunk` / initial chunk / ramp / dynamic、credit backpressure、codes 常驻 GPU、固定 KV 上限，以及 TTFA/RTF/gap/underrun 的 benchmark；repetition 检测与 `max_frames` 也在。下面仍是**未做或只做了一半**的项，按笔记分类列出，便于排期。

### 性能（位置明确、可单独推进）

| # | 缺口 | 笔记出处 |
| --- | --- | --- |
| 1 | **Code2Wav CUDA Graph**：stage 1 仍是 eager 小算子 + 卷积，launch-bound 与 talker 同类，图捕获未接 | 5.1、6.2、8.2 #4924 |
| 2 | **Talker 多步 CUDA Graph**：一帧两次 graph replay，中间夹着 eager 的 logits / penalty / sampling | 8.1 #7093 |
| 3 | **graph-safe sampling**：带 seed 时 code predictor 退回 eager（约 4×）；批内 per-row seed 未支持 | 8.4、6.3 #4970/#4869 |
| 4 | **每帧 D2H 同步**：`talker_runner` 里 `finished.tolist()` / `layer0.tolist()`；`_recent_ids` 每帧 host 建 tensor | 8.5 #4855 |
| 5 | **persistent batched GPU buffer**：`last_hidden`、`_text_step` 每帧 `stack` 重拼 | 8.5 |
| 6 | **stateful 下 stage 1 批处理**：`next_batch` 在 stateful 强制 budget=1，多流首块串行 | 5.1 stateful batch grouping |
| 7 | **stateless bucket 分桶**：按精确帧数分组，未 padding 到 bucket 再按 `lengths` 裁回 | 5.1 |
| 8 | **启动 warmup**：首请求付 graph capture / Triton JIT / cuDNN autotune 成本 | 6.2 #3424 |
| 9 | **上采样残差块融合**：`ConvTranspose1d → SnakeBeta → Conv` 中间物化；需先 profiling | 5.2、6.2 |

建议优先顺序（与笔记 ROI 一致）：**1、6、8** 都在现有架构内且有数字支撑；**2+3** 一组、收益大但工作量大。

### 结构性（需补能力或改架构）

| # | 缺口 | 笔记出处 |
| --- | --- | --- |
| 10 | **Base / ICL voice clone**：无 codec encoder、无 ECAPA；encoder 权重加载时跳过 | 2.4、6.2 条件输入 |
| 11 | **Stage 0/1 进程级 disaggregation（WS-5）**：`async_chunk` + credit 背压已在**单进程**内实现「攒够帧就解码」与 TTFA；但每步仍是单进程 talker → code2wav 顺序执行（见上文「已知缺口」），且无独立 Stage 1 多 replica 实例 / 跨进程部署 | 6.1、6.2 #3535 WS-5 |
| 12 | **Token-level 流式文本**：仅有 submit 时定稿的 `trailing_text`；无零 lookahead、容量分段、跨段继承 | 8.3 #6496 |
| 13 | **Stage 0 prompt prefix caching**：每请求整段 prefill | 6.3 #3665 等 |
| 14 | **请求生命周期**：无取消/超时/失败传播；重复 `request_id` 可能顶掉 outbox | 8.5 #6453 |

10 缺则整条链落空：ref-audio 缓存、`SpeakerCacheManager`、cache-miss 批量 prefill、FP8 ref encoder、#4256 重复编码等。

### 质量与正确性（基本空白）

| # | 缺口 | 笔记出处 |
| --- | --- | --- |
| 15 | **感知质量门禁**：无 HNR / UTMOS / 说话人相似度 / ASR WER、长跑与泄漏测试；parity 只保证与参考一致 | 6.3、8.7 #2115 |
| 16 | **起始静音**：x-vector-only 屏蔽开头 silence codec 词表 | #4966 |
| 17 | **chunk 边界听感测试**：stateless 只量幅度误差，无爆音/尾裁/时长压缩用例 | #4706/#2480/#4870 |
| 18 | **fp16 路径**：code predictor 在 fp16 下 fp32 中间量未做；fp16 未系统验证 | #3253 |
| 19 | **fp8 weight-only** 未实现；fp8 W8A8 无 benchmark 行；NPU/XPU（8.8）未覆盖 | 6.2、8.8 |

**调度策略**（fifo / deadline / round_robin）在「八份相同句子同时提交」的 sweep 里无法区分，笔记 8.4 的预期结果；要测出差异需混合长度、错峰到达或慢消费者负载——属于 benchmark 缺口而非实现缺失。

更细的测量背景见 [PERFORMANCE.md](PERFORMANCE.md)；英文 README 见 [README.en.md](README.en.md)。
