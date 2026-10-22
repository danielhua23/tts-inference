# tts-inference 优化任务细化（Qwen3-TTS）

> 依据 `qwen3tts-sortout.md`（核对基线 vllm-omni main `bc0c9f4b`）逐项对照 qtts 源码，只列出
> **还没做或只做了一半**的优化。本地 vllm-omni 已更新到 `2f860489a`（2026-10-13），
> 其中 [#8163](https://github.com/vllm-project/vllm-omni/pull/8163)、
> [#8259](https://github.com/vllm-project/vllm-omni/pull/8259) 新增了 sortout 没写到的实现
> （`streaming_decoder.py`、`fused_code_predictor.py` 等），本文把它们作为对应任务的实现参考，
> 汇总见附录。P1.5 不在 sortout 里，来自 SGLang-Omni 对 Higgs、MOSS 的
> [TTS 优化文章](https://zhuanlan.zhihu.com/p/2054821717529343214)。
>
> 数字来源是 `PERFORMANCE.md`：H100 80GB（SXM），torch 2.7.0a0+nv25.04，triton 3.2.0，
> `Qwen3-TTS-12Hz-1.7B-CustomVoice`，speaker `vivian`，26 字中文，约 59 帧、4.7 s 音频，
> 不固定 seed。整理本文时本机（H200）上没有 Qwen3-TTS checkpoint，没有复测。凡标
> **推算** 的数字，都是根据代码里的 shape 和 `PERFORMANCE.md` 的表推出来的，需要 P0.1 在
> 同一张卡上实测确认。

## sortout 里的优化，哪些已经做了

先把边界划清楚。下表是 sortout 第五、六、八节列出的优化中，qtts 已经实现的部分，不在本文范围：

| sortout 中的优化 | 出处 | qtts 的实现 |
| :-- | :-- | :-- |
| SnakeBeta 融合 Triton kernel，缓存 `exp(α)`、`1/β` | 5.1 | `layers/activation.py` |
| code predictor fused QKV / gate-up | 5.1、6.2 #5791 | `models/code_predictor.py`，`TalkerConfig.fuse_*` |
| residual embedding 堆成 `[Q-1, V, H]`，一次 gather + sum | 5.1、6.2 #5791 | `TalkerRunner._next_inputs_embeds` |
| code predictor 无 KV cache 的短序列重预填充 | 5.1 | `CodePredictor.forward`（精确前缀 `prefix_len = step + 1`） |
| stateful 增量 Code2Wav + sliding window 滚动 | 5.1、6.2 #5202 | `Qwen3TTSCode2Wav.decode_window`：逐层 pre-RoPE KV + 两段卷积缓存 |
| decoder 权重 fp32 | 5.1 | `Code2WavConfig.dtype` |
| 不加载 encoder（释放未使用的解码器） | 6.2 #2429 | `loader._SKIP_CODEC` |
| `async_chunk` 边生成边解码 | 6.2 #5011 | `SchedulerConfig.pipeline_mode` |
| initial chunk、ramp、按负载选首块（dynamic TTFA） | 6.2 #5011/#1714/#3485 | `engine/chunking.py` |
| credit 背压 | 6.2 #3535 WS-4 | `max_pending_chunks_per_request` |
| 面向 underrun 的 chunk 调度 | 8.4 #2568 | `chunk_policy = fifo / deadline / round_robin` |
| codes 常驻 GPU | 6.2 #4879 | `RequestTracker.frames` |
| 固定 KV cache 上限 | 6.2 #4279 | `TalkerKVCache` 启动时预分配 |
| talker 与 code predictor 的 CUDA Graph | 8.4、8.8 | `compilation.py` |
| `max_frames` 与 repetition guard | 8.7 #6158 | `TalkerConfig`（只做了最简单的版本，见 P3.3） |
| TTFA / RTF / gap / underrun benchmark | 8.4 | `entrypoints/benchmark.py`（负载太单一，见 P0.1） |
| int8 weight-only + 按 shape 校准；fp8 W8A8 | 6.2 #3036 | `quant/` |

下表是**没做或只做了一半**的部分，也是本文的任务清单：

| sortout 中的优化 | 出处 | qtts 现状 | 本文任务 |
| :-- | :-- | :-- | :-- |
| 性能拆分、混合负载 benchmark | 8.4 | 只有同构负载；"code predictor" 列混入了帧间胶水 | P0.1 |
| 感知质量门禁（HNR / UTMOS / 说话人相似度 / WER）、边界伪影、长跑 | 6.3、8.7 #2115 | 只有与参考实现的 parity | P0.2 |
| stateful Stage 1 批处理 | 5.1 #5202 | stateful 模式每 step 只解 1 个 chunk | P0.3 |
| Code2Wav CUDA Graph，runner 统一管理 graph 与显存记账 | 5.1、6.2、8.2 #4924 | Code2Wav 全程 eager | P0.4 |
| 避免逐步 D2H；persistent batched GPU buffer | 6.2 #4879、8.5 #4855 | 每帧两次 `.tolist()`，多处 Python 循环重建 tensor | P0.5 |
| 启动 warmup | 6.2 #3424 | graph 懒捕获，server 无 warmup | P0.6 |
| graph-safe 采样；batched MTP 下的 per-row seed | 6.3 #4970/#4889/#4869、8.4 | 带 seed 走 eager（慢 4 倍）；batch 内 seed 被静默丢弃 | P1.1 |
| Talker 多步 CUDA Graph | 8.1 #7093 | 一帧两次 replay，中间夹 eager 胶水 | P1.2 |
| Talker / code predictor 的编译与 kernel | 8.4 | compile 与手工 graph 互斥；code predictor 约 1500 kernel/帧 | P1.3、P1.4 |
| cache-miss 批量 / 分块 prefill；Stage 0 前缀缓存 | 6.2 #4256、6.3 #3665 | prefill 逐个请求 eager 执行；无前缀缓存 | P2.1 |
| stateless bucket 分桶、裁桶 | 5.1 | 按精确帧数分组，不 padding | P2.2 |
| 上采样残差块算子融合 | 5.2、6.2 | 未做，需先 profiling | P2.3 |
| fp16 下 code predictor 用 fp32 中间量；SnakeBeta bf16/fp16 修复 | 6.3 #3253/#3472 | fp16 未验证；bf16 Code2Wav 未测 | P2.4 |
| 按并发动态调整 codec window | 8.4 | 只有首块随负载变化 | P2.5 |
| FP8 | 6.2 #3036 | fp8 weight-only 未实现；fp8 W8A8 无 benchmark 行 | P2.6 |
| 取消、超时、失败传播、请求 ID 复用 | 8.5 #6453 | 都没有 | P3.1 |
| Stage 1 显存预算、可观测性 | 8.5 | 状态显存估算函数没有调用方 | P3.2 |
| 重复检测、异常时长限制 | 8.7 #6158 | 只检测"code-0 连续相同"，时长上限是固定值 | P3.3 |
| 两个 stage 流水并行 | 6.1 | 单线程顺序执行（README 中说明是刻意的选择） | P3.4 |
| 异步 decode：CPU 记账与下一步 GPU 计算重叠（launch / resolve 错开一步） | sortout 之外：SGLang-Omni 的 TTS 优化文章 | `TTSEngine.step` 中 CPU 与 GPU 完全串行；`_deliver` 每个 chunk 同步一次 | P1.5 |
| Stage 1 多副本、disaggregation | 6.1、6.2 #3535 WS-5 | 单进程单卡 | P3.5 |
| token 级流式文本输入 | 8.3 #6496 | 文本在 submit 时就全部确定 | P3.6 |
| ref-audio 缓存、`SpeakerCacheManager`、FP8 ref encoder、重复编码、起始静音屏蔽 | 6.2、8.6、6.3 #4966 | 都依赖 Base（voice clone），qtts 不支持 Base | P4 |
| NPU / XPU | 8.8 | 不在范围内 | 不做 |

有两项需要单独说明。

- **起始静音屏蔽（#4966）对 CustomVoice 不适用。** vllm-omni 只对 Base 的 x-vector-only
  请求生效（`qwen3_tts_talker.py` L338–346 的注释写明"ICL onsets are legitimate audio and
  must not be masked"），所以放在 P4，不作为 CustomVoice 的任务。
- **fp8 W8A8 不作为 decode 的优化方向。** `quant/fp8.py` 实测在 talker 的四种 shape 上只有
  bf16 的 0.20–0.71 倍，原因是激活量化多出的三次逐元素 pass。P2.6 只补一个大 batch 的
  benchmark 行，用来找交叉点。

## 四个指标，以及每种手段动的是哪一个

TTS 推理的目标比 DiT 多一个维度：除了吞吐和显存，流式输出还要求首包快、中途不断音。

- **吞吐 / RTF**：wall clock 除以音频时长，多流时取聚合值，低于 1 表示比实时快。
- **TTFA**：time to first audio，听众判断系统快慢只看这一个数。
- **流式连续性**：最差 inter-chunk gap 和 underrun 次数。平均值好看没用，一次超过已缓冲音频长度的 gap 就是一次断音。
- **显存**：权重、talker KV cache、每请求的 Code2Wav 状态、graph pool。H100 上不是瓶颈，但决定能开多少并发。

另外，**音频质量**是所有优化的硬约束：TTS 的 bug 听起来是"流畅地说了略微不对的话"，只看 RTF 发现不了。

| 手段 | 吞吐 / RTF | TTFA | 最差 gap / underrun | 显存 | 数值 | 本文任务 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| stateful Stage 1 批处理 | 高并发时 ↑ | 并发下 p95 ↓（主要收益） | ↓ | 状态池化，总量不变 | batch 改变 GEMM shape，近似无损 | P0.3 |
| Code2Wav CUDA Graph | 单流下小 | ↓ | ↓ | +graph pool | 同 shape 下逐位一致 | P0.4 |
| 去掉每帧 host 同步 | ↑，上限为 replay 间的 GPU 空闲 | ↓ | ↓ | +约 1 MB | 逐位一致 | P0.5 |
| 启动 warmup | 不变 | 首请求 ↓ | 首请求 ↓ | graph pool 提前占用 | 无损 | P0.6 |
| graph-safe 采样 | seeded 请求 ↑约 4 倍 | seeded 请求 ↓ | — | 不变 | 分布等价；固定 batch 组成时可复现 | P1.1 |
| 多步 talker graph | ↑ | ↓ | ↓ | +graph pool | 与单步路径逐位一致 | P1.2 |
| code predictor 降 kernel 数 | ↑（stage 0 的大头） | ↓ | ↓ | 折叠表 +约 63 MB | 近似无损 | P1.3 |
| talker decode 走 flash decode + 层内融合 | ↑ | ↓ | — | 不变 | 近似无损 | P1.4 |
| 异步 decode（CPU 与 GPU 重叠） | 多流时 ↑，上限为每步的 CPU 时间 | 单流不变（退回同步）；多流小幅 ↓ | ↓ | +两个 pinned staging buffer，KB 级 | 逐位一致 | P1.5 |
| 批量 / 分块 prefill，prompt 缓存 | 小 | 突发到达时 ↓ | 运行中的流 ↓ | 前缀缓存 +少量 | 近似无损 | P2.1 |
| stateless bucket 分桶 | 只影响 stateless | — | — | 不变 | 因果右填充，无损 | P2.2 |
| 上采样残差块融合 | 高并发时 ↑ | 小 | — | 不变 | 近似无损 | P2.3 |
| bf16 Code2Wav | Stage 1 ↑ | 小 | — | Stage 1 权重和状态减半 | 有损，需听 | P2.4 |
| 按负载调整稳态 chunk | 高负载时 ↑ | — | ↓ | 不变 | 无损 | P2.5 |
| fp8 weight-only | 约等于 int8 | — | — | 与 int8 相同 | 有损 | P2.6 |
| Stage 0/1 两个 stream 重叠 | 高并发时 ↑ | — | ↓ | 不变 | 逐位一致 | P3.4 |
| Stage 1 多副本 | 整机 ↑ | — | — | 每卡 ↓ | 无损 | P3.5 |
| token 级流式文本 | — | 上游 LLM 流式输出时大幅 ↓ | — | 不变 | 改变断句边界 | P3.6 |

表中的箭头是方向判断，具体幅度见各节；凡没有实测支撑的都在 P0.1 的测量清单里。

## 基线：时间花在哪

### 一帧的分解

一帧音频 = 一次 talker step + 十五次 code predictor 前缀 forward。`PERFORMANCE.md` 的数据
（每帧毫秒）：

| 配置 | step | talker forward | code predictor |
| :-- | --: | --: | --: |
| batch 1，eager norm，无 graph | 66.97 | 18.51 | 48.47 |
| batch 1，fused norm，无 graph | 64.98 | 17.56 | 47.42 |
| batch 1，eager norm，有 graph | 17.57 | 5.74 | 11.83 |
| batch 1，fused norm，有 graph | **11.85** | 4.16 | 7.69 |
| batch 4，fused norm，有 graph | **13.16** | 4.93 | 8.23 |

注意测法："code predictor" 一列是 `TalkerRunner.decode_step` 的总耗时减去单独测的 talker
forward，所以它**还包含两次 replay 之间所有 eager 执行的胶水代码**：采样、embedding、
repetition penalty、文本 embedding 拼接、host 同步。code predictor graph 本身占多少，目前
没有数据（P0.1-1）。

### 端到端与并发

| 配置（单流） | RTF | TTFA p50 | 最差 gap | underrun | 峰值 MiB |
| :-- | --: | --: | --: | --: | --: |
| baseline（eager norm、eager snake） | 0.849 | 0.357 | 1.670 | 3 | 6365 |
| + CUDA graph | **0.162** | **0.082** | 0.312 | 0 | 6461 |
| + int8 weight-only（校准） | 0.159 | 0.083 | 0.297 | 0 | 6046 |
| 带固定 seed（退回 eager code predictor） | 0.656 | — | — | — | — |

| 流数（开 graph） | RTF（聚合） | TTFA p50 | TTFA p95 | 最差 gap | underrun |
| :-- | --: | --: | --: | --: | --: |
| 1 | 0.165 | 0.082 | 0.082 | 0.301 | 0 |
| 2 | 0.102 | 0.111 | 0.120 | 0.448 | 0 |
| 4 | 0.060 | 0.165 | 0.191 | 0.486 | 0 |
| 8 | 0.039 | 0.276 | 0.338 | 0.806 | 0 |

从 1 流到 8 流，吞吐涨 4.2 倍，TTFA p50 涨 3.4 倍。下面四个判断解释这些数字背后的瓶颈。

### 关键判断一：stage 0 离权重带宽下界还差约 7 倍，差距在串行小 kernel，不在读权重

DiT 的大 GEMM 已经到了 bf16 峰值的 83%，所以 DiT 的结论是"减少计算量"。TTS 的 stage 0
完全不同：它是 batch 1–64 的 decode，每一帧都要把 talker 和 code predictor 的权重各读一遍，
理论下界由 HBM 带宽决定。

**talker 每帧读多少字节**（shape 取自 `PERFORMANCE.md` 的 int8 kernel 表）：

```text
qkv      4096 × 2048 =  8.39M
o_proj   2048 × 2048 =  4.19M
gate_up 12288 × 2048 = 25.17M
down     2048 × 6144 = 12.58M
每层 50.33M × 28 层 = 1.409B 参数 = 2.82 GB（bf16）
```

**code predictor 每帧读多少字节**（推算，按 config 默认的 hidden 1024、16 个 q head、
8 个 kv head、head_dim 128、intermediate 3072）：

```text
每层：qkv 1024→4096 4.19M + o 2048→1024 2.10M + gate_up 1024→6144 6.29M + down 3072→1024 3.15M
     = 15.73M × 5 层 = 78.6M 参数 = 157 MB / 次前缀 forward
× 15 次前缀 forward                         = 2.36 GB
+ 15 个 lm_head（1024 × 2048）与投影层         ≈ 0.07 GB
合计                                        ≈ 2.43 GB / 帧
```

157 MB 放不进 H100 的 50 MB L2，所以 15 次 forward 每次都要从 HBM 重读。

```text
每帧权重读取 ≈ 2.82 + 2.43 = 5.25 GB
H100 SXM HBM3 3.35 TB/s → 理论下界 ≈ 1.6 ms / 帧
实测（batch 1，有 graph）            11.85 ms / 帧，约为下界的 7.5 倍
```

GEMM 本身并不慢。按 int8 kernel 表里 cuBLAS bf16 在 M=4 下的耗时，talker 一层四个 GEMM
合计 38.0 µs，28 层约 1.06 ms，折合 2.65 TB/s，已经达到 HBM 带宽的 79%（gate_up 单个
kernel 的 2.6 TB/s 也说明这是 SXM 卡，PCIe 版只有 2.0 TB/s）。也就是说，talker graph
forward 的 4.16 ms 里，GEMM 只占约 1/4，剩下约 3.1 ms 花在 attention、mask、RoPE、
KV 写入、norm、残差加法这些小 kernel 上。`compilation.py` 的注释给出了 eager 下的规模：
一个 talker step 有 2071 个 kernel，平均每层约 74 个。即使在 graph 里没有 launch 开销，每个
kernel 也有 1–3 µs 的最小执行时间，它们是串行依赖的，所以累加起来就是这 3 ms。code
predictor 更极端：约 1500 个 kernel / 帧，平均每个约 5 µs。

这个判断也解释了 int8 为什么只把 RTF 从 0.162 降到 0.159：它减少的是读权重的字节，但字节
不是瓶颈。

因此 stage 0 的收益优先级是：

```text
减少每帧串行依赖的 kernel 数（P1.3 code predictor、P1.4 talker decode）
  ≈ 消除帧间的 eager 胶水和 host 同步（P0.5 → P1.2 多步 graph）
  > 减少读权重的字节（int8 / fp8，P2.6）
```

这个顺序只在 decode batch 较小时成立。batch 增大到 GEMM 变成 compute-bound 之后（按 H100 的
989 TFLOPS / 3.35 TB/s 估算，M 约 300 行），量化和 GEMM 效率的权重才会上升；`PERFORMANCE.md` 里 batch 4 只比
batch 1 慢 11%，说明 8 流以内远没到那个区间。

### 关键判断二：帧间胶水还在 graph 外面，而且每帧有两次强制同步

即使开了 graph，一帧仍然是"talker replay → eager 胶水 → code predictor replay → eager
胶水"。胶水部分在 `engine/talker_runner.py` 里，逐项列出：

| 位置 | 每帧做什么 | 问题 |
| :-- | :-- | :-- |
| `decode_step` L189 | `torch.stack` 每行的 `last_hidden` | 一次拷贝；`last_hidden` 是 graph 输出 buffer 的 view |
| `_sample_layer0` → `_recent_ids` L159–177 | 在 CPU 上用 Python 循环拼 penalty 窗口，再 H2D | 每帧一次 host 建 tensor + 拷贝 |
| `_sample_layer0` L147–157 | codec head、mask、penalty、temperature、top-k、Gumbel | 约十几个小 kernel，eager |
| `decode_step` L203–204 | `finished.tolist()`、`layer0.tolist()` | **两次 D2H 同步**：CPU 必须等 code predictor graph 执行完才能继续，下一帧的任何 launch 都无法提前 |
| `decode_step` L212 | `tracker.frames.append(codes[row:row+1])` | 每行一个切片，后面 `collect_chunks` 再 `torch.cat` |
| `_text_step` L128–145 | Python 循环取每行的文本 embedding，再 `stack` | 每帧重建 |
| `_advance` L269–309 | Python 循环填 `embeds[slot]`，`trackers.index` 是 O(n²) | 每行一次切片赋值 kernel |
| `TalkerDecodeGraphs.run` L168–169 | 用 Python list 建 `seq_lens`、`active` | 小，但每帧都有 |

胶水之外，`TTSEngine.step` 里还有一处同步：`_deliver`（`engine.py` L221–225）对每个交付的 chunk
做一次 `.to("cpu")`，要等 Code2Wav 跑完才返回。

两次 `.tolist()` 的影响最大：它让 CPU 和 GPU 在每帧里完全串行。这部分能省多少，取决于
replay 之间的 GPU 空闲时间，目前没有测（P0.1-1）。消除这些同步是 P1.2 多步 graph 的前提，
否则多步 graph 无从谈起。

### 关键判断三：并发下 TTFA 变差，主要来自三处串行

单流 TTFA 0.082 s ≈ prefill + 5 帧 × 11.85 ms（59 ms）+ 一次首块 decode，所以
prefill 加首块 decode 约 23 ms（推算）。8 流时 TTFA p50 0.276 s、p95 0.338 s，多出来的
部分来自三处串行：

1. **prefill 逐个执行。** `TTSEngine.step` 在同一个 step 里接纳全部 8 个请求，然后
   `TalkerRunner.prefill` 一个一个跑（`engine.py` L182–184），而且 prefill 不走 graph。
   `Qwen3TTSTalker.prefill` 的 docstring 说明这是刻意的，理由是 prompt 只有约 50 个
   token。但在突发到达时，这 8 次 prefill 的时间会加到每个请求的 TTFA 上，同时也会让已经
   在运行的流停顿。
2. **stateful 模式下，每个 step 只解一个 chunk。** `Scheduler.next_batch` 在 stateful 模式
   把 budget 固定为 1（`scheduler.py` L216–218），`decode_window` 也拒绝 B>1
   （`code2wav.py` L589–590）。8 条相同的句子会在同一个 step 凑齐首块，然后分 8 个连续
   step 依次解码，第 k 条要多等 k−1 个"talker step + 一次 decode"。
   `tests/test_scheduler.py::test_stateful_dispatch_takes_one_chunk_at_a_time` 固定了这个
   行为。
3. **talker step 随 batch 变慢。** batch 1 是 11.85 ms，batch 4 是 13.16 ms。这一项最小，
   是正常的。

三项各占多少需要 P0.1 拆分。从机制看，第 2 项是 TTFA p95 恶化的主因，所以 P0.3 排在
P0 最前面的工程任务里。

### 关键判断四：Code2Wav 单流下算力可以忽略，问题在 launch 和串行；高并发时才会追上 stage 0

Code2Wav 的 FLOPs 集中在 decoder 的卷积上。按 config 默认的 `decoder_dim=1536`、
`upsample_rates=(8,5,4,3)`、`upsampling_ratios=(2,2)` 推算每帧的计算量
（ResidualUnit = k7 空洞卷积 + k1 卷积，每个 decoder block 有 3 个）：

```text
decoder 入口 conv(k7, 1024→1536)，4 步 / 帧          ≈ 0.09 GFLOP
block0：convT(k16, 1536→768) + 3×RU(768)，32 步      ≈ 1.06
block1：convT(k10,  768→384) + 3×RU(384)，160 步     ≈ 1.32
block2：convT(k8,   384→192) + 3×RU(192)，640 步     ≈ 1.32
block3：convT(k6,   192→96)  + 3×RU(96)，1920 步     ≈ 0.99
upsample：2 × (convT + ConvNeXt)，1024 通道            ≈ 0.11
pre_transformer（宽 512）与 quantizer                ≈ 0.03
合计                                                ≈ 4.9 GFLOP / 帧 ≈ 61 GFLOP / 秒音频
```

H100 fp32 SIMT 峰值约 67 TFLOPS；PyTorch 默认 `cudnn.allow_tf32=True`，卷积还可能走
TF32 tensor core（约 495 TFLOPS）。即使按 fp32 SIMT 计算，1 秒音频也只需要约 1 ms，而
stage 0 生成 1 秒音频要 12.5 × 11.85 ≈ 148 ms。所以单流下 Code2Wav 的算力可以忽略。

它的问题在别处：

- **launch。** 按模块数估算，一次 `decode_window` 约 450 个 kernel（quantizer 约 35 个、
  8 层 transformer 约 250 个、upsample 约 26 个、decoder 约 125 个，另外每个卷积前都有一次
  `F.pad`），全部 eager 执行（推算，P0.1-5 用 profiler 确认）。5 帧的首块 GPU 工作量很小，
  耗时基本由 CPU launch 决定。
- **串行。** 见关键判断三的第 2 项。

所以 Code2Wav 的优化目标首先是 TTFA 和 gap，手段是批处理（P0.3）和 CUDA Graph（P0.4）；
降 FLOPs 的 bf16、卷积融合（P2.3、P2.4）优先级靠后。

高并发时情况会变。64 流时 stage 0 每秒音频的开销按 batch 摊薄，而 Code2Wav 线性增长：
64 × 61 GFLOP ≈ 3.9 TFLOP / 秒音频。如果 cuDNN 对这些"长而窄"的卷积选了非 tensor core
kernel（vllm-omni `time_major_conv.py` 的 docstring 说它确实会这样），按 fp32 SIMT 30% 的
效率算约需 0.2 s；而 stage 0 生成这 1 秒音频要 12.5 个 batch-64 的 step，即使每个 step 只有
20 ms 左右，也是 0.25 s，两者已经是同一量级。这与 sortout 6.1 的判断一致：CustomVoice 在高
并发下可能转为 Stage 1 瓶颈。这时 P2.3、P2.4、P3.4 的优先级才会上升。

### 显存账本

| 项目 | 大小 | 说明 |
| :-- | --: | :-- |
| talker KV cache | 每 (slot, position) 112 KiB | 28 层 × kv 宽 1024 × K/V 2 份 × 2 B |
| ↳ 17 slot × 1024 position | 1.86 GiB | sweep 使用的配置（`max_num_seqs` 16 + 1 个 scratch） |
| ↳ 17 slot × 4096 position | 7.44 GiB | 默认 `max_model_len` |
| ↳ 65 slot × 4096 position | 28.4 GiB | `max_num_seqs` 64 |
| 单流峰值（实测） | 6365 MiB | max_model_len 1024，无 graph |
| int8 全量（实测） | 5040 MiB | 比 bf16 少约 1.4 GiB |
| Code2Wav 状态 | 每请求几 MB | `state_bytes_estimate()`，`test_code2wav_modes` 校验过 |

H100 上显存不是瓶颈。并发开到 64 时 KV cache 会成为最大一项，但 sortout 没有列这方面的
优化（paged KV、fp8 KV），本文不展开。

## 框架已经具备的能力

| 能力 | qtts 的实现 |
| :-- | :-- |
| 两条自回归轴分开：talker 沿时间，code predictor 沿码本 | `models/talker.py`、`models/code_predictor.py` |
| 按 slot 索引的 KV cache，scratch slot 吸收 padding 行的写入 | `TalkerKVCache` |
| talker step、code predictor 的 CUDA Graph，按 (batch, key_len bucket) 懒捕获 | `compilation.py` |
| 三条 Code2Wav 解码路径：整句精确、stateful 精确、stateless 近似 | `models/code2wav.py` |
| chunk 策略：固定、ramp、按负载选首块 | `engine/chunking.py` |
| 准入、slot 低号优先、credit 背压、三种 chunk 调度策略 | `engine/scheduler.py` |
| 量化挂钩（加载后挂到 `LinearBase`）与按 shape 校准 | `quant/` |
| 与官方 `qwen-tts` 的 bit 级 parity（greedy，eager norm） | `scripts/dump_reference.py`、`tests/test_parity.py` |
| stateful 与 stateless 的波形误差测试 | `tests/test_code2wav_modes.py` |
| 每行一个独立进程的 sweep | `examples/benchmark/sweep.py` |
| aiohttp server：单线程独占 engine，handler 只碰队列 | `serving/server.py` |

## P0：最高优先级任务

### P0.1 度量基础设施

**优先级依据**：本文多数收益判断是推算；`PERFORMANCE.md` 的 "code predictor" 一列混入了
帧间胶水；Code2Wav 单次调用的耗时完全没有数据；三种调度策略在现有 suite 下无法区分。后面
每一项任务的验收都需要这些数字。另外，基线是 H100 上测的，而现在能用的是 H200，需要在
H200 上重新建立基线。

**实现内容**：

1. **按阶段拆分每帧耗时。** 把 `StageTimer` 改成默认用 CUDA event，不再每个阶段
   `synchronize`；保留同步模式作为选项。拆分为：talker replay、帧间胶水、code predictor
   replay、Code2Wav（按 phase、`new_frames`、batch 大小分别统计），输出到 `--json-path`。
2. **kernel 分解。** `benchmark --torch-profile` 导出 trace，并按 GEMM、attention、norm、
   逐元素、卷积、Triton、memcpy、GPU 空闲分类，报告每帧 kernel 数和 replay 间的空闲时间。
   用"每帧权重字节 / 每帧耗时"报告 stage 0 的带宽利用率，作用类似 DiT 的 MFU。
3. **TTFA 拆分。** 每个请求记录：排队（submit → admit）、prefill、首块帧生成、首块在
   Stage 1 前的等待、decode、交付，写入 `TTSOutput`。
4. **新的 benchmark 负载。** 混合文本长度、泊松到达、慢消费者（一条流按实时速率读取，用
   播放时间轴判定 underrun）、长文本（超过 568 帧，覆盖 stateful 的稳态窗口）、突发到达
   （已有 8 条流在跑时再同时到达 8 条）。现有 `--suite scheduling` 下三种策略无法区分，
   `PERFORMANCE.md` 已经说明这是负载的问题。
5. **Code2Wav 微基准。** 单独测 `decode_window`（按 B、`new_frames`、首块 / 后续块）和
   `decode_exact`（按帧数）的耗时和 kernel 数，分别覆盖 eager / graph、fp32 / bf16，为
   P0.3、P0.4、P2.3、P2.4 提供基线。

**验收标准**：

- `PERFORMANCE.md` 中的表能在同一张卡上用一条 sweep 命令复现，误差在 3% 以内；
- 各阶段耗时之和与 step 耗时的差不超过 2%，帧间胶水单独成项；
- 混合负载下三种调度策略的 TTFA p95 / 最差 gap 有可区分的结果；如果仍然区分不出，写明原因。

### P0.2 质量度量与门禁

**优先级依据**：现有 parity 只保证 greedy 解码时与参考实现一致。后面的任务里，seed 采样
（P1.1）、融合 kernel（P1.3、P1.4）、bf16 Code2Wav（P2.4）、量化（P2.6）、stateless 模式
都会改变音频，而 RTF 可以一边变好看、一边让音频变差。sortout 6.3、8.7 把这一项列为 TTS
区别于 LLM 的关键约束。

**实现内容**：

1. **固定评测集。** 约 32 条句子，覆盖中英文、数字和单位、长句、方言 speaker、带 instruct
   的 VoiceDesign，每条配固定 speaker 和多个 seed。在 P1.1 完成前，带 seed 的评测走 eager
   路径即可，评测时不关心速度。
2. **指标。** ASR 往返 CER/WER（中文 Paraformer 或 Whisper large-v3）、说话人相似度（WavLM-SV
   或 ECAPA embedding 的 cosine）、UTMOS、HNR、时长比、首尾静音长度。容器里没有
   `transformers`，评测和 `scripts/dump_reference.py` 一样放在另一个环境里跑。
3. **两类参照，不能混用。**
   - Code2Wav 的改动对给定 codes 是确定性的，用 dump 出来的固定 codes 比较波形：最大误差、
     SNR、mel 谱 L1，参照是 fp32 eager 的 `decode_exact`。
   - stage 0 的改动会改变采样轨迹，只能比较多个 seed 上的指标分布。先测同一配置重复运行的
     方差带，用它来定门限。
4. **chunk 边界测试**（#4706 / #2480 / #4870）：总采样点数 = 帧数 × 1920，尾块不丢；
   chunk 边界附近的高频能量与非边界处对比，用来检测爆音；覆盖 ramp 和 dynamic 首块。
5. **长跑。** 连续 1000 个请求，检查 Code2Wav 状态、graph pool、server outbox 没有泄漏。

**验收标准**：输出一份基线质量报告，包含每个指标的均值、最差值和重复运行的方差带；后续
有损任务都引用这份报告的门限。

### P0.3 stateful Stage 1 批处理（状态池化）

**收益基线**：8 流时 TTFA p50 0.276 s、p95 0.338 s，单流是 0.082 s。关键判断三的第 2 项
是主要原因：stateful 模式下每个 step 只解一个 chunk。

**实现内容**：

1. **状态池。** 把 `Code2WavRequestState` 改成按 slot 预分配的池，slot 复用 talker 的 slot 号：
   - `kv_k / kv_v`：`[slots, layers, W-1, kv_heads, head_dim]`，外加 `history_len[slots]`；
   - `quantized_ctx`：`[slots, codebook_dim, 2]`；
   - `latent_ctx`：`[slots, latent_dim, 12]`，外加 `latent_len[slots]`。
2. **`decode_window_batched(codes [B, Q, F], slots)`。** 同一 batch 内三种上下文的处理方式不同：
   - **KV 历史长度不同**：可以混在一个 batch 里。key 统一排成 `[W-1 个历史位置 | F 个新位置]`，
     无效的历史位置用 `history_len` 生成的 mask 屏蔽。
   - **`pre_conv` 的 2 帧上下文**：`pre_conv` 是第一层卷积，它本来就在左侧补零。把上下文初始化
     成 2 帧全零，再统一丢掉前 2 个输出，和"首块没有上下文"在数学上等价。所以所有请求都可以
     统一成"有 2 帧上下文"。
   - **latent 的 12 帧上下文**：**不能**用零代替。下游的 ConvTranspose 有 bias，ConvNeXt 和
     SnakeBeta 对 bias 的响应不为零。所以 `latent_len` 必须作为分组键。按默认 chunk 计划
     （首块 5 帧，之后 25 帧），它只会取 {0, 5, 12} 三个值。
   - 分组键是 `(new_frames, latent_len)`。同样长度的流（最常见的情况）会落进同一组。
3. **调度。** 去掉 `next_batch` 里 stateful 模式 budget=1 的限制：先按 `chunk_policy` 排序，
   再把与排在最前的 job 同组的 job 一起取出，上限为 `max_chunk_batch_size`；一个 step 内允许
   解多个组，总量受帧数预算约束，避免拖慢 talker 的节奏。同步更新
   `test_stateful_dispatch_takes_one_chunk_at_a_time`。
4. **释放。** `Code2WavRunner.release` 把 slot 的 `history_len`、`latent_len` 清零即可，不需要
   清空 tensor。

**验收标准**：

- batched 输出与逐请求 `decode_window` 相比，最大误差在 fp32 余量内（GEMM shape 变了，不保证
  逐位一致）；与 `decode_exact` 相比仍满足现有 `test_decode_window_is_exact_in_fp32` 的门限；
- 混合 `history_len` 的 batch（一条流在第 2 块、另一条在第 20 块）结果与各自单独解码一致；
- 报告 8 流 / 16 流的 TTFA p50、p95 和最差 gap，以及单次 decode 耗时随 B 的变化。

参考 vllm-omni：`qwen3_tts_code2wav.py`（stateful batch grouping，#5202）、
`segmented_graph_wrapper.py` 的 `_decode_suffix_batch` / `_copy_request_caches_to_batch`；
更新的实现是 `tokenizer_12hz/streaming_decoder.py`：每个请求占一个 slot 的逐层状态，一次调用为
B 个请求各解 T 帧。

### P0.4 Code2Wav CUDA Graph

**优先级依据**：一次 `decode_window` 约 450 个 eager kernel（推算），5 帧首块的耗时主要是
launch。单流下它直接计入 TTFA；并发下每次串行的 decode 都会加到其他流的等待里。sortout 5.1
把 Code2Wav graph 列为 vllm-omni 已完成的优化，8.2（#4924）进一步要求由 runner 统一管理
graph 的生命周期和显存。

**依赖**：P0.3。graph 要求状态地址固定，池化后的状态正好满足。

**实现内容**：

1. **`compilation.py` 新增 `Code2WavGraphs`。** stateful 路径按
   `(B_bucket, new_frames, latent_len)` 捕获；stateless 路径按 `(B_bucket, frames_bucket)`
   捕获（依赖 P2.2）。输入是静态的 codes buffer 和 slot 索引 buffer，graph 内部用
   `index_select` / `index_copy_` 读写状态池。
2. **去掉路径上依赖数据的 Python 逻辑。** `PreTransformerCache.extend`（`code2wav.py`
   L319–341）用 Python 变量 `history` 决定切片长度，并且先 `cat` 再 `copy_`。改成固定长度：
   始终拼接 W-1 个历史位置和新位置，用 mask 屏蔽无效部分，写回时用索引操作。
3. **batch padding。** 与 talker 一样，补齐行指向一个 scratch slot，不能写坏真实请求的状态。
4. **尾块。** 尾块是剩余帧，长度可能是 1–25 中的任意值。把它右填充到稳态大小：decoder 完全
   因果，右侧填充不影响前面帧的输出；填充污染的状态随请求结束释放。这样尾块不需要单独的
   graph。
5. **graph 管理（对应 #4924）。** 新增统一的 graph manager，负责三类 graph（talker、code
   predictor、Code2Wav）：捕获策略、每类各自的 pool、显存记账（pool 大小写入 `/metrics`
   和 benchmark 的 json）、命中次数和回退次数（带原因）、一个 `--code2wav-cuda-graph` 开关。
   不允许静默回退到 eager。

**验收标准**：

- 相同 shape 下 graph replay 与 eager 输出逐位一致；
- 按 phase 报告单次 decode 的 eager / graph 耗时，以及单流、8 流的 TTFA；
- 默认 chunk 计划下回退次数为 0；报告每类 graph 的 pool 显存。

参考 vllm-omni：`segmented_graph_wrapper.py`（`warmup`、`_capture_stateless`、
`_capture_combined_suffix`、`_split_for_graph_buckets`）、`cuda_graph_decoder_wrapper.py`、
`tokenizer_12hz/streaming_decoder.py::StreamingDecodeGraphs`；RFC
[#4924](https://github.com/vllm-project/vllm-omni/issues/4924)。

### P0.5 去掉每帧 host 同步，状态改为常驻 GPU 的 buffer

**收益基线**：上限是 replay 之间的 GPU 空闲时间，P0.1 测出之前没有数字。即使收益不大，
这一项也是 P1.2 多步 graph 的前提。

**实现内容**：

1. **按 slot 索引的常驻 GPU 状态**（对应 sortout 8.5 的 persistent batched GPU buffer）：
   - `last_hidden[slots+1, H]`：talker graph 直接写入，不再 `stack`；
   - `seq_lens[slots]`、`active[slots]`；
   - `text_queue[slots, T_max, H]`、`text_offset[slots]`、`text_len[slots]`，替代
     `_text_step` 的 Python 循环；
   - `recent_layer0[slots, 64]` 环形缓冲 + 计数，替代 `_recent_ids` 的 host 端拼装；
   - `frames[slots, max_frames, Q]`：码本 id 小于 2048，int16 就够，
     17 × 2048 × 16 × 2 B ≈ 1.1 MB。chunk 直接从这里切片，不再 `torch.cat`。
2. **采样、penalty、文本 embedding、下一步输入都在 slot 前缀上计算**，非活跃行用 mask 处理，
   这样 shape 与 graph 一致，为 P1.2 做准备。
3. **停止判定。** `finished[slots]`、`max_frames` 和 repetition guard 都在 GPU 上计算，并直接更新
   `active[slots]`。本步的 codes、finished 标志和停止原因 pack 进一个 staging buffer
   `[slots, Q + 2]`，每个 step 只读回一次，替代现在的两次 `.tolist()`。把这次读回改成异步、让已结束的
   行多跑一帧，是 P1.5 的内容。
4. chunk 策略和背压在 host 上，使用这次合并读回得到的 `frames_generated[slots]`。

**验收标准**：

- greedy 输出与改动前逐位一致；
- 在 `torch.cuda.set_sync_debug_mode("error")` 下跑完 `decode_step` 不报错（合并读回那一次放在例外
  里），即除这一次读回外，每帧 host 同步次数为 0；
- 报告 step 耗时的变化，并与 P0.1 测得的 replay 间空闲时间对照。

参考 vllm-omni：`qwen3_tts_talker.py` L348–351（`gpu_resident_buffer_keys`，#4879）；
RFC [#4855](https://github.com/vllm-project/vllm-omni/issues/4855)。

### P0.6 启动 warmup

**优先级依据**：talker graph 是懒捕获的（`TalkerDecodeGraphs` 的 docstring），server 运行中
每遇到一个新的 (batch, key_len bucket) 组合，就要在请求进行中执行 3 次预热和 1 次捕获；
code predictor graph、Triton JIT（RMSNorm、SnakeBeta、int8 kernel）、cuDNN 对新卷积长度的
算法选择也一样。benchmark 会先跑一轮 warmup，但 `serve` 不会，所以 server 的头几个请求付出
的是启动成本（sortout 6.2 #3424）。工作量小，收益确定。

**实现内容**：

1. `TTSEngine.warmup()`：按配置捕获 talker graph（batch ≤ `max_num_seqs`，key_len bucket ≤
   `max_model_len`；默认 `max_num_seqs=16` 时 batch 只有 1/2/4/8/16 五种，key_len bucket 有
   4096 / 256 = 16 种，共 80 张。要报告捕获时间和 pool 大小，并允许只预捕获常用长度的 bucket，
   更长的保持懒捕获）、每种 batch 大小的 code predictor graph（使用
   checkpoint 的采样参数）、P0.4 的 Code2Wav graph；对每种 chunk 大小跑一次 dummy decode，
   让 Triton 和 cuDNN 完成编译和算法选择。
2. 捕获期间 `active` 全为 False，key 写进 scratch slot，不会污染真实 cache。
3. `serve` 在 warmup 完成前，`/health` 返回 `warming`。

**验收标准**：server 第一个请求的 TTFA 与稳态一致；服务期间日志中不再出现 capture；报告
warmup 耗时和各类 graph pool 的显存。

## P1：stage 0 的 kernel 与 graph

### P1.1 graph-safe 采样与 per-row seed

**收益基线**：带 seed 的请求会退回 eager code predictor，RTF 从 0.162 变成 0.656。另外还有
一个正确性问题：`TalkerRunner._row_generator`（L256–267）只在 batch 中只有一行时才使用请求的
generator，**seeded 请求一旦与其他请求同批，seed 就被静默忽略**，现在只有 graph 回退会告警，
这一点没有告警。

**实现内容**：

1. **counter-based RNG。** 用 Philox 在 Triton kernel 里生成 Gumbel 噪声，随机数由
   `(row_seed, frame_idx, codebook_idx, vocab_idx)` 决定。`row_seed` 和 `frame_idx` 是 GPU
   tensor，可以被 graph 捕获；每行一个独立的随机流，不受 batch 组成影响。没有 seed 的行从一个
   全局计数器取种子。
2. talker 的 layer-0 采样和 code predictor 的 15 次采样都改用这个 kernel，把 temperature、
   top-k 和 Gumbel argmax 融合在一起。top-k 阈值可以像 vllm-omni 那样用二分查找代替排序。
3. repetition penalty 使用 P0.5 的 GPU 环形缓冲。
4. **batch 不变性的边界要写清楚。** logits 本身会随 batch 变化（GEMM 的 M 变了，舍入可能
   不同），所以同一个 seed 在 batch 1 和 batch 8 下不保证生成相同的 codes。验收只要求"同 seed、
   同 batch 组成时逐位一致"；单独与混批之间只比 P0.2 的统计指标。

**验收标准**：

- 带 seed 的 benchmark RTF 与不带 seed 的 graph 路径相差不超过 3%；
- 同一个 seed 跑两次输出完全一致；
- 在合成 logits 上做卡方检验，采样频率与 `torch.multinomial` 一致；保留现有
  `-log(-log u)` 的回归测试。

参考 vllm-omni：`code_predictor_sampling.py`（由外层 MTP graph 提供 uniform 随机数）、
`fused_code_predictor.py::_cp_sample_kernel`；#4970、#4889、#4869。

### P1.2 Talker 多步 CUDA Graph

**优先级依据**：做完 P0.5 后，每帧仍然有两次 replay 加中间的一串胶水 kernel 由 CPU 逐个发射。
RFC [#7093](https://github.com/vllm-project/vllm-omni/issues/7093) 的目标是一次 replay 连续
执行 K 个完整的帧，这里的 K 是时间帧数，不是码本数。

**依赖**：P0.5、P1.1。

#### P1.2-A：单帧整图

把"从 `last_hidden` 采样 layer 0 → embedding → code predictor → residual gather + 文本
embedding → 下一步输入 → talker decode → 新的 `last_hidden`"捕获成一张 graph，按
(B_bucket, key_len bucket) 管理。注意顺序：这张 graph 消费上一帧的 `last_hidden`，产出这一帧
的 codes 和下一帧的 `last_hidden`。

#### P1.2-B：K 帧

1. 在一次 capture 中展开 K 个单帧（K 是静态的）；key_len bucket 必须覆盖 `seq_len + K`。
2. K 帧内遇到 EOS 的行在 graph 内把 `active` 置 False，后续帧的 key 写进 scratch slot，产出的
   帧被丢弃；每个请求最多浪费 K−1 帧的计算。
3. chunk 发射、背压、准入都只在 K 的边界上做。K 不应大于首块帧数，否则会推迟 TTFA。
4. graph 数量 = B bucket × key_len bucket × K 的取值；捕获时间和 pool 显存都随 K 增加，需要报告。

本任务与 RFC 一致，不包括 prefill 捕获、speculative decoding 和把 Code2Wav 合并进来。

**验收标准**：greedy codes 与单步路径逐位一致；P1.1 之后带 seed 的结果也逐位一致；报告
K ∈ {1, 2, 4, 8} 时的每帧耗时，以及每个请求在 EOS 后浪费的帧数。

### P1.3 code predictor：减少每帧的 kernel 数

**收益基线**：每帧约 1500 个 kernel（`compilation.py` 的注释），权重读取的下界约 0.73 ms
（推算），实测"code predictor"列为 7.69 ms（含胶水）。这是 stage 0 最大的单项。

#### P1.3-A：允许 torch.compile 与手工 graph 叠加

现在 `_graphs_possible`（`compilation.py` L300–308）在 `compile_mode != none` 时直接禁用手工
graph，于是只能二选一：要么有 Inductor 的融合但没有 graph，要么有 graph 但全是未融合的小 kernel。
改为用不带 cudagraph 的模式（`mode="max-autotune-no-cudagraphs"` 或默认模式）编译
`_forward_prefix`，完成编译预热后再手工捕获。前缀长度只有 15 种，可以用静态 shape 编译，而不是
现在的 `dynamic=True`。这是成本最低的一步，先用它测出融合的收益上限。

#### P1.3-B：embedding + 投影折叠成查表

每个残差码本的下一步输入是 `small_to_mtp_projection(codec_embedding[i](code))`
（`code_predictor.py` L242）。这是两个线性运算的串联，可以在加载时预先算出
`[15, 2048, 1024]` 的表（bf16，15 × 2048 × 1024 × 2 B ≈ 63 MB），每帧省掉 14 次 embedding
和 14 次 GEMM，换成 14 次 gather。原始的 embedding 表还要保留，因为 talker 计算下一步输入时
也会用到。预计算的 GEMM 与逐行 GEMM 的归约顺序不同，结果会有 1 个 ULP 量级的差异。

#### P1.3-C：frame-local KV cache

每一步只计算新 token，attention 读取本帧已经算好的 K/V。GEMM 的 M 从 `prefix_len` 降到 1，
更重要的是 15 步的 shape 完全相同，graph 和融合都更简单。已有位置的 K/V 不再被重算，数值上
与重预填充不保证逐位一致，需要重新跑 parity，确认是否仍与参考实现逐位一致；不一致时用
teacher-forced logits 的 cosine 验收。

#### P1.3-D：手写融合 kernel

在 A–C 都量过之后再决定是否做。vllm-omni 的做法是每层只保留少数几个 kernel：RMSNorm、
一个 QKV GEMM、一个 attention kernel（内含 q/k RMSNorm、RoPE、K/V 写入和 attention）、
用 `addmm_` 把 O 投影累加进残差、RMSNorm、gate/up GEMM、SiLU-mul、用 `addmm_` 累加 down 投影。

**验收标准**：用 P0.1 的拆分单独报告 code predictor 每帧耗时和 kernel 数；greedy parity 与
teacher-forced logits cosine；P1.3-B 报告折叠表的显存。

参考 vllm-omni：`fused_code_predictor.py`、`cached_code_predictor.py`。

### P1.4 talker decode 层内：flash decode 与融合

**收益基线**：talker graph forward 4.16 ms 中，GEMM 约 1.06 ms（推算），其余约 3.1 ms 是小 kernel。

有一个容易忽略的事实：**decode 阶段从来没有用过 flash attention。** `Qwen3TTSTalker.decode`
（L333–350）总是传入 dense mask，而 `FlashAttentionBackend.supports` 遇到 mask 就拒绝
（`flash_attn.py` L59–60），于是每一层的 decode attention 都走 SDPA + 布尔 mask，还要对 key_len
bucket 中最多 255 个被屏蔽的位置做无用计算。`--talker-attention-backend flash_attn` 实际上只
影响 prefill 和 code predictor。vllm-omni 的 talker 是 vLLM 的 AR 模型，直接用 vLLM 的 decode
attention，这一点 qtts 需要自己补上。

**实现内容**：

1. **换成 `flash_attn_with_kvcache`。** 它支持按行的 `cache_seqlens`，可以就地追加新 K/V；
   `cache_batch_idx` 能把行映射到任意 cache slot，非活跃行直接映射到 scratch slot。KV cache 的
   布局 `[slots+1, max_len, kv_heads, head_dim]` 正好符合它的 `k_cache` 格式。q/k 的 RMSNorm
   要在 RoPE 之前做：先在外面做完 norm，再把 rotary 表交给 kernel（Qwen 是非交错的 neox 风格）。
   这样 shape 不再依赖 key_len，key_len bucket 只用来限制 graph 数量，被屏蔽位置的计算也没了。
2. 如果不用 kernel 自带的 rotary，就用一个 Triton kernel 把 q/k norm、RoPE 和 KV 写入融合在一起。
3. 残差加法 + RMSNorm 融合（vLLM 的 fused add-RMSNorm），SiLU-mul 融合。
4. 每一步单独测量。融合会改变 ULP，README 已经记录过 fused RMSNorm 会让接近平局的 argmax
   翻转，所以 parity 继续使用 eager 路径，新路径按 teacher-forced logits cosine 和 P0.2 验收。

**验收标准**：talker graph forward 每帧耗时下降；teacher-forced logits cosine ≥ 0.9999；
报告 talker graph 数量的变化。

### P1.5 异步 decode：launch 与 resolve 错开一步

**收益基线**：`TTSEngine.step`（`engine.py` L179–207）里 CPU 和 GPU 完全串行。GPU 跑完 code
predictor 之后，CPU 才能读回停止标志、逐个请求记账、在 `_advance` 里用 Python 循环拼下一步输入，
然后 launch talker graph；接着 `collect_chunks`、`next_batch`、发射 Code2Wav，再在 `_deliver`
（L221–225）里对每个 chunk 做一次同步的 `.to("cpu")`，最后是准入和 prefill。这些 CPU 工作进行时
GPU 都是空闲的。收益上限就是每个 step 里 CPU 工作的总时长，P0.1 测出之前没有数字。单请求加 graph
时这部分不大；它随运行中的请求数和每步交付的 chunk 数增长，所以主要收益在多并发。

大致思路是：每一步拆成 GPU 侧的 kernel launch 和 CPU 侧的 resolve，launch
第 N 步之后再 resolve 第 N−1 步，batch 小于 2 时退回同步。

**与其他任务的边界**：

- P0.5 让一个 step 内部没有 host 同步，读回合并成一次；本任务把这一次读回改成异步，并重排
  `TTSEngine.step`，让 CPU 记账、chunk 调度和交付与下一步的 GPU 计算重叠。
- P1.2 减少的是一个 step 内 GPU 侧的 kernel 和空隙，本任务隐藏的是 step 之间的 CPU 时间，两者叠加。
  做了 P1.2-B 之后，resolve 改为每 K 帧一次。
- P3.4 是两个 stage 的 GPU 工作在两个 stream 上重叠；本任务是同一个 stream 上 CPU 与 GPU 重叠。

**依赖**：P0.5（下一步的输入必须完全在 GPU 上算出，host 只需要知道它本来就能推出的信息：哪些 slot
在运行、各自的 `seq_len`、该用哪个 key_len bucket）。

**实现内容**：

1. **拆分 step。** 每次迭代按这个顺序执行：
   - launch(N)：从常驻 GPU 状态出发 replay 第 N 帧（采样、code predictor、下一步输入、talker
     decode）；把本步的 16 个 code、finished 标志和停止原因 pack 进 staging buffer
     `[slots, Q + 2]`，`non_blocking` 拷到 pinned host buffer，紧接着 record 一个 CUDA event。
   - resolve(N−1)：先 `event.query()`，没完成才 `event.synchronize()` 并计数；然后逐个请求记账、
     设置 `DRAINING`、跑 chunk 策略和背压，选出 Code2Wav job 并发射（在 GPU 队列里排在第 N 帧之后）。
   - 交付：Code2Wav 输出的音频同样异步拷到 pinned buffer 并 record event，在下一次 resolve 时交付，
     替代 `_deliver` 里的同步 `.to("cpu")`。
   - 准入：resolve 释放的 slot 在这之后才能分给新请求，新请求的 prefill 在同一个 stream 上排在
     launch(N) 之后，顺序天然正确。prefill 路径内部不能有 host 同步，否则会把重叠打断，用
     `set_sync_debug_mode` 检查。
2. **pinned buffer 的读写冲突。**
   - D2H：GPU 往一个 host buffer 写第 N 步结果时，CPU 正在读第 N−1 步的结果，所以用两个 pinned
     staging buffer 交替使用。CUDA stream 的顺序只约束 GPU 侧的操作，管不到 CPU 读 pinned 内存。
     音频 buffer 用一个小的环形池，每块等它的 event 完成后才复用。
   - H2D：`TalkerDecodeGraphs.run`（`compilation.py` L168–173）现在是先改写 pinned 的
     `host_seq_lens`、`host_active`，再 `non_blocking` 拷到 GPU。这在今天是安全的，只因为每帧的
     `.tolist()` 强制了同步；异步之后，host 可能在上一步的拷贝真正执行之前就改写了这块内存。P0.5 把
     `seq_lens`、`active` 改成 GPU 上推进之后这条路径就不需要了；准入时写入单个 slot 的状态，用带 Python
     标量参数的 `fill_` 或 index 写入，不经过可复用的 host buffer。
3. **lookahead：已结束的请求多跑一帧。** 在第 N−1 步遇到 EOS 的请求，CPU 还没 resolve，它仍然在第
   N 步的 batch 里。规则：
   - 第 N−1 步在 GPU 上已经把 `active[slot]` 置为 False（P0.5 的 GPU 停止判定），所以第 N 步这一行的
     K/V 走现有的 scratch slot 分流（`talker.py` L343），`frames` buffer 的写入也要按 `active` 屏蔽，
     staging 里这一行标记为无效，resolve 时丢弃；
   - slot 的释放（`scheduler.finish`、Code2Wav 状态的 `release`）只在 resolve 里做，保证新请求不会在
     旧请求的最后一帧还没执行完时拿到这个 slot；
   - `MAX_FRAMES` 是 host 能提前算出的，可以在 launch 前直接排除；repetition guard 在 GPU 上算，与 EOS
     走同一条路径。
4. **TTFA 的代价。** 同步循环里，第 N−1 帧凑齐首块后，Code2Wav 紧接着就在同一步执行；异步之后，首块
   要等 resolve(N−1)，而这时 GPU 队列里第 N 帧已经排在前面，所以首块晚一个 talker step（约 12 ms，单流
   TTFA 0.082 s 的 15% 左右，推算）。缓解办法：
   - 运行中的请求少于 `--async-decode-min-rows`（默认 2）时退回同步，与 SGLang-Omni 的做法一致；
   - 帧数在没有 EOS 时是确定的，host 能提前知道哪一步会凑齐某个请求的首块，那一步改走同步 resolve；
   - 做了 P3.4 之后，Code2Wav 在 side stream 上只等第 N−1 步的 event，这个延迟自然消失。
5. **计时。** `StageTimer.measure`（`utils.py` L72–87）每个 stage 前后都 `torch.cuda.synchronize()`，
   开着它异步就没有意义。P0.1 的计时要改成 CUDA event，另外报告 host 侧每步耗时、resolve 时 event
   未完成需要阻塞的比例，以及 Nsight 里 step 之间的 GPU 空闲。
6. **异常处理。** resolve 抛异常时第 N 步还在途：先 synchronize，丢弃在途结果，再按 P3.1 的失败传播
   处理受影响的请求。
7. 开关 `--async-decode`，默认关闭，验收通过后再改为默认开启。

**验收标准**：

- greedy codes 和 Code2Wav 波形与同步路径逐位一致（kernel 相同，只是 host 操作的顺序变了）；
- 稳态下 `set_sync_debug_mode("warn")` 只在 event 未完成时报告同步，报告这个比例；
- 1 / 4 / 8 / 16 流下的每帧耗时、聚合 RTF、TTFA p50 / p95、最差 gap，以及 step 间 GPU 空闲的前后对比；
- 单流 TTFA 不比同步路径差；
- 专门的测试：请求 A 在第 N−1 步 EOS、第 N 步还在途时，新请求 B 被分到同一个 slot，B 的输出与它单独
  运行时一致；在途时取消请求不会泄漏 slot 或 Code2Wav 状态。

参考：SGLang-Omni 的 `model_runner/base.py`（`execute_launch`、`execute_resolve`、`_next_host_staging`）
和 `higgs_tts/model_runner.py`（`_decode_pack_gpu`）；思路来自 SGLang 的 overlap scheduler。

## P2：Stage 1 的计算与条件输入

### P2.1 批量 / 分块 prefill，prompt 缓存，Stage 0 前缀缓存

**优先级依据**：关键判断三的第 1 项。另外，server 中 `engine.submit` 在 engine 线程里执行
（`_drain_inbox`），prompt 构造（tokenizer 加若干次 `embed_text` / `embed_codec`）会阻塞 step
循环；新请求的 prefill 也会让已经在运行的流停顿，直接体现为它们的 gap。这一点要用 P0.1 的
"突发到达"负载来测。sortout 6.2 把"cache-miss 批量 / 分块 prefill"列在 ref-audio 路径下，
但 #3535 WS-3 的 prefill 批处理同样适用于 CustomVoice。

**实现内容**：

1. **批量 prefill。** 把同一 step 接纳的多个 prompt 打包成 varlen（`flash_attn_varlen_func`，
   causal），用 `cu_seqlens` 隔离请求，KV 按 token 写到各自的 slot。`_run_layers` 已经支持
   `write_index`，主要改动在 attention。
2. **分块 prefill。** 限制每个 step 的 prefill token 数，与 decode 交错执行，限制它对运行中流的
   影响。
3. **prompt 缓存。** 按 `(speaker, language, instruct, streaming mode)` 缓存与文本无关的 embedding
   片段：role header、codec 控制标签和 speaker、`tts_bos` / `tts_eos` / `tts_pad`。按字节数做 LRU。
4. **Stage 0 前缀 KV 缓存。** CustomVoice 的公共前缀只有约 9 个 token（role header + 控制标签 +
   speaker），收益很小；VoiceDesign 的长 instruct 放在最前面，多个请求复用同一个 instruct 时
   收益明显。命中时把前缀 KV 从缓存 slot `index_copy_` 到请求的 slot。缓存键必须包含所有影响
   这些位置的因素（speaker、language id、instruct 文本、streaming mode、模型），vllm-omni 在这里
   出过串音 bug（#3665、#4317、#4373）。

**验收标准**：8 个请求同时到达时的 prefill 总耗时；8 流 TTFA p95；突发到达时运行中流的最差
gap；与逐个 prefill 相比，greedy codes 一致或 teacher-forced cosine ≥ 0.9999。

### P2.2 stateless bucket 分桶

**优先级依据**：`Code2WavRunner._run_stateless`（L106–109）按精确的总帧数分组。有 ramp、dynamic
首块或尾块时，分组会碎掉，模块 docstring 中"首块之后自然对齐"的假设就不成立了。stateless 模式
本身是近似的（左上下文 25 帧时峰值误差接近满量程），所以优先级低，只有在接受这个质量取舍时才需要。

**实现内容**：

1. 关键事实：decoder 完全因果，**右侧填充不影响前面帧的输出**，填充后裁掉多出的采样点，结果
   无损。
2. 按帧数 bucket（例如 8、16、32、64）右填充 codes（用任意合法 id），解码后先裁掉左上下文，
   再裁到 `new_frames × 1920`。
3. B 也做 bucket，为 P0.4 的 stateless graph 提供固定 shape。
4. 像 vllm-omni 的 `_record_decode_batch_stats` 一样统计实际帧数与填充帧数。

**验收标准**：真实采样点与不分桶的结果一致（因果填充，应当逐位一致，若 cuDNN 因长度不同选了
不同算法，则在 fp32 余量内）；报告填充比例。

### P2.3 上采样残差块融合

**优先级依据**：sortout 5.2 的结论是"只有在 profiling 确认后才值得做"，关键判断四也说明单流下
Code2Wav 的算力可以忽略，所以必须在 P0.4 之后，用 P0.1-5 的微基准确认剩余 GPU 时间确实在这里。
从代码看，可融合的点很多：每个 ResidualUnit 是 SnakeBeta → `F.pad`（一次拷贝）→ 卷积 →
SnakeBeta → `F.pad` → k1 卷积 → 加法；`CausalConvTranspose1d` 的右裁剪后接 `.contiguous()`，
又是一次拷贝。一次调用约有 30 次 pad、6 次裁剪拷贝和 30 次 SnakeBeta，作用在 `[B, C, T]` 的大
tensor 上。

#### P2.3-A：加载期线性折叠（低成本）

vllm-omni 的 `streaming_decoder.py` 在加载时把几组串联的线性运算合并：RVQ 的 output projection、
`pre_conv` 和 transformer 的 `input_proj` 合并成一个 k=3 卷积；LayerScale 折进 `o_proj` /
`down_proj`；ConvNeXt 的 `gamma` 折进 `pwconv2`。这几项改动小，近似无损。

#### P2.3-B：去掉 pad 和拷贝

在 stateful 模式下，每个卷积的左上下文可以直接从状态中取，不再补零后拷贝（这是
`streaming_decoder.py` 的做法：每个因果卷积只保留最后 `(k-1) × dilation` 行输入）；转置卷积的
裁剪改成 view。

#### P2.3-C：SnakeBeta 融进卷积输入

time-major 布局 `[B, T, C]` 下，每个稠密卷积可以写成 implicit GEMM，SnakeBeta 在读入时完成，
k1 卷积和残差加法放在 GEMM 的 epilogue 里，并能用上 tensor core。可以先用 `torch.compile`
编译整个 decoder stack（每种 chunk 大小一个静态 shape）作为低成本对照。

**验收标准**：Code2Wav 单次调用的 GPU 耗时；按 P0.2 的确定性波形指标与 fp32 eager 参照对比。

参考 vllm-omni：`tokenizer_12hz/time_major_conv.py`、`time_major_stack.py`、`streaming_decoder.py`。

### P2.4 精度路径：bf16 Code2Wav 与 fp16

**优先级依据**：`PERFORMANCE.md` 明确写了 bf16 Code2Wav 没有测过。vllm-omni 最新的流式 decoder
用的是 bf16（`qwen3_tts_talker.py` L1298：`StreamingCodecDecoder(..., dtype=torch.bfloat16)`），
说明那边认为质量可以接受，但这里需要自己验证。fp16：`--dtype float16` 能被接受，但 code predictor
没有做 #3253 要求的 fp32 中间量。

**实现内容**：

1. Code2Wav 按模块设精度：transformer bf16，卷积 bf16 或 TF32，最后一层卷积可保留 fp32；用 P0.2
   的确定性波形指标（SNR、mel L1）和 ASR、说话人相似度评估每种组合。
2. SnakeBeta 的 Triton kernel 在 fp32 中计算 `sin`，结果转回输入精度再相加；eager 路径在 bf16 下
   全程使用 bf16，两条路径在 bf16 下结果不同。需要确认用哪一条作为参照，并核对 #3472 修复的问题
   在这里是否存在。
3. fp16：要么在 code predictor 的 RMSNorm、attention softmax 和 logits 处升到 fp32，要么在配置
   阶段直接拒绝 fp16，二选一，不能保持现状。

**验收标准**：输出一张"精度组合 × SNR / mel L1 / WER / UTMOS / 单次耗时"的表；默认保持 fp32，
除非某个组合满足 P0.2 的门限。

### P2.5 按负载调整稳态 chunk

**优先级依据**：现在只有首块随负载变化（`chunking.build_policy`），稳态 chunk 固定为 25 帧
（sortout 8.4"根据并发动态调整 codec window"）。高负载时更大的 chunk 能分摊每次调用约 450 个
kernel 的固定开销；当某条流的缓冲余量很小时，应当先给它较小的块。收益只能在 P0.1 的混合负载下
看到。

**实现内容**：在 `ChunkPolicy` 中按 `deadline` 策略已经计算的 slack（已交付音频减去已过时间）
调整下一块的大小：slack 大就放大，slack 小就缩小；大小取有限集合（例如 {10, 25, 50}），以便与
P0.4 的 graph 配合。

**验收标准**：16 / 32 流混合负载下的 underrun 次数、最差 gap 和聚合 RTF。

### P2.6 量化补全

**优先级依据**：关键判断一说明读权重的字节不是当前瓶颈，所以这一项优先级最低，主要价值在显存和
大 batch。

**实现内容**：

1. fp8 weight-only：`quant/fp8.py` 的 docstring 已经写明做法，权重按 int8 存储，在 Triton kernel
   内 bitcast 为 `tl.float8e4nv`。显存与 int8 相同，速度预计接近 int8。
2. 在 batch 64、128 下补 fp8 W8A8 的 benchmark 行，找到它开始比 bf16 快的交叉点。
3. int8 kernel 在窄输出上不如 cuBLAS（down_proj 0.64 倍、qkv / o_proj 0.90 倍），可以考虑 split-K。
   这一项不在 sortout 里，按需处理。

**验收标准**：RTF、显存和 P0.2 质量指标三列同时报告。

## P3：serving 与可靠性

### P3.1 请求生命周期：取消、超时、失败传播、ID 去重

**优先级依据**（sortout 8.5，#6453 / #4855），代码里的事实：

- 没有取消：客户端中途断开，请求仍会生成完（`server.py` 的模块 docstring）。
- `RequestState.FAILED` 从来没有被赋值，只在 `collect_chunks` 中被读取。
- `engine.step()` 抛异常时，`EngineThread._loop`（L97–104）调用 `_fail_all` 清空所有 outbox，但
  `scheduler.running` 中的 tracker、slot 和 Code2Wav 状态都没有清理，下一个 step 会在同样的状态上
  继续执行。结果要么反复抛同一个异常，要么继续为已经断开的客户端生成音频。
- 重复的 `request_id` 会覆盖 `outboxes`（L126），`round_robin` 也按 `request_id` 分组。

**实现内容**：

1. `TTSEngine.abort(request_id)`：标记为 ABORTED，从 waiting / running 中移除，丢弃它的 pending
   ChunkJob，释放 slot 和 Code2Wav 状态。只在两个 step 之间执行，因为 engine 由单线程独占。
2. server 检测客户端断开（写入时的 `ConnectionResetError`，或 `transport.is_closing()`），向 engine
   线程投递一条 abort 命令。
3. 每个请求设置截止时间（总时长上限、无进展时长上限），超时则置为 FAILED 并给出原因。
4. step 抛异常时，尽量定位到出错的请求；定位不到就 abort 所有运行中的请求，并把 scheduler 重置到
   干净状态（释放全部 slot 和状态）。
5. submit 时拒绝重复的 `request_id`，或者由 server 生成内部 id。

**验收标准**：测试覆盖以下场景：流中途 abort 后 `slots_in_use` 恢复，之后不再发出 chunk；被释放的
slot 立刻分配给新请求，新请求的音频正确（KV cache 不需要清零，因为 `seq_len` 重置后旧位置被 mask
屏蔽，这一点要用测试钉住）；注入异常后 server 能继续服务新请求。

### P3.2 Stage 1 显存预算与可观测性

**优先级依据**（sortout 8.5）：`Code2WavRunner.state_bytes_per_request`（L75–76）没有任何调用方，
尽管 `state_bytes_estimate` 的 docstring 写的是"Used by the scheduler to size how many streaming
requests fit"；graph pool 的显存只在捕获时打一行日志；`/metrics` 没有 graph 命中 / 回退、各阶段
等待时间和 underrun。

**实现内容**：

1. 启动时输出显存表：各子模型权重、KV cache、Code2Wav 状态池（P0.3 之后是固定值）、三类 graph
   pool、workspace；准入时检查总预算。
2. `/metrics` 增加：每类 graph 的 replay / capture / fallback 次数、chunk 队列深度、被背压的请求数、
   每个请求的 TTFA 和最差 gap 分布。

**验收标准**：启动显存表的总和与 `memory_allocated` 相差不超过 5%；`/metrics` 字段齐全。

### P3.3 运行时保护：重复与缺失 EOS

**优先级依据**（sortout 8.7，#6158）：`repetition_guard_frames` 默认是 0（关闭），而且只检测
"code-0 连续 N 帧完全相同"，抓不到周期性重复；`max_frames` 固定为 2048 帧（约 164 s），与文本
长度无关，一句 10 个字的话失控时也要跑满 164 s。

**实现内容**：

1. 按文本长度设帧数上限：`frames ≤ a × text_tokens + b`，a、b 用 P0.2 评测集的"帧数 / 文本 token
   数"分布拟合（例如取 p99.9 再乘 1.5）。
2. 在 code-0 的滑动窗口上检测周期性重复（周期 ≤ N 帧、重复 ≥ M 次），基于 P0.5 的 GPU 环形缓冲
   计算。
3. 以保守阈值默认开启，用不同的 `FinishReason` 区分触发原因。

**验收标准**：人为制造的失控生成（例如压低 EOS 的 logit）能被截断；P0.2 的正常评测集上误触发为 0。

### P3.4 Stage 0 / Stage 1 在两个 stream 上重叠

**优先级依据**：sortout 6.1 的流水并行；README 的"已知缺口"中说明了现在刻意不做的理由：会给
每次测量引入"重叠效率"这一项。P0.3、P0.4 之后 Stage 1 的单次调用会变短，低并发下重叠的收益不大；
但关键判断四指出高并发时 Code2Wav 会追上 stage 0，那时重叠最多接近 2 倍。

**实现内容**：Code2Wav 放到一个 side CUDA stream 上，第 N 步的 decode 与第 N+1 步的 talker 重叠，
依赖关系用 event 表达（codes 从 P0.5 的 frames buffer 读取）；P0.1 的计时改为每个 stream 各自的
CUDA event；开关 `--overlap-stages` 默认关闭；报告重叠效率
`(t_stage0 + t_stage1 − t_step) / min(t_stage0, t_stage1)`。两个 stage 都是 launch-bound 时，
同卡并发才有收益，这一点要用数据确认。

**验收标准**：8 / 16 / 32 流的聚合 RTF；输出与不重叠时逐位一致（kernel 相同）。

### P3.5 Stage 1 多副本与进程级拆分

**优先级依据**：sortout 6.1 明确说拆分是解决阶段负载失衡的 serving 手段，低并发和单机时反而增加
传输和调度成本。另外，vllm-omni 最新的 #8259 走的是相反方向：把流式 codec decode 放进 talker 进程
（`stream_decode.py`、`first_frame_decoder.py`），用来压低 TTFA。qtts 本来就是单进程，与这个方向
一致。所以这一项排在 P3.4 之后，只在 P3.4 证明 Stage 1 确实成为瓶颈后才做。

**实现内容**：最简单的版本是同进程内 stage 0 和 stage 1 放在不同 GPU 上；完整版本是独立的 Stage 1
worker 进程（各自持有 Code2Wav 和状态池），codes 通过 IPC 或共享内存传递，按 request id 粘性路由，
背压跨进程（扩展 `max_pending_chunks_per_request`），失败传播依赖 P3.1。

**验收标准**：整机吞吐与每卡效率分别报告；粘性路由下的音频与单进程逐位一致。

### P3.6 token 级流式文本输入

**优先级依据**：sortout 8.3，RFC [#6496](https://github.com/vllm-project/vllm-omni/issues/6496)，
Qwen3-TTS 是首个目标模型。qtts 已经有 streaming 文本条件（`non_streaming_mode=False` 时每帧从
`trailing_text` 队列取一个文本 token），但文本在 submit 时就全部确定，队列用完后一直停在
`tts_eos` 这一行。

**实现内容**：

1. API：`append_text(request_id, delta, final=False)`。
2. 稳定前缀提交：数字、单位、缩写等可能有歧义的尾部先挂起，等语义确定再提交（音频无法撤回）。
3. 文本队列为空但还没 `final` 时，不能喂 `tts_eos`（模型会以为文本结束），而是让请求进入类似
   THROTTLED 的等待状态，暂停 talker。
4. 分段边界上继承 Code2Wav 状态（已经是每请求的）和 talker 最近一段对齐状态；容量自适应分段按
   RFC 的"声学 decode step 数 / 文本 token 数"比例 ρ 选边界。

**验收标准**：上游按真实 LLM 的 token 速率喂文本时，从第一个文本 token 到第一段音频的延迟；分段处
的能量和基频跳变；P0.2 质量指标不劣于整段输入。

## P4：依赖 Base（voice clone）支持的任务

qtts 只实现了 CustomVoice 和 VoiceDesign，`PromptBuilder.build_prompt` 对 Base 直接报错，
`loader` 跳过了 codec encoder 的权重。下面这些都以"先支持 Base"为前提，排在最后：

| 任务 | 出处 | 说明 |
| :-- | :-- | :-- |
| 支持 Base：codec encoder 生成 `ref_code`，ECAPA 生成 x-vector | 6.2 条件输入 | 前提 |
| ICL：用 `ref_code` 初始化 Code2Wav 状态，裁掉参考音频的波形 | 3.4、4.2 | qtts 的逐层 KV 设计下，等价于先对 ref codes 跑一次 `decode_window` 并丢弃输出 |
| ref-audio 编码缓存、`SpeakerCacheManager`、可复用的 prefix KV | 6.2 #4200/#2630/#3492 | 参考 `vllm_omni/utils/speaker_cache.py` |
| 避免同一参考音频重复编码 | 8.6 #4256 | |
| cache-miss 时批量 ref-audio prefill、FP8 ref encoder | 6.2 #3535 WS-3 | |
| x-vector-only 请求屏蔽开头的静音 codec 词表 | 6.3 #4966 | 参考 `qwen3_tts_talker.py` L326–346、L1131–1170 |
| Base ICL 音质回退用例 | 6.3 #1554/#1731/#1664 | 纳入 P0.2 评测集 |

## 判定汇总

| 判断 | 判定 | 依据 |
| :-- | :-- | :-- |
| stage 0 离权重带宽下界约 7 倍 | 推算 | 每帧 5.25 GB → 1.6 ms，实测 11.85 ms |
| stage 0 的剩余时间在串行小 kernel，不在 GEMM | 推算，P0.1 确认 | GEMM 约 1.06 ms，talker forward 4.16 ms |
| "code predictor 7.69 ms"包含帧间胶水 | 已确认（测法） | 该列是 step 减去 talker forward |
| int8 收益小，因为字节不是瓶颈 | 已坐实 | RTF 0.162 → 0.159 |
| 每帧两次 host 同步 | 已坐实（代码） | `talker_runner.py` L203–204 |
| 每个 step 内 CPU 与 GPU 完全串行，交付时每个 chunk 再同步一次 | 已坐实（代码） | `engine.py` L179–207、L221–225 |
| 异步 decode 会让首块晚一个 talker step | 推算（机制） | Code2Wav 排在下一帧之后；P1.5 用同步回退和首块同步 resolve 规避 |
| decode 从不使用 flash attention | 已坐实（代码） | decode 总是传 mask，flash 后端遇 mask 拒绝 |
| 并发下 TTFA 恶化的主因是 stateful 每 step 只解 1 块 | 机制已坐实，占比待 P0.1 | `scheduler.py` L216–218 |
| 单流下 Code2Wav 算力可以忽略 | 推算 | 约 4.9 GFLOP / 帧 |
| 高并发时 Code2Wav 会追上 stage 0 | 推算 | 取决于 cuDNN 是否走 tensor core |
| stateless 不是免费的加速 | 已坐实 | `test_code2wav_modes.py`，左上下文 25 帧时峰值误差约 0.9 |
| 带 seed 的请求混批时 seed 被静默忽略 | 已坐实（代码） | `_row_generator` |
| step 异常后 scheduler 不清理 | 已坐实（代码） | `_fail_all` 只清 outbox |
| Code2Wav 状态显存估算没有调用方 | 已坐实（代码） | `state_bytes_per_request` |
| 起始静音屏蔽不适用于 CustomVoice | 已确认 | vllm-omni 只对 x-vector-only 生效 |
| fp8 W8A8 在 decode batch 下更慢 | 已坐实 | 0.20–0.71 倍 |
| NPU / XPU | 不做 | 不在范围内 |

## 推进顺序

### 第一波：度量与低成本修复

1. **P0.1 性能度量**、**P0.2 质量度量**：后续所有任务的前提，并在 H200 上重建基线。
2. **P0.6 启动 warmup**：工作量小，server 首请求直接受益。
3. **P3.1 生命周期**：属于正确性问题，越晚修越难。
4. **P0.5 的第一部分**：两次 `.tolist()` 合并成一次读回，penalty 窗口移到 GPU。

预期：数值不变，server 首请求 TTFA 回到稳态水平，P0.1 给出本文所有推算数字的实测值。

### 第二波：Stage 1，目标是并发下的 TTFA 和 gap

1. **P0.3 stateful 批处理** → **P0.4 Code2Wav CUDA Graph**。
2. **P3.2 显存预算与可观测性**：P0.4 引入了新的 graph pool，需要记账。
3. **P2.2 stateless bucket**：只在使用 stateless 模式时需要。

目标（不是预测）：8 流 TTFA p95 接近单流水平，数值在 fp32 余量内不变。

### 第三波：stage 0，目标是每帧耗时

1. **P0.5 剩余部分** → **P1.1 graph-safe 采样** → **P1.2-A 单帧整图**。
2. **P1.3-A compile 与 graph 叠加**、**P1.3-B 折叠表** → **P1.4 flash decode 与融合**。
3. **P1.5 异步 decode**：P0.5 完成后即可开始，和 1、2 两项互不依赖。它的收益在多流，单流的每帧耗时
   基本不变，所以按 P0.1 测出的 step 间 CPU 时间决定排在前面还是后面。
4. **P1.2-B K 帧**、**P1.3-C/D**：根据前面测得的剩余时间决定是否做。

目标（不是预测）：每帧权重读取的下界约 1.6 ms，离下界 2–3 倍即 4–5 ms / 帧，相当于把现在的
11.85 ms 降到一半以下。风险集中在数值：每一步都要有 teacher-forced cosine 和 P0.2 的指标。

### 第四波：serving 与结构性能力

1. **P2.1 批量 prefill 与 prompt 缓存**、**P2.5 动态 chunk**：需要 P0.1 的混合负载才能体现收益。
2. **P3.4 两个 stream 重叠** → 视结果决定 **P3.5 多副本**。
3. **P2.4 精度**、**P2.3 卷积融合**、**P2.6 量化**：高并发下 Code2Wav 成为瓶颈后才有意义。
4. **P3.3 运行时保护**、**P3.6 流式文本**、**P4 Base**：功能性工作，按需求排期。

### 按场景选配置

| 场景 | 推荐组合 |
| :-- | :-- |
| 单流最低 TTFA | talker graph + P0.4 + P0.5 + P0.6 + 小首块（5 帧）；之后加 P1.2-A |
| 8–64 流并发流式 | 上一行 + P0.3 + P1.5 + P2.1 + `deadline` 策略；高并发再加 P3.4 |
| 离线批量导出 | `sync` 模式或大 chunk；只有接受质量取舍时才用 stateless + P2.2；缺显存时用 int8 |
| 可复现评测 | P1.1 |
| 上游 LLM 流式输出文本 | P3.6 |

## 每步的质量门槛

- **无损改动**（P0.4、P0.5、P0.6、P1.2、P1.5、P2.2、P3.1、P3.4）：greedy codes 与 Code2Wav 波形逐位一致，
  `pytest tests -q` 全部通过，parity 使用 `--talker-rmsnorm-backend eager`。
- **改变计算顺序的改动**（P0.3、P1.3、P1.4、P2.1、P2.3-A/B/C）：说明不能逐位一致的原因（GEMM 的 M
  变了、kernel 不同、预计算改变了归约顺序）；stage 0 要求 teacher-forced logits cosine ≥ 0.9999；
  Code2Wav 要求与 `decode_exact` 的误差在现有测试门限内；P0.2 的指标落在重复运行的方差带内。
- **有损改动**（P2.4、P2.6、stateless 模式、P1.1 的随机流变化）：报告 P0.2 的全部指标，给出均值和
  最差值，确定性的 Code2Wav 改动额外报告 SNR 和 mel L1。
- **不静默回退**：graph 回退、seed 被忽略、attention 后端回退都必须计数并出现在日志和 `/metrics` 中。
- **分开报告**：每项单独开启的结果和组合结果都要给出，`sweep.py` 每行一个独立进程的约定不变。

## 验证方法

每项改动在相同负载、相同 warmup 下对比改动前后，至少收集：

1. `python examples/benchmark/sweep.py --model $MODEL --suite <suite>`：RTF、TTFA p50 / p95、最差 gap、
   underrun、峰值显存。
2. P0.1 的每帧拆分：talker replay、胶水、code predictor replay、Code2Wav（按 phase 和 B），以及每帧
   kernel 数和 GPU 空闲时间。
3. 数值：`tests/test_parity.py`（greedy，与参考实现逐位一致）、`tests/test_code2wav_modes.py`（波形），
   以及 P0.2 的质量脚本。
4. P0.5 用 `torch.cuda.set_sync_debug_mode("error")` 确认每帧没有 host 同步。
5. Nsight Systems：P3.4 的两个 stream 重叠情况、P1.2 的 replay 间空闲、P1.5 的 step 间空闲。

流式相关的改动，验收看的是**最差 gap 和 underrun**，不是平均值；吞吐类的改动，验收看的是**聚合
RTF 在相同质量门限下的变化**。

## 推荐负载

| 场景 | 变量 | 验证目标 |
| :-- | :-- | :-- |
| 单流 | 26 字中文，CustomVoice，`vivian` | 每帧耗时、TTFA，所有 stage 0 任务的主指标 |
| 并发同构 | 1 / 2 / 4 / 8 / 16 / 32 流同时提交 | 吞吐、TTFA p95（P0.3、P0.4） |
| 并发异构 | 10–300 字混合，泊松到达 | 调度策略、最差 gap、underrun（P2.5） |
| 突发到达 | 8 流运行中再同时到达 8 流 | prefill 对运行中流的影响（P2.1） |
| 慢消费者 | 一条流按实时速率读取 | 背压、server 队列 |
| 长文本 | 超过 568 帧（约 45 s） | stateful 稳态窗口、`max_frames`、P3.3 |
| 带 seed | 固定 seed，batch 1 与 batch 8 | P1.1 |
| 质量集 | 固定句子 × speaker × seed | 所有有损和改变计算顺序的任务 |

## 指标定义

| 指标 | 含义与用途 |
| :-- | :-- |
| RTF（聚合） | 总 wall clock / 总音频时长；多流时看聚合值，单流 RTF 大于 1 不一定有问题 |
| TTFA | 从 submit 到第一段音频交付；报告 p50 / p95 / max |
| 最差 inter-chunk gap | 相邻两次交付之间最长的间隔，决定一条流能不能听 |
| underrun | 某个 gap 超过了此前已交付的音频时长，即听众实际听到的断音 |
| 每帧耗时 | 一次 `decode_step`，按 P0.1 拆成 talker replay、胶水、code predictor replay |
| 带宽利用率 | 每帧权重字节 / 每帧耗时 / HBM 峰值带宽；stage 0 的 "MFU" |
| 每帧 kernel 数、GPU 空闲 | 融合与 graph 类任务的直接验收 |
| 常驻显存与各类 pool | 权重、KV、Code2Wav 状态池、三类 graph pool 分别报告 |
| teacher-forced logits cosine | 改变计算顺序的 stage 0 改动用来判断实现是否正确 |
| SNR / mel L1 | 确定性的 Code2Wav 改动，以 fp32 eager `decode_exact` 为参照 |
| CER/WER、说话人相似度、UTMOS、HNR | 有损改动的感知质量，报告均值和最差值 |

## 附录：sortout 之后 vllm-omni 新增、可以直接参考的实现

sortout 的核对基线是 `bc0c9f4b`（2026-09-12），之后 vllm-omni 合入了
[#8163](https://github.com/vllm-project/vllm-omni/pull/8163)（CUDA 上 Qwen3-TTS 的流式吞吐与首包优化）
和 [#8259](https://github.com/vllm-project/vllm-omni/pull/8259)（单阶段 Qwen3-TTS，在 talker 内做流式
codec decode）。下面的文件都在 `vllm_omni/model_executor/models/qwen3_tts/` 下：

| 文件 | 内容 | 对应任务 |
| :-- | :-- | :-- |
| `tokenizer_12hz/streaming_decoder.py` | 每个请求一个 slot 的逐层状态；一次调用为 B 个请求各解 T 帧；im2col + GEMM；加载期线性折叠；`StreamingDecodeGraphs`；bf16 | P0.3、P0.4、P2.3、P2.4 |
| `tokenizer_12hz/time_major_conv.py`、`time_major_stack.py` | time-major 布局的因果卷积 implicit GEMM，SnakeBeta 在读入时完成 | P2.3-C |
| `fused_code_predictor.py` | code predictor 每层少量 kernel；embedding + 投影折叠成表；二分查找 top-k | P1.1、P1.3-B、P1.3-D |
| `cached_code_predictor.py` | 帧内 KV cache，帧与帧之间不保留状态 | P1.3-C |
| `code_predictor_sampling.py` | 由外层 graph 提供 uniform 随机数的 top-k Gumbel 采样 | P1.1 |
| `first_frame_decoder.py` | 首帧在 talker 进程内直接解成 PCM；单帧无上下文时 attention 退化为 V 投影，省掉约 2/3 的 kernel | sortout 之外的 TTFA 优化，可在 P0.4 之后评估 |
| `stream_decode.py` | 单阶段流水线的启用条件 | 与 P3.5 方向相反的选择，见 P3.5 |
