# 测量结果

英文版：[PERFORMANCE.en.md](PERFORMANCE.en.md)

所有数字都来自 `danie_torch_2504` 容器里的一张 H100 80GB（torch 2.7.0a0+nv25.04、
triton 3.2.0、CUDA 12.9），checkpoint 为 `Qwen3-TTS-12Hz-1.7B-CustomVoice`，
speaker `vivian`，文本是 `RunConfig.text` 里那句 26 字的中文，生成出来约 59 帧、约
4.7 秒音频。

复现某张表：

```bash
python examples/benchmark/sweep.py --model $MODEL --suite optimizations   # 另有 concurrency、chunking、scheduling
```

每一行都是一个独立进程：丢弃一次 warmup 后测三轮，带 `--no-collect-stage-timings`，
不设固定 seed。采样是随机的，所以各行之间音频长度会差几个百分点；RTF 把它除掉了，
TTFA 本身不依赖它。

不设 seed 是故意的。捕获好的 CUDA graph 无法从调用方传入的 generator 取随机数，于是
带 seed 的请求会退回 eager code predictor，同一份配置测出来是 0.656 RTF 而不是
0.162——一个为了让数字可复现而存在的开关，把它要复现的数字翻了四倍。两者同时打开时
engine 会告警。

**怎么读这些列。** RTF 是 wall clock 除以音频时长，并在并发流之间聚合——低于 1.0 表
示比实时更快。TTFA 是 time-to-first-audio。"最差 gap"是相邻 chunk 之间最长的那个间
隔；而 underrun 指的是某个 gap 超过了它之前已交付的音频长度，也就是听众真的会听到的
那种断音。

## 一帧的时间去了哪里

一帧音频 = 一次 talker step + 十五次 code predictor prefill。下面的拆分是直接量出来
的（`TalkerRunner.decode_step` 与单独的 talker forward 对比），batch 1 和 batch 4，
单位是每帧毫秒：

| | step | talker forward | code predictor |
| --- | --- | --- | --- |
| batch 1, eager norm, 无 graph | 66.97 | 18.51 | 48.47 |
| batch 1, fused norm, 无 graph | 64.98 | 17.56 | 47.42 |
| batch 1, eager norm, 有 graph | 17.57 | 5.74 | 11.83 |
| batch 1, fused norm, 有 graph | **11.85** | 4.16 | 7.69 |
| batch 4, fused norm, 无 graph | 67.85 | 18.76 | 49.08 |
| batch 4, fused norm, 有 graph | **13.16** | 4.93 | 8.23 |

这张表里有三件事值得带走。

**一帧里的大头是 code predictor，不是 talker。** 它是个 5 层、宽 1024 的模型——占参数
量的 6%——却花掉 28 层 talker 的 2.6 倍时间，因为它每帧要跑十五次，而且没有 K/V
cache 可以复用。先去优化 talker 是最自然的选择，也是错的那个。

**两半都是 launch-bound，不是 compute-bound。** 把 step 捕获进 CUDA graph 能让它从
67.0 ms 降到 11.9 ms，5.6 倍，而算术一点没变。这一页上其余所有东西相比它都很小，而
且有几项优化是在它之后才显形的。

**batch 4 的代价和 batch 1 差不多。** 13.16 ms 对 11.85 ms，换来四倍的音频：每条流每
帧 3.3 ms。在这个规模下 stage 0 的 decode 完全受带宽约束，所以在 batch 大到让 GEMM
变成 compute-bound 之前，并发几乎是免费的。

## 优化项，端到端

`--suite optimizations`，单流：

| 配置 | RTF | TTFA p50 | 最差 gap | underrun | 峰值 MiB |
| --- | --- | --- | --- | --- | --- |
| baseline（eager norm、eager snake） | 0.849 | 0.357 | 1.670 | 3 | 6365 |
| + fused RMSNorm | 0.834 | 0.349 | 1.639 | 3 | 6365 |
| + fused SnakeBeta | 0.834 | 0.352 | 1.639 | 3 | 6365 |
| + CUDA graph | **0.162** | **0.082** | 0.312 | 0 | 6461 |
| + int8 weight-only，经校准 | 0.159 | 0.083 | 0.297 | 0 | 6046 |
| + int8 weight-only，全量 | 0.167 | 0.086 | 0.334 | 0 | 5040 |

baseline 在单流下每句话就 underrun 三次：每帧 67 ms 对每帧 80 ms 的音频，留给一次
decoder 调用的余量是零，于是第一个 chunk 边界一旦晚到就一直晚。CUDA graph 消掉了这些
underrun，并把 TTFA 砍到 1/4.3。

最后两行是 per-shape calibration 的论据。只量化那些 int8 kernel 实测更快的 shape，得
到 0.159 RTF 并省回 415 MiB；把所有符合条件的 shape 都量化，能省回 1.4 GiB，但*比不
量化还慢*（0.167 对 0.162）。哪个对，取决于你缺的是内存还是时间——但被当成"weight-only
量化没用"报出去的，恰恰是全量那一行，而它说的其实是四个特定矩阵 shape 的事，不是量化
本身的事。

### 为什么这里的 fusion 看起来收益很小

fused RMSNorm 在上面这张表里端到端只值 2%，在一个 graph 重放的 step 里值 32%
（17.57 → 11.85 ms）。两个数都是真的：没有 graph 时 step 在等 kernel launch，而把一次
normalization 的六个 kernel 去掉五个，去掉的是本来就与其他 launch 重叠着的 launch。
在没解决 launch overhead 之前去测一个 fusion，你会得出它没用的结论。

### int8 kernel，逐 shape

`python examples/kernels/tune_w8_gemm.py`，M=4、bf16、在 CUDA graph 内计时：

| layer | shape | cuBLAS bf16 | fused int8 | 朴素反量化 | 比值 |
| --- | --- | --- | --- | --- | --- |
| qkv | 4096x2048 | 5.2 µs | 5.8 µs | 49.7 µs | 0.90x |
| o_proj | 2048x2048 | 5.1 µs | 5.6 µs | 25.3 µs | 0.90x |
| gate_up | 12288x2048 | 19.3 µs | 9.9 µs | 157.9 µs | **1.94x** |
| down_proj | 2048x6144 | 8.4 µs | 13.1 µs | 78.6 µs | 0.64x |

"朴素反量化"那一列是 `F.linear(x, w_int8.to(dtype) * scale)`——写 weight-only 量化最自
然的方式，比不量化还*慢* 2.5 到 8 倍，因为它把权重的一份全精度拷贝写进 global memory，
再让 GEMM 读回来。fused kernel 只在输出宽到能填满整台机器时才赢：M=4 时唯一的并行维度
是 N，所以 N=12288 能跑到 2.5 TB/s，而 N=2048 只有 0.9 TB/s，对面 cuBLAS 是 3.3。详细
推理在 `qtts/quant/kernels.py` 里。

## 并发

`--suite concurrency`，开 CUDA graph，`max_num_seqs=8`：

| 流数 | RTF（聚合） | TTFA p50 | TTFA p95 | 最差 gap | underrun | 峰值 MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.165 | 0.082 | 0.082 | 0.301 | 0 | 5565 |
| 2 | 0.102 | 0.111 | 0.120 | 0.448 | 0 | 5629 |
| 4 | 0.060 | 0.165 | 0.191 | 0.486 | 0 | 5675 |
| 8 | 0.039 | 0.276 | 0.338 | 0.806 | 0 | 5726 |

从 1 流到 8 流，吞吐涨 4.2 倍，代价是 TTFA 涨 3.4 倍加 161 MiB。内存基本不变，是因为全
部 16 个 slot 的 K/V cache 在启动时就一次性分配好了（`max_num_seqs` × `max_model_len`），
所以并发在运行期不额外花内存；涨的是 per-request 的 Code2Wav 状态，每个几 MB。

TTFA 变差是因为 stateful 模式下 stage 1 是串行的：八个首块排在同一个 decoder 后面。要管
的是这个队列，不是 talker——这正是 `--chunk-policy` 和
`--code2wav-dynamic-initial-chunk` 存在的意义。

## chunking

`--suite chunking`，单流，开 CUDA graph：

| 配置 | RTF | TTFA p50 | 最差 gap | 峰值 MiB |
| --- | --- | --- | --- | --- |
| `--pipeline-mode sync` | 0.158 | 0.911 | 0.000 | 6645 |
| 首块 5 帧 | 0.161 | 0.083 | 0.302 | 6461 |
| 首块 25 帧 | 0.160 | 0.319 | 0.325 | 6462 |
| ramp 5,10,25 | 0.161 | 0.083 | 0.323 | 6461 |
| stateless decoder | 0.161 | 0.082 | 0.302 | 6521 |

chunking 在吞吐上几乎免费（RTF 0.158 到 0.161，2%），在 TTFA 上值 11 倍。首块大小是唯一
的杠杆：5 帧给出 83 ms，25 帧给出 319 ms，差的就是听众多等的那 20 帧音频。`sync` 完全没
有 gap，因为它只交付一次——它是正确的 baseline，同时也无法用于流式。

`stateless` 这一行看着像免费的收益，其实不是：它在这里每一列都和 stateful 打平，但改变了
音频——因为 decoder 真实的感受野是约 568 帧，而这个模式只重读 25 帧。取自
`tests/test_code2wav_modes.py`，在幅度被限制在 1.0 的信号上，与整句解码相比的峰值采样点
误差：

| left context | 25 | 71 | 120 | stateful（任意 chunk 大小） |
| --- | --- | --- | --- | --- |
| 峰值误差 | ~0.9 | 更低 | 更低 | ~1e-8（fp32 的余量） |

它换来的是把不同请求的 chunk batch 进同一次 decoder 调用的能力，而这是让 stage 1 不再串
行的唯一办法。在负载高的 server 上这是一个实打实的吞吐论据；同时它是一次质量取舍，不是
加速。

## 调度策略

`--suite scheduling`，8 条相同的流：

| policy | RTF | TTFA p50 | TTFA p95 | 最差 gap |
| --- | --- | --- | --- | --- |
| fifo | 0.038 | 0.275 | 0.337 | 0.484 |
| deadline | 0.037 | 0.276 | 0.338 | 0.605 |
| round_robin | 0.037 | 0.275 | 0.342 | 0.518 |

无法区分，而对这个负载来说这是预期结果，不是 bug：同一时刻提交的八份相同句子会锁步到达各
自的 chunk 边界，于是每种 policy 看到的都是一队等价的 chunk，任何顺序都一样好。只有请求之
间存在差异时，policy 才可能有收益——混合的文本长度、错峰到达、某条流的消费者很慢。把那种
负载做成可测量的，是这里接下来最该做的事；在那之前，`deadline` 的优势是一个关于本套件不会
产生的负载的断言，所以默认是 `fifo`。

## 没有测过的东西

- 超过 8 条并发流，或 `max_model_len` 超过 1024 帧的情况。
- 这个模型上的 fp8 W8A8（`--quantization fp8`）。路径写好了也能跑；这里没有一行用它，是因
  为适合受带宽约束的 decode 循环的是 weight-only，而 fp8 的对比要在 compute-bound 的 batch
  size 下才有意思。
- bf16 的 Code2Wav（`--code2wav-dtype bfloat16`）。这个 decoder 每帧要在 fp32 下跑约 20 级
  串行卷积；加速容易测，而累积的舍入误差需要比波形，不是看一个 RTF 数字。
