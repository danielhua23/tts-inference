"""A fused dequantize-and-matmul kernel for int8 weight-only quantization.

Weight-only quantization ought to pay off in a talker decode step: batch 1-64
against 1.4B of decoder weights means the step is bound by how fast the weights
can be read, so halving the bytes per weight should halve the time. It only pays
if the dequantization is *fused into the GEMM*. The obvious implementation --

    F.linear(x, weight_int8.to(x.dtype) * scale)

-- reads the int8 weight, writes a full-precision copy of it back to global
memory, and then has the GEMM read that. The step now moves more bytes than the
unquantized model did. It is numerically correct and reliably slower, which
makes it the most popular way to conclude that weight-only quantization does
not work. At the talker's gate_up shape (12288x2048) on an H100: 21 us
unquantized, 163 us for that one-liner.

This kernel keeps the dequantization in registers -- each block converts its
int8 tile of B after loading it, and the dequantized weight is never stored --
and it is still only sometimes faster. Two reasons, neither of them arithmetic:

**Launch overhead.** Launching a Triton kernel from Python costs ~20 us; these
GEMMs run in 5-20 us. Measured eagerly, the kernel loses on every shape by
roughly that constant, and the loss has nothing to do with quantization.
Numbers here are therefore all CUDA-graph numbers, and ``--talker-cuda-graph``
is a precondition for quantization paying off rather than an independent knob.

**Bandwidth efficiency at low occupancy.** cuBLAS reaches ~3.3 TB/s on all of
these shapes, i.e. HBM peak. This kernel reaches that only when the output is
wide enough to fill the machine: with M <= 64 the only parallelism is over N,
so N=12288 (gate_up) gives ~380 blocks and 2.5 TB/s, while N=2048 (down_proj)
gives 64 blocks for 132 SMs and 0.9 TB/s. Halving the bytes at 0.9 TB/s is
slower than reading all of them at 3.3.

The break-even is therefore ``BW_kernel > BW_cublas / 2``, which some shapes in
this model clear and some do not. Measured, M=4, H100, CUDA graph:

    layer     shape        bf16     int8-w8    ratio
    qkv       4096x2048     5.1 us   6.3 us    0.81x
    o_proj    2048x2048     5.1 us   6.2 us    0.82x
    gate_up  12288x2048    19.8 us  10.2 us    1.94x
    down      2048x6144     8.4 us  14.4 us    0.59x

That is why :mod:`qtts.quant.int8` measures each shape at load time instead of
quantizing everything: applied blindly the net effect on the talker is nil,
and applied where it wins it is a 1.3x cut in linear-layer time plus a quarter
off the weight footprint.

Split-K was tried and removed. It raises occupancy for the narrow shapes, but
the partials have to be reduced in a second pass, and the reduction costs more
than the occupancy gains (down_proj: 14.4 us direct, 16.5 us with SPLIT_K=4).

The tile shape below is the best of a sweep over BLOCK_N in {32..256}, BLOCK_K
in {32,64,128}, SPLIT_K in {1,2,4}, warps in {4,8} and stages in {2,3,4}, run
per shape. The same configuration won every shape, so it is a constant rather
than a lookup table.
"""

from __future__ import annotations

import torch

# Imported at module scope, not lazily, because Triton resolves the names in a
# kernel's annotations against the module globals of the function it compiles. A
# kernel defined inside a function that does ``import triton.language as tl``
# locally will compile its body and then fail on ``x: tl.constexpr`` with a bare
# NameError.
try:
    import triton
    import triton.language as tl

    _triton_error: str | None = None
except Exception as exc:  # pragma: no cover - depends on the install
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _triton_error = f"triton unavailable: {exc}"

BLOCK_N = 32
BLOCK_K = 128
NUM_WARPS = 4
NUM_STAGES = 3


def kernel_available() -> tuple[bool, str]:
    return (triton is not None), (_triton_error or "")


if triton is not None:

    @triton.jit
    def _w8_gemm_kernel(
        a_ptr, b_ptr, scale_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_valid = offs_k + k < K
            a = tl.load(a_ptrs, mask=m_mask & k_valid[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask & k_valid[:, None], other=0)
            # The point of the file: the convert happens here, on a tile already
            # in registers. Converting to the activation dtype rather than to
            # f32 keeps the dot on tensor cores, and loses nothing -- int8 holds
            # values in [-127, 127], which bf16 and fp16 represent exactly --
            # ahead of the f32 accumulator.
            acc += tl.dot(a, b.to(a.dtype))
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # The scale is per output channel, so it factors out of the dot product
        # and is applied once here rather than per element of B.
        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc * scale[None, :]
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask & n_mask)


def w8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ (weight * scale).T + bias``, dequantizing ``weight`` in registers.

    ``x`` is ``[..., K]`` floating, ``weight`` is ``[N, K]`` int8, and ``scale``
    is ``[N]``: one per output channel, which is the granularity that lets the
    scale move outside the dot product.
    """
    if triton is None:
        raise RuntimeError(_triton_error or "triton unavailable")

    shape = x.shape
    a = x.reshape(-1, shape[-1])
    if not a.is_contiguous():
        a = a.contiguous()
    rows, k = a.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise ValueError(f"weight is [{n}, {weight.shape[1]}] but x has K={k}")

    out = torch.empty((rows, n), device=a.device, dtype=x.dtype)
    block_m = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    grid = ((rows + block_m - 1) // block_m, (n + BLOCK_N - 1) // BLOCK_N)
    _w8_gemm_kernel[grid](
        a, weight, scale, bias if bias is not None else a, out,
        rows, n, k,
        a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out.reshape(*shape[:-1], n)
