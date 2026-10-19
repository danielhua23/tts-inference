"""Code2Wav: 16 codebooks per frame at 12.5 Hz to a 24 kHz waveform.

Stage 1 of the pipeline. Unlike the talker this is not autoregressive -- it is a
deterministic function of the codes -- but it is *causal and stateful*, and that
is what makes streaming it interesting.

The chain, with the tensor width at each step:

    codes        [B, 16, F]  int64
    quantizer    [B, 512, F]      sum of 16 codebook lookups, projected
    pre_conv     [B, 1024, F]     causal k=3
    pre_transformer  [B, 1024, F] 8 layers, width 512, sliding window 72
    upsample     [B, 1024, 4F]    2x (ConvTranspose k2s2 + ConvNeXt)
    decoder      [B, 1, 1920F]    4 blocks upsampling 8/5/4/3
    clamp(-1, 1)

Total upsample is 2*2*8*5*4*3 = 1920, so one frame is 1920 samples and 12.5
frames per second gives 24 kHz.

Submodule names deliberately mirror the checkpoint's tensor paths
(``decoder.pre_transformer.layers.0...``), so loading is a ``load_state_dict``
rather than a mapping table: a structural mistake then shows up as a missing or
unexpected key instead of as quiet noise in the audio.

Three decode entry points, which is the point of this file:

* :meth:`Qwen3TTSCode2Wav.decode_exact` -- whole utterance at once. The
  reference for everything else.
* :meth:`Qwen3TTSCode2Wav.decode_window` -- incremental and exact, via a
  per-layer K/V cache plus two small convolution caches.
* :meth:`Qwen3TTSCode2Wav.chunked_decode` -- incremental and *approximate*.
  Keeps no state, re-reading a fixed ``left_context`` before each chunk.

The gap between the last two is larger than it looks, and measuring it was
worth the trouble. ``pre_transformer`` has ``sliding_window = 72``, so the
obvious reading is that 71 frames of left context make a chunk exact. It does
not: the window bounds *each layer*, and 8 layers compose into a reach of
``8 * 71 = 568`` frames, about 45 seconds. So ``chunked_decode`` with the usual
``left_context = 25`` is not slightly approximate, it is very approximate, and
feeding it more context converges only slowly. Measured against
``decode_exact`` on 300 random frames, peak sample error runs 1.0 at
``left_context = 25`` and is still 0.09 at 120, on a signal bounded by 1.0.

That also settles why :meth:`decode_window` caches K/V rather than retaining
raw codes and recomputing. Recomputing exactly would mean re-running the
8-layer transformer over ~570 frames to emit 25 -- a 23x overhead -- whereas
each layer's own cache needs only its own 71 positions. The reference
implementation makes the same choice for the same reason.

So ``chunked_decode`` looks like a performance knob and is really a quality
knob. Benchmarks that compare the modes should report audio error next to
latency; ``tests/test_code2wav_modes.py`` does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.backends.torch_sdpa import build_mask
from ..attention.layer import attention
from ..layers.activation import GatedMLP, SnakeBeta
from ..layers.conv import CausalConv1d, CausalConvTranspose1d, ConvNeXtBlock
from ..layers.linear import LinearBase
from ..layers.norm import LayerScale, RMSNorm
from ..layers.rope import RotaryEmbedding, apply_rotary_pos_emb
from ..utils import get_logger
from .hf_config import Code2WavHFConfig

logger = get_logger("code2wav")

#: Left context ``pre_conv`` (causal, kernel 3) needs to produce a correct
#: output frame.
_CONV_CONTEXT_FRAMES = 2

#: Latent frames the upsample + decoder stack needs ahead of a frame before its
#: samples are exact. Inherited from the reference implementation, where it
#: covers the widest causal receptive field in the stack (the dilation-9
#: kernel-7 residual convolutions, at each upsampled rate).
_DOWNSTREAM_CONTEXT_FRAMES = 12


class EuclideanCodebook(nn.Module):
    """A codebook stored as EMA accumulators rather than as vectors.

    Training keeps a running sum per entry and a usage count; the usable
    codebook is the quotient. Dividing on every lookup would be wasteful when
    the weights never change after loading, so :meth:`precompute` does it once
    and registers the result.
    """

    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim), requires_grad=False)
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size), requires_grad=False)
        self.register_buffer("embedding", torch.zeros(codebook_size, dim), persistent=False)

    @torch.no_grad()
    def precompute(self) -> None:
        self.embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes, self.embedding)


class VectorQuantization(nn.Module):
    """One codebook. ``project_out`` is Identity for this checkpoint."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self._codebook = EuclideanCodebook(dim, codebook_size)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # [B, F, dim] -> [B, dim, F]
        return self._codebook(codes).transpose(1, 2)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, num_quantizers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(VectorQuantization(dim, codebook_size) for _ in range(num_quantizers))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` is ``[Q, B, F]``; the residuals sum."""
        out = self.layers[0](codes[0])
        for index in range(1, codes.shape[0]):
            out = out + self.layers[index](codes[index])
        return out


class ResidualVectorQuantizer(nn.Module):
    """A group of codebooks plus the projection back to the model width.

    ``input_proj`` exists in the checkpoint but only the *encoder* direction
    uses it, so it is not built here. Loading skips the tensor explicitly
    rather than by accident.
    """

    def __init__(self, dim: int, output_dim: int, num_quantizers: int, codebook_size: int) -> None:
        super().__init__()
        self.vq = ResidualVectorQuantization(dim, codebook_size, num_quantizers)
        self.output_proj = nn.Conv1d(dim, output_dim, 1, bias=False)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` is ``[B, Q, F]``."""
        return self.output_proj(self.vq(codes.transpose(0, 1)))


class SplitResidualVectorQuantizer(nn.Module):
    """Codebook 0 is semantic, codebooks 1..15 are acoustic residuals.

    They are split because they are projected separately, not because they are
    combined differently: the two groups' outputs are summed.
    """

    def __init__(self, config: Code2WavHFConfig) -> None:
        super().__init__()
        dim = config.codebook_dim // 2
        self.n_q_semantic = config.num_semantic_quantizers
        self.rvq_first = ResidualVectorQuantizer(
            dim, config.codebook_dim, self.n_q_semantic, config.codebook_size
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dim, config.codebook_dim, config.num_quantizers - self.n_q_semantic, config.codebook_size
        )

    def precompute(self) -> None:
        for module in self.modules():
            if isinstance(module, EuclideanCodebook):
                module.precompute()

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` is ``[B, Q, F]`` -> ``[B, codebook_dim, F]``."""
        quantized = self.rvq_first(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest(codes[:, self.n_q_semantic :])
        return quantized


class PreTransformerAttention(nn.Module):
    """Sliding-window self-attention. No q/k norm, unlike the talker."""

    def __init__(self, config: Code2WavHFConfig, backend: str) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.backend = backend
        hidden = config.hidden_size
        self.q_proj = LinearBase(hidden, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = LinearBase(hidden, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = LinearBase(hidden, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = LinearBase(self.num_heads * self.head_dim, hidden, bias=config.attention_bias)

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, S, hidden]`` to q, k, v shaped ``[B, S, heads, head_dim]``.

        Kept separate from :meth:`forward` so the incremental path can stash k
        and v *before* the rotary embedding is applied. That is what lets the
        cache be re-labelled: a stored post-rotary key is pinned to the
        absolute position it was computed at, whereas a pre-rotary one can be
        re-embedded at whatever position the current window assigns it.
        """
        bsz, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        return query, key, value

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        cos_k: torch.Tensor,
        sin_k: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate, attend, project out. Inputs are ``[B, S, heads, head_dim]``."""
        bsz, q_len = query.shape[0], query.shape[1]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = apply_rotary_pos_emb(query, query, cos_q, sin_q)[0]
        key = apply_rotary_pos_emb(key, key, cos_k, sin_k)[0]
        out = attention(query, key, value, scale=self.scale, backend=self.backend, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))

    def forward(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = self.project(hidden)
        return self.attend(query, key, value, cos, sin, cos, sin, attn_mask)


class PreTransformerLayer(nn.Module):
    """Pre-norm block with a LayerScale on each residual branch.

    The LayerScale initialises near zero, so at the start of training the block
    is close to the identity. It stays in the architecture, which is why the
    branch outputs are scaled before being added.
    """

    def __init__(self, config: Code2WavHFConfig, backend: str, norm_backend: str) -> None:
        super().__init__()
        self.self_attn = PreTransformerAttention(config, backend)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size, fuse=False)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, backend=norm_backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, backend=norm_backend)
        self.self_attn_layer_scale = LayerScale(config.hidden_size)
        self.mlp_layer_scale = LayerScale(config.hidden_size)

    def forward(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        attn = self.self_attn(self.input_layernorm(hidden), cos, sin, attn_mask)
        return self._finish(hidden, attn)

    def forward_incremental(
        self,
        hidden: torch.Tensor,
        cache: PreTransformerCache,
        layer_idx: int,
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        cos_k: torch.Tensor,
        sin_k: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self.self_attn.project(self.input_layernorm(hidden))
        key, value = cache.extend(layer_idx, key, value)
        attn = self.self_attn.attend(query, key, value, cos_q, sin_q, cos_k, sin_k, attn_mask)
        return self._finish(hidden, attn)

    def _finish(self, hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn_layer_scale(attn)
        mlp = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + self.mlp_layer_scale(mlp)


class PreTransformerCache:
    """Per-layer sliding-window K/V for one request.

    Holds ``sliding_window - 1`` positions per layer, which is exactly the
    history a new frame's attention can reach *within a layer*. Stacked over 8
    layers the model's reach is 568 frames, but it never needs 568 frames of
    input: each layer's cache already summarises its own history. That is the
    whole reason this class exists instead of simply retaining raw codec frames
    and recomputing -- recomputing exactly would mean re-running the 8-layer
    transformer over ~570 frames for every 25-frame chunk.

    Keys are stored *before* the rotary embedding, so each call re-labels the
    window from zero and relative offsets stay correct forever. There is no
    position-table ceiling and no wraparound to get wrong.
    """

    __slots__ = ("k", "v", "length", "capacity")

    def __init__(
        self,
        num_layers: int,
        capacity: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_layers, capacity, kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0
        self.capacity = capacity

    def extend(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend the cached history to ``key``/``value`` and retain the tail.

        ``key``/``value`` are ``[1, N, kv_heads, head_dim]``. Returns the full
        window to attend over. Truncation happens on the way in, so the stored
        window is already the right size for the next call.
        """
        history = self.length
        cached_k = self.k[layer_idx, :history].unsqueeze(0)
        cached_v = self.v[layer_idx, :history].unsqueeze(0)
        full_k = torch.cat([cached_k, key], dim=1)
        full_v = torch.cat([cached_v, value], dim=1)

        keep = min(self.capacity, full_k.shape[1])
        self.k[layer_idx, :keep].copy_(full_k[0, -keep:])
        self.v[layer_idx, :keep].copy_(full_v[0, -keep:])
        # Every layer sees the same number of positions, so recording the
        # length once per chunk (on the last layer) keeps them in lockstep.
        if layer_idx == self.k.shape[0] - 1:
            self.length = keep
        return full_k, full_v

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


class PreTransformer(nn.Module):
    """Sequence modelling over codec frames, at width 512 inside 1024 I/O.

    Positions restart at zero on every call. That is safe because the only
    position-dependent pieces are the rotary embedding, which depends on the
    *difference* between query and key positions, and the sliding window, which
    is likewise relative.

    Note that "sliding window 72" bounds each layer, not the model: with 8
    layers the output at one frame depends on 8 * 71 = 568 earlier frames.
    Anything that reasons about how much history streaming needs has to use the
    stacked figure, not the per-layer one.
    """

    def __init__(self, config: Code2WavHFConfig, backend: str, norm_backend: str) -> None:
        super().__init__()
        self.sliding_window = config.sliding_window
        self.input_proj = LinearBase(config.latent_dim, config.hidden_size, bias=True)
        self.layers = nn.ModuleList(
            PreTransformerLayer(config, backend, norm_backend) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, backend=norm_backend)
        self.output_proj = LinearBase(config.hidden_size, config.latent_dim, bias=True)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` is ``[B, F, latent_dim]``."""
        seq_len = hidden.shape[1]
        if seq_len > self.rotary.max_position:
            raise ValueError(
                f"{seq_len} frames exceeds the codec decoder's {self.rotary.max_position}-position RoPE table "
                f"({seq_len / 12.5:.0f}s of audio). Decode in chunks: decode_window and chunked_decode both "
                "restart positions per call and have no such limit."
            )
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        cos, sin = self.rotary(positions, dtype=hidden.dtype)
        mask = build_mask(
            query_positions=positions,
            key_positions=positions,
            key_valid=None,
            sliding_window=self.sliding_window,
            causal=True,
        )
        hidden = self.input_proj(hidden)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask)
        return self.output_proj(self.norm(hidden))

    def new_cache(self, dtype: torch.dtype, device: torch.device) -> PreTransformerCache:
        attn = self.layers[0].self_attn
        return PreTransformerCache(
            len(self.layers), self.sliding_window - 1, attn.num_kv_heads, attn.head_dim, dtype, device
        )

    def forward_incremental(self, hidden: torch.Tensor, cache: PreTransformerCache) -> torch.Tensor:
        """Extend a request by ``hidden``'s frames, reusing ``cache``.

        ``hidden`` is ``[1, N, latent_dim]`` covering only the new frames. The
        result equals :meth:`forward` over the whole sequence, restricted to
        those frames, up to floating-point reassociation.
        """
        history = cache.length
        new = hidden.shape[1]
        device = hidden.device
        key_positions = torch.arange(history + new, device=device).unsqueeze(0)
        query_positions = key_positions[:, history:]
        cos_k, sin_k = self.rotary(key_positions, dtype=hidden.dtype)
        cos_q, sin_q = cos_k[:, history:], sin_k[:, history:]
        mask = build_mask(
            query_positions=query_positions,
            key_positions=key_positions,
            key_valid=None,
            sliding_window=self.sliding_window,
            causal=True,
        )
        hidden = self.input_proj(hidden)
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.forward_incremental(
                hidden, cache, layer_idx, cos_q, sin_q, cos_k, sin_k, mask
            )
        return self.output_proj(self.norm(hidden))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, snake_backend: str) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim, backend=snake_backend)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim, backend=snake_backend)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecoderBlock(nn.Module):
    """SnakeBeta, transposed convolution, then three dilated residual units.

    Held as a numbered ``block`` list rather than named attributes so the
    parameter paths match the checkpoint's ``decoder.1.block.2.conv1...``.
    """

    def __init__(self, config: Code2WavHFConfig, layer_idx: int, snake_backend: str) -> None:
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        rate = config.upsample_rates[layer_idx]
        block: list[nn.Module] = [
            SnakeBeta(in_dim, backend=snake_backend),
            CausalConvTranspose1d(in_dim, out_dim, kernel_size=2 * rate, stride=rate),
        ]
        block += [ResidualUnit(out_dim, dilation, snake_backend) for dilation in (1, 3, 9)]
        self.block = nn.ModuleList(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.block:
            x = module(x)
        return x


class Code2WavRequestState:
    """Everything one streaming request must carry between chunks.

    Three caches, one per causal dependency in the chain, each the minimum the
    stage actually reaches back over:

    * ``quantized`` -- 2 frames, for ``pre_conv``'s kernel of 3.
    * ``kv`` -- ``sliding_window - 1`` positions per transformer layer.
    * ``latent`` -- 12 frames, for the upsample and decoder stack.
    """

    __slots__ = ("quantized", "kv", "latent", "frames_emitted")

    def __init__(self, kv: PreTransformerCache) -> None:
        self.quantized: torch.Tensor | None = None
        self.kv = kv
        self.latent: torch.Tensor | None = None
        self.frames_emitted = 0

    def nbytes(self) -> int:
        total = self.kv.nbytes()
        for tensor in (self.quantized, self.latent):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


class Qwen3TTSCode2Wav(nn.Module):
    def __init__(
        self,
        config: Code2WavHFConfig,
        *,
        attention_backend: str = "auto",
        snake_backend: str = "triton",
        norm_backend: str = "fused",
    ) -> None:
        super().__init__()
        self.config = config
        self.total_upsample = config.total_upsample
        self.sample_rate = config.output_sample_rate

        self.quantizer = SplitResidualVectorQuantizer(config)
        self.pre_conv = CausalConv1d(config.codebook_dim, config.latent_dim, kernel_size=3)
        self.pre_transformer = PreTransformer(config, attention_backend, norm_backend)
        self.upsample = nn.ModuleList(
            nn.ModuleList(
                [
                    CausalConvTranspose1d(config.latent_dim, config.latent_dim, kernel_size=r, stride=r),
                    ConvNeXtBlock(config.latent_dim),
                ]
            )
            for r in config.upsampling_ratios
        )
        decoder: list[nn.Module] = [CausalConv1d(config.latent_dim, config.decoder_dim, kernel_size=7)]
        decoder += [DecoderBlock(config, i, snake_backend) for i in range(len(config.upsample_rates))]
        out_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(out_dim, backend=snake_backend),
            CausalConv1d(out_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)

        #: How far back a frame's samples actually depend on earlier codes.
        #: ``sliding_window`` bounds one layer; the 8 layers compose, so the
        #: true reach is ``8 * 71 = 568`` frames -- 45 seconds of audio. This
        #: is what makes :meth:`chunked_decode` approximate and
        #: :meth:`decode_window` worth its cache.
        self.receptive_field_frames = (
            config.num_hidden_layers * (config.sliding_window - 1)
            + _CONV_CONTEXT_FRAMES
            + _DOWNSTREAM_CONTEXT_FRAMES
        )

    @property
    def device(self) -> torch.device:
        return self.pre_conv.conv.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pre_conv.conv.weight.dtype

    def precompute_caches(self) -> None:
        """Fold load-time-constant arithmetic into buffers. Call after loading."""
        self.quantizer.precompute()
        for module in self.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_cache()

    # ---------------------------------------------------------------- decoding

    def _latent(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, Q, F]`` codes to ``[B, latent_dim, F]`` latents."""
        quantized = self.quantizer(codes)
        return self._latent_from_quantized(quantized)

    def _latent_from_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        hidden = self.pre_conv(quantized).transpose(1, 2)
        return self.pre_transformer(hidden).transpose(1, 2)

    def _synthesize(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, latent_dim, F]`` latents to ``[B, 1, 1920F]`` samples."""
        hidden = latent
        for transpose_conv, convnext in self.upsample:
            hidden = convnext(transpose_conv(hidden))
        for module in self.decoder:
            hidden = module(hidden)
        return hidden.clamp(-1.0, 1.0)

    @torch.inference_mode()
    def decode_exact(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode a whole utterance. ``[B, Q, F]`` -> ``[B, 1920F]``."""
        codes = self._validate(codes)
        return self._synthesize(self._latent(codes)).squeeze(1)

    @torch.inference_mode()
    def decode_window(self, new_codes: torch.Tensor, state: Code2WavRequestState) -> torch.Tensor:
        """Decode the next chunk exactly, advancing ``state``.

        ``new_codes`` is ``[1, Q, F_new]``; returns ``F_new * 1920`` samples
        that match :meth:`decode_exact` over the same frames.
        """
        new_codes = self._validate(new_codes)
        if new_codes.shape[0] != 1:
            raise ValueError("decode_window is per-request; use chunked_decode to batch")
        new_frames = int(new_codes.shape[-1])

        quantized = self.quantizer(new_codes)
        conv_context = 0
        if state.quantized is not None:
            conv_context = int(state.quantized.shape[-1])
            quantized = torch.cat([state.quantized, quantized], dim=-1)
        state.quantized = quantized[..., -_CONV_CONTEXT_FRAMES:].contiguous()

        # pre_conv left-pads with zeros, which is right only at the true start
        # of the request. With context available the first `conv_context`
        # outputs are the padded ones and get dropped.
        hidden = self.pre_conv(quantized)[..., conv_context:].transpose(1, 2)
        latent = self.pre_transformer.forward_incremental(hidden, state.kv).transpose(1, 2)

        lead_in = 0
        if state.latent is not None:
            lead_in = int(state.latent.shape[-1])
            latent = torch.cat([state.latent, latent], dim=-1)
        state.latent = latent[..., -_DOWNSTREAM_CONTEXT_FRAMES:].contiguous()

        audio = self._synthesize(latent)
        if lead_in:
            audio = audio[..., lead_in * self.total_upsample :]
        state.frames_emitted += new_frames
        return audio.reshape(-1)

    @torch.inference_mode()
    def chunked_decode(
        self, codes: torch.Tensor, chunk_frames: int, left_context_frames: int
    ) -> torch.Tensor:
        """Decode in chunks holding no state, re-reading ``left_context_frames``.

        Approximate unless ``left_context_frames`` reaches
        :attr:`receptive_field_frames` (568 for this checkpoint), which no
        practical setting does. See the module docstring for measurements.
        """
        codes = self._validate(codes)
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
        total = int(codes.shape[-1])
        pieces = []
        for start in range(0, total, chunk_frames):
            end = min(start + chunk_frames, total)
            context = min(left_context_frames, start)
            window = codes[..., start - context : end]
            audio = self._synthesize(self._latent(window))
            if context:
                audio = audio[..., context * self.total_upsample :]
            pieces.append(audio)
        return torch.cat(pieces, dim=-1).squeeze(1)

    def new_state(self) -> Code2WavRequestState:
        return Code2WavRequestState(self.pre_transformer.new_cache(self.dtype, self.device))

    def state_bytes_estimate(self) -> int:
        """Bytes one request's :class:`Code2WavRequestState` occupies once warm.

        Used by the scheduler to size how many streaming requests fit; the
        transformer K/V dominates, the two convolution caches are rounding.
        """
        cfg = self.config
        element = torch.finfo(self.dtype).bits // 8
        kv = 2 * cfg.num_hidden_layers * (cfg.sliding_window - 1) * cfg.num_key_value_heads * cfg.head_dim
        conv = cfg.codebook_dim * _CONV_CONTEXT_FRAMES
        latent = cfg.latent_dim * _DOWNSTREAM_CONTEXT_FRAMES
        return (kv + conv + latent) * element

    def _validate(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(f"expected codes shaped [B, Q, F], got {tuple(codes.shape)}")
        expected = self.config.num_quantizers
        if codes.shape[1] != expected:
            raise ValueError(f"expected {expected} codebooks, got {codes.shape[1]}")
        if codes.shape[2] == 0:
            raise ValueError("no frames to decode")
        return codes.to(device=self.device, dtype=torch.long)
