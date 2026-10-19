"""The residual code predictor: autoregression along the codebook axis.

The talker is autoregressive in time and produces only codebook 0 of each RVQ
frame. The remaining Q-1 codebooks are residuals that refine that frame's
acoustics, and they are produced here, autoregressively along the codebook
axis within a single frame:

    time frame t:  talker -> code[t, 0]
                   this module -> code[t, 1], code[t, 2], ... code[t, Q-1]

Splitting the two axes is what keeps the cost down. If the talker unrolled both
axes into one sequence it would run ``frame_rate * Q`` large-model decode steps
per second of audio; instead it runs ``frame_rate`` of them and a 5-layer,
1024-wide model fills in the rest.

The sequence this model sees is only Q+1 long and it is rebuilt from scratch
for every codebook, so there is nothing to cache: a KV cache for a 16-token
sequence costs more bookkeeping than the attention it saves. What does matter
at that shape is launch count, which is why q/k/v and gate/up are fused and the
whole prefix forward is offered to ``torch.compile``.

Position 0 of the sequence is the talker's hidden state, position 1 is the
embedding of codebook 0, and position ``s+1`` is the embedding of codebook
``s``. All of them are projected from the talker's 2048-wide space into this
model's 1024-wide one by the same ``small_to_mtp_projection``; the residual
embedding tables live in talker-width space because the talker also sums them
into its own next-step input.
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention import attention
from ..config import TalkerConfig
from ..layers import GatedMLP, LinearBase, MergedLinear, RMSNorm, RotaryEmbedding, apply_rotary_pos_emb
from ..utils import get_logger
from .hf_config import CodePredictorHFConfig

logger = get_logger("code_predictor")


def sample_gumbel(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """``argmax(logits + Gumbel)``, distributionally equal to sampling softmax.

    Cheaper than softmax-then-multinomial at the ``[B, 2048]`` shapes here, and
    well defined for rows where top-k has left a single finite entry, where
    ``multinomial`` on an all-zero probability row would raise instead.
    """
    uniform = torch.rand(logits.shape, device=logits.device, dtype=torch.float32, generator=generator)
    # Both clamps guard a log against an argument that underflows to zero, and
    # both have to apply to the value actually being logged. Clamping
    # ``torch.log(uniform)`` instead of its negation looks equivalent and is
    # not: the log of a probability is always negative, so the clamp would pin
    # every element to +1e-20, and the outer log of the negated result is NaN
    # for the whole tensor. argmax over all-NaN then returns index 0 on every
    # row, which is a valid-looking token id and silently disallowed.
    gumbel = -torch.log((-torch.log(uniform.clamp_min(1e-20))).clamp_min(1e-20))
    return (logits.float() + gumbel).argmax(dim=-1, keepdim=True)


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


class PredictorAttention(nn.Module):
    def __init__(
        self,
        config: CodePredictorHFConfig,
        dtype: torch.dtype,
        fuse_qkv: bool,
        backend: str,
        norm_backend: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.fuse_qkv = fuse_qkv
        self.backend = backend

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        if fuse_qkv:
            self.qkv_proj = MergedLinear(config.hidden_size, [q_size, kv_size, kv_size], bias=False, dtype=dtype)
        else:
            self.q_proj = LinearBase(config.hidden_size, q_size, bias=False, dtype=dtype)
            self.k_proj = LinearBase(config.hidden_size, kv_size, bias=False, dtype=dtype)
            self.v_proj = LinearBase(config.hidden_size, kv_size, bias=False, dtype=dtype)
        self.o_proj = LinearBase(q_size, config.hidden_size, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, backend=norm_backend)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, backend=norm_backend)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        if self.fuse_qkv:
            q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        else:
            q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
        q = self.q_norm(q.view(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch, seq, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = attention(q, k, v, scale=self.scaling, backend=self.backend, causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, -1))


class PredictorLayer(nn.Module):
    def __init__(self, config: CodePredictorHFConfig, dtype: torch.dtype, runtime: TalkerConfig) -> None:
        super().__init__()
        self.self_attn = PredictorAttention(
            config, dtype, runtime.fuse_qkv, runtime.attention_backend, runtime.rmsnorm_backend
        )
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size, fuse=runtime.fuse_gate_up, dtype=dtype)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, backend=runtime.rmsnorm_backend
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, backend=runtime.rmsnorm_backend
        )

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class CodePredictor(nn.Module):
    def __init__(
        self,
        config: CodePredictorHFConfig,
        talker_hidden_size: int,
        *,
        runtime: TalkerConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime = runtime
        self.dtype = dtype
        self.device = device
        self.num_groups = config.num_code_groups
        self.num_residual = self.num_groups - 1
        if self.num_residual < 1:
            raise ValueError(f"num_code_groups={self.num_groups} leaves no residual codebooks to predict")

        self.small_to_mtp_projection = LinearBase(talker_hidden_size, config.hidden_size, bias=True, dtype=dtype)
        # One table and one head per residual codebook: index i handles
        # codebook i+1. They are not tied, so an off-by-one here swaps two
        # codebooks' vocabularies and degrades the audio without erroring.
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, talker_hidden_size, dtype=dtype) for _ in range(self.num_residual)]
        )
        self.lm_head = nn.ModuleList(
            [LinearBase(config.hidden_size, config.vocab_size, bias=False, dtype=dtype) for _ in range(self.num_residual)]
        )
        self.layers = nn.ModuleList([PredictorLayer(config, dtype, runtime) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, backend=runtime.rmsnorm_backend)
        self.rotary = RotaryEmbedding(
            config.head_dim, self.num_groups + 1, theta=config.rope_theta, device=device, dtype=torch.float32
        )

        self._stacked_embed: torch.Tensor | None = None
        self._compiled = None

    def stacked_embedding_weight(self) -> torch.Tensor:
        """``[Q-1, vocab, talker_hidden]``, built once after weight loading.

        Lets the talker turn Q-1 embedding lookups into one gather, which is
        the residual-embedding fusion the doc lists under stage-0 optimizations.
        """
        if self._stacked_embed is None:
            self._stacked_embed = torch.stack([e.weight for e in self.codec_embedding], dim=0)
        return self._stacked_embed

    def _forward_prefix(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rotary(position_ids, dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        return self.norm(hidden)

    def maybe_compile(self) -> None:
        if self.runtime.compile_mode in ("code_predictor", "full"):
            # Dynamic shapes on purpose: the prefix grows by one position per
            # residual codebook, so a static compile would recompile Q-1 times
            # per process and then specialize on the last shape it saw.
            self._compiled = torch.compile(self._forward_prefix, dynamic=True)
            logger.info("code predictor: torch.compile enabled (dynamic shapes)")

    def _prefix(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        fn = self._compiled or self._forward_prefix
        return fn(hidden, position_ids)

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        talker_hidden: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Fill in codebooks 1..Q-1. Returns ``[batch, Q]`` of code ids.

        ``layer0_code`` is ``[batch]``, ``layer0_embed`` and ``talker_hidden``
        are ``[batch, talker_hidden]``.
        """
        batch = int(layer0_code.shape[0])
        projection = self.small_to_mtp_projection
        seq = torch.zeros(batch, self.num_groups, self.config.hidden_size, dtype=self.dtype, device=self.device)
        seq[:, 0] = projection(talker_hidden.to(self.dtype))
        seq[:, 1] = projection(layer0_embed.to(self.dtype))

        codes = torch.empty(batch, self.num_groups, dtype=torch.long, device=self.device)
        codes[:, 0] = layer0_code.reshape(batch)

        inv_temperature = 1.0 / max(temperature, 1e-6)
        sampling = do_sample and temperature > 0
        positions = torch.arange(self.num_groups, device=self.device, dtype=torch.long)

        for step in range(1, self.num_groups):
            # Exact prefix rather than the full Q positions: under a causal
            # mask position `step` cannot see anything past it, so the result
            # is identical and the attention is smaller.
            prefix_len = step + 1
            hidden = self._prefix(seq[:, :prefix_len], positions[:prefix_len].expand(batch, prefix_len))
            logits = self.lm_head[step - 1](hidden[:, step])
            if sampling:
                code = sample_gumbel(apply_top_k(logits * inv_temperature, top_k), generator)
            else:
                code = logits.argmax(dim=-1, keepdim=True)
            codes[:, step] = code.reshape(batch)
            if step < self.num_groups - 1:
                seq[:, step + 1] = projection(self.codec_embedding[step - 1](code.reshape(batch)))
        return codes
