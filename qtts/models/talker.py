"""Stage 0: the Qwen3-TTS talker.

A 28-layer Qwen3 decoder that emits one codebook-0 id per audio frame. Two
things make it not just an LLM:

* **The input embedding is a sum of two streams.** Every position carries a
  text-side embedding (``text_projection(text_embedding(id))``) added to a
  codec-side embedding (``codec_embedding(id)``). During decode the codec side
  is the frame just generated and the text side is either the next text token
  (streaming) or a constant pad embedding (non-streaming). Mixing these up
  produces fluent-sounding audio that says the wrong thing, which no shape
  check will catch.
* **The logits are masked to the codec vocabulary.** The head is 3072 wide but
  only ``[1, 2048)`` are real codebook-0 indices; the rest are speaker tags,
  language tags and control tokens the codec decoder cannot decode. Only
  ``codec_eos_token_id`` is additionally allowed, as the stop signal.

KV cache layout is ``[layers, slots, positions, kv_heads, head_dim]`` and a
request is pinned to a slot for its lifetime. Position last-but-one is what
makes the writes cheap: ``(slot, position)`` flattens to a single index, so
appending a decode step's keys is one ``index_copy_`` over the batch instead of
a per-row loop. Reads always take the contiguous slot prefix ``[0, n_rows)``,
which is a view; indexing with a slot tensor would instead copy
``[rows, heads, positions, dim]`` per layer per step, which at 28 layers is
gigabytes of traffic per decode step. The cost of that choice is that the
scheduler has to hand out low slots first, and rows in the prefix that are idle
are computed and discarded.

No paging. A TTS request is a few hundred frames of a fixed-width sequence, so
the fragmentation paging solves does not arise here.
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention import attention, build_mask
from ..config import TalkerConfig
from ..layers import GatedMLP, LinearBase, MergedLinear, RMSNorm, RotaryEmbedding, apply_rotary_pos_emb
from ..utils import get_logger, human_bytes
from .hf_config import Qwen3TTSHFConfig, TalkerHFConfig

logger = get_logger("talker")


class TalkerKVCache:
    """``[layers, slots, positions, kv_heads, head_dim]`` for keys and values.

    One slot more than ``max_seqs`` is allocated. The extra one is a scratch
    slot, and it is what makes a fixed-shape decode step possible: the step
    covers the whole slot prefix ``[0, n)``, but not every slot in that prefix
    is running -- some are throttled, some just finished. Those rows still go
    through the transformer, and their keys have to land *somewhere*. Writing
    them at their own ``(slot, seq_len)`` would overwrite a live request's
    cache, so the scratch slot absorbs them instead and their output is
    discarded. Every padded row writing the same scratch index is fine; nothing
    reads it.
    """

    def __init__(
        self,
        num_layers: int,
        max_seqs: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        slots = max_seqs + 1
        shape = (num_layers, slots, max_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seqs = max_seqs
        self.max_len = max_len
        self.scratch_slot = max_seqs
        self.scratch_index = max_seqs * max_len
        # Flat views over (slot, position) so a decode append is one
        # index_copy_ instead of one slice assignment per row.
        self.k_flat = self.k.view(num_layers, slots * max_len, num_kv_heads, head_dim)
        self.v_flat = self.v.view(num_layers, slots * max_len, num_kv_heads, head_dim)
        logger.info(
            "talker KV cache: %d slots (+1 scratch) x %d positions = %s",
            max_seqs,
            max_len,
            human_bytes(self.nbytes()),
        )

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class TalkerAttention(nn.Module):
    """Qwen3 attention: GQA plus per-head RMSNorm on q and k before RoPE."""

    def __init__(
        self,
        config: TalkerHFConfig,
        layer_idx: int,
        dtype: torch.dtype,
        fuse_qkv: bool,
        norm_backend: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.fuse_qkv = fuse_qkv

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

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns q/k/v as ``[batch, seq, heads, head_dim]``."""
        batch, seq, _ = hidden.shape
        if self.fuse_qkv:
            q, k, v = self.qkv_proj.split(self.qkv_proj(hidden))
        else:
            q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
        q = self.q_norm(q.view(batch, seq, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(batch, seq, self.num_kv_heads, self.head_dim))
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim)
        return q, k, v


class TalkerLayer(nn.Module):
    def __init__(self, config: TalkerHFConfig, layer_idx: int, dtype: torch.dtype, runtime: TalkerConfig) -> None:
        super().__init__()
        self.self_attn = TalkerAttention(
            config, layer_idx, dtype, fuse_qkv=runtime.fuse_qkv, norm_backend=runtime.rmsnorm_backend
        )
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size, fuse=runtime.fuse_gate_up, dtype=dtype)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, backend=runtime.rmsnorm_backend
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, backend=runtime.rmsnorm_backend
        )


class Qwen3TTSTalker(nn.Module):
    def __init__(
        self,
        config: Qwen3TTSHFConfig,
        *,
        runtime: TalkerConfig,
        dtype: torch.dtype,
        device: torch.device,
        max_num_seqs: int,
        max_model_len: int,
    ) -> None:
        super().__init__()
        tc = config.talker
        self.config = config
        self.tc = tc
        self.runtime = runtime
        self.dtype = dtype
        self.device = device
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.num_quantizers = tc.num_code_groups

        self.codec_embedding = nn.Embedding(tc.vocab_size, tc.hidden_size, dtype=dtype)
        self.text_embedding = nn.Embedding(tc.text_vocab_size, tc.text_hidden_size, dtype=dtype)
        # text_projection is the reference's "ResizeMLP": fc1, SiLU, fc2, both
        # with bias. It is not a plain projection, and replacing it with one
        # changes the text conditioning rather than failing.
        self.text_proj_fc1 = LinearBase(tc.text_hidden_size, tc.text_hidden_size, bias=True, dtype=dtype)
        self.text_proj_fc2 = LinearBase(tc.text_hidden_size, tc.hidden_size, bias=True, dtype=dtype)

        self.layers = nn.ModuleList([TalkerLayer(tc, i, dtype, runtime) for i in range(tc.num_hidden_layers)])
        self.norm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps, dtype=dtype, backend=runtime.rmsnorm_backend)
        self.codec_head = LinearBase(tc.hidden_size, tc.vocab_size, bias=False, dtype=dtype)

        self.rotary = RotaryEmbedding(
            tc.head_dim, max_model_len, theta=tc.rope_theta, device=device, dtype=torch.float32
        )
        self.cache = TalkerKVCache(
            tc.num_hidden_layers, max_num_seqs, max_model_len, tc.num_key_value_heads, tc.head_dim, dtype, device
        )

        # Only [1, 2048) plus codec EOS may be sampled. Stored in disallowed
        # form so a decode step is one masked_fill rather than a negation.
        vocab = tc.vocab_size
        allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
        allowed[1 : min(tc.code_predictor.vocab_size, vocab)] = True
        if 0 <= tc.codec_eos_token_id < vocab:
            allowed[tc.codec_eos_token_id] = True
        self.register_buffer("codec_disallowed", ~allowed, persistent=False)
        self.register_buffer(
            "tts_pad_embed", torch.zeros(1, tc.hidden_size, dtype=dtype, device=device), persistent=False
        )

    # -- embeddings ---------------------------------------------------------

    def embed_codec(self, ids: torch.Tensor) -> torch.Tensor:
        return self.codec_embedding(ids)

    def embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_proj_fc2(nn.functional.silu(self.text_proj_fc1(self.text_embedding(ids))))

    @torch.inference_mode()
    def init_runtime_buffers(self) -> None:
        """Cache the pad embedding that every decode step adds in.

        ``text_projection(text_embedding(tts_pad_token_id))`` depends only on
        frozen weights, so in non-streaming mode this single row is the entire
        text stream for the whole generation.
        """
        pad_id = torch.tensor([[self.config.tts_pad_token_id]], device=self.device, dtype=torch.long)
        self.tts_pad_embed.copy_(self.embed_text(pad_id).reshape(1, -1))

    # -- transformer --------------------------------------------------------

    def _run_layers(
        self,
        hidden: torch.Tensor,
        *,
        row_start: int,
        write_index: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``hidden`` is ``[rows, seq, hidden]``; rows map to slots ``row_start + i``.

        ``write_index`` is the flattened ``slot * max_len + position`` for every
        (row, seq) pair, in row-major order.
        """
        rows, seq, _ = hidden.shape
        row_end = row_start + rows
        cos, sin = self.rotary(query_positions, dtype=hidden.dtype)
        for layer in self.layers:
            attn = layer.self_attn
            residual = hidden
            q, k, v = attn.project(layer.input_layernorm(hidden))
            q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)

            self.cache.k_flat[attn.layer_idx].index_copy_(
                0, write_index, k.transpose(1, 2).reshape(rows * seq, attn.num_kv_heads, attn.head_dim)
            )
            self.cache.v_flat[attn.layer_idx].index_copy_(
                0, write_index, v.reshape(rows * seq, attn.num_kv_heads, attn.head_dim)
            )

            keys = self.cache.k[attn.layer_idx, row_start:row_end, :key_len].transpose(1, 2)
            values = self.cache.v[attn.layer_idx, row_start:row_end, :key_len].transpose(1, 2)
            out = attention(
                q,
                keys,
                values,
                scale=attn.scaling,
                backend=self.runtime.attention_backend,
                causal=attn_mask is None,
                attn_mask=attn_mask,
            )
            hidden = residual + attn.o_proj(out.transpose(1, 2).reshape(rows, seq, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden

    @torch.inference_mode()
    def prefill(self, inputs_embeds: torch.Tensor, slot: int) -> torch.Tensor:
        """Run one request's prompt. Returns the last position's hidden state.

        One request at a time on purpose: a TTS prompt is the chat-template
        header plus the text, on the order of 50 tokens, against hundreds of
        decode steps. Batching it would add padding bookkeeping to something
        that is not where the time goes.
        """
        prompt_len = int(inputs_embeds.shape[0])
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"prompt is {prompt_len} tokens but max_model_len is {self.max_model_len}; "
                "raise --max-model-len or shorten the text"
            )
        hidden = inputs_embeds.reshape(1, prompt_len, -1).to(self.dtype)
        positions = torch.arange(prompt_len, device=self.device, dtype=torch.long)
        hidden = self._run_layers(
            hidden,
            row_start=slot,
            write_index=positions + slot * self.cache.max_len,
            query_positions=positions.unsqueeze(0),
            key_len=prompt_len,
            attn_mask=None,
        )
        return self.norm(hidden)[0, -1]

    @torch.inference_mode()
    def decode(
        self,
        inputs_embeds: torch.Tensor,
        seq_lens: torch.Tensor,
        key_len: int,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One decode step over the slot prefix ``0..rows-1``.

        Row ``i`` *is* slot ``i``. That identity is load-bearing: the cache is
        read as the contiguous prefix, so a caller that packed only its running
        requests into rows 0..k would have them attend to, and write into,
        whichever slots happen to sit at those indices.

        ``seq_lens[i]`` is how many positions slot ``i`` already holds, so it is
        both where the new key goes and the query's rotary position.

        ``active[i]`` false means slot ``i`` is in the prefix but not running
        this step -- throttled, draining, or waiting. Its row is still computed
        (that is the cost of the contiguous-prefix layout) but its key is
        diverted to the scratch slot so it cannot corrupt a live cache, and its
        output is ignored.

        ``key_len`` comes from the caller rather than ``seq_lens.max()`` because
        reading a max off the device is a host sync in the middle of the hottest
        loop in the engine, and because a graph-captured step needs the shape
        fixed before it runs.
        """
        rows = int(inputs_embeds.shape[0])
        hidden = inputs_embeds.reshape(rows, 1, -1).to(self.dtype)
        positions = seq_lens.reshape(rows, 1)
        key_positions = torch.arange(key_len, device=self.device, dtype=torch.long).expand(rows, key_len)
        mask = build_mask(
            query_positions=positions,
            key_positions=key_positions,
            key_valid=key_positions <= positions,
            causal=True,
        )
        row_offsets = torch.arange(rows, device=self.device, dtype=torch.long) * self.cache.max_len
        write_index = row_offsets + seq_lens
        if active is not None:
            write_index = torch.where(active, write_index, write_index.new_full((), self.cache.scratch_index))
        hidden = self._run_layers(
            hidden,
            row_start=0,
            write_index=write_index,
            query_positions=positions,
            key_len=key_len,
            attn_mask=mask,
        )
        return self.norm(hidden)[:, 0]

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.codec_head(hidden).float().masked_fill(self.codec_disallowed, float("-inf"))
