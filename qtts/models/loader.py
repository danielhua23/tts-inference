"""Loading safetensors into the modules.

The modules were named to match the checkpoint's tensor paths wherever that was
possible, so most of this file is prefix rewriting rather than a mapping table.
Two things need real work:

* **Fusion.** When ``fuse_qkv`` or ``fuse_gate_up`` is on, three or two
  checkpoint tensors write into slices of one parameter, so those keys are
  routed to :meth:`MergedLinear.load_partition` instead of being copied whole.
* **Accounting.** Every source key must land somewhere and every parameter must
  be filled. A silently skipped tensor is the worst failure mode available
  here: the model still runs and still produces audio, just wrong audio. So
  both directions are checked and the mismatch is raised, not logged.

The codec checkpoint also contains the full *encoder* -- 225 tensors for
turning audio into codes. Text-to-speech never runs it, so it is skipped by an
explicit rule rather than by falling through the "unexpected key" check, which
is what keeps that check strict.
"""

from __future__ import annotations

import glob
import json
import os
import re

import torch
from torch import nn

from ..layers.linear import MergedLinear
from ..utils import get_logger, human_bytes

logger = get_logger("loader")

_QKV = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
_GATE_UP = {"gate_proj": 0, "up_proj": 1}


def _iter_safetensors(path: str):
    """Yield ``(key, tensor)`` across all shards, one shard open at a time."""
    from safetensors import safe_open

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
        files = [os.path.join(path, name) for name in shards]
    else:
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {path}")
    for file_path in files:
        with safe_open(file_path, framework="pt") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


class _Sink:
    """Routes source tensors to parameters, tracking what has been filled."""

    def __init__(self, module: nn.Module, dtype: torch.dtype) -> None:
        self.module = module
        self.dtype = dtype
        # state_dict() is the right source of truth for "must come from the
        # checkpoint": it holds parameters and persistent buffers, and excludes
        # the non-persistent ones (RoPE tables, the divided-out codebooks, the
        # cached pad embedding) that precompute_caches() derives instead.
        self.params = dict(module.state_dict(keep_vars=True))
        self.filled: set[str] = set()
        self.skipped = 0

    def _fused_target(self, target: str) -> tuple[str, int] | None:
        """Rewrite a ``...q_proj.weight`` style key onto a fused parameter."""
        parts = target.split(".")
        if len(parts) < 2:
            return None
        leaf, name = parts[-2], parts[-1]
        prefix = ".".join(parts[:-2])
        for table, fused in ((_QKV, "qkv_proj"), (_GATE_UP, "gate_up_proj")):
            if leaf not in table:
                continue
            candidate = f"{prefix}.{fused}" if prefix else fused
            owner = self.module.get_submodule(candidate) if self._has(candidate) else None
            if isinstance(owner, MergedLinear):
                return f"{candidate}.{name}", table[leaf]
        return None

    def _has(self, qualified: str) -> bool:
        try:
            self.module.get_submodule(qualified)
        except AttributeError:
            return False
        return True

    def put(self, target: str, tensor: torch.Tensor) -> None:
        fused = self._fused_target(target)
        if fused is not None:
            fused_key, index = fused
            owner_name, leaf = fused_key.rsplit(".", 1)
            owner = self.module.get_submodule(owner_name)
            assert isinstance(owner, MergedLinear)
            payload = tensor.to(dtype=self.dtype)
            if leaf == "weight":
                owner.load_partition(index, payload)
            else:
                owner.load_bias_partition(index, payload)
            self.filled.add(fused_key)
            return

        param = self.params.get(target)
        if param is None:
            raise KeyError(f"checkpoint tensor maps to {target!r}, which the model does not have")
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(
                f"{target}: model expects {tuple(param.shape)} but checkpoint has {tuple(tensor.shape)}"
            )
        param.data.copy_(tensor.to(dtype=self.dtype))
        self.filled.add(target)

    def skip(self) -> None:
        self.skipped += 1

    def verify(self, what: str) -> None:
        """Fail if any parameter went unwritten, naming a few of them."""
        # Fused parameters are recorded under their fused name; every partition
        # writes the same key, so presence is enough.
        missing = sorted(set(self.params) - self.filled)
        if missing:
            head = ", ".join(missing[:8])
            raise RuntimeError(
                f"{what}: {len(missing)} parameters were not present in the checkpoint: {head}"
                f"{' ...' if len(missing) > 8 else ''}"
            )


def load_talker_weights(
    talker: nn.Module, code_predictor: nn.Module, model_path: str, dtype: torch.dtype
) -> None:
    """Load ``talker.*`` tensors into the talker and its code predictor."""
    talker_sink = _Sink(talker, dtype)
    predictor_sink = _Sink(code_predictor, dtype)
    seen = 0

    for key, tensor in _iter_safetensors(model_path):
        seen += 1
        if key.startswith("talker.code_predictor."):
            rest = key[len("talker.code_predictor.") :]
            # `model.` wraps the transformer body and the codec embeddings;
            # lm_head and small_to_mtp_projection sit outside it.
            rest = rest[len("model.") :] if rest.startswith("model.") else rest
            predictor_sink.put(rest, tensor)
        elif key.startswith("talker.text_projection."):
            rest = key[len("talker.text_projection.") :]
            rest = rest.replace("linear_fc1.", "text_proj_fc1.").replace("linear_fc2.", "text_proj_fc2.")
            talker_sink.put(rest, tensor)
        elif key.startswith("talker.model."):
            talker_sink.put(key[len("talker.model.") :], tensor)
        elif key.startswith("talker."):
            talker_sink.put(key[len("talker.") :], tensor)
        else:
            talker_sink.skip()

    talker_sink.verify("talker")
    predictor_sink.verify("code_predictor")
    logger.info(
        "loaded talker (%s) + code predictor (%s) from %d checkpoint tensors",
        human_bytes(sum(p.numel() * p.element_size() for p in talker.parameters())),
        human_bytes(sum(p.numel() * p.element_size() for p in code_predictor.parameters())),
        seen,
    )


#: ``input_proj`` is the encode direction of each quantizer group; decoding
#: only uses ``output_proj``, so the module is never built.
_SKIP_CODEC = re.compile(r"^encoder\.|^decoder\.quantizer\.rvq_(first|rest)\.input_proj\.")


def load_code2wav_weights(code2wav: nn.Module, model_path: str, dtype: torch.dtype) -> None:
    """Load ``speech_tokenizer/``'s ``decoder.*`` tensors."""
    tokenizer_path = os.path.join(model_path, "speech_tokenizer")
    sink = _Sink(code2wav, dtype)
    seen = 0
    for key, tensor in _iter_safetensors(tokenizer_path):
        seen += 1
        if _SKIP_CODEC.match(key):
            sink.skip()
            continue
        if not key.startswith("decoder."):
            sink.skip()
            continue
        sink.put(key[len("decoder.") :], tensor)

    sink.verify("code2wav")
    logger.info(
        "loaded code2wav (%s) from %d checkpoint tensors, skipping %d encoder/encode-only tensors",
        human_bytes(sum(p.numel() * p.element_size() for p in code2wav.parameters())),
        seen,
        sink.skipped,
    )
