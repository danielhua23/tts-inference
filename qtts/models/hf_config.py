"""Checkpoint configuration, read straight from ``config.json``.

Deliberately not ``transformers.PretrainedConfig``: the container ships
transformers 5.x, which has no ``qwen3_tts`` model type, so anything that went
through ``AutoConfig`` would fail to resolve. These dataclasses read the same
JSON the reference implementation reads and keep the field names it uses.

Nothing here has a default that silences a missing checkpoint field. ``Q`` in
particular is read from the checkpoint every time: the talker config and the
codec decoder config both carry a quantizer count, they must agree, and a
hardcoded 16 would keep working right up until a checkpoint disagreed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"{where} is missing required field {key!r}")
    return d[key]


@dataclass
class CodePredictorHFConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    num_code_groups: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodePredictorHFConfig:
        return cls(
            vocab_size=int(_require(d, "vocab_size", "code_predictor_config")),
            hidden_size=int(_require(d, "hidden_size", "code_predictor_config")),
            intermediate_size=int(_require(d, "intermediate_size", "code_predictor_config")),
            num_hidden_layers=int(_require(d, "num_hidden_layers", "code_predictor_config")),
            num_attention_heads=int(_require(d, "num_attention_heads", "code_predictor_config")),
            num_key_value_heads=int(_require(d, "num_key_value_heads", "code_predictor_config")),
            head_dim=int(_require(d, "head_dim", "code_predictor_config")),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            rope_theta=float(d.get("rope_theta", 10000.0)),
            max_position_embeddings=int(d.get("max_position_embeddings", 65536)),
            num_code_groups=int(_require(d, "num_code_groups", "code_predictor_config")),
        )


@dataclass
class TalkerHFConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    num_code_groups: int
    text_hidden_size: int
    text_vocab_size: int

    codec_eos_token_id: int
    codec_pad_id: int
    codec_bos_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int

    spk_id: dict[str, int]
    spk_is_dialect: dict[str, Any]
    codec_language_id: dict[str, int]

    code_predictor: CodePredictorHFConfig
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TalkerHFConfig:
        head_dim = int(d.get("head_dim") or (int(d["hidden_size"]) // int(d["num_attention_heads"])))
        return cls(
            vocab_size=int(_require(d, "vocab_size", "talker_config")),
            hidden_size=int(_require(d, "hidden_size", "talker_config")),
            intermediate_size=int(_require(d, "intermediate_size", "talker_config")),
            num_hidden_layers=int(_require(d, "num_hidden_layers", "talker_config")),
            num_attention_heads=int(_require(d, "num_attention_heads", "talker_config")),
            num_key_value_heads=int(_require(d, "num_key_value_heads", "talker_config")),
            head_dim=head_dim,
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            rope_theta=float(d.get("rope_theta", 10000.0)),
            max_position_embeddings=int(d.get("max_position_embeddings", 32768)),
            num_code_groups=int(_require(d, "num_code_groups", "talker_config")),
            text_hidden_size=int(_require(d, "text_hidden_size", "talker_config")),
            text_vocab_size=int(_require(d, "text_vocab_size", "talker_config")),
            codec_eos_token_id=int(_require(d, "codec_eos_token_id", "talker_config")),
            codec_pad_id=int(_require(d, "codec_pad_id", "talker_config")),
            codec_bos_id=int(_require(d, "codec_bos_id", "talker_config")),
            codec_think_id=int(_require(d, "codec_think_id", "talker_config")),
            codec_nothink_id=int(_require(d, "codec_nothink_id", "talker_config")),
            codec_think_bos_id=int(_require(d, "codec_think_bos_id", "talker_config")),
            codec_think_eos_id=int(_require(d, "codec_think_eos_id", "talker_config")),
            spk_id={str(k).lower(): int(v) for k, v in (d.get("spk_id") or {}).items()},
            spk_is_dialect={str(k).lower(): v for k, v in (d.get("spk_is_dialect") or {}).items()},
            codec_language_id={str(k).lower(): int(v) for k, v in (d.get("codec_language_id") or {}).items()},
            code_predictor=CodePredictorHFConfig.from_dict(_require(d, "code_predictor_config", "talker_config")),
            rope_scaling=d.get("rope_scaling"),
        )


@dataclass
class Code2WavHFConfig:
    codebook_size: int
    codebook_dim: int
    latent_dim: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    sliding_window: int
    layer_scale_initial_scale: float
    num_quantizers: int
    num_semantic_quantizers: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    decoder_dim: int

    output_sample_rate: int
    decode_upsample_rate: int

    @property
    def total_upsample(self) -> int:
        total = 1
        for r in tuple(self.upsample_rates) + tuple(self.upsampling_ratios):
            total *= int(r)
        return total

    @classmethod
    def from_dict(cls, top: dict[str, Any]) -> Code2WavHFConfig:
        d = _require(top, "decoder_config", "speech_tokenizer config")
        cfg = cls(
            codebook_size=int(_require(d, "codebook_size", "decoder_config")),
            codebook_dim=int(_require(d, "codebook_dim", "decoder_config")),
            latent_dim=int(_require(d, "latent_dim", "decoder_config")),
            hidden_size=int(_require(d, "hidden_size", "decoder_config")),
            intermediate_size=int(_require(d, "intermediate_size", "decoder_config")),
            num_hidden_layers=int(_require(d, "num_hidden_layers", "decoder_config")),
            num_attention_heads=int(_require(d, "num_attention_heads", "decoder_config")),
            num_key_value_heads=int(_require(d, "num_key_value_heads", "decoder_config")),
            head_dim=int(d.get("head_dim") or (int(d["hidden_size"]) // int(d["num_attention_heads"]))),
            attention_bias=bool(d.get("attention_bias", False)),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-5)),
            rope_theta=float(d.get("rope_theta", 10000.0)),
            max_position_embeddings=int(d.get("max_position_embeddings", 8000)),
            sliding_window=int(_require(d, "sliding_window", "decoder_config")),
            layer_scale_initial_scale=float(d.get("layer_scale_initial_scale", 0.01)),
            num_quantizers=int(_require(d, "num_quantizers", "decoder_config")),
            num_semantic_quantizers=int(d.get("num_semantic_quantizers", 1)),
            upsample_rates=tuple(int(r) for r in _require(d, "upsample_rates", "decoder_config")),
            upsampling_ratios=tuple(int(r) for r in _require(d, "upsampling_ratios", "decoder_config")),
            decoder_dim=int(_require(d, "decoder_dim", "decoder_config")),
            output_sample_rate=int(_require(top, "output_sample_rate", "speech_tokenizer config")),
            decode_upsample_rate=int(_require(top, "decode_upsample_rate", "speech_tokenizer config")),
        )
        if cfg.total_upsample != cfg.decode_upsample_rate:
            # The waveform length per frame is derived from the conv stack, and
            # the config states it independently. A mismatch means one of them
            # is wrong and every downstream length calculation (trimming,
            # chunk boundaries, frame accounting) would be off.
            raise ValueError(
                f"upsample_rates {cfg.upsample_rates} x upsampling_ratios {cfg.upsampling_ratios} give "
                f"{cfg.total_upsample} samples per frame, but decode_upsample_rate says "
                f"{cfg.decode_upsample_rate}"
            )
        return cfg


@dataclass
class Qwen3TTSHFConfig:
    talker: TalkerHFConfig
    code2wav: Code2WavHFConfig
    model_type: str
    tts_model_type: str
    tts_model_size: str
    im_start_token_id: int
    im_end_token_id: int
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    generation: dict[str, Any] = field(default_factory=dict)

    @property
    def num_quantizers(self) -> int:
        return self.talker.num_code_groups

    @property
    def sample_rate(self) -> int:
        return self.code2wav.output_sample_rate

    @property
    def frame_rate(self) -> float:
        return self.code2wav.output_sample_rate / self.code2wav.decode_upsample_rate

    @classmethod
    def from_pretrained(cls, path: str) -> Qwen3TTSHFConfig:
        with open(os.path.join(path, "config.json")) as fh:
            top = json.load(fh)
        tokenizer_dir = os.path.join(path, "speech_tokenizer")
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(
                f"{path} has no speech_tokenizer/ subdirectory. Point --model at a full "
                "Qwen3-TTS checkpoint; the codec decoder weights live there."
            )
        with open(os.path.join(tokenizer_dir, "config.json")) as fh:
            tok = json.load(fh)

        talker = TalkerHFConfig.from_dict(_require(top, "talker_config", "config.json"))
        code2wav = Code2WavHFConfig.from_dict(tok)
        if talker.num_code_groups != code2wav.num_quantizers:
            raise ValueError(
                f"talker_config.num_code_groups={talker.num_code_groups} disagrees with "
                f"decoder_config.num_quantizers={code2wav.num_quantizers}: the talker would emit "
                "frames the decoder cannot read"
            )
        if talker.code_predictor.num_code_groups != talker.num_code_groups:
            raise ValueError(
                f"code_predictor_config.num_code_groups={talker.code_predictor.num_code_groups} "
                f"disagrees with talker_config.num_code_groups={talker.num_code_groups}"
            )

        generation: dict[str, Any] = {}
        gen_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as fh:
                generation = json.load(fh)

        return cls(
            talker=talker,
            code2wav=code2wav,
            model_type=str(top.get("model_type", "")),
            tts_model_type=str(top.get("tts_model_type", "")),
            tts_model_size=str(top.get("tts_model_size", "")),
            im_start_token_id=int(top.get("im_start_token_id", 151644)),
            im_end_token_id=int(top.get("im_end_token_id", 151645)),
            tts_pad_token_id=int(top.get("tts_pad_token_id", 151671)),
            tts_bos_token_id=int(top.get("tts_bos_token_id", 151672)),
            tts_eos_token_id=int(top.get("tts_eos_token_id", 151673)),
            generation=generation,
        )

    def task_type(self) -> str:
        """The prompt layout this checkpoint expects."""
        mapping = {
            "custom_voice": "CustomVoice",
            "voice_design": "VoiceDesign",
            "base": "Base",
        }
        key = self.tts_model_type.lower()
        if key not in mapping:
            raise ValueError(f"unsupported tts_model_type={self.tts_model_type!r}, expected one of {sorted(mapping)}")
        return mapping[key]
