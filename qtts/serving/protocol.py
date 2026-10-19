"""Request schema and audio encoding for the HTTP layer.

Field names match vllm-omni's ``additional_information`` entries, so a payload
written against one server works against the other. There it arrives as
one-element lists after msgspec transport; both forms are accepted here, which
is what :meth:`TTSRequest.from_additional_information` unwraps.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any

import torch

from ..engine import TTSRequest

_MAX_TEXT_CHARS = 4096


@dataclass
class GenerateRequest:
    text: str
    speaker: str = ""
    language: str = "Auto"
    instruct: str = ""
    max_new_tokens: int | None = None
    non_streaming_mode: bool | None = None
    initial_codec_chunk_frames: int | None = None
    request_id: str | None = None

    @classmethod
    def parse(cls, payload: Any) -> GenerateRequest:
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        # vllm-omni nests the fields under additional_information; accept both
        # that and the flat form.
        fields = payload.get("additional_information", payload)
        if not isinstance(fields, dict):
            raise ValueError("additional_information must be a JSON object")

        request = TTSRequest.from_additional_information(fields, request_id=payload.get("request_id"))
        if not request.text.strip():
            raise ValueError("text is required and must not be empty")
        if len(request.text) > _MAX_TEXT_CHARS:
            # At 12.5 Hz the talker emits real time, so a huge text is minutes
            # of generation holding a slot. Rejecting it is kinder than
            # accepting it and starving every other stream.
            raise ValueError(f"text is {len(request.text)} characters; the limit is {_MAX_TEXT_CHARS}")
        return cls(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            instruct=request.instruct,
            max_new_tokens=request.max_new_tokens,
            non_streaming_mode=request.non_streaming_mode,
            initial_codec_chunk_frames=request.initial_codec_chunk_frames,
            request_id=request.request_id,
        )

    def to_engine_request(self) -> TTSRequest:
        kwargs: dict[str, Any] = {
            "text": self.text,
            "speaker": self.speaker,
            "language": self.language,
            "instruct": self.instruct,
            "max_new_tokens": self.max_new_tokens,
            "non_streaming_mode": self.non_streaming_mode,
            "initial_codec_chunk_frames": self.initial_codec_chunk_frames,
        }
        if self.request_id:
            kwargs["request_id"] = self.request_id
        return TTSRequest(**kwargs)


def pcm16_bytes(audio: torch.Tensor) -> bytes:
    """Mono float waveform to 16-bit little-endian PCM.

    Clamped before scaling: the decoder already clamps to [-1, 1], but a value
    of exactly 1.0 scaled by 32768 overflows int16 and wraps to full-scale
    negative, which is an audible click at the loudest moment of the audio.
    """
    samples = audio.detach().reshape(-1).to(torch.float32).clamp(-1.0, 1.0).cpu()
    return (samples * 32767.0).to(torch.int16).numpy().tobytes()


def encode_wav(audio: torch.Tensor, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16_bytes(audio))
    return buffer.getvalue()
