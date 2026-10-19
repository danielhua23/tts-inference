"""Building the talker's prefill prompt.

The talker does not consume token ids. Every prefill position is an embedding
that is the *sum* of a text-side and a codec-side vector, and this file is where
that sum is assembled. Getting the layout wrong does not raise -- it produces
fluent audio saying something other than the text -- so the structure is spelled
out rather than inferred.

The prompt, for CustomVoice, position by position:

    [ <|im_start|> assistant \\n ]        role header, text stream only
    [ nothink, think_bos, think_eos ]    codec control tags, or the 4-tag
                                         variant carrying an explicit language
    [ speaker ]                          the voice, as a codec vocabulary id
    [ codec_pad ]                        summed with tts_bos on the text side
    [ text token 0 .. n, tts_eos ]       text stream, codec side all pad
    [ codec_bos ]                        summed with tts_pad; generation starts

The codec-side prefix is shifted by one against the text side
(``codec_input[:, :-1]``), which is what puts ``codec_bos`` at the final
position instead of inside the tag block.

Two text-conditioning modes, and the checkpoint picks the default:

* **non-streaming** (CustomVoice, VoiceDesign) puts the whole text in the
  prefill prompt, and every decode step adds the same pad embedding. The text
  is fully visible from the first frame.
* **streaming** puts only the first text token in the prompt and feeds one more
  per decode step from a queue. This is what lets text arrive while audio is
  already being produced, and it is why the talker's text stream is a queue
  rather than a constant.

Base / voice-clone is deliberately absent. It needs the codec *encoder* to turn
reference audio into ``ref_code`` plus an ECAPA speaker encoder, neither of
which CustomVoice ships weights for; :func:`build_prompt` rejects it rather
than quietly producing a prompt with an uninitialized speaker vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hf_config import Qwen3TTSHFConfig
from .talker import Qwen3TTSTalker


def assistant_template(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def instruct_template(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"


#: The template's fixed framing, in tokens: ``<|im_start|> assistant \n`` in
#: front, and ``<|im_end|> \n <|im_start|> assistant \n`` behind. The text sits
#: between them, so the slices below are ``[:3]`` and ``[3:-5]``.
_ROLE_TOKENS = 3
_TAIL_TOKENS = 5


def load_tokenizer(model_path: str):
    """Load the text tokenizer.

    ``fix_mistral_regex`` is deliberately not passed. transformers 5.x warns
    that this checkpoint's pre-tokenizer regex matches a known-bad Mistral one
    and offers to patch it; applying the patch would change how text splits and
    therefore diverge from the reference implementation, which runs on
    transformers 4.57. The default path was checked token-for-token against
    that version.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


@dataclass
class TalkerPrompt:
    #: ``[prompt_len, hidden]`` prefill embeddings.
    embeds: torch.Tensor
    #: ``[T, hidden]`` text embeddings consumed one per decode step. In
    #: non-streaming mode this is a single pad row reused forever.
    trailing_text: torch.Tensor
    num_text_tokens: int


class PromptBuilder:
    def __init__(self, config: Qwen3TTSHFConfig, talker: Qwen3TTSTalker, tokenizer) -> None:
        self.config = config
        self.talker = talker
        self.tokenizer = tokenizer
        self.task_type = config.task_type()

    def _tokenize(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, padding=False)["input_ids"]
        return torch.tensor([ids], device=self.talker.device, dtype=torch.long)

    def _resolve_language_id(self, language: str, speaker: str) -> int | None:
        """Map a language name to its codec tag, honouring dialect voices.

        A dialect speaker with the language left at Chinese or Auto gets its
        dialect tag instead. The reference does this because the dialect voices
        generate more stable codes when the tag agrees with the timbre.
        """
        tc = self.config.talker
        key = (language or "").lower()
        if key and key != "auto":
            resolved = tc.codec_language_id.get(key)
            if resolved is not None:
                return resolved
        if key in ("", "auto", "chinese") and speaker:
            dialect = tc.spk_is_dialect.get(speaker.lower())
            if isinstance(dialect, str) and dialect:
                return tc.codec_language_id.get(dialect)
        return None

    def _codec_control(self, language_id: int | None) -> torch.Tensor:
        tc = self.config.talker
        if language_id is None:
            tags = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
        else:
            tags = [tc.codec_think_id, tc.codec_think_bos_id, int(language_id), tc.codec_think_eos_id]
        ids = torch.tensor([tags], device=self.talker.device, dtype=torch.long)
        return self.talker.embed_codec(ids)

    def _speaker_embed(self, speaker: str) -> torch.Tensor:
        tc = self.config.talker
        key = (speaker or "").lower().strip()
        if not key:
            raise ValueError("CustomVoice requires a speaker; see config.json talker_config.spk_id")
        if key not in tc.spk_id:
            raise ValueError(f"unknown speaker {speaker!r}; this checkpoint supports {sorted(tc.spk_id)}")
        ids = torch.tensor([[tc.spk_id[key]]], device=self.talker.device, dtype=torch.long)
        return self.talker.embed_codec(ids)

    def build_prompt(
        self,
        *,
        text: str,
        speaker: str = "",
        language: str = "Auto",
        instruct: str = "",
        non_streaming_mode: bool | None = None,
    ) -> TalkerPrompt:
        if self.task_type == "Base":
            raise NotImplementedError(
                "Base (voice clone) needs the codec encoder and the ECAPA speaker encoder to turn "
                "reference audio into ref_code and an x-vector. This runtime implements CustomVoice "
                "and VoiceDesign; point --model at a CustomVoice or VoiceDesign checkpoint."
            )
        if not text.strip():
            raise ValueError("text is empty")
        if non_streaming_mode is None:
            non_streaming_mode = self.task_type in ("CustomVoice", "VoiceDesign")

        talker = self.talker
        tc = self.config.talker
        device = talker.device

        input_ids = self._tokenize(assistant_template(text))
        if int(input_ids.shape[1]) <= _ROLE_TOKENS + _TAIL_TOKENS:
            raise ValueError(f"text {text!r} produced no text tokens after the chat template")
        text_ids = input_ids[:, _ROLE_TOKENS:-_TAIL_TOKENS]

        special = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id]], device=device, dtype=torch.long
        )
        tts_bos_embed, tts_eos_embed = talker.embed_text(special).chunk(2, dim=1)
        tts_pad_embed = talker.tts_pad_embed.reshape(1, 1, -1)

        language_id = self._resolve_language_id(language, speaker)
        parts = [self._codec_control(language_id)]
        if self.task_type == "CustomVoice":
            parts.append(self._speaker_embed(speaker))
        tail_ids = torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=torch.long)
        parts.append(talker.embed_codec(tail_ids))
        codec_input = torch.cat(parts, dim=1)

        role_embed = talker.embed_text(input_ids[:, :_ROLE_TOKENS])
        # Text side of the tag block: pad everywhere except the last position,
        # which carries tts_bos. Codec side is shifted left by one so codec_bos
        # lands after the text instead of inside the tags.
        codec_prefix = torch.cat(
            [tts_pad_embed.expand(-1, int(codec_input.shape[1]) - 2, -1), tts_bos_embed], dim=1
        )
        codec_prefix = codec_prefix + codec_input[:, :-1]
        prompt = torch.cat([role_embed, codec_prefix], dim=1)

        if non_streaming_mode:
            text_all = torch.cat([talker.embed_text(text_ids), tts_eos_embed], dim=1)
            pad_ids = torch.full(
                (1, int(text_all.shape[1])), tc.codec_pad_id, device=device, dtype=torch.long
            )
            bos_id = torch.tensor([[tc.codec_bos_id]], device=device, dtype=torch.long)
            prompt = torch.cat(
                [prompt, text_all + talker.embed_codec(pad_ids), tts_pad_embed + talker.embed_codec(bos_id)],
                dim=1,
            )
            trailing_text = tts_pad_embed
        else:
            first_text = talker.embed_text(text_ids[:, :1]) + codec_input[:, -1:]
            prompt = torch.cat([prompt, first_text], dim=1)
            trailing_text = torch.cat([talker.embed_text(text_ids[:, 1:]), tts_eos_embed], dim=1)

        if instruct.strip():
            prompt = torch.cat([talker.embed_text(self._tokenize(instruct_template(instruct))), prompt], dim=1)

        return TalkerPrompt(
            embeds=prompt.squeeze(0).contiguous(),
            trailing_text=trailing_text.squeeze(0).contiguous(),
            num_text_tokens=int(text_ids.shape[1]),
        )
