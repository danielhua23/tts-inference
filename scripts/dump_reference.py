"""Run the official ``qwen-tts`` package and record what it computed.

This is half of the parity check. It has to run in a separate interpreter with
a separate ``transformers`` on ``PYTHONPATH``, because the official package
pins 4.57 while the container ships 5.x and the two cannot coexist in one
process. So instead of importing both, this script dumps tensors and
``tests/test_parity.py`` compares them.

    PYTHONPATH=/path/to/.ref-env python scripts/dump_reference.py \
        --model /path/to/checkpoint --out /tmp/reference.pt

What it captures, and why each one is worth capturing separately:

* ``prompt_embeds`` -- the talker's prefill input. Catches prompt-assembly
  mistakes, which are the ones that produce fluent audio saying the wrong
  thing.
* ``first_logits`` -- codebook-0 logits at the first generated frame. Catches
  the talker forward pass, isolated from any sampling.
* ``codes`` -- the whole greedy code sequence. Catches the decode loop, the
  code predictor, and the embedding sum that feeds the next step.

Greedy throughout. With sampling on, two correct implementations disagree
immediately, and with greedy they agree until they don't -- so the first
divergent frame localises the bug.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import sys
import types

import torch


def _stub_unused_imports() -> None:
    """Satisfy imports that only the 25 Hz tokenizer needs.

    ``qwen_tts/core/__init__`` imports both tokenizer generations eagerly, and
    the 25 Hz one pulls in ``torchaudio`` for Kaldi feature extraction. This
    checkpoint is 12 Hz and never touches that code path. Installing
    ``torchaudio`` would risk resolving a different ``torch`` build into the
    reference environment, which would make the parity comparison meaningless,
    so the module is stubbed instead. If the 25 Hz path is ever actually
    reached, the stub raises rather than returning something plausible.
    """

    class _Unavailable(types.ModuleType):
        def __getattr__(self, name: str):
            # Dunder lookups have to behave normally: `inspect` walks every
            # module in sys.modules probing __file__ and friends while building
            # tracebacks, and torch's fake-op registration triggers exactly
            # that during import. Raising there turns an unrelated import into
            # a confusing failure inside this stub.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise RuntimeError(
                f"{self.__name__}.{name} was stubbed out because only the 25 Hz tokenizer needs it; "
                "this checkpoint should not reach that code"
            )

    for name in ("torchaudio", "torchaudio.compliance", "torchaudio.compliance.kaldi"):
        module = _Unavailable(name)
        # importlib.util.find_spec raises outright on a module whose __spec__
        # is None, and transformers probes for torchaudio at import time, so
        # the stub needs a spec even though nothing will load from it.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.__version__ = "0.0.0+stub"
        module.__path__ = []  # marks it a package so submodule imports resolve
        sys.modules.setdefault(name, module)
        if "." in name:
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, sys.modules[name])


def main() -> None:
    _stub_unused_imports()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--speaker", default="serena")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import transformers
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    print(f"reference transformers {transformers.__version__}")
    model = Qwen3TTSModel.from_pretrained(args.model, dtype=torch.float32, device_map="cuda")

    captured: dict[str, torch.Tensor] = {}

    talker_body = model.model.talker.model
    original_forward = talker_body.forward

    def record_prefill(*fn_args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None and embeds.shape[1] > 1 and "prompt_embeds" not in captured:
            captured["prompt_embeds"] = embeds[0].detach().float().cpu()
        return original_forward(*fn_args, **kwargs)

    talker_body.forward = record_prefill

    codec_head = model.model.talker.codec_head
    original_head = codec_head.forward

    def record_logits(hidden, *fn_args, **kwargs):
        out = original_head(hidden, *fn_args, **kwargs)
        if "first_logits" not in captured:
            captured["first_logits"] = out.reshape(-1, out.shape[-1])[-1].detach().float().cpu()
        return out

    codec_head.forward = record_logits

    # Per-frame inputs to the residual predictor. Recorded separately from the
    # final codes so a divergence can be attributed: if these match and the
    # codes do not, the code predictor is at fault; if these already differ,
    # the talker decode step is.
    predictor = model.model.talker.code_predictor
    original_generate = predictor.generate
    frame_inputs: list[torch.Tensor] = []
    frame_outputs: list[torch.Tensor] = []

    def record_frames(*fn_args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        result = original_generate(*fn_args, **kwargs)
        if embeds is not None and len(frame_inputs) < 8:
            frame_inputs.append(embeds[0].detach().float().cpu())
            frame_outputs.append(result.sequences[0].detach().reshape(-1).cpu())
        return result

    predictor.generate = record_frames

    # Whatever tensor reaches the codec decoder *is* the generated code
    # sequence, so hooking here needs no knowledge of the generate loop.
    tokenizer = model.model.speech_tokenizer
    original_decode = tokenizer.decode

    def record_codes(codes, *fn_args, **kwargs):
        # The 12 Hz tokenizer takes a list of per-request dicts.
        if "codes" not in captured:
            first = codes[0]["audio_codes"] if isinstance(codes, (list, tuple)) else codes
            captured["codes"] = first.detach().reshape(first.shape[-2], first.shape[-1]).cpu()
        return original_decode(codes, *fn_args, **kwargs)

    tokenizer.decode = record_codes

    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct or None,
        do_sample=False,
        subtalker_dosample=False,
        repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens,
    )

    payload = {
        "text": args.text,
        "speaker": args.speaker,
        "language": args.language,
        "instruct": args.instruct,
        "sample_rate": int(sample_rate),
        "audio": torch.as_tensor(wavs[0]).float().reshape(-1).cpu(),
        #: ``[frames, 2, talker_hidden]``: the talker hidden state and the
        #: codebook-0 embedding, for the first few frames.
        "predictor_inputs": torch.stack(frame_inputs) if frame_inputs else torch.empty(0),
        #: ``[frames, Q-1]``: the residual codebooks each of those produced.
        "predictor_outputs": torch.stack(frame_outputs) if frame_outputs else torch.empty(0),
        **captured,
    }
    torch.save(payload, args.out)

    print(f"saved {args.out}")
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key}: {value!r}")


if __name__ == "__main__":
    main()
