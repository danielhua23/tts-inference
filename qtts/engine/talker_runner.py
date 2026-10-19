"""Stage 0 execution: one codec frame per decode step, batched across requests.

The per-step loop, which is where all of stage 0's time goes:

1. ``talker.decode`` advances one position for every active row and returns its
   hidden state.
2. That hidden state produces codebook-0 logits, masked to the legal codec
   range and sampled.
3. The code predictor walks codebooks 1..15 *inside* the frame, conditioned on
   the talker hidden state and codebook 0.
4. The 16 codes are embedded and summed with the next text embedding to form
   the input for step 1 of the next iteration.

Step 3 is worth noticing: it is 15 sequential forward passes of a 5-layer model
per frame, against one pass of a 28-layer model in step 1. Whether stage 0 is
dominated by the talker or by the code predictor is a question about launch
overhead rather than FLOPs, and it is why ``--talker-compile-mode
code_predictor`` exists as a separate setting.

Both autoregressive axes sample independently, with the checkpoint's own
defaults (the talker additionally applies a repetition penalty; the code
predictor does not).
"""

from __future__ import annotations

import torch

from ..compilation import build_code_predictor_graphs, build_talker_graphs
from ..config import EngineConfig, SamplingConfig
from ..models.code_predictor import CodePredictor, apply_top_k, sample_gumbel
from ..models.prompt import TalkerPrompt
from ..models.talker import Qwen3TTSTalker
from ..utils import get_logger
from .request import FinishReason, RequestState, RequestTracker

logger = get_logger("talker_runner")


def apply_repetition_penalty(
    logits: torch.Tensor, seen: torch.Tensor, seen_valid: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide the logits of already-emitted ids, as HuggingFace does.

    ``seen`` is ``[rows, window]`` of previously generated codebook-0 ids with
    ``seen_valid`` marking the real entries. Positive logits are divided and
    negative ones multiplied, so the penalty always moves a score toward
    ``-inf`` regardless of sign.

    For TTS this is not the usual "stop the model repeating itself
    textually" -- it is what keeps the talker from locking into a periodic
    codebook-0 pattern, which sounds like a held buzz rather than like repeated
    words. The checkpoint ships 1.05.
    """
    if penalty == 1.0:
        return logits
    gathered = logits.gather(1, seen.clamp(min=0))
    adjusted = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    adjusted = torch.where(seen_valid, adjusted, gathered)
    return logits.scatter(1, seen.clamp(min=0), adjusted)


class TalkerRunner:
    def __init__(
        self,
        talker: Qwen3TTSTalker,
        code_predictor: CodePredictor,
        config: EngineConfig,
        sampling: SamplingConfig,
    ) -> None:
        self.talker = talker
        self.code_predictor = code_predictor
        self.config = config
        self.sampling = sampling
        self.device = talker.device
        self.dtype = talker.dtype
        self.num_quantizers = talker.num_quantizers
        self.eos_id = talker.tc.codec_eos_token_id
        self.codebook_size = code_predictor.config.vocab_size
        self._residual_embed: torch.Tensor | None = None
        self._residual_rows: torch.Tensor | None = None
        self.graphs = build_talker_graphs(talker, config.talker)
        self.predictor_graphs = build_code_predictor_graphs(
            code_predictor, config.talker, config.scheduler.max_num_seqs
        )
        self._warned_seed_vs_graphs = False

    # -- setup --------------------------------------------------------------

    def residual_embedding(self) -> torch.Tensor:
        """``[Q-1, vocab, hidden]``, the code predictor's tables stacked.

        Codebook 0's embedding comes from the talker's own table; codebooks
        1..15 are embedded with the *code predictor's* tables, at talker width.
        Reading them from the wrong module is a silent-quality bug, so the two
        sources are kept visibly distinct here.
        """
        if self._residual_embed is None:
            self._residual_embed = self.code_predictor.stacked_embedding_weight()
        return self._residual_embed

    @torch.inference_mode()
    def prefill(self, tracker: RequestTracker, prompt: TalkerPrompt) -> None:
        tracker.last_hidden = self.talker.prefill(prompt.embeds, tracker.slot)
        tracker.trailing_text = prompt.trailing_text
        tracker.text_offset = 0
        tracker.seq_len = prompt.embeds.shape[0]
        tracker.state = RequestState.DECODING
        if tracker.request.seed is not None:
            tracker.generator = torch.Generator(device=self.device)
            tracker.generator.manual_seed(tracker.request.seed)
            if self.predictor_graphs is not None and not self._warned_seed_vs_graphs:
                # A captured graph cannot draw from a caller-supplied generator,
                # so a seeded request falls back to the eager code predictor --
                # which is 4x slower per frame. Saying so matters because the
                # fallback is invisible otherwise, and a benchmark that seeds
                # its requests for reproducibility ends up measuring a path the
                # server will never take.
                logger.warning(
                    "request %s is seeded, so its code predictor runs eagerly instead of from a "
                    "captured graph; drop the seed to measure the graph path",
                    tracker.request.request_id,
                )
                self._warned_seed_vs_graphs = True

    # -- per-step -----------------------------------------------------------

    def _text_step(self, trackers: list[RequestTracker]) -> torch.Tensor:
        """The text-side embedding each row adds this step.

        In non-streaming mode ``trailing_text`` is a single pad row and every
        step reuses it. In streaming mode it is a queue of the remaining text
        tokens, consumed one per frame and held at the final row once drained.
        """
        rows = []
        for tracker in trackers:
            tail = tracker.trailing_text
            if tail is None or tail.shape[0] == 0:
                rows.append(self.talker.tts_pad_embed.reshape(-1))
                continue
            index = min(tracker.text_offset, tail.shape[0] - 1)
            rows.append(tail[index])
            if tracker.text_offset < tail.shape[0] - 1:
                tracker.text_offset += 1
        return torch.stack(rows, dim=0)

    def _sample_layer0(self, hidden: torch.Tensor, trackers: list[RequestTracker]) -> torch.Tensor:
        logits = self.talker.logits(hidden)
        params = self.sampling
        if params.repetition_penalty != 1.0:
            seen, valid = self._recent_ids(trackers)
            logits = apply_repetition_penalty(logits, seen, valid, params.repetition_penalty)
        if not (params.do_sample and params.temperature > 0):
            return logits.argmax(dim=-1)
        logits = logits / max(params.temperature, 1e-6)
        logits = apply_top_k(logits, params.top_k)
        return sample_gumbel(logits, self._row_generator(trackers)).reshape(-1)

    def _recent_ids(self, trackers: list[RequestTracker]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack each row's penalty window into one padded tensor.

        Padded slots point at id 0, which is never a legal codebook-0 value, and
        are masked out of the update rather than left to shift a real logit.
        """
        window = max((len(t.recent_layer0) for t in trackers), default=0)
        rows = len(trackers)
        if window == 0:
            empty = torch.zeros(rows, 1, dtype=torch.long, device=self.device)
            return empty, torch.zeros_like(empty, dtype=torch.bool)
        packed = torch.zeros(rows, window, dtype=torch.long)
        valid = torch.zeros(rows, window, dtype=torch.bool)
        for row, tracker in enumerate(trackers):
            seen = tracker.recent_layer0
            if seen:
                packed[row, : len(seen)] = torch.tensor(seen, dtype=torch.long)
                valid[row, : len(seen)] = True
        return packed.to(self.device), valid.to(self.device)

    @torch.inference_mode()
    def decode_step(self, trackers: list[RequestTracker]) -> None:
        """Produce one frame for every tracker in ``trackers``.

        ``trackers`` are the running requests in slot order. Sampling and the
        code predictor run on exactly these rows; only the talker forward has to
        cover the whole slot prefix, because that is how its cache is laid out.
        """
        if not trackers:
            return
        hidden = torch.stack([t.last_hidden for t in trackers], dim=0)  # type: ignore[arg-type]
        layer0 = self._sample_layer0(hidden, trackers)

        layer0_embed = self.talker.embed_codec(layer0.reshape(-1, 1)).squeeze(1)
        codes = self._residual_codes(layer0, layer0_embed, hidden, trackers)

        # A frame whose codebook 0 is out of range (EOS, or a control id that
        # slipped the mask) has no meaning for the codec decoder. Zero the whole
        # frame so it decodes to silence if it is ever shipped, and record the
        # stop separately.
        finished = (layer0 < 1) | (layer0 >= self.codebook_size)
        codes = torch.where(finished.reshape(-1, 1), torch.zeros_like(codes), codes)

        next_embeds = self._next_inputs_embeds(layer0_embed, codes, trackers)
        stops = finished.tolist()
        layer0_ids = layer0.tolist()

        keep: list[RequestTracker] = []
        for row, tracker in enumerate(trackers):
            if stops[row]:
                tracker.finish_reason = FinishReason.EOS
                tracker.state = RequestState.DRAINING
                continue
            tracker.frames.append(codes[row : row + 1])
            tracker.frames_generated += 1
            self._note_layer0(tracker, layer0_ids[row])
            if tracker.frames_generated >= self._frame_budget(tracker):
                tracker.finish_reason = FinishReason.MAX_FRAMES
                tracker.state = RequestState.DRAINING
            elif self._repeating(tracker):
                tracker.finish_reason = FinishReason.REPETITION
                tracker.state = RequestState.DRAINING
            else:
                keep.append(tracker)
            tracker.next_embed = next_embeds[row]

        self._advance(keep, trackers, next_embeds)

    def _residual_codes(
        self,
        layer0: torch.Tensor,
        layer0_embed: torch.Tensor,
        hidden: torch.Tensor,
        trackers: list[RequestTracker],
    ) -> torch.Tensor:
        """Codebooks 1..Q-1 for every row, graph-replayed when possible."""
        params = self.sampling
        generator = self._row_generator(trackers)
        if self.predictor_graphs is not None and generator is None:
            return self.predictor_graphs.run(
                layer0,
                layer0_embed,
                hidden,
                do_sample=params.subtalker_do_sample,
                temperature=params.subtalker_temperature,
                top_k=params.subtalker_top_k,
            )
        return self.code_predictor(
            layer0,
            layer0_embed,
            hidden,
            do_sample=params.subtalker_do_sample,
            temperature=params.subtalker_temperature,
            top_k=params.subtalker_top_k,
            generator=generator,
        )

    @staticmethod
    def _row_generator(trackers: list[RequestTracker]) -> torch.Generator | None:
        """A per-request generator only applies when it is the only row.

        Sampling is one batched call, so two seeded requests in the same step
        cannot each get their own stream of randomness. Rather than give one of
        them the other's, a seeded request is only honoured when it is alone in
        the batch -- which is the case for the reproducibility runs that use it.
        """
        if len(trackers) != 1:
            return None
        return trackers[0].generator

    def _advance(
        self,
        keep: list[RequestTracker],
        trackers: list[RequestTracker],
        next_embeds: torch.Tensor,
    ) -> None:
        """Run the next position for rows that continue, refreshing their hidden.

        The batch handed to the talker is the *slot prefix*, not the list of
        survivors: row ``i`` has to be slot ``i`` or the contiguous-prefix cache
        read attends to the wrong request. So a survivor at slot 3 goes to row 3
        even if slots 1 and 2 stopped this step, and the gaps are filled with
        inactive rows whose keys are diverted to the scratch slot.

        ``seq_len`` is the count of positions already written, so it is both the
        index this step's embedding goes to and its rotary position, and it must
        therefore be read *before* being advanced. Incrementing it earlier leaves
        position ``seq_len`` holding zeros and shifts every later rotary position
        by one -- which does not fail. It makes the talker condition on a gap,
        and the audio stays fluent while no longer matching the text.
        """
        if not keep:
            return
        prefix = max(t.slot for t in keep) + 1
        hidden_size = next_embeds.shape[-1]
        embeds = next_embeds.new_zeros(prefix, hidden_size)
        seq_lens = [0] * prefix
        active = [False] * prefix
        for tracker in trackers:
            if tracker.slot < prefix:
                seq_lens[tracker.slot] = tracker.seq_len
        for tracker in keep:
            embeds[tracker.slot] = next_embeds[trackers.index(tracker)]
            seq_lens[tracker.slot] = tracker.seq_len
            active[tracker.slot] = True

        key_len = max(seq_lens) + 1
        hidden = self._forward(embeds, seq_lens, active, key_len)
        for tracker in keep:
            tracker.last_hidden = hidden[tracker.slot]
            tracker.seq_len += 1

    def _forward(
        self, embeds: torch.Tensor, seq_lens: list[int], active: list[bool], key_len: int
    ) -> torch.Tensor:
        if self.graphs is not None:
            return self.graphs.run(embeds, seq_lens, active, key_len)
        return self.talker.decode(
            embeds,
            torch.tensor(seq_lens, dtype=torch.long, device=self.device),
            key_len,
            torch.tensor(active, dtype=torch.bool, device=self.device),
        )

    def _next_inputs_embeds(
        self, layer0_embed: torch.Tensor, codes: torch.Tensor, trackers: list[RequestTracker]
    ) -> torch.Tensor:
        residual = self.residual_embedding()
        rows = codes.shape[0]
        if self._residual_rows is None or self._residual_rows.shape[0] < rows:
            index = torch.arange(self.num_quantizers - 1, device=self.device)
            self._residual_rows = index.unsqueeze(0).expand(max(rows, 1), -1).contiguous()
        row_index = self._residual_rows[:rows]
        gathered = residual[row_index, codes[:, 1:]]
        text = self._text_step(trackers)
        return layer0_embed + gathered.sum(dim=1) + text.to(self.dtype)

    # -- stopping -----------------------------------------------------------

    def _frame_budget(self, tracker: RequestTracker) -> int:
        requested = tracker.request.max_new_tokens
        budget = self.config.talker.max_frames
        if requested is not None:
            budget = min(budget, int(requested))
        # A request cannot outrun its KV cache, whatever it asked for.
        return min(budget, self.config.scheduler.max_model_len - tracker.prompt_len)

    def _note_layer0(self, tracker: RequestTracker, code: int) -> None:
        window = max(self.config.talker.repetition_guard_frames, 64)
        tracker.recent_layer0.append(code)
        if len(tracker.recent_layer0) > window:
            del tracker.recent_layer0[0]

    def _repeating(self, tracker: RequestTracker) -> bool:
        """Detect a stuck talker: the same codebook-0 id N frames running.

        Off by default. It is a safety valve for experiments that change
        sampling -- a talker that latches produces a constant tone and then
        runs to ``max_frames``, burning the whole budget on one bad request.
        """
        guard = self.config.talker.repetition_guard_frames
        if guard <= 0 or len(tracker.recent_layer0) < guard:
            return False
        tail = tracker.recent_layer0[-guard:]
        return all(code == tail[0] for code in tail)
