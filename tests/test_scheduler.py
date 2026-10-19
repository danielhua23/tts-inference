"""Chunk emission, slot allocation and chunk dispatch, without a checkpoint.

These are the parts of the engine whose bugs do not look like bugs. A chunk
policy that emits one frame twice produces audio that is fluent and slightly
wrong; a slot pool that hands out a high index while a low one is free produces
correct audio at a fraction of the throughput. Both survive a listening test,
so they are pinned here instead.
"""

from __future__ import annotations

import pytest
import torch

from qtts.config import Code2WavConfig, EngineConfig, SchedulerConfig
from qtts.engine.chunking import ChunkPolicy, build_policy, dynamic_initial_chunk, max_initial_for_chunk
from qtts.engine.request import RequestState, RequestTracker, TTSRequest
from qtts.engine.scheduler import Scheduler, SlotPool
from qtts.serving.protocol import GenerateRequest, encode_wav, pcm16_bytes


def _drain(policy: ChunkPolicy, total_frames: int) -> list[tuple[int, int, bool]]:
    """Feed frames one at a time and collect every emission.

    Mimics the engine loop: one frame per talker step, poll the policy after
    each, then poll once more with ``finished=True``.
    """
    emissions = []
    for available in range(1, total_frames + 1):
        plan = policy.plan(available, finished=False)
        if plan.emit:
            emissions.append((plan.new_frames, plan.left_context, plan.final))
    while not policy.done:
        plan = policy.plan(total_frames, finished=True)
        if not plan.emit:
            break
        emissions.append((plan.new_frames, plan.left_context, plan.final))
    return emissions


# -- chunk policy -----------------------------------------------------------


@pytest.mark.parametrize("total", [1, 4, 5, 6, 29, 30, 31, 74, 100])
def test_every_frame_is_emitted_exactly_once(total: int) -> None:
    """The property that matters: no frame dropped, none sent twice.

    Off-by-one errors here are audible as a stutter or a clipped syllable at a
    chunk boundary, and the boundary cases are the ones that hit it -- a
    request that ends exactly on a boundary, one frame before, one frame after.
    """
    policy = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5)
    emissions = _drain(policy, total)
    assert sum(new for new, _, _ in emissions) == total
    assert emissions[-1][2] is True, "the last emission must be marked final"
    assert sum(1 for _, _, final in emissions if final) == 1


def test_first_chunk_is_the_initial_size() -> None:
    """Time-to-first-audio is set here, so the first boundary is pinned."""
    policy = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5)
    assert [new for new, _, _ in _drain(policy, 100)] == [5, 25, 25, 25, 20]


def test_a_request_shorter_than_the_first_chunk_goes_out_whole() -> None:
    """Three frames must not become a 5-frame wait and then a remainder.

    A request that finishes before its first boundary has nothing to gain by
    waiting for it, and splitting it would double its time-to-first-audio for
    no throughput in return.
    """
    policy = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5)
    assert _drain(policy, 3) == [(3, 0, True)]


def test_an_empty_request_still_gets_a_final_chunk() -> None:
    """The consumer is waiting on ``finished``; it has to arrive even at zero
    frames, or a request that produced no audio never closes its stream."""
    policy = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5)
    plan = policy.plan(0, finished=True)
    assert (plan.emit, plan.new_frames, plan.final) == (True, 0, True)
    assert policy.done


def test_the_final_chunk_is_handed_out_once() -> None:
    """The engine polls every request every step, including after it finished."""
    policy = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5)
    _drain(policy, 30)
    for _ in range(5):
        assert policy.plan(30, finished=True).emit is False


def test_ramp_follows_the_written_schedule() -> None:
    """90 frames is exactly 5 + 10 + 25 + 25 + 25, so the schedule consumes
    everything and the trailing emission is the empty end-of-stream marker."""
    policy = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5, ramp=(5, 10, 25))
    emissions = _drain(policy, 90)
    assert [new for new, _, _ in emissions] == [5, 10, 25, 25, 25, 0]
    assert sum(new for new, _, _ in emissions) == 90


def test_left_context_is_only_requested_for_the_stateless_decoder() -> None:
    """Sending history to the stateful decoder would decode it twice, and the
    runner asserts against it, so the request has to originate here."""
    stateful = ChunkPolicy(chunk_frames=25, left_context_frames=25, initial_chunk_frames=5)
    assert all(context == 0 for _, context, _ in _drain(stateful, 100))

    stateless = ChunkPolicy(
        chunk_frames=25, left_context_frames=25, initial_chunk_frames=5, needs_left_context=True
    )
    contexts = [context for _, context, _ in _drain(stateless, 100)]
    assert contexts[0] == 0, "the first chunk has no history to re-read"
    assert contexts[1:] == [5, 25, 25, 25], "context is capped by what was actually sent"


def test_initial_chunk_larger_than_the_chunk_is_clamped() -> None:
    policy = ChunkPolicy(chunk_frames=10, left_context_frames=0, initial_chunk_frames=40)
    assert policy.initial_chunk_frames == 10
    assert sum(new for new, _, _ in _drain(policy, 35)) == 35


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_frames": 0},
        {"chunk_frames": 25, "left_context_frames": -1},
        {"chunk_frames": 25, "ramp": (5,)},
        {"chunk_frames": 25, "ramp": (5, 0, 25)},
    ],
)
def test_bad_chunk_settings_are_rejected(kwargs: dict) -> None:
    base = {"chunk_frames": 25, "left_context_frames": 25, "initial_chunk_frames": 5}
    with pytest.raises(ValueError):
        ChunkPolicy(**{**base, **kwargs})


def test_dynamic_initial_grows_the_first_chunk_with_load() -> None:
    """Under light load latency is free; under load a tiny first chunk is a
    decoder call every other stream waits behind."""
    sizes = [dynamic_initial_chunk(active, 8, max_initial_for_chunk(25)) for active in (1, 4, 8)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1], sizes
    assert all(size <= 16 for size in sizes), "must stay below the steady chunk size"


def test_a_ramp_overrides_dynamic_initial() -> None:
    """The ramp states every chunk size including the first; letting load
    rewrite its first entry would silently change a schedule someone wrote."""
    policy = build_policy(
        chunk_frames=25,
        left_context_frames=25,
        initial_chunk_frames=5,
        ramp=(7, 14, 25),
        needs_left_context=False,
        active_requests=8,
        max_num_seqs=8,
        dynamic_initial=True,
    )
    assert [new for new, _, _ in _drain(policy, 46)] == [7, 14, 25, 0]


# -- slot allocation --------------------------------------------------------


def test_slots_are_reused_lowest_index_first() -> None:
    """The talker reads its K/V cache as the row prefix ``[0, n)`` because that
    is a view. A hole in the prefix is computed and thrown away, so a freed low
    slot has to be preferred over an untouched high one."""
    pool = SlotPool(4)
    held = [pool.acquire() for _ in range(4)]
    assert held == [0, 1, 2, 3]
    assert pool.acquire() is None, "capacity is a hard cap"

    pool.release(2)
    pool.release(0)
    assert pool.acquire() == 0
    assert pool.acquire() == 2


def test_releasing_a_slot_twice_is_an_error() -> None:
    """Two requests sharing a slot interleave their K/V, which sounds like one
    voice reading two texts. Failing loudly at the release is far cheaper."""
    pool = SlotPool(2)
    slot = pool.acquire()
    pool.release(slot)
    with pytest.raises(AssertionError):
        pool.release(slot)


# -- chunk dispatch ---------------------------------------------------------


class _FakeRunner:
    """Stands in for Code2WavRunner: the scheduler only needs state hooks."""

    sample_rate = 24000
    stateful = True

    def prepare(self, tracker: RequestTracker) -> None:
        tracker.decoder_state = object()

    def release(self, tracker: RequestTracker) -> None:
        tracker.decoder_state = None


def _scheduler(**scheduler_kwargs) -> Scheduler:
    config = EngineConfig(
        scheduler=SchedulerConfig(max_num_seqs=4, **scheduler_kwargs),
        code2wav=Code2WavConfig(mode="stateful", codec_chunk_frames=10, initial_codec_chunk_frames=5),
    )
    return Scheduler(config, _FakeRunner())


def _admit(scheduler: Scheduler, count: int) -> list[RequestTracker]:
    for index in range(count):
        scheduler.add_request(TTSRequest(text="x", request_id=f"r{index}"), prompt_len=8)
    trackers = scheduler.admit()
    for tracker in trackers:
        tracker.state = RequestState.DECODING
    return trackers


def _generate(tracker: RequestTracker, frames: int) -> None:
    for _ in range(frames):
        tracker.frames.append(torch.zeros(1, 16, dtype=torch.long))
        tracker.frames_generated += 1


def test_admission_stops_at_capacity_and_resumes_on_finish() -> None:
    scheduler = _scheduler()
    trackers = _admit(scheduler, 6)
    assert len(trackers) == 4 and len(scheduler.waiting) == 2

    scheduler.finish(trackers[1])
    admitted = scheduler.admit()
    assert [t.slot for t in admitted] == [1], "the freed low slot goes to the next waiter"
    assert admitted[0].decoder_state is not None, "stage-1 state is allocated at admission"
    assert trackers[1].decoder_state is None, "a finished request must release its state"


def test_idle_rows_inside_the_prefix_are_counted() -> None:
    """A request finishing in the middle leaves a hole the talker still pays
    for. The waste is reported rather than hidden, because it is the reason
    slot allocation is written the way it is."""
    scheduler = _scheduler()
    trackers = _admit(scheduler, 4)
    assert [t.slot for t in scheduler.active_rows()] == [0, 1, 2, 3]
    assert scheduler.max_idle_rows == 0

    trackers[1].state = RequestState.FINISHED
    assert [t.slot for t in scheduler.active_rows()] == [0, 2, 3]
    assert scheduler.max_idle_rows == 1


def test_backpressure_pauses_and_resumes_one_request() -> None:
    scheduler = _scheduler(max_pending_chunks_per_request=2)
    trackers = _admit(scheduler, 2)
    trackers[0].pending_chunks = 2

    scheduler.apply_backpressure()
    assert trackers[0].state == RequestState.THROTTLED
    assert trackers[1].state == RequestState.DECODING, "backpressure is per request"
    assert trackers[0] not in scheduler.active_rows()

    trackers[0].pending_chunks = 0
    scheduler.apply_backpressure()
    assert trackers[0].state == RequestState.DECODING


def test_sync_mode_emits_one_chunk_per_request() -> None:
    """The no-chunking baseline: nothing leaves until that request's talker
    stopped, and then it leaves whole."""
    scheduler = _scheduler(pipeline_mode="sync")
    trackers = _admit(scheduler, 1)
    _generate(trackers[0], 40)

    scheduler.collect_chunks()
    assert scheduler.pending == [], "sync mode must not emit mid-generation"

    trackers[0].state = RequestState.DRAINING
    scheduler.collect_chunks()
    assert [job.new_frames for job in scheduler.pending] == [40]


def test_async_chunk_emits_at_the_policy_boundaries() -> None:
    scheduler = _scheduler()
    trackers = _admit(scheduler, 1)
    tracker = trackers[0]

    emitted = []
    for _ in range(40):
        _generate(tracker, 1)
        scheduler.collect_chunks()
        emitted += [job.new_frames for job in scheduler.next_batch()]
    assert emitted == [5, 10, 10, 10], "35 of 40 frames are past the last boundary"

    tracker.state = RequestState.DRAINING
    scheduler.collect_chunks()
    assert [job.new_frames for job in scheduler.next_batch()] == [5]


def test_emitted_codes_are_the_new_frames_in_order() -> None:
    """``[1, Q, F]`` with the frames the policy asked for and no others. A
    transposed or misaligned slice here is audible as noise, not as a crash."""
    scheduler = _scheduler()
    tracker = _admit(scheduler, 1)[0]
    for index in range(15):
        tracker.frames.append(torch.full((1, 16), index, dtype=torch.long))
        tracker.frames_generated += 1

    scheduler.collect_chunks()
    first = scheduler.next_batch()[0]
    assert first.codes.shape == (1, 16, 5)
    assert first.codes[0, 0].tolist() == [0, 1, 2, 3, 4]

    # The backlog drains at the scheduled size rather than going out as one
    # oversized chunk, then stops: nothing is emitted twice.
    scheduler.collect_chunks()
    second = scheduler.next_batch()[0]
    assert second.codes.shape == (1, 16, 10)
    assert second.codes[0, 0].tolist() == list(range(5, 15))

    scheduler.collect_chunks()
    assert scheduler.pending == [], "all 15 frames are already sent"


def test_stateful_dispatch_takes_one_chunk_at_a_time() -> None:
    """``decode_window`` mutates one request's cache, so it cannot be batched;
    the budget has to be capped regardless of what was configured."""
    scheduler = _scheduler(max_chunk_batch_size=8)
    trackers = _admit(scheduler, 3)
    for tracker in trackers:
        _generate(tracker, 5)
    scheduler.collect_chunks()
    assert len(scheduler.pending) == 3
    assert len(scheduler.next_batch()) == 1


def test_fifo_dispatches_in_emission_order() -> None:
    scheduler = _scheduler(chunk_policy="fifo")
    trackers = _admit(scheduler, 3)
    for tracker in reversed(trackers):
        _generate(tracker, 5)
        scheduler.collect_chunks()

    order = [scheduler.next_batch()[0].tracker.request.request_id for _ in range(3)]
    assert order == ["r2", "r1", "r0"]


def test_deadline_serves_the_stream_closest_to_running_dry() -> None:
    """A listener notices a gap only when the audio already delivered runs out,
    so the request with the least buffered audio goes first -- and a request
    still waiting for its first chunk outranks every request that has audio."""
    scheduler = _scheduler(chunk_policy="deadline")
    trackers = _admit(scheduler, 3)
    for tracker in trackers:
        _generate(tracker, 5)
    scheduler.collect_chunks()

    now = trackers[0].submitted_at
    for tracker, buffered in zip(trackers, (2.0, 0.2, None), strict=True):
        if buffered is None:
            continue
        tracker.first_audio_at = now
        tracker.audio_pieces = [torch.zeros(int(buffered * _FakeRunner.sample_rate))]

    order = [scheduler.next_batch()[0].tracker.request.request_id for _ in range(3)]
    assert order == ["r2", "r1", "r0"], "silent first, then least buffered"


def test_round_robin_gives_every_stream_a_turn() -> None:
    """One request with a backlog must not push the others behind all of it."""
    scheduler = _scheduler(chunk_policy="round_robin", max_pending_chunks_per_request=0)
    trackers = _admit(scheduler, 2)
    for _ in range(25):
        _generate(trackers[0], 1)
        scheduler.collect_chunks()
    _generate(trackers[1], 5)
    scheduler.collect_chunks()
    assert len(scheduler.pending) == 4, "three chunks queued for r0, one for r1"

    order = [scheduler.next_batch()[0].tracker.request.request_id for _ in range(4)]
    assert order[:2] == ["r0", "r1"] or order[:2] == ["r1", "r0"]
    assert order.count("r1") == 1 and order.count("r0") == 3
    assert order[1] == "r1", "r1 must not wait behind r0's whole backlog"


def test_unknown_chunk_policy_is_rejected_at_configuration() -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(chunk_policy="shortest_first")


# -- serving protocol -------------------------------------------------------


def test_flat_and_vllm_omni_payloads_parse_the_same() -> None:
    """vllm-omni nests the fields and wraps each in a one-element list after
    msgspec transport. Accepting both is what makes its example inputs runnable
    here without a translation step."""
    flat = GenerateRequest.parse({"text": "hello", "speaker": "vivian", "max_new_tokens": 64})
    nested = GenerateRequest.parse(
        {"additional_information": {"text": ["hello"], "speaker": ["vivian"], "max_new_tokens": [64]}}
    )
    assert (flat.text, flat.speaker, flat.max_new_tokens) == ("hello", "vivian", 64)
    assert (nested.text, nested.speaker, nested.max_new_tokens) == ("hello", "vivian", 64)


def test_request_defaults_survive_the_round_trip() -> None:
    request = GenerateRequest.parse({"text": "hello"}).to_engine_request()
    assert request.language == "Auto" and request.speaker == ""
    assert request.max_new_tokens is None


@pytest.mark.parametrize(
    "payload", [{"text": ""}, {"text": "   "}, {}, {"text": "x" * 5000}, [], {"additional_information": 3}]
)
def test_bad_payloads_are_rejected(payload) -> None:
    """Including the long one: at 12.5 Hz a huge text holds a slot for minutes,
    so it is rejected rather than admitted and left to starve everyone else."""
    with pytest.raises(ValueError):
        GenerateRequest.parse(payload)


def test_full_scale_audio_does_not_wrap_to_a_click() -> None:
    """1.0 scaled by 32768 overflows int16 and wraps to full-scale negative,
    which is an audible click at the loudest sample of the audio."""
    pcm = torch.frombuffer(
        bytearray(pcm16_bytes(torch.tensor([1.0, -1.0, 0.0, 2.0, -2.0]))), dtype=torch.int16
    )
    assert pcm.tolist() == [32767, -32767, 0, 32767, -32767]


def test_wav_header_describes_the_payload() -> None:
    audio = torch.zeros(2400)
    wav = encode_wav(audio, 24000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) == 44 + audio.numel() * 2
    import io
    import wave

    with wave.open(io.BytesIO(wav)) as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 24000)
        assert handle.getnframes() == audio.numel()
