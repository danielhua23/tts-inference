"""An HTTP server for the engine, in as little machinery as the problem allows.

aiohttp rather than FastAPI because the container has aiohttp and the request
schema is four optional strings; a validation framework would be the largest
dependency in the project.

The interesting part is not the HTTP. It is that the engine is synchronous and
single-threaded by design (see ``engine/engine.py``), while a server has to
accept requests at any time and stream several responses at once. The
arrangement below is the smallest one that does not lie about either:

* **One background thread owns the engine.** It loops: drain the submission
  queue, call ``step()``, push the resulting chunks onto per-request queues.
  Nothing else ever touches the engine, so the scheduler's invariants -- slot
  allocation, the dense row prefix, per-request decoder state -- hold without
  locks.
* **Handlers only touch queues.** A request puts a submission on the inbox and
  reads chunks off its own outbox. The event loop is never blocked by model
  work, which is what makes concurrent streaming responses actually stream.
* **Backpressure is the engine's, not the transport's.** The scheduler already
  pauses a talker whose chunks are piling up
  (``max_pending_chunks_per_request``), so the queues here are unbounded on
  purpose: bounding them would add a second, redundant mechanism that
  interacts with the first in ways that are hard to reason about.

One deliberate limitation: a request that the client abandons mid-stream is
still generated to completion. Cancellation means telling the engine thread to
abort a request that may be mid-step, and doing that correctly needs an abort
path through the scheduler. It is a real gap, not an oversight.

Endpoints:

    POST /generate         JSON in, one WAV out
    POST /generate_stream  JSON in, chunked raw PCM out (16-bit LE mono)
    GET  /health           readiness plus engine counters
    GET  /metrics          request and chunk counters
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from ..config import EngineConfig, SamplingConfig, config_summary
from ..engine import TTSEngine, TTSRequest
from ..utils import get_logger
from .protocol import GenerateRequest, encode_wav, pcm16_bytes

logger = get_logger("serving")

#: Pushed onto a request's outbox to mean "no more chunks".
_DONE = object()


@dataclass
class _Submission:
    request: TTSRequest
    outbox: queue.Queue = field(default_factory=queue.Queue)


class EngineThread:
    """Owns the engine and runs its loop on one thread."""

    def __init__(self, engine: TTSEngine, idle_sleep: float = 0.002) -> None:
        self.engine = engine
        self.inbox: queue.Queue[_Submission] = queue.Queue()
        self.outboxes: dict[str, queue.Queue] = {}
        self.idle_sleep = idle_sleep
        self.requests_total = 0
        self.chunks_total = 0
        self.errors_total = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qtts-engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def submit(self, request: TTSRequest) -> queue.Queue:
        submission = _Submission(request)
        self.inbox.put(submission)
        return submission.outbox

    def _loop(self) -> None:
        while not self._stop.is_set():
            admitted = self._drain_inbox()
            try:
                chunks = self.engine.step()
            except Exception:
                # One bad request must not take the server down, and the engine
                # cannot be trusted to be in a usable state afterwards either,
                # so every stream in flight is failed rather than left hanging.
                logger.exception("engine step failed; failing all in-flight requests")
                self.errors_total += 1
                self._fail_all()
                continue
            for chunk in chunks:
                outbox = self.outboxes.get(chunk.request_id)
                if outbox is not None:
                    outbox.put(chunk)
                    self.chunks_total += 1
                    if chunk.finished:
                        outbox.put(_DONE)
                        self.outboxes.pop(chunk.request_id, None)
            if not admitted and not chunks and not self.engine.scheduler.has_work():
                # Nothing queued and nothing running: sleep rather than spin, so
                # an idle server does not hold a core at 100%.
                time.sleep(self.idle_sleep)

    def _drain_inbox(self) -> int:
        admitted = 0
        while True:
            try:
                submission = self.inbox.get_nowait()
            except queue.Empty:
                return admitted
            self.outboxes[submission.request.request_id] = submission.outbox
            self.engine.submit(submission.request)
            self.requests_total += 1
            admitted += 1

    def _fail_all(self) -> None:
        for outbox in self.outboxes.values():
            outbox.put(_DONE)
        self.outboxes.clear()

    def stats(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "chunks_total": self.chunks_total,
            "errors_total": self.errors_total,
            "in_flight": len(self.outboxes),
            **{f"sched_{k}": v for k, v in self.engine.scheduler.stats().items()},
        }


async def _drain(outbox: queue.Queue, poll: float = 0.002):
    """Yield chunks from a thread queue without blocking the event loop.

    ``queue.Queue.get`` blocks the whole loop, and ``run_in_executor`` per chunk
    costs a thread hop per few milliseconds of audio. Polling with a short sleep
    is cruder and is what keeps other streams responsive.
    """
    while True:
        try:
            item = outbox.get_nowait()
        except queue.Empty:
            await asyncio.sleep(poll)
            continue
        if item is _DONE:
            return
        yield item


def build_app(engine_thread: EngineThread, engine_config: EngineConfig):
    from aiohttp import web

    engine = engine_thread.engine
    routes = web.RouteTableDef()

    @routes.post("/generate")
    async def generate(http_request: web.Request) -> web.Response:
        try:
            payload = GenerateRequest.parse(await http_request.json())
        except ValueError as exc:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": str(exc)}), content_type="application/json"
            ) from exc

        outbox = engine_thread.submit(payload.to_engine_request())
        pieces = [chunk.audio async for chunk in _drain(outbox) if chunk.audio.numel()]
        audio = torch.cat(pieces) if pieces else torch.zeros(0)
        return web.Response(
            body=encode_wav(audio, engine.sample_rate),
            content_type="audio/wav",
            headers={"X-Sample-Rate": str(engine.sample_rate)},
        )

    @routes.post("/generate_stream")
    async def generate_stream(http_request: web.Request) -> web.StreamResponse:
        try:
            payload = GenerateRequest.parse(await http_request.json())
        except ValueError as exc:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": str(exc)}), content_type="application/json"
            ) from exc

        response = web.StreamResponse(
            headers={
                # Raw PCM rather than a WAV stream: a WAV header needs the total
                # length, which is not known until the talker hits EOS, and
                # writing a placeholder length is how streamed WAVs end up
                # unplayable.
                "Content-Type": "audio/L16",
                "X-Sample-Rate": str(engine.sample_rate),
                "X-Audio-Format": "pcm_s16le_mono",
            }
        )
        response.enable_chunked_encoding()
        await response.prepare(http_request)

        outbox = engine_thread.submit(payload.to_engine_request())
        async for chunk in _drain(outbox):
            if chunk.audio.numel():
                await response.write(pcm16_bytes(chunk.audio))
        await response.write_eof()
        return response

    @routes.get("/health")
    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "sample_rate": engine.sample_rate,
                "config": config_summary(engine_config),
                **engine_thread.stats(),
            }
        )

    @routes.get("/metrics")
    async def metrics(_: web.Request) -> web.Response:
        return web.json_response({**engine_thread.stats(), "timings": engine.timing_summary()})

    app = web.Application()
    app.add_routes(routes)
    return app


def serve(engine_config: EngineConfig, sampling: SamplingConfig, host: str, port: int) -> None:
    from aiohttp import web

    engine = TTSEngine(engine_config, sampling=sampling)
    thread = EngineThread(engine)
    thread.start()
    logger.info("serving on http://%s:%d (POST /generate, /generate_stream)", host, port)
    try:
        web.run_app(build_app(thread, engine_config), host=host, port=port, print=None)
    finally:
        thread.stop()
