import asyncio
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Streams idle longer than this (seconds) are pruned by the maintenance loop.
STREAM_MAX_AGE_SECONDS = 3600
# Maximum events buffered per stream before back-pressure kicks in.
STREAM_QUEUE_MAX = 512


class StreamHub:
    """Thread-safe SSE hub: worker threads emit events; async consumers listen."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._created_at: dict[str, float] = {}

    def open(self, loop: asyncio.AbstractEventLoop) -> str:
        stream_id = uuid.uuid4().hex
        self._queues[stream_id] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        self._loops[stream_id] = loop
        self._created_at[stream_id] = time.monotonic()
        logger.debug("SSE stream opened: %s", stream_id)
        return stream_id

    def exists(self, stream_id: str) -> bool:
        return stream_id in self._queues

    def emit(self, stream_id: str, payload: dict[str, Any]) -> None:
        """Emit *payload* to the stream from any thread.  Drops silently if the
        stream is gone or its queue is full (prevents unbounded growth under a
        slow consumer)."""
        queue = self._queues.get(stream_id)
        loop = self._loops.get(stream_id)
        if not queue or not loop:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, payload)
        except asyncio.QueueFull:
            logger.warning("SSE queue full for stream %s — dropping event.", stream_id)
        except RuntimeError:
            # Event loop already closed (server shutdown race)
            pass

    def close(self, stream_id: str, payload: dict[str, Any] | None = None) -> None:
        if stream_id not in self._queues:
            return
        if payload:
            self.emit(stream_id, payload)
        self.emit(stream_id, {"event": "close"})
        self._queues.pop(stream_id, None)
        self._loops.pop(stream_id, None)
        self._created_at.pop(stream_id, None)
        logger.debug("SSE stream closed: %s", stream_id)

    def cleanup_stale(self) -> int:
        """Close streams that have been open longer than STREAM_MAX_AGE_SECONDS.

        Returns the number of streams pruned.
        """
        cutoff = time.monotonic() - STREAM_MAX_AGE_SECONDS
        stale = [sid for sid, ts in self._created_at.items() if ts < cutoff]
        for sid in stale:
            logger.warning("Closing stale SSE stream: %s", sid)
            self.close(sid)
        return len(stale)

    async def listen(self, stream_id: str):
        """Async generator that yields payloads until the stream is closed."""
        queue = self._queues.get(stream_id)
        if not queue:
            return
        while True:
            payload = await queue.get()
            if payload.get("event") == "close":
                break
            yield payload


hub = StreamHub()
