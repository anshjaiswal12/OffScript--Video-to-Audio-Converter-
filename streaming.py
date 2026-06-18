import asyncio
import uuid
from typing import Any


class StreamHub:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}

    def open(self, loop: asyncio.AbstractEventLoop) -> str:
        stream_id = uuid.uuid4().hex
        self._queues[stream_id] = asyncio.Queue()
        self._loops[stream_id] = loop
        return stream_id

    def exists(self, stream_id: str) -> bool:
        return stream_id in self._queues

    def emit(self, stream_id: str, payload: dict[str, Any]) -> None:
        queue = self._queues.get(stream_id)
        loop = self._loops.get(stream_id)
        if not queue or not loop:
            return
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def close(self, stream_id: str, payload: dict[str, Any] | None = None) -> None:
        if stream_id not in self._queues:
            return
        if payload:
            self.emit(stream_id, payload)
        self.emit(stream_id, {"event": "close"})
        self._queues.pop(stream_id, None)
        self._loops.pop(stream_id, None)

    async def listen(self, stream_id: str):
        queue = self._queues.get(stream_id)
        if not queue:
            return
        while True:
            payload = await queue.get()
            if payload.get("event") == "close":
                break
            yield payload


hub = StreamHub()
