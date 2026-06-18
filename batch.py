import uuid
from pathlib import Path
from typing import Any

_batches: dict[str, dict[str, Any]] = {}


def create_batch(
    paths: list[str] | None = None,
    directory: str | None = None,
) -> dict[str, Any]:
    batch_id = uuid.uuid4().hex
    queue: list[str] = []

    if directory:
        root = Path(directory).expanduser().resolve()
        if root.is_dir():
            queue.extend(str(item) for item in sorted(root.iterdir()) if item.is_file())

    if paths:
        for raw in paths:
            resolved = str(Path(raw).expanduser().resolve())
            if resolved not in queue:
                queue.append(resolved)

    batch = {
        "batch_id": batch_id,
        "status": "queued",
        "total": len(queue),
        "queue": queue,
        "completed": [],
        "failed": [],
    }
    _batches[batch_id] = batch
    return batch


def get_batch(batch_id: str) -> dict[str, Any] | None:
    return _batches.get(batch_id)
