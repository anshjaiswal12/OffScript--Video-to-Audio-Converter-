import logging
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
            queue.extend(
                str(item) for item in sorted(root.iterdir()) if item.is_file()
            )
        else:
            logger.warning("Batch directory does not exist: %s", directory)

    if paths:
        for raw in paths:
            resolved = str(Path(raw).expanduser().resolve())
            if resolved not in queue:
                queue.append(resolved)

    batch: dict[str, Any] = {
        "batch_id": batch_id,
        "status": "queued",
        "total": len(queue),
        "queue": queue,
        "completed": [],
        "failed": [],
    }
    _batches[batch_id] = batch
    logger.info("Batch %s created with %d file(s).", batch_id, len(queue))
    return batch


def get_batch(batch_id: str) -> dict[str, Any] | None:
    return _batches.get(batch_id)


def record_batch_result(
    batch_id: str,
    file_path: str,
    *,
    success: bool,
    transcript_path: str | None = None,
    error: str | None = None,
) -> None:
    """Record the outcome of a single file in a batch.

    Called by the transcription endpoint once a file finishes (success or failure).
    """
    batch = _batches.get(batch_id)
    if not batch:
        return
    entry: dict[str, Any] = {"file": file_path}
    if success:
        entry["transcript"] = transcript_path
        batch["completed"].append(entry)
    else:
        entry["error"] = error
        batch["failed"].append(entry)

    done = len(batch["completed"]) + len(batch["failed"])
    if done >= batch["total"]:
        batch["status"] = "done"
    else:
        batch["status"] = "processing"

    logger.debug(
        "Batch %s: %d/%d done (%d failed).",
        batch_id, done, batch["total"], len(batch["failed"]),
    )


def build_transcripts_zip(filenames: list[str], outputs_dir: Path) -> bytes | None:
    """Build an in-memory ZIP of the requested transcript files.

    Returns the raw bytes or None if no files were found on disk.
    """
    buffer = BytesIO()
    added = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for raw in filenames:
            path = outputs_dir / Path(raw).name
            if path.is_file():
                archive.write(path, arcname=path.name)
                added += 1
            else:
                logger.warning("Transcript not found for ZIP: %s", raw)
    if not added:
        return None
    return buffer.getvalue()
