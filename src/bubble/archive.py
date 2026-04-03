import json
import os
from pathlib import Path

_ARCHIVE_DIR = os.getenv("BUBBLE_ARCHIVE_DIR", "./data/archive")
_MKDIR_DONE = False


def _path(user_id: str) -> Path:
    global _MKDIR_DONE
    p = Path(_ARCHIVE_DIR)
    if not _MKDIR_DONE:
        p.mkdir(parents=True, exist_ok=True)
        _MKDIR_DONE = True
    return p / f"{user_id}.jsonl"


def read_segments(user_id: str):
    """Yield all archived segment records for a user."""
    path = _path(user_id)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_segment(
    user_id: str,
    *,
    text: str,
    prior: str | None,
    intensity: float,
    valence: str,
    timestamp: str,
) -> None:
    """Append one segment record to the user's JSONL archive."""
    entry = {
        "text":      text,
        "prior":     prior,
        "intensity": intensity,
        "valence":   valence,
        "timestamp": timestamp,
    }
    with _path(user_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
