"""
archive.py — JSONL segment archive for bubble's event-sourcing replay.
"""

import json
from pathlib import Path

from . import config


def _path(user_id: str) -> Path:
    p = Path(config.ARCHIVE_DIR)
    p.mkdir(parents=True, exist_ok=True)
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
