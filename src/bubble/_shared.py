import os
from datetime import datetime, timezone

import numpy as np
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("BUBBLE_MODEL", "claude-haiku-4-5-20251001")
_client = AsyncAnthropic()

_SUMMARIZE_SYSTEM = """\
You distill one or more user statements into a single memory record.

Rules:
- Capture the belief, preference, event, or tendency the statements express.
- When multiple statements are given, identify the common pattern they share.
- Write exactly one sentence with no grammatical subject.
- Start with a verb or descriptor that names the belief, event, or pattern.
- Do not explain, qualify, or ask for clarification.\
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(vec: np.ndarray) -> list[float]:
    """L2-normalize a numpy vector and return as a Python list."""
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


def _centroid(nodes: list[dict]) -> list[float]:
    """Mean of source embeddings, L2-normalized."""
    matrix = np.array([n["embedding"] for n in nodes], dtype=np.float32)
    return _normalize(matrix.mean(axis=0))


async def _summarize(nodes: list[dict]) -> str:
    texts = "\n".join(f"- {n['raw_text']}" for n in nodes)
    response = await _client.messages.create(
        model=MODEL,
        max_tokens=128,
        system=_SUMMARIZE_SYSTEM,
        messages=[{"role": "user", "content": texts}],
    )
    return response.content[0].text.strip()
