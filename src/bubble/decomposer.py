"""
decomposer.py — segment decomposition via the configured LLM provider.
"""

import re
import logging

from .llm import get_llm

log = logging.getLogger(__name__)

# Paste detection thresholds
_PASTE_MIN_WORDS = 100
_PASTE_MAX_PRONOUN_RATIO = 0.02  # first-person pronouns / total words
_FIRST_PERSON = re.compile(
    r"\b(i|me|my|mine|myself|we|our|ours|ourselves)\b", re.IGNORECASE
)


def _is_paste(text: str) -> bool:
    words = text.split()
    if len(words) < _PASTE_MIN_WORDS:
        return False
    pronoun_ratio = len(_FIRST_PERSON.findall(text)) / len(words)
    return pronoun_ratio < _PASTE_MAX_PRONOUN_RATIO


async def decompose(message: str, prior: str | None = None) -> list[dict]:
    """
    Decompose a user message into segments with intensity and valence.
    Returns list of {text: str, intensity: float, valence: pos|neg|neu}.
    Paste-detected messages are returned as a single near-zero-intensity segment.

    prior: optional conversational context the user is responding to
           (agent reply, group chat message, etc.).
    """
    if _is_paste(message):
        return [{"text": message, "intensity": 0.05, "valence": "neu"}]

    log.info("decompose input: %s", message[:120])
    segments = await get_llm().decompose(message, prior)
    log.info("decompose segments: %s", segments)
    return segments
