"""
_shared.py — pure utility functions shared across the bubble pipeline.

This module no longer holds any LLM client or model constants.
See bubble.llm for the LLM abstraction layer.
See bubble.config for all environment configuration.
"""

from datetime import datetime, timezone

import numpy as np


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
