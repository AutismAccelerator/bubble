"""
embed.py — text embedding via the configured inference endpoint.
"""

import httpx

from . import config

EMBED_DIM = config.EMBED_DIM

_http = httpx.AsyncClient(timeout=30.0)


async def embed(text: str) -> list[float]:
    """
    Embed text via the configured inference endpoint (OpenAI-compatible /v1/embeddings).
    Returns an EMBED_DIM-dimensional vector as a Python list.
    Configure via: BUBBLE_EMBED_ENDPOINT, BUBBLE_EMBED_DIM.
    """
    response = await _http.post(
        config.EMBED_ENDPOINT,
        json={"input": text},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
