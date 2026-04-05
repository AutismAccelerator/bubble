import os

import httpx
from dotenv import load_dotenv

load_dotenv()

EMBED_DIM = int(os.getenv("BUBBLE_EMBED_DIM", "768"))
_EMBED_ENDPOINT = os.getenv("BUBBLE_EMBED_ENDPOINT", "http://localhost:8997/v1/embeddings")

_http = httpx.AsyncClient(timeout=30.0)


async def embed(text: str) -> list[float]:
    """
    Embed text via the configured inference endpoint (OpenAI-compatible /v1/embeddings).
    Returns an EMBED_DIM-dimensional vector as a Python list.
    Configure via: BUBBLE_EMBED_ENDPOINT, BUBBLE_EMBED_MODEL, BUBBLE_EMBED_DIM.
    """
    response = await _http.post(
        _EMBED_ENDPOINT,
        json={"input": text},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
