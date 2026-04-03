import os

import httpx
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv(
    "BUBBLE_EMBED_MODEL", "jinaai/jina-embeddings-v5-text-nano-clustering"
)
EMBED_DIM = int(os.getenv("BUBBLE_EMBED_DIM", "768"))
_EMBED_ENDPOINT = os.getenv("BUBBLE_EMBED_ENDPOINT", "http://localhost:8997")

_http = httpx.AsyncClient(timeout=30.0)


async def embed(text: str) -> list[float]:
    """
    Embed text via the configured inference endpoint (OpenAI-compatible /v1/embeddings).
    Returns an EMBED_DIM-dimensional vector as a Python list.
    Configure via: BUBBLE_EMBED_ENDPOINT, BUBBLE_EMBED_MODEL, BUBBLE_EMBED_DIM.
    """
    response = await _http.post(
        f"{_EMBED_ENDPOINT}/v1/embeddings",
        json={"model": EMBED_MODEL, "input": text},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
