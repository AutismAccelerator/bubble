import os

import httpx
from dotenv import load_dotenv

load_dotenv()

_RERANK_MODEL    = os.getenv("BUBBLE_RERANK_MODEL",    "cross-encoder/ms-marco-MiniLM-L6-v2")
_RERANK_ENDPOINT = os.getenv("BUBBLE_RERANK_ENDPOINT", "http://localhost:7997")

_http = httpx.AsyncClient(timeout=30.0)


async def rerank(query: str, texts: list[str]) -> list[float]:
    """
    Score (query, document) pairs via the configured inference endpoint
    (OpenAI-compatible /v1/rerank, e.g. Infinity or HF TEI).
    Returns relevance scores in the same order as documents (higher = more relevant).
    Configure via: BUBBLE_RERANK_ENDPOINT, BUBBLE_RERANK_MODEL.
    """
    response = await _http.post(
        f"{_RERANK_ENDPOINT}/rerank",
        json={"model": _RERANK_MODEL, "query": query, "texts": texts},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response.raise_for_status()
    # Results come back sorted by score; restore original document order by index.
    scores = [0.0] * len(texts)
    for r in response.json():
        scores[r["index"]] = r["score"]
    return scores
