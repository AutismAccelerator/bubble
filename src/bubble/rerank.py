"""
rerank.py — document reranking via the configured inference endpoint.
"""

import httpx

from . import config

_http = httpx.AsyncClient(timeout=30.0)


async def rerank(query: str, texts: list[str]) -> list[float]:
    """
    Score (query, document) pairs via the configured inference endpoint
    (OpenAI-compatible /v1/rerank, e.g. Infinity or HF TEI).
    Returns relevance scores in the same order as documents (higher = more relevant).
    Configure via: BUBBLE_RERANK_ENDPOINT.
    """
    response = await _http.post(
        config.RERANK_ENDPOINT,
        json={"query": query, "texts": texts},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response.raise_for_status()
    # Results come back sorted by score; restore original document order by index.
    scores = [0.0] * len(texts)
    for r in response.json():
        scores[r["index"]] = r["score"]
    return scores
