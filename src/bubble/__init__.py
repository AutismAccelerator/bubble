"""
bubble — Hierarchical Memory Consolidation System

Typical agent usage
-------------------
    import bubble

    # Once, when a user session starts:
    await bubble.init_graph(user_id)

    # On every user message — retrieve and store in one call (preferred):
    result = await bubble.observe(user_id, message, prior=agent_reply)
    context = result["retrieved"]  # SnapshotNode results relevant to this message
    stored  = result["stored"]     # ingested node descriptors

    # Or separately:
    await bubble.process(user_id, message, prior=agent_reply)
    context = await bubble.retrieve(user_id, query)

    # Periodically (runs HDBSCAN + promotion):
    await bubble.consolidate(user_id)

    # retrieved is a list of dicts:
    #   {id, summary, members: [{id, summary, confidence_label}],
    #    context: [{rel, id, summary, confidence_label}]}
"""

import asyncio

from .db import get_graph, init_graph
from .embed import embed as _embed
from .decomposer import decompose as _decompose
from .ingest import _route_segments, ingest, replay
from .promote import promote
from .retrieve import _retrieve_from_vecs, retrieve


async def observe(user_id: str, message: str, prior: str | None = None, top_k: int = 3, verbose: bool = False) -> dict:
    """
    Decompose once, retrieve relevant memories, then store — all in a single call.

    Shares the decompose+embed step between retrieval and ingestion.
    Retrieval runs before storage so newly ingested segments don't appear in results.

    Returns:
      {
        "retrieved": [...],  # same format as retrieve()
        "stored":    [...],  # same format as process()
      }
    """
    segments = await _decompose(message, prior)
    embeddings = list(await asyncio.gather(*[_embed(s["text"]) for s in segments]))

    g = get_graph(user_id)
    stored = await _route_segments(user_id, segments, embeddings, prior)
    retrieved = await _retrieve_from_vecs(g, message, embeddings, top_k, verbose)
    return {"retrieved": retrieved, "stored": stored}


async def process(user_id: str, message: str, prior: str | None = None) -> list[dict]:
    """
    Ingest a message into the user's memory graph.

    Routes each segment to:
      - Episodic Episode (intensity >= 0.6): JSONL + Layer 1 node immediately
      - Layer 0 active pool (everything else): waits for consolidate()

    prior: optional conversational context the user is responding to.
    Returns the list of created node descriptors.
    """
    nodes = await ingest(user_id, message, prior)
    await promote(user_id)
    return nodes


async def consolidate(user_id: str) -> dict:
    """
    Run the full consolidation pipeline on a user's graph:
      1. HDBSCAN on the Layer 0 active pool
      2. Promote clusters crossing the t_promo_score threshold to Episodes
         (includes JSONL archival, SegmentNode deletion, L2 assignment)

    Returns:
      {"promoted": [...]}  # newly created Episode descriptors

    Call periodically rather than on every message.
    """
    promoted = await promote(user_id)
    return {"promoted": promoted}


__all__ = [
    "init_graph",
    "observe",
    "process",
    "consolidate",
    "retrieve",
    "ingest",
    "promote",
    "replay",
]
