"""
Retrieval pipeline — spec §4.

retrieve(user_id, query) → ranked list of SnapshotNode result dicts.

Flow:
  1. Decompose long queries via decomposer; embed each segment in parallel.
  2. Wide k-NN against SnapshotNode centroids (_CANDIDATES); scores merged by similarity.
  3. Ensure snapshot summaries are valid (lazy LLM if needed).
  4. Rerank candidates with reranker; keep top_k.
  5. For each result: traverse SYNTHESIZES → Episode members (ordered by timestamp).
  6. Translate confidence floats to natural-language labels.
"""

import asyncio
import os

from .contradict import ensure_snapshot_summary
from .db import get_graph, init_graph
from .embed import embed
from .decomposer import decompose
from .rerank import rerank

_LONG_QUERY_WORDS = 30
_CANDIDATES = 3
_TOP_K = 5
_RERANK_ENABLED = os.getenv("BUBBLE_RERANK_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# §4.3  Confidence translation
# ---------------------------------------------------------------------------

def _confidence_label(confidence: float, episodic: bool = False) -> str:
    if episodic:
        return "vivid, unverified by convergence"
    if confidence > 0.75:
        return "established"
    if confidence >= 0.50:
        return "likely"
    if confidence >= 0.30:
        return "uncertain"
    return "contested"


# ---------------------------------------------------------------------------
# ANN on SnapshotNode
# ---------------------------------------------------------------------------

async def _ann_candidates(g, query_vec: list[float], k: int) -> list[tuple[dict, float]]:
    """k-NN against SnapshotNode.centroid. Returns [(snap_dict, score)] descending."""
    try:
        result = await g.query(
            "CALL db.idx.vector.queryNodes('SnapshotNode', 'centroid', $k, vecf32($vec)) "
            "YIELD node, score "
            "RETURN node.id, node.summary, node.valid, score",
            {"k": k, "vec": query_vec},
        )
    except Exception:
        # Index missing (uninitialized graph) — ensure it exists, return empty.
        await init_graph(g.name.removeprefix("bubble:"))
        return []
    out = []
    for row in result.result_set:
        snap = {
            "id":      row[0],
            "summary": row[1],
            "valid":   bool(row[2]),
        }
        out.append((snap, float(row[3])))
    return out


# ---------------------------------------------------------------------------
# Traversal: SnapshotNode → members → edge context
# ---------------------------------------------------------------------------

async def _get_members_with_context(g, snap_id: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (members, context).
    members: Episode dicts belonging to this SnapshotNode, ordered by timestamp.
    context: empty — chain members are already fully captured via SYNTHESIZES.
    """
    result = await g.query(
        "MATCH (snap:SnapshotNode {id: $id})-[:SYNTHESIZES]->(t:Episode) "
        "RETURN t.id, t.summary, t.confidence, t.episodic, t.timestamp "
        "ORDER BY t.timestamp",
        {"id": snap_id},
    )
    members = [
        {
            "id":         row[0],
            "summary":    row[1],
            "confidence": float(row[2]) if row[2] is not None else 0.0,
            "episodic":   bool(row[3]),
        }
        for row in result.result_set
    ]
    return members, []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _retrieve_from_vecs(
    g,
    query: str,
    query_vecs: list[list[float]],
    top_k: int,
) -> list[dict]:
    """
    Core retrieval given pre-computed query vectors.
    ANN on SnapshotNode → lazy summary generation → rerank → traversal → confidence labels.
    """
    results_per_vec = await asyncio.gather(*[
        _ann_candidates(g, qvec, _CANDIDATES)
        for qvec in query_vecs
    ])

    score_map: dict[str, float] = {}
    snap_map:  dict[str, dict]  = {}
    for candidates in results_per_vec:
        for snap, sim in candidates:
            nid = snap["id"]
            score_map[nid] = score_map.get(nid, 0.0) + sim
            snap_map.setdefault(nid, snap)

    if not score_map:
        return []

    candidate_ids   = sorted(score_map, key=lambda nid: score_map[nid], reverse=True)
    candidate_snaps = [snap_map[nid] for nid in candidate_ids]

    # Ensure summaries are valid before reranking (lazy LLM generation).
    # Pass already-fetched valid/summary from ANN to skip the redundant DB query.
    summaries = await asyncio.gather(*[
        ensure_snapshot_summary(g, snap["id"], valid=snap["valid"], summary=snap["summary"])
        for snap in candidate_snaps
    ])
    for snap, summary in zip(candidate_snaps, summaries):
        snap["summary"] = summary or ""

    if _RERANK_ENABLED:
        rerank_scores = await rerank(query, [s["summary"] for s in candidate_snaps])
        ranked = sorted(
            zip(candidate_snaps, rerank_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
    else:
        ranked = [(s, score_map[s["id"]]) for s in candidate_snaps[:top_k]]

    traversals = await asyncio.gather(*[
        _get_members_with_context(g, snap["id"]) for snap, _ in ranked
    ])

    results = []
    for (snap, _), (members, context) in zip(ranked, traversals):
        results.append({
            "id":      snap["id"],
            "summary": snap["summary"],
            "members": [
                {
                    "id":               m["id"],
                    "summary":          m["summary"],
                    "confidence_label": _confidence_label(m["confidence"], m["episodic"]),
                }
                for m in members
            ],
            "context": context,
        })

    return results


async def retrieve(user_id: str, query: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Retrieve relevant SnapshotNodes for a query.

    Returns a ranked list of result dicts:
      id, summary, members [{id, summary, confidence_label}],
      context [{rel, id, summary, confidence_label}]
    """
    g = get_graph(user_id)

    words = query.split()
    if len(words) > _LONG_QUERY_WORDS:
        segments   = await decompose(query)
        query_vecs = list(await asyncio.gather(*[embed(s["text"]) for s in segments]))
    else:
        query_vecs = [await embed(query)]
    return await _retrieve_from_vecs(g, query, query_vecs, top_k)
