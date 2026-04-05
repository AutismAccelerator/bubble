import asyncio
import math
import os
import uuid
from collections import Counter

from ._shared import _centroid, _now, _summarize
from .archive import write_segment
from .cluster import get_clusters
from .chain import check_new
from .db import get_graph

_PROMOTE_THRESHOLD = float(os.getenv("BUBBLE_PROMOTE_THRESHOLD", "0.2"))


def _promo_score(nodes: list[dict]) -> tuple[float, float]:
    """
    Compute t_promo_score = log(n) * avg_intensity.
    Returns (score, avg_intensity).
    """
    n = len(nodes)
    avg_intensity = sum(nd["intensity"] for nd in nodes) / n
    if n < 2:
        return 0.0, avg_intensity
    return math.log(n) * avg_intensity, avg_intensity


def _dominant_valence(nodes: list[dict]) -> str:
    counts = Counter(n.get("valence") or "neu" for n in nodes)
    return counts.most_common(1)[0][0] if counts else "neu"


async def promote(user_id: str, theta: float = _PROMOTE_THRESHOLD) -> list[dict]:
    """
    Evaluate all Layer 0 clusters for promotion to Layer 1.

    For each cluster where t_promo_score > theta:
      - LLM-summarize the cluster into a new immutable Episode
      - Write each SegmentNode to JSONL archive
      - Delete SegmentNodes from graph
      - Run check_new (chain assignment + L2 SnapshotNode)

    Returns a list of promoted Episode descriptors.
    """
    clusters = await get_clusters(user_id)
    if not clusters:
        return []

    g = get_graph(user_id)

    qualifying: list[tuple[list[dict], float, float, list[float]]] = []
    for nodes in clusters.values():
        score, avg_intensity = _promo_score(nodes)
        if score > theta:
            qualifying.append((nodes, score, avg_intensity, _centroid(nodes)))

    if not qualifying:
        return []

    summaries = await asyncio.gather(*[_summarize(nodes) for nodes, *_ in qualifying])

    ts = _now()
    episode_ids = [str(uuid.uuid4()) for _ in qualifying]

    await asyncio.gather(*[
        g.query(
            "CREATE (t:Episode {"
            "  id: $id,"
            "  summary: $summary,"
            "  confidence: $confidence,"
            "  centroid: vecf32($centroid),"
            "  valence: $valence,"
            "  episodic: false,"
            "  timestamp: $ts"
            "})",
            {
                "id":         tid,
                "summary":    summary,
                "confidence": avg_intensity,
                "centroid":   centroid,
                "valence":    _dominant_valence(nodes),
                "ts":         ts,
            },
        )
        for (nodes, _, avg_intensity, centroid), summary, tid
        in zip(qualifying, summaries, episode_ids)
    ])

    for (nodes, *_), tid in zip(qualifying, episode_ids):
        for node in nodes:
            write_segment(
                user_id,
                text=node["raw_text"],
                prior=node.get("prior"),
                intensity=node["intensity"],
                valence=node["valence"],
                timestamp=node.get("timestamp") or ts,
            )

    all_seg_ids = [n["id"] for nodes, *_ in qualifying for n in nodes]
    await g.query(
        "UNWIND $ids AS sid MATCH (s:SegmentNode {id: sid}) DELETE s",
        {"ids": all_seg_ids},
    )

    await asyncio.gather(*[check_new(user_id, tid) for tid in episode_ids])

    return [
        {
            "id":            tid,
            "summary":       summary,
            "confidence":    round(avg_intensity, 4),
            "t_promo_score": round(score, 4),
            "n_segments":    len(nodes),
        }
        for (nodes, score, avg_intensity, _), summary, tid
        in zip(qualifying, summaries, episode_ids)
    ]
