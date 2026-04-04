import os

import numpy as np
from sklearn.cluster import HDBSCAN

from .db import get_graph

_CLUSTER_MIN_SIZE = int(os.getenv("BUBBLE_CLUSTER_MIN_SIZE", "3"))
_CLUSTER_DIMS = int(os.getenv("BUBBLE_CLUSTER_DIMS", "128"))


async def get_clusters(user_id: str) -> dict[int, list[dict]]:
    """
    Run HDBSCAN on all SegmentNodes for a user.
    All SegmentNodes in the graph are the active pool — promoted nodes are deleted at promotion.

    Returns {cluster_label: [node_dicts]} — noise (label -1) is excluded.
    Returns empty dict if the pool is smaller than _CLUSTER_MIN_SIZE.

    Each node dict contains: id, raw_text, embedding (768-dim), intensity, valence,
    prior (str|None), timestamp (str).
    """
    g = get_graph(user_id)
    result = await g.query(
        "MATCH (n:SegmentNode) "
        "RETURN n.id, n.raw_text, n.embedding, n.intensity, n.valence, n.prior, n.timestamp"
    )

    if not result.result_set:
        return {}

    nodes = [
        {
            "id":        row[0],
            "raw_text":  row[1],
            "embedding": row[2],
            "intensity": row[3],
            "valence":   row[4],
            "prior":     row[5],
            "timestamp": row[6],
        }
        for row in result.result_set
    ]

    if len(nodes) < _CLUSTER_MIN_SIZE:
        return {}

    # Truncate to 128 dims (Matryoshka), then re-normalize.
    # The full 768-dim vector is unit-norm, but the truncated prefix is not —
    # re-normalizing restores the cosine distance relationship for euclidean HDBSCAN.
    raw = np.array([n["embedding"][:_CLUSTER_DIMS] for n in nodes], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    matrix = raw / np.where(norms > 0, norms, 1.0)

    labels = HDBSCAN(
        min_cluster_size=_CLUSTER_MIN_SIZE,
        metric="euclidean",  # equiv to cosine on unit vectors
    ).fit_predict(matrix)

    clusters: dict[int, list[dict]] = {}
    for node, label in zip(nodes, labels):
        if label == -1:
            continue  # noise — accumulates at Layer 0, never promotes
        clusters.setdefault(int(label), []).append(node)

    return clusters
