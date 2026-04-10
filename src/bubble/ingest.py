import asyncio
import logging
import uuid

from .archive import read_segments, write_segment
from ._shared import _now
from .chain import check_new
from .db import get_graph
from .embed import embed
from .decomposer import decompose
from . import config

log = logging.getLogger(__name__)
_EPISODIC_THRESHOLD = config.EPISODIC_THRESHOLD


async def _store_segment(
    g, seg: dict, embedding: list[float], prior: str | None, timestamp: str | None = None
) -> dict:
    node_id = str(uuid.uuid4())
    await g.query(
        "CREATE (n:SegmentNode {"
        "  id: $id,"
        "  raw_text: $text,"
        "  embedding: $emb,"
        "  intensity: $intensity,"
        "  valence: $valence,"
        "  prior: $prior,"
        "  timestamp: $ts"
        "})",
        {
            "id":        node_id,
            "text":      seg["text"],
            "emb":       embedding,
            "intensity": seg["intensity"],
            "valence":   seg["valence"],
            "prior":     prior,
            "ts":        timestamp or _now(),
        },
    )
    return {"type": "SegmentNode", "id": node_id, **seg}


async def _create_episodic_node(
    g, seg: dict, embedding: list[float], prior: str | None, timestamp: str
) -> dict:
    """Create an episodic Episode in the graph. Does not write to archive."""
    episode_id = str(uuid.uuid4())
    await g.query(
        "CREATE (t:Episode {"
        "  id: $id,"
        "  summary: $summary,"
        "  confidence: $confidence,"
        "  centroid: vecf32($centroid),"
        "  valence: $valence,"
        "  episodic: true,"
        "  timestamp: $ts"
        "})",
        {
            "id":         episode_id,
            "summary":    seg["text"],
            "confidence": seg["intensity"],
            "centroid":   embedding,
            "valence":    seg["valence"],
            "ts":         timestamp,
        },
    )
    return {"type": "Episode", "id": episode_id, "episodic": True, **seg}


async def _store_episodic(
    g, user_id: str, seg: dict, embedding: list[float], prior: str | None
) -> dict:
    """High-intensity segment: write JSONL directly, create Episode — no SegmentNode."""
    ts = _now()
    write_segment(
        user_id,
        text=seg["text"],
        prior=prior,
        intensity=seg["intensity"],
        valence=seg["valence"],
        timestamp=ts,
    )
    return await _create_episodic_node(g, seg, embedding, prior, ts)


async def _route_segments(
    user_id: str, segments: list[dict], embeddings: list[list[float]], prior: str | None
) -> list[dict]:
    g = get_graph(user_id)

    episodic_items = [(i, seg, emb) for i, (seg, emb) in enumerate(zip(segments, embeddings))
                      if seg["intensity"] >= _EPISODIC_THRESHOLD]
    regular_items  = [(i, seg, emb) for i, (seg, emb) in enumerate(zip(segments, embeddings))
                      if seg["intensity"] < _EPISODIC_THRESHOLD]

    regular_nodes = await asyncio.gather(
        *[_store_segment(g, seg, emb, prior) for _, seg, emb in regular_items]
    )

    episodic_nodes = []
    for _, seg, emb in episodic_items:
        node = await _store_episodic(g, user_id, seg, emb, prior)
        await check_new(user_id, node["id"])
        episodic_nodes.append(node)

    results: list[dict | None] = [None] * len(segments)
    for (i, _, _), node in zip(regular_items, regular_nodes):
        results[i] = node
    for (i, _, _), node in zip(episodic_items, episodic_nodes):
        results[i] = node
    return [r for r in results if r is not None]


async def replay(user_id: str) -> dict:
    """
    Rebuild graph state for a user from their JSONL archive.

    Re-embeds every archived segment, routes episodic entries directly to
    Episodes (no re-archive write), stores regular entries as SegmentNodes,
    then runs promote() to form L0→L1 clusters.

    Returns {"replayed": n, "promoted": n}.
    """
    from .promote import promote

    entries = list(read_segments(user_id))
    if not entries:
        return {"replayed": 0, "promoted": 0}

    g = get_graph(user_id)
    embeddings = await asyncio.gather(*[embed(e["text"]) for e in entries])

    episodic_items = [(e, emb) for e, emb in zip(entries, embeddings) if e["intensity"] >= _EPISODIC_THRESHOLD]
    regular_items  = [(e, emb) for e, emb in zip(entries, embeddings) if e["intensity"] < _EPISODIC_THRESHOLD]

    await asyncio.gather(*[
        _store_segment(
            g,
            {"text": e["text"], "intensity": e["intensity"], "valence": e["valence"]},
            emb, e.get("prior"), e["timestamp"],
        )
        for e, emb in regular_items
    ])

    for e, emb in sorted(episodic_items, key=lambda x: x[0].get("timestamp", "")):
        node = await _create_episodic_node(
            g,
            {"text": e["text"], "intensity": e["intensity"], "valence": e["valence"]},
            emb, e.get("prior"), e["timestamp"],
        )
        await check_new(user_id, node["id"])
    promoted = await promote(user_id)

    return {"replayed": len(entries), "promoted": len(promoted)}


async def ingest(user_id: str, message: str, prior: str | None = None) -> list[dict]:
    """
    Decompose a message, embed each segment, and route.

    prior: optional conversational context the user is responding to.
    Returns a list of created node descriptors.
    """
    segments = await decompose(message, prior)
    log.info("segments: %s", segments)
    embeddings = list(await asyncio.gather(*[embed(seg["text"]) for seg in segments]))
    return await _route_segments(user_id, segments, embeddings, prior)
