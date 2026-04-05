"""
Chain assignment pipeline — spec §3.8.

When a new Episode is created, assign it to a topic chain:
  1 — ANN top-1 against Episode centroids.
  2 — Similarity threshold check. Below → new isolated SnapshotNode.
  3 — LLM relatedness check (binary). Not related → new isolated SnapshotNode.
  4 — Related → traverse FOLLOWED_BY edges to chain tail, wire FOLLOWED_BY edge, join SnapshotNode.
"""

import os
import uuid

import numpy as np

from ._shared import MODEL, _client, _normalize, _now
from .db import get_graph

_CHAIN_MAX_DISTANCE = float(os.getenv("BUBBLE_CHAIN_MAX_DISTANCE", "0.4"))
_NLI_ENABLED = os.getenv("BUBBLE_ENABLE_NLI", "false").lower() == "true"
_NLI_ENDPOINT = os.getenv("BUBBLE_NLI_ENDPOINT", "http://localhost:8999/predict")

_SNAPSHOT_SYSTEM = """\
A sequence of memory records about the user is listed below, from earliest to most recent.

Rules:
- The most recent memory takes precedence over earlier ones.
- Earlier memory provides historical context.
- Synthesize all memory into a single simplified coherent narrative that represents the full arc.
- Output one concise paragraph.
- No subject, start with verb.
- Do not explain or justify.\
"""

# ---------------------------------------------------------------------------
# Relatedness check (LLM or NLI)
# ---------------------------------------------------------------------------


async def _related(summary_a: str, summary_b: str) -> bool:
    """True if the two summaries are about the same topic.

    BUBBLE_CHAIN_BACKEND=nli  — local NLI model, argmax != neutral → related.
    BUBBLE_CHAIN_BACKEND=llm  — LLM yes/no (default).
    """
    if _NLI_ENABLED:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _NLI_ENDPOINT,
                json={"inputs": [summary_a, summary_b]},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            resp.raise_for_status()
            scores = resp.json()  # [{"label": ..., "score": ...}, ...]
        return max(scores, key=lambda x: x["score"])["label"].lower() != "neutral"

    answer = (
        (
            await _client.messages.create(
                model=MODEL,
                max_tokens=8,
                system="You are a memory topic classifier. Reply with exactly 'yes' or 'no'.",
                messages=[
                    {
                        "role": "user",
                        "content": (f"Are these two beliefs about the same topic or subject?\n\nA: {summary_a}\nB: {summary_b}"),
                    }
                ],
            )
        )
        .content[0]
        .text.strip()
        .lower()
    )
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# Chain traversal
# ---------------------------------------------------------------------------


async def _traverse_to_tail(g, start_id: str) -> str:
    """
    Follow FOLLOWED_BY edges from start_id to the chain tail
    (the node with no outgoing FOLLOWED_BY edge).
    """
    result = await g.query(
        "MATCH (start:Episode {id: $id})-[:FOLLOWED_BY*0..]->(tail:Episode) WHERE NOT (tail)-[:FOLLOWED_BY]->() RETURN tail.id LIMIT 1",
        {"id": start_id},
    )
    if result.result_set:
        return result.result_set[0][0]
    return start_id


# ---------------------------------------------------------------------------
# Graph writers
# ---------------------------------------------------------------------------


async def _wire_follows(g, from_id: str, to_id: str) -> None:
    await g.query(
        "MATCH (a:Episode {id: $a}), (b:Episode {id: $b}) CREATE (a)-[:FOLLOWED_BY]->(b)",
        {"a": from_id, "b": to_id},
    )


# ---------------------------------------------------------------------------
# Graph loaders
# ---------------------------------------------------------------------------


def _rows_to_episode_dicts(rows) -> list[dict]:
    return [
        {
            "id": r[0],
            "summary": r[1],
            "centroid": r[2],
            "episodic": bool(r[3]),
            "timestamp": r[4],
            "valence": r[5] or "neu",
        }
        for r in rows
        if r[2] is not None
    ]


async def _load_node(g, episode_id: str) -> dict | None:
    result = await g.query(
        "MATCH (t:Episode {id: $id}) RETURN t.id, t.summary, t.centroid, t.episodic, t.timestamp, t.valence",
        {"id": episode_id},
    )
    rows = _rows_to_episode_dicts(result.result_set)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# SnapshotNode management
# ---------------------------------------------------------------------------


async def _recompute_snapshot_centroid(g, snap_id: str) -> None:
    result = await g.query(
        "MATCH (snap:SnapshotNode {id: $id})-[:SYNTHESIZES]->(t:Episode) RETURN t.centroid",
        {"id": snap_id},
    )
    centroids = [row[0] for row in result.result_set if row[0] is not None]
    if not centroids:
        return
    new_centroid = _normalize(np.array(centroids, dtype=np.float32).mean(axis=0))
    await g.query(
        "MATCH (snap:SnapshotNode {id: $id}) SET snap.centroid = vecf32($centroid)",
        {"id": snap_id, "centroid": new_centroid},
    )


async def _join_snapshot(g, snap_id: str, new_node_id: str) -> None:
    await g.query(
        "MATCH (snap:SnapshotNode {id: $snap_id}), (t:Episode {id: $tid}) CREATE (snap)-[:SYNTHESIZES]->(t) SET snap.valid = false",
        {"snap_id": snap_id, "tid": new_node_id},
    )
    await _recompute_snapshot_centroid(g, snap_id)


async def _create_snapshot(g, episode_id: str, summary: str, centroid: list[float]) -> None:
    snap_id = str(uuid.uuid4())
    await g.query(
        "CREATE (snap:SnapshotNode {  id: $id, summary: $summary, centroid: vecf32($centroid), valid: true, timestamp: $ts})",
        {"id": snap_id, "summary": summary, "centroid": centroid, "ts": _now()},
    )
    await g.query(
        "MATCH (snap:SnapshotNode {id: $snap_id}), (t:Episode {id: $tid}) CREATE (snap)-[:SYNTHESIZES]->(t)",
        {"snap_id": snap_id, "tid": episode_id},
    )


# ---------------------------------------------------------------------------
# Lazy snapshot summary generation
# ---------------------------------------------------------------------------


async def ensure_snapshot_summary(
    g,
    snap_id: str,
    *,
    valid: bool | None = None,
    summary: str | None = None,
) -> str | None:
    """
    Return the SnapshotNode summary, generating it lazily if valid=false or summary is null.

    Pass `valid` and `summary` when already fetched (e.g. from ANN query) to skip the
    redundant DB round-trip.

    Single-member chains: copy Episode summary directly (no LLM).
    Multi-member chains: LLM synthesis ordered by timestamp, episodic members last.
    """
    if valid is None or summary is None:
        result = await g.query(
            "MATCH (snap:SnapshotNode {id: $id}) RETURN snap.summary, snap.valid",
            {"id": snap_id},
        )
        if not result.result_set:
            return None
        summary, valid = result.result_set[0]
    if valid and summary:
        return summary

    result = await g.query(
        "MATCH (snap:SnapshotNode {id: $id})-[:SYNTHESIZES]->(t:Episode) RETURN t.id, t.summary, t.episodic, t.timestamp",
        {"id": snap_id},
    )
    members = [
        {
            "id": r[0],
            "summary": r[1],
            "episodic": bool(r[2]),
            "timestamp": r[3] or "",
        }
        for r in result.result_set
    ]

    if not members:
        return None

    if len(members) == 1:
        new_summary = members[0]["summary"]
    else:
        non_ep = sorted([m for m in members if not m["episodic"]], key=lambda m: m["timestamp"])
        ep = sorted([m for m in members if m["episodic"]], key=lambda m: m["timestamp"])
        ordered = non_ep + ep

        count = len(ordered)
        lines = []
        for i, m in enumerate(ordered):
            tag = " [episodic]" if m["episodic"] else ""
            if i == 0:
                label = f"Belief 1 (earliest){tag}"
            elif i == count - 1:
                label = f"Belief {i + 1} (most recent){tag}"
            else:
                label = f"Belief {i + 1}{tag}"
            lines.append(f"{label}: {m['summary']}")

        new_summary = (
            (
                await _client.messages.create(
                    model=MODEL,
                    max_tokens=256,
                    system=_SNAPSHOT_SYSTEM,
                    messages=[{"role": "user", "content": "\n\n".join(lines)}],
                )
            )
            .content[0]
            .text.strip()
        )

    await g.query(
        "MATCH (snap:SnapshotNode {id: $id}) SET snap.summary = $summary, snap.valid = true",
        {"id": snap_id, "summary": new_summary},
    )
    return new_summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check_new(user_id: str, new_episode_id: str) -> None:
    """
    Assign a newly created Episode to a topic chain.
    Called by promote() and ingest._store_episodic() after Episode creation.
    """
    g = get_graph(user_id)
    new_node = await _load_node(g, new_episode_id)
    if new_node is None or new_node["centroid"] is None:
        return

    # Step 1 — ANN top-1 (k=3: self may occupy a slot if already indexed)
    result = await g.query(
        "CALL db.idx.vector.queryNodes('Episode', 'centroid', 2, vecf32($vec)) YIELD node, score WHERE node.id <> $id RETURN node.id, node.summary, score LIMIT 1",
        {"vec": new_node["centroid"], "id": new_episode_id},
    )

    if not result.result_set:
        # No other Episodes exist yet
        await _create_snapshot(g, new_episode_id, new_node["summary"], new_node["centroid"])
        return

    closest_id, closest_summary, score = result.result_set[0]

    # Step 2 — Similarity threshold
    if score > _CHAIN_MAX_DISTANCE:
        await _create_snapshot(g, new_episode_id, new_node["summary"], new_node["centroid"])
        return

    # Step 3 — relatedness check
    if not await _related(new_node["summary"], closest_summary):
        await _create_snapshot(g, new_episode_id, new_node["summary"], new_node["centroid"])
        return

    # Step 4 — Append to chain
    tail_id = await _traverse_to_tail(g, closest_id)
    await _wire_follows(g, tail_id, new_episode_id)

    snap_result = await g.query(
        "MATCH (snap:SnapshotNode)-[:SYNTHESIZES]->(t:Episode {id: $id}) RETURN snap.id",
        {"id": closest_id},
    )
    if snap_result.result_set:
        await _join_snapshot(g, snap_result.result_set[0][0], new_episode_id)
    else:
        await _create_snapshot(g, new_episode_id, new_node["summary"], new_node["centroid"])
