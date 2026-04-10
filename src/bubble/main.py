"""
bubble CLI -- manual test harness for the ingest -> cluster -> promote -> retrieve pipeline.

Usage:
    python -m bubble.main decompose "<message>"
    python -m bubble.main ingest    <user_id> "<message>"
    python -m bubble.main promote   <user_id>
    python -m bubble.main query     <user_id> "<query>"
    python -m bubble.main status    <user_id>
    python -m bubble.main diagnose  <user_id>
    python -m bubble.main reset     <user_id>
"""

import asyncio
import json
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from . import config
from .cluster import get_clusters
from .db import get_graph, init_graph
from .decomposer import decompose
from .ingest import ingest, replay
from .promote import _promo_score, promote
from .retrieve import retrieve


async def _decompose(message: str, prior: str | None = None) -> None:
    """Dry-run the decomposer — no graph writes."""
    segments = await decompose(message, prior)
    for seg in segments:
        bar   = "#" * int(seg["intensity"] * 20)
        route = "episodic" if seg["intensity"] >= 0.6 else ("notable" if seg["intensity"] >= 0.3 else "low")
        print(f"  [{seg['valence']:>3}] {seg['intensity']:.2f} {bar:<20} {route}")
        print(f"        \"{seg['text']}\"")


async def _ingest(user_id: str, message: str, prior: str | None = None) -> None:
    await init_graph(user_id)
    nodes = await ingest(user_id, message, prior)
    print(json.dumps(nodes, indent=2))


async def _promote(user_id: str) -> None:
    results = await promote(user_id)
    if not results:
        print("No clusters met the promotion threshold.")
    else:
        print(json.dumps(results, indent=2))


async def _status(user_id: str) -> None:
    g = get_graph(user_id)

    counts = {}
    for label, cypher in [
        ("SegmentNode (active pool)", "MATCH (n:SegmentNode) RETURN count(n)"),
        ("Episode (episodic)",        "MATCH (n:Episode {episodic: true})  RETURN count(n)"),
        ("Episode (distilled)",       "MATCH (n:Episode {episodic: false}) RETURN count(n)"),
        ("SnapshotNode",              "MATCH (n:SnapshotNode) RETURN count(n)"),
        ("FOLLOWED_BY edges",         "MATCH ()-[r:FOLLOWED_BY]->() RETURN count(r)"),
        ("SYNTHESIZES edges",         "MATCH ()-[r:SYNTHESIZES]->() RETURN count(r)"),
    ]:
        result = await g.query(cypher)
        counts[label] = result.result_set[0][0] if result.result_set else 0

    for k, v in counts.items():
        print(f"  {k:<28} {v}")

    # Show Episode summaries
    episodes = await g.query(
        "MATCH (t:Episode) RETURN t.id, t.summary, t.confidence, t.episodic"
    )
    if episodes.result_set:
        print("\nEpisodes:")
        for row in episodes.result_set:
            tag = "[episodic]" if row[3] else "[distilled]"
            print(f"  {tag} ({row[2]:.3f}) {row[1]}")
            print(f"         id={row[0]}")

    # Show SnapshotNode summaries
    snaps = await g.query(
        "MATCH (s:SnapshotNode) RETURN s.id, s.summary, s.valid"
    )
    if snaps.result_set:
        print("\nSnapshotNodes (L2):")
        for row in snaps.result_set:
            valid = "valid" if row[2] else "stale"
            summary = row[1] or "(pending)"
            print(f"  [{valid}] {summary[:80]}")
            print(f"           id={row[0]}")


async def _query(user_id: str, query: str) -> None:
    await init_graph(user_id)
    results = await retrieve(user_id, query)
    if not results:
        print("No relevant memory found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['summary']}")
        if r.get("members"):
            for m in r["members"]:
                print(f"     member ({m['confidence_label']}) {m['summary']}")
        for ctx in r.get("context", []):
            print(f"     {ctx['rel']} ({ctx['confidence_label']}) {ctx['summary']}")


async def _diagnose(user_id: str, json_out: bool = False) -> None:
    theta            = config.PROMOTE_THRESHOLD
    min_cluster      = config.CLUSTER_MIN_SIZE
    t_similarity     = config.T_SIMILARITY
    cluster_join_sim = config.CLUSTER_JOIN_SIM
    nli_enabled      = config.NLI_ENABLED
    nli_model        = config.NLI_MODEL

    g = get_graph(user_id)

    # ── Layer 0 ──────────────────────────────────────────────────
    clusters, pool_result = await asyncio.gather(
        get_clusters(user_id),
        g.query("MATCH (n:SegmentNode) RETURN count(n)"),
    )
    total_active = pool_result.result_set[0][0] if pool_result.result_set else 0

    cluster_rows = []
    for label, nodes in sorted(clusters.items()):
        score, avg_int = _promo_score(nodes)
        cluster_rows.append({
            "label":         label,
            "n":             len(nodes),
            "avg_intensity": round(avg_int, 4),
            "t_promo_score": round(score,   4),
            "action":        "PROMOTE" if score > theta else "skip",
        })
    clustered = sum(r["n"] for r in cluster_rows)

    # ── Episode pairwise distances ───────────────────────────────
    result = await g.query("MATCH (t:Episode) RETURN t.id, t.summary, t.centroid")
    episodes = [(r[0], r[1], r[2]) for r in result.result_set if r[2] is not None]
    episode_distances = None
    if len(episodes) >= 2:
        centroids = np.array([nd[2] for nd in episodes], dtype=np.float32)
        dmat      = cosine_distances(centroids)
        n_t       = len(episodes)
        upper     = dmat[np.triu_indices(n_t, k=1)]
        pair_list = [
            {
                "distance":             round(float(dmat[i, j]), 4),
                "passes_t_similarity":  bool(dmat[i, j] <= t_similarity),
                "a":                    episodes[i][1],
                "b":                    episodes[j][1],
            }
            for i in range(n_t)
            for j in range(i + 1, n_t)
        ]
        episode_distances = {
            "pairs":                        len(upper),
            "min":                          round(float(upper.min()),  4),
            "mean":                         round(float(upper.mean()), 4),
            "max":                          round(float(upper.max()),  4),
            "pairs_checked_at_t_similarity": int((upper <= t_similarity).sum()),
            "pair_list":                    pair_list,
        }

    data = {
        "constants": {
            "BUBBLE_PROMOTE_THRESHOLD": theta,
            "BUBBLE_CLUSTER_MIN_SIZE":  min_cluster,
            "BUBBLE_T_SIMILARITY":      t_similarity,
            "BUBBLE_CLUSTER_JOIN_SIM":  cluster_join_sim,
            "BUBBLE_NLI_ENABLED":       nli_enabled,
            "BUBBLE_NLI_MODEL":         nli_model,
        },
        "layer0": {
            "active_pool": total_active,
            "clustered":   clustered,
            "noise":       total_active - clustered,
            "clusters":    cluster_rows,
        },
        "episode_distances": episode_distances,
    }

    if json_out:
        print(json.dumps(data, indent=2))
        return

    # ── human-readable output ────────────────────────────────────
    co = data["constants"]
    print("-- Current constants -------------------------------------------")
    print(f"  BUBBLE_PROMOTE_THRESHOLD {co['BUBBLE_PROMOTE_THRESHOLD']}   (t_promo_score promotion threshold)")
    print(f"  BUBBLE_CLUSTER_MIN_SIZE  {co['BUBBLE_CLUSTER_MIN_SIZE']}    (HDBSCAN min_cluster_size)")
    print(f"  BUBBLE_T_SIMILARITY      {co['BUBBLE_T_SIMILARITY']}   (contradiction distance ceiling)")
    print(f"  BUBBLE_CLUSTER_JOIN_SIM  {co['BUBBLE_CLUSTER_JOIN_SIM']}   (ANN fallback sim floor for L2 cluster join)")
    print(f"  BUBBLE_NLI_ENABLED       {co['BUBBLE_NLI_ENABLED']}  (NLI stage in contradiction pipeline)")
    if co["BUBBLE_NLI_ENABLED"]:
        print(f"  BUBBLE_NLI_MODEL         {co['BUBBLE_NLI_MODEL']}")

    l0 = data["layer0"]
    print("\n-- Layer 0 cluster report --------------------------------------")
    if not l0["clusters"]:
        print(f"  No clusters found. Active pool size: {l0['active_pool']}")
        print(f"  (BUBBLE_CLUSTER_MIN_SIZE={co['BUBBLE_CLUSTER_MIN_SIZE']} — lower if pool is large but nothing clusters)")
    else:
        print(f"  Active pool: {l0['active_pool']}  clustered: {l0['clustered']}  noise: {l0['noise']}")
        print()
        print(f"  {'cluster':>7}  {'n':>4}  {'avg_int':>7}  {'t_promo':>7}  {'action'}")
        print(f"  {'-------':>7}  {'--':>4}  {'-------':>7}  {'-------':>7}  {'------'}")
        for cl in l0["clusters"]:
            print(f"  {cl['label']:>7}  {cl['n']:>4}  {cl['avg_intensity']:>7.3f}  {cl['t_promo_score']:>7.4f}  {cl['action']}")
        print()
        print(f"  BUBBLE_PROMOTE_THRESHOLD = {co['BUBBLE_PROMOTE_THRESHOLD']}  -- raise to promote fewer clusters, lower to promote more")

    td = data["episode_distances"]
    print("\n-- Episode pairwise distances (tune BUBBLE_T_SIMILARITY) -------")
    if td is None:
        print("  Need >= 2 Episodes to compute distances.")
    else:
        print(f"  pairs: {td['pairs']}   min: {td['min']:.4f}   mean: {td['mean']:.4f}   max: {td['max']:.4f}")
        print(f"  t_similarity = {co['BUBBLE_T_SIMILARITY']}  -- pairs with distance <= t_similarity enter contradiction check")
        print(f"  {td['pairs_checked_at_t_similarity']}/{td['pairs']} pairs would pass Stage 1 at current t_similarity")
        print()
        for p in td["pair_list"]:
            flag = "*" if p["passes_t_similarity"] else " "
            print(f"  {flag} [{p['distance']:.4f}] {p['a'][:45]}")
            print(f"         <> {p['b'][:45]}")


async def _reset(user_id: str) -> None:
    g = get_graph(user_id)
    await g.query("MATCH (n) DETACH DELETE n")
    print(f"Graph bubble:{user_id} cleared.")

async def _replay(user_id: str) -> None:
    await init_graph(user_id)
    await replay(user_id)

_COMMANDS = {
    "decompose": (1, _decompose, "decompose <message> [prior]"),
    "ingest":    (2, _ingest,    "ingest    <user_id> <message> [prior]"),
    "promote":   (1, _promote,   "promote   <user_id>"),
    "query":     (2, _query,     "query     <user_id> <query>"),
    "diagnose":  (1, _diagnose,  "diagnose  <user_id>"),
    "status":    (1, _status,    "status    <user_id>"),
    "reset":     (1, _reset,     "reset     <user_id>"),
    "replay":     (1, _replay,     "replay     <user_id>"),
}


def main() -> None:
    args = sys.argv[1:]
    json_out = "--json" in args
    if json_out:
        args = [a for a in args if a != "--json"]

    if not args or args[0] not in _COMMANDS:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    n_args, fn, usage = _COMMANDS[cmd]
    if len(args) - 1 < n_args:
        print(f"Usage: python -m bubble.main {usage}")
        sys.exit(1)

    if json_out and cmd == "diagnose":
        asyncio.run(fn(*args[1:], json_out=True))
    else:
        asyncio.run(fn(*args[1:]))


if __name__ == "__main__":
    main()
