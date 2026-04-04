# bubble

> *Segments are small bubbles. They stick to one another, grow heavier, and float upward — through an active pool, into crystallized beliefs, and finally into a synthesized view of who you are.*

An event-sourced belief engine for persistent AI agent memory.

---

## The Core Idea

Most memory systems ask: *how do we store this?*  
Bubble asks: *what does this mean, and when does it become a belief?*

A single signal is noise. A signal that keeps returning from different directions, at different moments, with consistent weight — that is belief. Bubble models memory the same way experience shapes personality: not through direct writing, but through accumulation, clustering, and emergence.

Every message is broken into atomic segments and appended to an **immutable event log** (JSONL archive). The graph database is not the source of truth — it is a *derived projection* of that log. Wipe the graph, replay the archive, and the same beliefs re-emerge.

This is **unverifiable replayability**: given the same history of events, a near-identical worldview would crystallize. You cannot step into the same river twice to confirm it, but the river would look the same.

---

## How Memory Forms

```
User message
    │
    ▼
decompose()
  LLM splits the message into atomic segments.
  Each segment gets:
    intensity  0.0–1.0  personal significance (object weight × expression certainty)
    valence    pos | neg | neu
  Pasted content (low pronoun density, >100 words) is stored as-is — no decomposition.
    │
    ├─── intensity ≥ 0.6 ──────────────────────────────────────────────────────────────┐
    │    Episodic path: vivid, committed experience.                                    │
    │    Written immediately to JSONL archive → Episode node → chain assignment.        │
    │                                                                                   │
    └─── intensity < 0.6 ──────────────────────────────────────────────────────────────┤
         Distilled path: softer signal, needs time to prove itself.                    │
         Stored as a SegmentNode in the active pool (Layer 0).                         │
                │                                                                       │
          promote()                                                                     │
          HDBSCAN clusters the pool (128-dim Matryoshka slice).                        │
          Cluster score: log(n) × avg_intensity > θ ?                                  │
                │                                                                       │
               YES                                                                      │
          LLM summarizes → Episode → archived → SegmentNodes deleted.                  │
                │                                                                       ▼
          check_new()  ←──────────────────────────────────────────────────────────────┘
          ANN top-1 against existing Episodes.
          Similarity threshold → LLM relatedness check → chain or isolate.
          Episode appended to an existing chain, or begins its own.
                │
                ▼
          SnapshotNode (Layer 2)
          Synthesized view of the full belief chain.
          Lazy: not generated until first retrieval.
          Invalidated whenever a new Episode joins the chain.
```

### Three Layers

| Layer | Node | Lifecycle | Role |
|-------|------|-----------|------|
| 0 | `SegmentNode` | Ephemeral — deleted on promotion | Active pool, awaiting consolidation |
| 1 | `Episode` | Immutable, HNSW-indexed | A crystallized belief |
| 2 | `SnapshotNode` | Mutable summary, lazy LLM | Retrieval target — synthesized arc of a belief chain |

### Graph Relationships

| Edge | Meaning |
|------|---------|
| `Episode -[:FOLLOWED_BY]-> Episode` | Temporal chain — how a belief has evolved |
| `SnapshotNode -[:SYNTHESIZES]-> Episode` | This snapshot covers these beliefs |
| `Episode -[:SUPPORTS]-> Episode` | Corroborating evidence |
| `Episode -[:CONTRADICTS]-> Episode` | Conflicting evidence |

---

## Notable Mechanics

**Intensity is two-dimensional.** Object significance (how personal and lasting is the subject?) and expression certainty (how committed is the claim?) are both required for a high score. Hedging drags it down. Direct commitment raises it. Neither dimension alone is sufficient.

**Paste detection.** Messages over 100 words with a first-person pronoun ratio below 2% are treated as copied content — stored at near-zero intensity without decomposition.

**HDBSCAN on a Matryoshka slice.** Embeddings are stored at 768 dimensions. Clustering runs on the first 128 (re-normalized), where topical density is cheaper to find without the curse of dimensionality.

**Promotion threshold.** `score = log(n) × avg_intensity > θ`. More segments in a cluster raises the log term; higher average intensity raises the weight. The `diagnose` command shows each cluster's score against the current `θ` so you can tune without guessing.

**Lazy SnapshotNode summaries.** Snapshots are created immediately when a new chain starts, but not summarized until first retrieval. Single-member chains copy the Episode text directly — no LLM call. Multi-member chains synthesize ordered by timestamp (distilled first, episodic last), with the most recent belief taking precedence.

**Chain assignment.** Every new Episode undergoes: ANN top-1 → cosine distance threshold → LLM binary relatedness check. Three gates must pass before an Episode joins an existing chain. If any gate fails, it starts its own isolated chain.

**Replayability.** `replay(user_id)` reads the JSONL archive, re-embeds every segment, reconstructs the graph from scratch, and runs promotion. The archive is the only durable state. The graph is always reconstructible.

---

## Setup

### Requirements

- Python ≥ 3.13
- [FalkorDB](https://docs.falkordb.com/) (graph DB with HNSW vector index)
- An OpenAI-compatible `/v1/embeddings` server (e.g. [Infinity](https://github.com/michaelfeil/infinity))
- Anthropic API key

### Install

```bash
git clone https://github.com/<your-org>/bubble.git
cd bubble
pip install -e .
```

### FalkorDB

```bash
docker run -p 6379:6379 falkordb/falkordb:latest
```

### Embedding server (Infinity example)

```bash
pip install "infinity-emb[all]"
infinity_emb v2 \
  --model-id jinaai/jina-embeddings-v5-text-nano-clustering \
  --port 8997
```

### Environment

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY
```

| Variable | Default | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` | — | Required |
| `FALKORDB_HOST` | `localhost` | |
| `FALKORDB_PORT` | `6379` | |
| `BUBBLE_MODEL` | `claude-sonnet-4-6` | Decomposition and summarization |
| `BUBBLE_EMBED_ENDPOINT` | `http://localhost:8997` | OpenAI-compatible embeddings |
| `BUBBLE_EMBED_MODEL` | `jinaai/jina-embeddings-v5-text-nano-clustering` | |
| `BUBBLE_EMBED_DIM` | `768` | |
| `BUBBLE_RERANK_ENABLED` | `false` | Cross-encoder reranking on retrieval |
| `BUBBLE_RERANK_ENDPOINT` | `http://localhost:8998` | |

**Tuning constants:**

| Variable | Default | Effect |
|----------|---------|--------|
| `BUBBLE_EPISODIC_THRESHOLD` | `0.6` | Intensity floor for episodic storage — segments at or above become Episodes |
| `BUBBLE_PROMOTE_THRESHOLD` | `0.2` | Promotion threshold — lower promotes more clusters, noisier beliefs |
| `BUBBLE_CLUSTER_MIN_SIZE` | `3` | HDBSCAN minimum segments to form a cluster |
| `BUBBLE_CHAIN_MAX_DISTANCE` | `0.4` | Cosine distance ceiling for chain assignment — higher merges more aggressively |

Run `diagnose` to see how your current values behave against actual data before changing them.

---

## CLI

```bash
# See what segments a message would produce (no writes)
python -m bubble.main decompose "<message>"

# Ingest a message
python -m bubble.main ingest <user_id> "<message>"

# Ingest with prior context (what the agent said before this message)
python -m bubble.main ingest <user_id> "<message>" "<prior>"

# Promote Layer 0 clusters to Episodes
python -m bubble.main promote <user_id>

# Query memory
python -m bubble.main query <user_id> "<query>"

# Layer counts and summaries
python -m bubble.main status <user_id>

# Cluster scores + Episode distance matrix (tune θ and T_DISTANCE here)
python -m bubble.main diagnose <user_id>
python -m bubble.main diagnose <user_id> --json

# Clear graph (archive is preserved — replay() can rebuild)
python -m bubble.main reset <user_id>
```

---

## Python API

```python
import asyncio
from bubble.db import init_graph
from bubble.ingest import ingest, replay
from bubble.promote import promote
from bubble.retrieve import retrieve

user_id = "alice"

async def main():
    await init_graph(user_id)

    # Ingest a message
    nodes = await ingest(user_id, "I've decided to quit and start a company.")

    # Ingest with conversational context
    nodes = await ingest(
        user_id,
        "I think I'm ready.",
        prior="Are you sure about the career change?",
    )

    # Promote Layer 0 clusters
    promoted = await promote(user_id)

    # Retrieve relevant memory
    results = await retrieve(user_id, "career goals")
    for r in results:
        print(r["summary"])
        for m in r.get("members", []):
            print(" ", m["confidence_label"], m["summary"])

    # Rebuild graph from JSONL archive
    stats = await replay(user_id)  # {"replayed": N, "promoted": N}

asyncio.run(main())
```

### `ingest(user_id, message, prior=None) → list[dict]`

Decomposes a message, embeds each segment, routes high-intensity segments directly to Layer 1 and the rest to Layer 0. Returns created node descriptors.

### `promote(user_id) → list[dict]`

Runs HDBSCAN on the Layer 0 pool, promotes qualifying clusters to Episodes, archives their SegmentNodes, and runs chain assignment. Returns promoted Episode descriptors.

### `retrieve(user_id, query, top_k=5) → list[dict]`

ANN search on SnapshotNode centroids. Long queries are decomposed first. Triggers lazy LLM summarization for stale snapshots. Returns SnapshotNodes with their member Episodes labeled by confidence.

### `replay(user_id) → dict`

Rebuilds the entire graph from the JSONL archive. Safe to run after `reset`. Returns `{"replayed": N, "promoted": N}`.

---

## Project Structure

```
src/bubble/
├── main.py         CLI
├── decomposer.py   LLM decomposition, paste detection
├── ingest.py       Routing (episodic / segment), replay()
├── promote.py      HDBSCAN promotion, Layer 0 → Layer 1
├── contradict.py   Chain assignment, SnapshotNode management, lazy summarization
├── retrieve.py     ANN retrieval, confidence labels
├── cluster.py      HDBSCAN on Matryoshka slice
├── archive.py      Append-only JSONL  ./data/archive/{user_id}.jsonl
├── embed.py        Embedding client
├── rerank.py       Optional cross-encoder reranker
├── db.py           FalkorDB client, HNSW index creation
└── _shared.py      Anthropic client, shared utilities
```

---

*If this is interesting to you, consider starring the repo — it's how we know the ideas are worth continuing.*
