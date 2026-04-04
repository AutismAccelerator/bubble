# Bubble

**Belief formation as memory for AI agents.**

Most memory systems index everything and retrieve later. Bubble treats memory as a *formation* problem: a single weak signal is noise; a signal that returns from different directions, at different moments, with consistent weight, becomes belief.

Built on event sourcing — the archive is the ground truth, the graph is always reconstructible from it.

---

## Paper

[Bubble: Belief Formation as Memory](link) — arXiv preprint

## How it works
[ raw input ]
                               │
                         ┌─────▼─────┐
                         │ decompose │
                         └─────┬─────┘
                               │
              ┌────────────────┴────────────────┐
          ι ≥ θ                             ι < θ
              │                                 │
        vivid signal                       weak signal
              │                                 │
         ┌────▼────┐                     ┌──────▼──────┐
         │ archive │                     │    pool     │
         └────┬────┘                     │  · · · · ·  │
              │                          │ ·  · ·  · · │
              │                          │  · · · ·  · │
              │                          └──────┬──────┘
              │                                 │
              │                          enough gathered?
              │                                 │
              │                    no ──────────┘
              │                                 │ yes
              │                          ┌──────▼──────┐
              │                          │   cluster   │
              │                          │   + score   │
              │                          └──────┬──────┘
              │                                 │
              └──────────────┬──────────────────┘
                             │
                       ┌─────▼─────┐
                       │  episode  │  immutable
                       └─────┬─────┘
                             │
                     same topic chain?
                      yes │       │ no
                          │       │
               ┌──────────▼─┐   ┌▼────────────┐
               │ ... ──► e  │   │     e       │  new chain
               └──────────┬─┘   └─────┬───────┘
                          │           │
                    ┌─────▼───────────▼─────┐
                    │       snapshot        │
                    │  centroid  │  summary │
                    │  (eager)   │  (lazy)  │
                    └───────────┬───────────┘
                                │
                           [ retrieve ]
                                │
               ┌────────────────┴─────────────────┐
           default                            verbose
               │                                  │
         snapshot summary              episode chain + labels

## Setup
### 1.run [Falkordb](https://github.com/falkordb/falkordb) 
docker run -e REDIS_ARGS="--appendonly yes --appendfsync everysec" -v <PATH>:/var/lib/falkordb/data -p 3000:3000 -p 6379:6379 -d --name falkordb falkordb/falkordb

### 2.embedding model
**note: command below is cpu version**
```bash
docker run --name tei-embedding -d -p 8997:80 -v <PATH>:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id jinaai/jina-embeddings-v5-text-nano-clustering
```

Or OpenAI compatible embedding api


### necessary configuration in your .env file
ANTHROPIC_API_KEY=
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
BUBBLE_EMBED_MODEL=jinaai/jina-embeddings-v5-text-nano-clustering
BUBBLE_EMBED_DIM=768
BUBBLE_EMBED_ENDPOINT=http://localhost:8997


See the [.env.example](.env.example) for ALL tunable arguments.


## 
currently works well with `jinaai/jina-embeddings-v5-text-nano-clustering`