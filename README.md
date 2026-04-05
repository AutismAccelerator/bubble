# Bubble

**Belief formation as memory for AI agents.**

Most memory systems index everything and retrieve later. Bubble treats memory as a *formation* problem: a single weak signal is noise; a signal that returns from different directions, at different moments, with consistent weight, becomes belief.

Built on event sourcing — the archive is the ground truth, the graph is always reconstructible from it. 

---

## Paper

[Bubble: Belief Formation as Memory](link) — arXiv preprint

## How it works
```
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
                     same topic chain?(NLI)
                      yes │       │ no
                          │       │
               ┌──────────▼─┐   ┌▼────────────┐
   joins chain │ ... ──► e  │   │     e       │  new chain
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
         snapshot summary             with episode chain + labels
```
## Setup
### 1.run [Falkordb](https://github.com/falkordb/falkordb) 
```bash
docker run -e REDIS_ARGS="--appendonly yes --appendfsync everysec" -v <PATH>:/var/lib/falkordb/data -p 3000:3000 -p 6379:6379 -d --name falkordb falkordb/falkordb
```
### 2.embedding model(Matryoshka)
**note: command below is cpu version**
```bash
docker run --name tei-embedding -d -p 8997:80 -v <PATH>:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id nomic-ai/nomic-embed-text-v1.5

```
Or embedding cloud api

### 3.NLI model(Optional, but recommended, saves some LLM calls)
**note: command below is cpu version**
```bash
docker run --name tei-nli -d -p 8999:80 -v /mnt/g/docker-data/volumes/tei:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id cross-encoder/nli-deberta-v3-small
```


### necessary configuration in your .env file
```python
ANTHROPIC_API_KEY=
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
BUBBLE_EMBED_DIM=768
BUBBLE_EMBED_ENDPOINT=http://localhost:8997/v1/embeddings

#If you have NLI setup
BUBBLE_ENABLE_NLI=true
BUBBLE_NLI_ENDPOINT=http://localhost:8999/predict
```

## How to use (extremely easy and clean)
### ingest
```python
import bubble
bubble.process(user_id, content, prior)
```
prior: the context of the content, for example prior messages
### retrieve
```python
import bubble
memory_user = await bubble.retrieve(user_id, query)
```

## Tuning/Customization
See [.env.example](.env.example) for ALL tunable arguments.

## Limitations
Bubble is an experimental project, not a production library. 
See [arxiv](link) for detailed specification.
**[Discord](https://discord.com/users/1319641673990672477)**
Contributions are welcome.