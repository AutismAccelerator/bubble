# Bubble
**Event Sourcing based belief formation system for long-term AI agent memory**
**[paper](https://doi.org/10.5281/zenodo.19438945)**   **[Discord](https://discord.com/users/1319641673990672477)**

---

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
docker run --name tei-nli -d -p 8999:80 -v <PATH>:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id cross-encoder/nli-deberta-v3-small
```


### Necessary configuration in your .env file
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
### installation
```bash
pip install bubble-memory
or
uv add bubble-memory
```
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

## Replayability 
Memory episodes are archived in `<project root>/data/archive` as jsonl\
You can reconstruct your whole memory graph by a single command ! WITHOUT A SINGLE LLM CALL !
```python
python -m bubble.main replay <user_id>
```

## Tuning/Customization
See [.env.example](.env.example) for ALL tunable arguments.

## Limitations
Bubble is currently an experimental project for personal use.\
Current `promotion formula`, env vars might not be the best.
`prompts` might have much room to improve. Patch **bubble.decomposer._SYSTEM** if it doesn't fit your use case.\
\
Leave a star if you like this work. Contributions are welcome.