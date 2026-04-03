
bubble
Hierarchical Memory Consolidation System
Architecture & Design Specification  —  Research / Experimental

A signal-driven, pseudo-emotionally routed, layered knowledge graph
for persistent AI agent memory

Abstract
bubble is a research architecture for persistent AI agent memory that departs from existing solutions by treating memory not as a storage problem, but as a belief formation problem. Rather than caching, summarizing, or indexing what a user says, bubble models what a user means — distilling raw signals into layered episodes that earn their existence through converging evidence and emotional weight.

The system is designed around three core insights: (1) LLMs are already the cortical knowledge layer — agent memory should store only what LLMs cannot know; (2) frequency of signal does not equal meaningfulness of belief; (3) human preference can shift episodically in a single moment, and any memory system that cannot accommodate this will fail in practice.


Table of Contents
1.  Motivation & Problem Statement
    1.1  Why Existing Solutions Fall Short
    1.2  The Core Insight
2.  Core Concepts
    2.1  Cluster of Truth
    2.2  Frequency vs. Intensity vs. Preference
3.  System Architecture
    3.1  Overview
    3.2  Scale Architecture
    3.3  Graph Database Schema
    3.4  Ingestion Pipeline
    3.5  Layer 0 — Raw Signal Buffer  (3.5.1 Clustering, 3.5.2 Archival, 3.5.3 Promotion Formula)
    3.6  Layer 1 — Episodes
    3.7  Episodic Override
    3.8  Chain Assignment
    3.9  Layer 2 — Snapshot Nodes
    3.10 Layer Propagation
4.  Retrieval Architecture
    4.1  Query Decomposition
    4.2  Graph Traversal
    4.3  LLM Presentation
5.  Open Research Questions
6.  Comparison with Existing Solutions
7.  Implementation Status

1. Motivation & Problem Statement
1.1 Why Existing Solutions Fall Short
Current AI memory solutions fall into three categories: vector stores, periodic summarization, and graph-based memory. All three treat memory as a storage resource rather than an evolving belief system.

Approach	Core Limitation
Vector Store	Surface-level recall. No belief formation. Retrieves similar text, not meaningful context.
Periodic Summarization	Clock-dependent. Loses episodic events. Treats all signals as equally important.
Graph Memory	Static relationships. No confidence modeling. No contradiction resolution.
bubble (this work)	Signal-weighted belief formation with episodic override and temporal contradiction history.

1.2 The Core Insight
LLMs are already the cortical knowledge layer. Agent memory only needs to store four categories:
•	Things specific to this user — preferences, behaviors, declared values
•	Things specific to this context — project state, decisions made, outcomes observed
•	Things that occurred after training — recent events, updated facts
•	Things the agent experienced or decided — action history, learned corrections

2. Core Concepts
2.1 Cluster of Truth
The fundamental memory unit in bubble is a cluster of truth — a belief that has earned its existence by having multiple independent signals converge in both topic proximity AND emotional weight. A single signal is noise. Only convergence creates belief.

Beliefs are immutable once formed. New evidence does not mutate existing Episodes — it forms new ones. The temporal chain of Episodes on the same topic is the belief's evolution. Layer 2 synthesizes that chain into a unified view for retrieval.

2.2 Frequency vs. Intensity vs. Preference
Signal Property	Behavior
High frequency, low intensity	Circumstantial noise. A user debugging all day generates high-volume signals with near-zero intensity. Never forms an episode.
Low frequency, high intensity	Episodic signal. Bypasses Layer 0 accumulation entirely. Written directly to JSONL archive and creates an Episode at Layer 1.
High frequency, high intensity	Strong convergent belief. Fast-tracks to Layer 1 with high confidence.
Low frequency, low intensity	Weak signal. Accumulates indefinitely at Layer 0. Never promotes without threshold crossing.

3. System Architecture
3.1 Overview
User Message
      │
      ▼
┌─────────────────────────────────────────┐
│           Decomposer (LLM)              │
│  decompose → intensity → valence        │
└─────────────────┬───────────────────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   LOW (0-0.3)  MED (0.3-0.6)  HIGH (0.6-1.0)
   Accumulate   Accumulate     Episodic Override
       │          │                │
       ▼          ▼                ▼
┌─────────────────────────┐   ┌──────────────────────┐
│       Layer 0           │   │  JSONL + Episode     │
│  HDBSCAN (embeddings)   │   │  (episodic=true)     │
│  Per-user graph         │   └──────────┬───────────┘
└────────────┬────────────┘              │
             │ t_promo_score > θ          │
             ▼                           │
      JSONL archive                      │
      SegmentNodes deleted               │
             │                           │
             ▼                           ▼
┌──────────────────────────────────────────────────┐
│              Layer 1 — Episodes                  │
│  LLM summarized · confidence scored · immutable · ANN-indexed  │
└────────────────────┬─────────────────────────────┘
                     │ FOLLOWED_BY edges (chain assignment)
                     ▼
┌──────────────────────────────────────────────────┐
│           Layer 2 — Snapshot Nodes               │
│  Synthesized view · ANN-indexed · lazy LLM       │
└──────────────────────────────────────────────────┘

3.2 Scale Architecture
Each user is a completely isolated graph partition. This is a critical architectural premise that drives all algorithm and infrastructure decisions.

Property	Value
Per-user dataset size	Tiny to medium — hundreds to low thousands of nodes typically
Scale dimension	Number of users, not per-user data volume
User isolation	Complete — no cross-user data, no shared graph
Storage model	Graph database, one partition per user — single source of truth
Algorithm implications	Per-user operations are cheap; horizontal scaling handles user volume
Event sourcing	Raw signals are archived to per-user JSONL at promotion time. Each entry includes the original text and conversational prior, making the signal log fully replayable. The graph holds only derived belief structure.

3.3 Graph Database Schema
All layers coexist in the same per-user graph partition, distinguished by node label.

Node Labels

Label	Layer	Key Properties
SegmentNode	0	raw_text: string, embedding: float[768], intensity: float, valence: pos|neg|neu, prior: string|null, timestamp: datetime
Episode	1	summary: string, confidence: float, centroid: float[768], valence: pos|neg|neu, episodic: bool, timestamp: datetime
SnapshotNode	2	summary: string|null, centroid: float[768], valid: bool, timestamp: datetime

SegmentNodes are transient. They live in the graph only while awaiting HDBSCAN clustering. At promotion, they are written to the JSONL archive and deleted from the graph. The graph never retains promoted SegmentNodes.

Episodes are immutable after creation. New evidence does not modify existing Episodes; it forms new ones. The temporal chain of connected Episodes represents belief evolution.

SnapshotNode.summary is null until first retrieval, then cached. SnapshotNode.centroid is always current — updated eagerly whenever the cluster membership changes. SnapshotNode.valid is set false whenever a new Episode joins the cluster, triggering lazy re-synthesis on next retrieval.

Relationship Types

Relationship	Between	Semantics
FOLLOWED_BY	Episode → Episode	Temporal chain link — appended to end of same-topic chain at creation
SYNTHESIZES	SnapshotNode → Episode	This L2 node covers this L1 member

Layer 0 clustering extracts the embedding matrix from all SegmentNodes (which are exclusively the active pool) and runs HDBSCAN directly on that matrix.
A vector index on SnapshotNode.centroid enables DB-side ANN for retrieval.
Chain assignment at promotion time uses a single ANN query against Episode centroids to find the closest existing node, then an LLM relatedness check if the similarity threshold is crossed.

JSONL Archive Entry
{
  text:      string        // raw_text
  prior:     string|null   // conversational context at ingestion time
  intensity: float
  valence:   pos|neg|neu
  timestamp: string        // ISO 8601
}

3.4 Ingestion Pipeline
3.4.1 Decomposer
Every incoming message passes through a lightweight LLM performing three jobs in one call. Domain tags were considered and explicitly dropped — embeddings capture topic proximity implicitly, and the contradiction pipeline handles detection without domain as a proxy.

Input:  raw user message
        optional prior: string — the conversational context the user is responding
        to (agent reply, group chat exchange, etc.), injected as a <prior> block
        before the user message. When the user's message is a fragment that only
        makes sense as a response, the decomposer restates it as a complete
        self-contained assertion. Caller is responsible for providing this when
        relevant; absent prior, the message is treated as self-contained.
        Multi-speaker exchanges use [Name]: prefix format; the user's own messages
        in the prior are normalized to [User].
Output: [
  {
    text:      string        // atomic sub-sentence
    intensity: float         // 0.0 – 1.0
    valence:   pos|neg|neu
  },
  ...
]

3.4.2 Paste Detection
Detected via: token count exceeds threshold, low first-person pronoun density, high lexical uniformity. Bypasses Decomposer entirely. Assigned near-zero intensity by rule. Can weakly reinforce existing clusters but cannot independently form episodes.

3.4.3 Routing
Intensity Range	Path
0.0 – 0.6  (Low / Med)	Normal accumulation. Stored as SegmentNode in Layer 0 active pool.
0.6 – 1.0  (High)	Episodic override. Written to JSONL archive. LLM-summarized Episode
               created at Layer 1 (episodic=true). No SegmentNode created.

No absorption at ingestion time. Segments are never routed directly into existing Episodes. All non-episodic segments enter the Layer 0 active pool and wait for HDBSCAN clustering.

3.5 Layer 0 — Raw Signal Buffer
Layer 0 Design Principles
• No eviction by clock — user interaction frequency is uncertain
• No eviction by storage pressure — operational concern, not memory concern
• Segments are the atomic unit, not whole messages
• Prior context retained on every SegmentNode — the JSONL archive is fully replayable
• Clustering via HDBSCAN on embedding matrix of all active SegmentNodes
• Promotion only when volume + intensity BOTH cross threshold
• Promoted SegmentNodes are deleted from graph — the active pool IS all SegmentNodes in graph

3.5.1 Clustering — HDBSCAN
The active pool is all SegmentNodes in the graph. On each promotion check, embeddings are extracted into a matrix and passed directly to HDBSCAN. No graph edges are required — HDBSCAN operates entirely in the raw embedding space.

HDBSCAN is run on promotion check, not on every insertion. At per-user scale the active pool stays small (promoted nodes are deleted), so a full re-cluster is fast enough that incremental approaches are unnecessary.

Property	Detail
Decay	Not applicable — all nodes retained at full weight until promotion
Variable density	HDBSCAN handles variable cluster density intrinsically — no fixed resolution parameter
Active pool	All SegmentNodes in graph; promoted nodes are deleted, keeping the pool small
Input	embedding[:128], re-normalized to unit length — the full-dim vector is unit-norm but the
       truncated prefix is not; re-normalizing restores the cosine distance relationship for
       euclidean HDBSCAN. Topic separation only.
Noise handling	HDBSCAN assigns outlier label (−1) to segments that belong to no cluster; these accumulate but never promote
Scale	Active pool stays bounded as promotions drain and delete it

3.5.2 Archival
When a cluster promotes to Layer 1, each SegmentNode in the cluster is written to the JSONL archive (text, prior, intensity, valence, timestamp, episode_id) and then deleted from the graph. The graph is immediately leaner; the JSONL is the immutable signal log.

Episodic segments bypass this entirely — they are written to JSONL at ingestion time and no SegmentNode is ever created.

3.5.3 Promotion Formula
A Layer 0 cluster promotes to Layer 1 only when both volume and intensity thresholds are satisfied simultaneously. High intensity segments never reach this path — they went episodic upstream.

t_promo_score = log(n) * avg_intensity > θ

where:
  n             =  segment count in cluster
  avg_intensity =  mean intensity across members
  θ             =  single fixed constant (tuned empirically)

log(n)        → diminishing returns on volume alone
multiplication → both factors must be non-zero to promote
                 high volume + zero intensity = never promotes
                 single segment = log(1) = 0 = never promotes

Diversity is not included in the promotion formula. HDBSCAN groups segments by embedding proximity, so intra-cluster diversity is inherently bounded by the clustering algorithm itself. log(n) already provides diminishing returns on repeated similar signals.

3.6 Layer 1 — Episodes
Episode {
  id:         uuid
  summary:    string    // LLM-generated belief statement, always third-person subject-free
  confidence: float     // set at creation from avg_intensity; internal only, never shown as number
  centroid:   float[]   // mean embedding of source segments, set at creation
  valence:    pos|neg|neu  // dominant valence of source segments
  episodic:   bool
  timestamp:  datetime  // belief formation time — immutable
}

Episodes are write-once. Once created, no field is modified. Belief evolution is represented as a temporal chain of Episodes connected by FOLLOWED_BY edges — each new node on the same topic appends to the end of the chain.

Confidence is set at promotion to avg_intensity of the promoting cluster. For episodic nodes it is set to the segment's intensity. It is never updated.

Edge Type	Semantics
FOLLOWED_BY	Temporal chain link — directed, one outgoing edge per node maximum

3.7 Episodic Override
High intensity segment (> 0.6) arrives
        │
        ▼  bypasses Layer 0 entirely
Written to JSONL archive (text, prior, intensity, valence, timestamp)
        │
        ▼  LLM summarizes to third-person
Episodic Episode created at Layer 1 (episodic=true, confidence=intensity)
        │
        ▼  chain assignment runs immediately
check_new — finds closest existing Episode, LLM relatedness check
        │
   ┌────┴────────────┐
   ▼                 ▼
Related          Not related
append to        new isolated
existing chain   SnapshotNode

Episodic nodes are ordinary Episodes with episodic=true. Their subsequent status evolves through the natural evidence accumulation process — new segments that accumulate and promote around the same topic will form new Episodes that append to the same chain. No explicit confirmation mechanism exists; the temporal chain of Episodes tells the story.

3.8 Chain Assignment
Runs at write time whenever a new Episode is created (both from cluster promotion and episodic override).

Step	Method & Cost
1 — ANN top-1	Single vector query against Episode centroids. Returns the single closest existing Episode. Near-zero cost.
2 — Similarity threshold	If cosine similarity < t_similarity → new isolated SnapshotNode. Done.
3 — LLM relatedness check	Single LLM call: "are these about the same topic or belief?" Binary yes/no. One call maximum per new Episode.
4 — Chain append	If related: traverse FOLLOWED_BY edges from the matched node to the tail (node with no outgoing FOLLOWED_BY edge). Wire FOLLOWED_BY edge from tail → new Episode. New Episode joins the matched chain's SnapshotNode. If not related: new isolated SnapshotNode.

Tension between beliefs on the same chain (contradictions, reversals, evolution) is not classified at write time. The ordered chain of summaries is passed to the L2 synthesis LLM at retrieval, which surfaces the full arc naturally.

3.9 Layer 2 — Snapshot Nodes
Every Episode belongs to exactly one SnapshotNode cluster at Layer 2. SnapshotNodes are the primary retrieval target. The ANN vector index is on SnapshotNode.centroid.

Cluster Membership
A new Episode is assigned to a L2 cluster by chain assignment (§3.8):
• Not related to any existing chain → new SnapshotNode created covering only this Episode
• Appended to an existing chain → joins that chain's SnapshotNode

Each SnapshotNode covers exactly one Episode chain. Chains are never merged.

SnapshotNode Properties
• centroid: mean of member Episode centroids — updated eagerly at membership change, no LLM required
• summary: null on creation; generated lazily by LLM at first retrieval after valid=false
• valid: set false whenever a new Episode joins the cluster → triggers lazy re-synthesis

Synthesis
When a SnapshotNode's summary is requested and valid=false, the LLM synthesizes all member Episodes into a single coherent narrative. Members are ordered by FOLLOWED_BY chain traversal (chronological). The most recent member takes precedence. Episodic members are always presented last in the temporal arc regardless of chain position — they represent discrete high-intensity events that override accumulated belief. Any tension or contradiction within the chain is resolved naturally by the synthesis LLM.

For a single-member SnapshotNode, the summary is copied from the Episode summary directly — no LLM call required.

Read-only at retrieval: SnapshotNode summaries are a cache of LLM reasoning. They do not affect clustering or propagation.

3.10 Layer Propagation
Same mechanics apply at every layer boundary with increasing inertia.

Layer Boundary	Promotion Trigger
Layer 0 → 1	t_promo_score crosses θ. LLM summarizes cluster into a new immutable Episode.
Layer 1 → 2	Automatic. Every Episode belongs to a SnapshotNode. Cluster membership determined by chain assignment (§3.8) at creation time.
Layer 2 → N	Deferred. Same pattern, higher threshold. Forms slowly over extended interaction history.

4. Retrieval Architecture
4.1 Query Decomposition
Long or multi-topic queries are decomposed by the Decomposer into segments before retrieval. Each segment is embedded independently and searched in parallel against Layer 2. Results merged and deduplicated. SnapshotNodes retrieved by multiple segments rank highest — cross-segment relevance is the primary signal.

4.2 Graph Traversal
ANN query against SnapshotNode.centroid returns the top-k SnapshotNodes. For each result:
1. Fetch or generate SnapshotNode summary (lazy LLM if valid=false)
2. Follow SYNTHESIZES edges to retrieve member Episodes
3. Follow FOLLOWED_BY edges among members to present the full chain in order

Edge Type	Traversal Behavior
FOLLOWED_BY	Always follow — reconstructs temporal belief chain for synthesis

4.3 LLM Presentation
Confidence scores are internal values only. Translated to natural language at the retrieval boundary — numbers never reach the LLM.

Internal Score	LLM Presentation
conf > 0.75	"established"
conf 0.50 – 0.75	"likely"
conf 0.30 – 0.50	"uncertain"
conf < 0.30	"contested" or "unclear"
Episodic node	"recently shifted, unverified"

5. Open Research Questions
Critical Open Questions
1. θ empirical tuning — single fixed constant, needs real interaction data to calibrate

2. t_similarity distance threshold — determined by natural valley in pairwise embedding
   distance distribution. Needs analysis on real user data.

3. Decomposer reliability — entire routing architecture depends on accurate
   intensity scoring. Evaluation framework not yet defined.

4. Retrieval at scale — graph traversal efficiency across many users.
   Partitioning and indexing strategy not yet defined.

6. JSONL retention policy — how much raw signal history to preserve per user.
   Storage growth is bounded by interaction volume, not graph size.

7. t_similarity threshold for chain assignment — cosine similarity floor that
   determines whether a new Episode is "close enough" to an existing chain to
   warrant the LLM relatedness check. Needs empirical calibration on real user data.

8. HDBSCAN re-cluster trigger — currently triggered on promotion check. Whether
   insertion-triggered re-clustering produces meaningfully better cluster quality
   at per-user scale is an open empirical question.

9. HDBSCAN min_cluster_size — the primary tuning parameter. Too small → noise segments
   incorrectly form beliefs. Too large → genuine sparse beliefs never promote. Needs
   empirical calibration; likely user-interaction-count-dependent.

10. Confidence formula — initial confidence set to avg_intensity of the promoting cluster.
    An Episode's confidence is fixed at creation and never updated. Whether this is
    sufficient or whether chain-based confidence propagation is needed is an open question.

11. LLM relatedness prompt calibration — the chain assignment LLM call asks "are these
    about the same topic or belief?" Threshold for what counts as "related" determines
    chain granularity. Too permissive → unrelated beliefs collapse into one chain. Too
    strict → same belief forms parallel chains. Needs empirical calibration.

12. Layer 2→N saturation metric — "Layer 2 nodes begin overlapping in meaning" is
    described but not measured. Deferred to Layer 3 implementation phase.


6. Comparison with Existing Solutions
Capability	Existing  vs  bubble
Memory unit	Fact, sentence, or summary  →  Immutable belief node with formation timestamp
Eviction	Time or storage pressure  →  Promotion-based archival, no clock, no mutation
Episodic shifts	Not modeled  →  Dedicated high-intensity path; direct L1 node, JSONL archive
Contradiction	Overwrite or ignore  →  Temporal chain of beliefs + L2 synthesis
Belief evolution	Overwrite or summarize  →  New Episode per evolution step; chain preserved
Confidence	Binary or absent  →  Continuous, intensity-weighted
LLM presentation	Raw facts or scores  →  Natural language confidence translation
Clustering	Standard equal-weight  →  HDBSCAN on active SegmentNode embedding matrix
Retrieval target	Raw beliefs  →  L2 SnapshotNode synthesis (unified view of belief evolution)
Maturity	Production-ready  →  Research / experimental



Engineering architecture: FalkorDB (graph database, per-user partitioning, HNSW vector index on
SnapshotNode.centroid), embedding inference server (OpenAI-compatible HTTP endpoint, Matryoshka —
full-dim for storage/retrieval, 128-dim re-normalized slice for HDBSCAN), optional reranker
inference server (BUBBLE_RERANK_ENABLED, OpenAI-compatible /rerank endpoint), Anthropic API —
claude-haiku-4-5-20251001 (decomposer + summarization + chain relatedness check),
scikit-learn HDBSCAN (clustering), httpx (async HTTP throughout).
