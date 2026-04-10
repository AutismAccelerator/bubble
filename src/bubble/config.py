"""
config — centralised environment variable registry for bubble.

Every tunable constant is read here once. All other modules import from this
module instead of calling os.getenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────────────
# Provider: "anthropic" | "openai" | "gemini"
LLM_PROVIDER: str = os.getenv("BUBBLE_LLM_PROVIDER", "anthropic").lower()
MODEL: str = os.getenv("BUBBLE_MODEL", "claude-sonnet-4-6")

# Anthropic
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# OpenAI-compatible (OpenAI, DeepSeek, Groq, Ollama, etc.)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Google Gemini
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# ── Graph ────────────────────────────────────────────────────────────────────
FALKORDB_HOST: str = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT: int = int(os.getenv("FALKORDB_PORT", "6379"))

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBED_DIM: int = int(os.getenv("BUBBLE_EMBED_DIM", "768"))
EMBED_ENDPOINT: str = os.getenv("BUBBLE_EMBED_ENDPOINT", "http://localhost:8997/v1/embeddings")

# ── Reranker ─────────────────────────────────────────────────────────────────
RERANK_ENABLED: bool = os.getenv("BUBBLE_RERANK_ENABLED", "false").lower() == "true"
RERANK_ENDPOINT: str = os.getenv("BUBBLE_RERANK_ENDPOINT", "http://localhost:8998/rerank")

# ── NLI (contradiction / relatedness check) ──────────────────────────────────
NLI_ENABLED: bool = os.getenv("BUBBLE_ENABLE_NLI", "false").lower() == "true"
NLI_ENDPOINT: str = os.getenv("BUBBLE_NLI_ENDPOINT", "http://localhost:8999/predict")
NLI_MODEL: str = os.getenv("BUBBLE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

# ── Memory pipeline tuning ───────────────────────────────────────────────────
EPISODIC_THRESHOLD: float = float(os.getenv("BUBBLE_EPISODIC_THRESHOLD", "0.6"))
PROMOTE_THRESHOLD: float = float(os.getenv("BUBBLE_PROMOTE_THRESHOLD", "0.2"))
CLUSTER_MIN_SIZE: int = int(os.getenv("BUBBLE_CLUSTER_MIN_SIZE", "3"))
CLUSTER_DIMS: int = int(os.getenv("BUBBLE_CLUSTER_DIMS", "128"))
CHAIN_MAX_DISTANCE: float = float(os.getenv("BUBBLE_CHAIN_MAX_DISTANCE", "0.4"))
T_SIMILARITY: float = float(os.getenv("BUBBLE_T_SIMILARITY", "0.4"))
CLUSTER_JOIN_SIM: float = float(os.getenv("BUBBLE_CLUSTER_JOIN_SIM", "0.7"))

# ── Archive ───────────────────────────────────────────────────────────────────
ARCHIVE_DIR: str = os.getenv("BUBBLE_ARCHIVE_DIR", "./data/archive")
