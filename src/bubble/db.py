"""
db.py — FalkorDB graph client and user-graph accessor.
"""

from falkordb.asyncio import FalkorDB

from . import config
from .embed import EMBED_DIM

_client: FalkorDB | None = None


def get_client() -> FalkorDB:
    global _client
    if _client is None:
        _client = FalkorDB(
            host=config.FALKORDB_HOST,
            port=config.FALKORDB_PORT,
        )
    return _client


def get_graph(user_id: str):
    return get_client().select_graph(f"bubble:{user_id}")


async def init_graph(user_id: str) -> None:
    """Create indexes for a user graph. Safe to call on an already-initialized graph."""
    g = get_graph(user_id)

    # HNSW vector index on SnapshotNode.centroid — retrieval (L2 entry point).
    try:
        await g.query(
            "CREATE VECTOR INDEX FOR (n:SnapshotNode) ON (n.centroid) "
            f"OPTIONS {{dimension: {EMBED_DIM}, similarityFunction: 'cosine'}}"
        )
    except Exception:
        pass  # already exists

    # HNSW vector index on Episode.centroid — chain assignment ANN search.
    try:
        await g.query(
            "CREATE VECTOR INDEX FOR (n:Episode) ON (n.centroid) "
            f"OPTIONS {{dimension: {EMBED_DIM}, similarityFunction: 'cosine'}}"
        )
    except Exception:
        pass  # already exists
