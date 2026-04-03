import os
from dotenv import load_dotenv
from falkordb.asyncio import FalkorDB

load_dotenv()

_client: FalkorDB | None = None


def get_client() -> FalkorDB:
    global _client
    if _client is None:
        _client = FalkorDB(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
        )
    return _client


def get_graph(user_id: str):
    return get_client().select_graph(f"bubble:{user_id}")


async def init_graph(user_id: str) -> None:
    """Create indexes for a user graph. Safe to call on an already-initialized graph."""
    g = get_graph(user_id)

    from .embed import EMBED_DIM

    # HNSW vector index on SnapshotNode.centroid — retrieval (L2 entry point).
    try:
        await g.query(
            "CREATE VECTOR INDEX FOR (n:SnapshotNode) ON (n.centroid) "
            f"OPTIONS {{dimension: {EMBED_DIM}, similarityFunction: 'cosine'}}"
        )
    except Exception:
        pass  # already exists

    # HNSW vector index on Episode.centroid — L2 cluster assignment ANN fallback.
    # Used by _assign_to_snapshot to find the nearest existing belief when no
    # SUPPORTS/CONTRADICTS edge connects the new Episode to an existing cluster.
    try:
        await g.query(
            "CREATE VECTOR INDEX FOR (n:Episode) ON (n.centroid) "
            f"OPTIONS {{dimension: {EMBED_DIM}, similarityFunction: 'cosine'}}"
        )
    except Exception:
        pass  # already exists
