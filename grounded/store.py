"""Qdrant wrapper: create the collection, upsert chunks, search."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config
from .chunk import Chunk
from .graph.types import GraphPath


@dataclass
class Hit:
    id: int
    score: float
    source: str
    heading: str
    text: str
    # When this hit is a graph traversal rendered as evidence, the path it came
    # from -- so evidence is self-describing (no parallel path list to re-zip).
    path: GraphPath | None = None


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    # One shared client. Server mode if QDRANT_URL is set, otherwise embedded
    # (a local directory) — embedded mode keeps a single open handle, so the
    # cache here is load-bearing, not just an optimisation.
    if config.settings.qdrant_url:
        return QdrantClient(url=config.settings.qdrant_url)
    if config.settings.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=config.settings.qdrant_path)


def reset_collection() -> None:
    c = _client()
    if c.collection_exists(config.settings.collection):
        c.delete_collection(config.settings.collection)
    c.create_collection(
        collection_name=config.settings.collection,
        vectors_config=VectorParams(size=config.settings.embed_dim, distance=Distance.COSINE),
    )


def upsert(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=i,
            vector=vec,
            payload={"source": ch.source, "heading": ch.heading, "text": ch.text},
        )
        for i, (ch, vec) in enumerate(zip(chunks, vectors))
    ]
    _client().upsert(collection_name=config.settings.collection, points=points)


def search(vector: list[float], k: int) -> list[Hit]:
    res = _client().query_points(collection_name=config.settings.collection, query=vector, limit=k)
    return [
        Hit(
            id=p.id,
            score=p.score,
            source=p.payload["source"],
            heading=p.payload["heading"],
            text=p.payload["text"],
        )
        for p in res.points
    ]
