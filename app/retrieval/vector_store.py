"""Qdrant: a database built to find the nearest vectors fast.

We store each chunk as a "point": id + vector + payload (the chunk's text and
metadata). Search = "give me the k points whose vectors are closest to the
question's vector" (cosine similarity).

Two modes, same code:
  * QDRANT_URL set   -> talk to a Qdrant server (docker-compose)
  * QDRANT_URL empty -> Qdrant runs inside our process, saved to a folder
                        (simplest for free deployment: one container)
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.models import Chunk, SearchResult


class VectorStore:
    def __init__(self, dim: int, location: str | None = None):
        if location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        # One collection per vector size: 384-number (local) and 768-number (Gemini)
        # vectors can't live in the same collection.
        self.name = f"{settings.collection}-{dim}"
        if not self.client.collection_exists(self.name):
            self.client.create_collection(
                self.name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def count(self) -> int:
        return self.client.count(self.name).count

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            ),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                # uuid5 = same chunk_id always gives the same point id -> no duplicates
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=v,
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 256):  # send in batches
            self.client.upsert(self.name, points=points[i : i + 256])

    def search(
        self, vector: list[float], k: int, owners: set[str] | None = None
    ) -> list[SearchResult]:
        """owners: whose chunks may be returned ("" = public library). None = everyone."""
        flt = None
        if owners is not None:
            # Qdrant does the filtering, so another user's private chunks never come back.
            flt = Filter(
                should=[
                    FieldCondition(key="owner", match=MatchAny(any=sorted(owners))),
                    IsEmptyCondition(is_empty=PayloadField(key="owner")),  # old public points
                ]
                if "" in owners
                else [FieldCondition(key="owner", match=MatchAny(any=sorted(owners)))]
            )
        hits = self.client.query_points(self.name, query=vector, query_filter=flt, limit=k).points
        return [
            SearchResult(chunk=Chunk(**h.payload), score=h.score, source="vector") for h in hits
        ]
