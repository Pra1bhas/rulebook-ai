"""Text -> vector (list of numbers that captures MEANING).

Sentences with similar meaning get vectors pointing in a similar direction,
even if the words are different. "Can the bank share my data?" lands near
"disclosure of customer information to third parties".

We use BAAI/bge-small-en-v1.5 through fastembed:
  * runs locally on CPU (ONNX), free, no rate limits
  * 384 numbers per vector, small and fast
"""
import hashlib
import math
import re
from typing import Protocol

from app.config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(settings.embedding_model, cache_dir=settings.model_cache_dir)
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed batches internally; returns normalized vectors
        return [v.tolist() for v in self.model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds the instruction prefix bge models expect for questions
        return next(iter(self.model.query_embed(text))).tolist()


class FakeEmbedder:
    """For tests only: hashes words into a small vector. No model download needed."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embedder() -> Embedder:
    return FakeEmbedder() if settings.embedder == "fake" else FastEmbedder()
