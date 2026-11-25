"""Text -> vector (list of numbers that captures MEANING).

Sentences with similar meaning get vectors pointing in a similar direction,
even if the words are different. "Can the bank share my data?" lands near
"disclosure of customer information to third parties".

Two real options (EMBEDDER setting):
  * fastembed (default): BAAI/bge-small-en-v1.5 runs locally on CPU (ONNX).
    Free, no rate limits, 384 numbers per vector. Needs ~500 MB RAM.
  * gemini: Google's gemini-embedding-001 API (free tier), 768 numbers per vector.
    Almost no RAM, so the app fits free 512 MB hosting (Render).
    Library vectors are cached in data/embeddings/*.npz (committed to git),
    so the server never re-embeds the library: 0 API calls at startup.
"""
import hashlib
import logging
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str], cache: bool = True) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(settings.embedding_model, cache_dir=settings.model_cache_dir)
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_documents(self, texts: list[str], cache: bool = True) -> list[list[float]]:
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

    def embed_documents(self, texts, cache=True):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class GeminiEmbedder:
    """Embeddings from the Gemini API + a vector cache file for the public library."""

    # Free tier: 100 embeddings per MINUTE, and every text in a batch counts.
    # So we send <=90 texts per call, and when we hit the limit we wait for the next minute.
    BATCH = 90
    PATIENT_WAIT = 65  # seconds; one-time library embedding can be slow, it must not fail

    def __init__(self, client=None):
        # Created lazily: when every text is already in the cache (Docker build of the
        # public library), no API key is needed at all.
        self._client = client
        self.dim = settings.gemini_embedding_dim
        slug = re.sub(r"[^a-z0-9]+", "-", settings.gemini_embedding_model.lower())
        self.cache_file = Path(settings.embedding_cache_dir) / f"{slug}-{self.dim}.npz"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_file.exists():
            data = np.load(self.cache_file)
            self.cache = dict(zip(data["keys"].tolist(), data["vecs"]))
            log.info("Embedding cache: %d vectors from %s", len(self.cache), self.cache_file)
        self.embed_query = lru_cache(maxsize=1000)(self._embed_query)  # repeated questions: free

    def embed_documents(self, texts: list[str], cache: bool = True) -> list[list[float]]:
        """cache=False for private user uploads: their vectors are never written to the file."""
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        new: dict[str, np.ndarray] = {}
        for start in range(0, len(missing), self.BATCH):
            idx = missing[start : start + self.BATCH]
            vecs = self._call([texts[i] for i in idx], "RETRIEVAL_DOCUMENT", patient=True)
            for i, v in zip(idx, vecs):
                new[keys[i]] = v
        if new and cache:
            self.cache.update(new)
            self._save()
        return [(new.get(k) if k in new else self.cache[k]).astype(float).tolist() for k in keys]

    def _embed_query(self, text: str) -> list[float]:
        return self._call([text], "RETRIEVAL_QUERY")[0].astype(float).tolist()

    def _call(self, texts: list[str], task: str, patient: bool = False) -> list[np.ndarray]:
        """patient=True (documents): on rate limit wait a full minute, up to 10 times.
        patient=False (a user's question): short exponential backoff, then give up fast."""
        from google.genai import errors, types

        from app.llm import RETRYABLE_CODES

        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        tries = 10 if patient else settings.llm_max_retries
        for attempt in range(tries + 1):
            try:
                resp = self.client().models.embed_content(
                    model=settings.gemini_embedding_model, contents=texts, config=config
                )
                break
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == tries:
                    raise
                wait = self.PATIENT_WAIT if patient else 2**attempt * 2  # 2s, 4s, 8s...
                log.warning("Embedding API busy/rate limited (%s), waiting %ss", e.code, wait)
                time.sleep(wait)
        out = []
        for e in resp.embeddings:
            v = np.asarray(e.values, dtype=np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))  # <3072 dims are not pre-normalized
        return out

    def client(self):
        if self._client is None:
            from google import genai

            if not settings.gemini_api_key:
                raise RuntimeError(
                    "EMBEDDER=gemini needs GEMINI_API_KEY for texts that are not in "
                    f"{self.cache_file}. Run the app once locally to refresh the cache."
                )
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def _save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.cache.keys()))
        vecs = np.array(list(self.cache.values()), dtype=np.float16)  # half size, enough precision
        np.savez_compressed(self.cache_file, keys=keys, vecs=vecs)


def get_embedder() -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder()
    if settings.embedder == "gemini":
        return GeminiEmbedder()
    return FastEmbedder()
