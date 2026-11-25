"""GeminiEmbedder with a fake API client: caching, privacy, batching. No key, no internet."""
from types import SimpleNamespace

import numpy as np

from app.config import settings
from app.retrieval.embedder import GeminiEmbedder


class FakeGeminiClient:
    """Counts API calls and returns a fixed vector per text."""

    def __init__(self, dim):
        self.calls, self.dim = [], dim
        self.models = SimpleNamespace(embed_content=self.embed_content)

    def embed_content(self, model, contents, config):
        self.calls.append(list(contents))
        vec = lambda t: [float(len(t))] + [1.0] * (self.dim - 1)  # noqa: E731 (not normalized)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec(t)) for t in contents])


def _embedder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embedding_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "gemini_embedding_dim", 8)
    client = FakeGeminiClient(8)
    return GeminiEmbedder(client=client), client


def test_library_vectors_are_cached_to_file(tmp_path, monkeypatch):
    emb, client = _embedder(tmp_path, monkeypatch)
    first = emb.embed_documents(["section 6 consent", "section 9 children"])
    assert len(client.calls) == 1
    # A NEW embedder (= server restart) loads the file: zero API calls for the same texts.
    again, client2 = GeminiEmbedder(client=FakeGeminiClient(8)), None
    assert np.allclose(again.embed_documents(["section 6 consent", "section 9 children"]), first, atol=1e-3)
    assert again._client.calls == []


def test_private_uploads_are_never_written_to_cache(tmp_path, monkeypatch):
    emb, client = _embedder(tmp_path, monkeypatch)
    emb.embed_documents(["my rent agreement text"], cache=False)
    assert not emb.cache_file.exists()
    assert "my rent agreement text" not in str(emb.cache.keys())


def test_vectors_are_normalized_and_batched(tmp_path, monkeypatch):
    emb, client = _embedder(tmp_path, monkeypatch)
    monkeypatch.setattr(GeminiEmbedder, "BATCH", 2)
    vecs = emb.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [len(c) for c in client.calls] == [2, 2, 1], "100 texts per call in real use, 2 here"
    assert all(abs(np.linalg.norm(v) - 1) < 1e-3 for v in vecs)


def test_same_question_is_embedded_once(tmp_path, monkeypatch):
    emb, client = _embedder(tmp_path, monkeypatch)
    emb.embed_query("can i withdraw consent")
    emb.embed_query("can i withdraw consent")
    assert len(client.calls) == 1


def test_build_needs_no_api_key_when_everything_is_cached(tmp_path, monkeypatch):
    emb, _ = _embedder(tmp_path, monkeypatch)
    emb.embed_documents(["chapter 1 text"])  # fills the cache file
    monkeypatch.setattr(settings, "gemini_api_key", "")  # like `docker build` on Render
    no_key = GeminiEmbedder()  # no client given, no key
    assert len(no_key.embed_documents(["chapter 1 text"])[0]) == 8
