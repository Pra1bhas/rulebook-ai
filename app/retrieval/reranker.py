"""Reranker: a second, more careful look at the top candidates.

Embeddings (bi-encoder) turn the question and the chunk into vectors SEPARATELY,
then compare. Fast, but it never reads them together.
A cross-encoder reads (question + chunk) TOGETHER and outputs one relevance
score. Much more accurate, but slow, so we only run it on ~20 candidates,
never on the whole database.

Pipeline: hybrid top 20 -> rerank -> keep top 5.
"""
from app.config import settings
from app.models import SearchResult


class Reranker:
    def __init__(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(settings.reranker_model, cache_dir=settings.model_cache_dir)

    def rerank(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        if not results:
            return []
        # Same input the embeddings see: heading + text ("6. Consent ..." gives the topic)
        docs = [f"{r.chunk.section}\n{r.chunk.text}" for r in results]
        scores = list(self.model.rerank(query, docs))
        paired = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:k]
        return [SearchResult(chunk=r.chunk, score=float(s), source="rerank") for r, s in paired]


class NoReranker:
    """Used when reranker is turned off: just keep the hybrid order."""

    def rerank(self, query, results, k):
        return results[:k]


def get_reranker():
    return Reranker() if settings.reranker == "fastembed" else NoReranker()
