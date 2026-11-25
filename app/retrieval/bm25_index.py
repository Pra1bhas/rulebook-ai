"""Keyword search with BM25 (the classic search-engine algorithm).

Idea: a chunk scores high if it contains the query words, especially RARE words
(like "KYC" or "8(6)"), and the score doesn't grow forever if a word repeats.
Great for exact terms, bad for synonyms ("share my data" vs "disclosure").
"""
import re

from rank_bm25 import BM25Okapi

from app.models import Chunk, SearchResult

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "for", "on",
    "by", "with", "as", "that", "this", "it", "at", "from", "shall", "any", "such",
    "what", "which", "who", "can", "do", "does", "my", "i", "me",
}


def tokenize(text: str) -> list[str]:
    # keep things like "8(6)" and "kyc" as tokens
    tokens = re.findall(r"[a-z0-9]+(?:\([a-z0-9]+\))*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(f"{c.section} {c.text}") for c in chunks]
        self.bm25 = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, k: int, owners: set[str] | None = None) -> list[SearchResult]:
        """owners: whose chunks may be returned ("" = public library). None = everyone."""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) == k:
                break
            if owners is None or self.chunks[i].owner in owners:
                out.append(SearchResult(chunk=self.chunks[i], score=float(scores[i]), source="bm25"))
        return out
