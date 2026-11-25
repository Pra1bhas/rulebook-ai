"""Merge keyword (BM25) and meaning (vector) results with Reciprocal Rank Fusion.

Problem: BM25 scores look like 12.7, vector scores look like 0.83. Different
scales, so we can't just add them.
RRF ignores the scores and uses only the RANK (1st, 2nd, 3rd...):

    rrf_score(chunk) = sum over each list of  1 / (k + rank)

A chunk that is near the top in BOTH lists wins. k (=60) stops the #1 spot
from dominating everything.

Worked example (k=60):
    chunk A: bm25 rank 1, vector rank 3 -> 1/61 + 1/63 = 0.0323
    chunk B: bm25 rank 2, not in vector -> 1/62        = 0.0161
    => A comes first.
"""
from app.models import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60, limit: int = 20
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, r)
    ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [
        SearchResult(chunk=by_id[cid].chunk, score=scores[cid], source="hybrid") for cid in ranked
    ]
