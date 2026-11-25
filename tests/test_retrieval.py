from app.models import Chunk, SearchResult
from app.retrieval.bm25_index import BM25Index, tokenize
from app.retrieval.hybrid import reciprocal_rank_fusion


def _chunk(i, text, section="s"):
    return Chunk(chunk_id=str(i), doc_id="d", doc_name="d.pdf", page=1,
                 section=section, chunk_index=i, text=text)


def test_tokenize_keeps_section_numbers():
    assert "8(6)" in tokenize("as per section 8(6) of the Act")
    assert "the" not in tokenize("the KYC rules")


def test_bm25_finds_exact_term():
    idx = BM25Index([_chunk(0, "rules about KYC verification"), _chunk(1, "cats and dogs"),
                     _chunk(2, "loan interest rates")])
    results = idx.search("KYC", k=3)
    assert results[0].chunk.chunk_id == "0"
    assert len(results) == 1  # chunks with zero score are not returned


def test_rrf_prefers_chunk_in_both_lists():
    a, b, c = _chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")
    bm25 = [SearchResult(chunk=b, score=9, source="bm25"), SearchResult(chunk=a, score=5, source="bm25")]
    vec = [SearchResult(chunk=c, score=.9, source="vector"), SearchResult(chunk=a, score=.8, source="vector")]
    merged = reciprocal_rank_fusion([bm25, vec], k=60)
    assert merged[0].chunk.chunk_id == "0"  # 'a' is 2nd in both lists -> wins
    assert abs(merged[0].score - 2 / 62) < 1e-9
