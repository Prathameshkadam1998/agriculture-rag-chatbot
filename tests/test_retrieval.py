from src.retrieval import bm25_search, build_bm25_index, reciprocal_rank_fusion

from .conftest import STOPWORDS


def test_rrf_rewards_items_ranked_by_both_lists():
    semantic = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}]
    keyword = [{"chunk_id": 2}, {"chunk_id": 4}]

    fused = reciprocal_rank_fusion([semantic, keyword], rrf_k=60)

    assert [item["chunk_id"] for item in fused] == [2, 1, 4, 3]
    assert fused[0]["score"] == 1 / 62 + 1 / 61


def test_bm25_matches_despite_query_punctuation_and_case(corpus):
    chunks, bm25_chunks, _ = corpus
    index = build_bm25_index(bm25_chunks)

    results = bm25_search("Brown spots on MANGO leaves?", index, bm25_chunks, chunks, top_k=3, stopwords=STOPWORDS)

    assert results[0]["chunk_id"] == 4
    assert results[0]["source"] == "bm25"


def test_bm25_returns_nothing_when_no_term_matches(corpus):
    chunks, bm25_chunks, _ = corpus
    index = build_bm25_index(bm25_chunks)

    assert bm25_search("bitcoin wallet", index, bm25_chunks, chunks, top_k=3, stopwords=STOPWORDS) == []
    assert bm25_search("how to the", index, bm25_chunks, chunks, top_k=3, stopwords=STOPWORDS) == []
