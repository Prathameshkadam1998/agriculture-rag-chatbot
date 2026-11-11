import pytest

from src.pipeline import AgriRAGPipeline, HybridCEPipeline, SemanticOnlyPipeline, sigmoid
from src.retrieval import build_bm25_index, build_faiss_index

from .conftest import STOPWORDS, FakeEmbedder, FakeReranker


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(0) == 0.5
    assert sigmoid(800) == 1.0
    assert sigmoid(-800) == 0.0


def test_pipeline_ranks_the_matching_passage_first(pipeline):
    results = pipeline.query("Which fertilizer is good for maize?")

    assert results[0]["metadata"]["original_question"] == "Which fertilizer is good for maize?"
    assert len(results) <= 3
    assert 0.0 <= results[0]["relevance"] <= 1.0
    assert results == sorted(results, key=lambda item: item["rerank_score"], reverse=True)


def test_results_are_cached_and_cache_is_bounded(pipeline):
    first = pipeline.query("When should I plant cassava?")
    assert pipeline.query("When should I plant cassava?") is first

    pipeline.cache.max_size = 2
    for crop in ("maize", "beans", "mango"):
        pipeline.query(f"tell me about {crop}")
    assert len(pipeline.cache) == 2


def test_expansion_can_be_limited_to_keyword_search(corpus):
    chunks, bm25_chunks, metadata = corpus
    seen = []

    class RecordingEmbedder(FakeEmbedder):
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return super().encode(texts, **kwargs)

    embedder = RecordingEmbedder()
    index = build_faiss_index(embedder.encode(chunks))
    seen.clear()

    pipe = AgriRAGPipeline(
        embedder,
        index,
        build_bm25_index(bm25_chunks),
        chunks,
        metadata,
        cross_encoder=FakeReranker(),
        bm25_chunks=bm25_chunks,
        bm25_stopwords=STOPWORDS,
        query_expander=lambda q: q + " anthracnose",
        expansion_target="bm25",
    )
    results = pipe.query("mango problem")

    assert seen == ["mango problem"]
    assert results[0]["expanded_query"] == "mango problem anthracnose"

    with pytest.raises(ValueError):
        AgriRAGPipeline(embedder, index, None, chunks, metadata, expansion_target="dense")


def test_other_variants_share_the_interface(corpus):
    chunks, bm25_chunks, metadata = corpus
    embedder = FakeEmbedder()
    common = dict(
        embedding_model=embedder,
        faiss_index=build_faiss_index(embedder.encode(chunks)),
        bm25_index=build_bm25_index(bm25_chunks),
        chunks=chunks,
        bm25_chunks=bm25_chunks,
        metadata=metadata,
        cross_encoder=FakeReranker(),
        bm25_stopwords=STOPWORDS,
        final_top_k=2,
    )

    for variant in (
        SemanticOnlyPipeline(**common),
        HybridCEPipeline(**common),
        AgriRAGPipeline(**common, use_reranker=False),
    ):
        results = variant.query("improve clay soil")
        assert results[0]["metadata"]["original_question"] == "How can I improve clay soil?"
        assert len(results) == 2
