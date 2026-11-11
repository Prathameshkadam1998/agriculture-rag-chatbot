"""End-to-end RAG retrieval pipelines.

The pipelines only need two duck-typed models:

* ``embedding_model.encode(list_of_texts, normalize_embeddings=True)``
* ``cross_encoder.predict(list_of_[query, passage]_pairs)``

so they work with Sentence Transformers (the research notebook) and with the
lightweight ONNX models in ``embeddings.py`` (the deployed app).
"""

import math
from collections import OrderedDict

import numpy as np

try:
    from .retrieval import bm25_search, reciprocal_rank_fusion, semantic_search
except ImportError:  # notebook style: ``src/`` itself is on sys.path
    from retrieval import bm25_search, reciprocal_rank_fusion, semantic_search


def default_query_expander(query):
    return query


def sigmoid(value):
    """Map a cross-encoder logit to a 0-1 relevance score."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class _QueryCache:
    """Small LRU cache so a long-running server cannot grow without bound."""

    def __init__(self, max_size=256):
        self.max_size = max_size
        self._items = OrderedDict()

    def __contains__(self, key):
        return key in self._items

    def __getitem__(self, key):
        self._items.move_to_end(key)
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def __len__(self):
        return len(self._items)

    def clear(self):
        self._items.clear()


class AgriRAGPipeline:
    """Hybrid FAISS + BM25 pipeline with optional cross-encoder reranking."""

    def __init__(
        self,
        embedding_model,
        faiss_index,
        bm25_index,
        chunks,
        metadata,
        cross_encoder=None,
        bm25_chunks=None,
        retrieval_pool_k=20,
        final_top_k=5,
        rrf_k=60,
        use_reranker=True,
        query_expander=None,
        bm25_stopwords=(),
        expansion_target="both",
        cache_size=256,
    ):
        if expansion_target not in ("both", "bm25"):
            raise ValueError("expansion_target must be 'both' or 'bm25'")

        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.metadata = metadata
        self.cross_encoder = cross_encoder
        self.bm25_chunks = bm25_chunks if bm25_chunks is not None else chunks
        self.retrieval_pool_k = retrieval_pool_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.query_expander = query_expander or default_query_expander
        self.bm25_stopwords = bm25_stopwords
        # "both": the expanded query feeds dense and keyword search (notebook
        # behaviour). "bm25": synonyms only widen keyword recall and the dense
        # query stays exactly what the user typed.
        self.expansion_target = expansion_target
        self.cache = _QueryCache(cache_size)

    # -- building blocks ---------------------------------------------------

    def _candidates(self, query, expanded_query):
        semantic_results = semantic_search(
            query=expanded_query if self.expansion_target == "both" else query,
            embedding_model=self.embedding_model,
            faiss_index=self.faiss_index,
            chunks=self.chunks,
            top_k=self.retrieval_pool_k,
        )

        bm25_results = bm25_search(
            query=expanded_query,
            bm25_index=self.bm25_index,
            bm25_chunks=self.bm25_chunks,
            embedding_chunks=self.chunks,
            top_k=self.retrieval_pool_k,
            stopwords=self.bm25_stopwords,
        )

        return semantic_results, bm25_results

    def _fuse(self, semantic_results, bm25_results):
        fused = reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            rrf_k=self.rrf_k,
        )
        return [(item["chunk_id"], item["score"]) for item in fused]

    def _rerank(self, query, results):
        if not (self.use_reranker and self.cross_encoder is not None and results):
            return results

        pairs = [[query, result["chunk"]] for result in results]
        scores = self.cross_encoder.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            result["score"] = float(score)
            result["relevance"] = sigmoid(float(score))

        return sorted(results, key=lambda item: item["rerank_score"], reverse=True)

    # -- public API --------------------------------------------------------

    def query(self, query, use_cache=True):
        if use_cache and query in self.cache:
            return self.cache[query]

        expanded_query = self.query_expander(query)
        semantic_results, bm25_results = self._candidates(query, expanded_query)

        results = [
            {
                "chunk_id": cid,
                "chunk": self.chunks[cid],
                "score": float(score),
                "metadata": self.metadata[cid],
                "original_query": query,
                "expanded_query": expanded_query,
            }
            for cid, score in self._fuse(semantic_results, bm25_results)[: self.retrieval_pool_k]
        ]

        results = self._rerank(query, results)[: self.final_top_k]

        if use_cache:
            self.cache[query] = results

        return results


class SemanticOnlyPipeline:
    """FAISS-only semantic retrieval pipeline."""

    def __init__(
        self,
        embedding_model,
        faiss_index,
        chunks,
        metadata,
        final_top_k=5,
        query_expander=None,
        cache_size=256,
        **_,
    ):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.metadata = metadata
        self.final_top_k = final_top_k
        self.query_expander = query_expander or default_query_expander
        self.cache = _QueryCache(cache_size)

    def query(self, query, use_cache=True):
        if use_cache and query in self.cache:
            return self.cache[query]

        expanded_query = self.query_expander(query)

        results = semantic_search(
            query=expanded_query,
            embedding_model=self.embedding_model,
            faiss_index=self.faiss_index,
            chunks=self.chunks,
            top_k=self.final_top_k,
        )

        for result in results:
            result["metadata"] = self.metadata[result["chunk_id"]]
            result["original_query"] = query
            result["expanded_query"] = expanded_query

        if use_cache:
            self.cache[query] = results

        return results


class HybridCEPipeline(AgriRAGPipeline):
    """Hybrid retrieval that blends normalised scores instead of ranks."""

    def __init__(self, *args, semantic_weight=0.6, bm25_weight=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def _fuse(self, semantic_results, bm25_results):
        blended = {}

        for result in semantic_results:
            blended[result["chunk_id"]] = self.semantic_weight * result["score"]

        bm25_scores = np.array([result["score"] for result in bm25_results], dtype=float)

        if bm25_scores.size and bm25_scores.max() > bm25_scores.min():
            bm25_scores = (
                bm25_scores - bm25_scores.min()
            ) / (
                bm25_scores.max() - bm25_scores.min()
            )

        for result, norm_score in zip(bm25_results, bm25_scores):
            blended[result["chunk_id"]] = (
                blended.get(result["chunk_id"], 0.0)
                + self.bm25_weight * float(norm_score)
            )

        ranked_ids = sorted(blended, key=blended.get, reverse=True)
        return [(cid, blended[cid]) for cid in ranked_ids]
