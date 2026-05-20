"""End-to-end RAG pipeline classes.

These classes keep the notebook simple while reusing the helper functions in
``retrieval.py``.
"""

import numpy as np

from retrieval import bm25_search, reciprocal_rank_fusion, semantic_search


def default_query_expander(query):
    return query


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
    ):
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
        self.cache = {}

    def query(self, query, use_cache=True):
        cache_key = query

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        expanded_query = self.query_expander(query)

        semantic_results = semantic_search(
            query=expanded_query,
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
        )

        fused = reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            rrf_k=self.rrf_k,
        )

        results = []
        for item in fused[: self.retrieval_pool_k]:
            cid = item["chunk_id"]
            result = {
                "chunk_id": cid,
                "chunk": self.chunks[cid],
                "score": item["score"],
                "metadata": self.metadata[cid],
                "original_query": query,
                "expanded_query": expanded_query,
            }
            results.append(result)

        if self.use_reranker and self.cross_encoder is not None and results:
            pairs = [[query, result["chunk"]] for result in results]
            scores = self.cross_encoder.predict(pairs)

            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
                result["score"] = float(score)

            results = sorted(
                results,
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

        results = results[: self.final_top_k]

        if use_cache:
            self.cache[cache_key] = results

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
        **_,
    ):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.metadata = metadata
        self.final_top_k = final_top_k
        self.query_expander = query_expander or default_query_expander
        self.cache = {}

    def query(self, query, use_cache=True):
        cache_key = query

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

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
            self.cache[cache_key] = results

        return results


class HybridCEPipeline(AgriRAGPipeline):
    """Hybrid retrieval pipeline with score blending and optional reranking."""

    def __init__(self, *args, semantic_weight=0.6, bm25_weight=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def query(self, query, use_cache=True):
        cache_key = query

        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        expanded_query = self.query_expander(query)

        semantic_results = semantic_search(
            query=expanded_query,
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
        )

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

        results = []
        for cid in ranked_ids[: self.retrieval_pool_k]:
            results.append({
                "chunk_id": cid,
                "chunk": self.chunks[cid],
                "score": float(blended[cid]),
                "metadata": self.metadata[cid],
                "original_query": query,
                "expanded_query": expanded_query,
            })

        if self.use_reranker and self.cross_encoder is not None and results:
            pairs = [[query, result["chunk"]] for result in results]
            scores = self.cross_encoder.predict(pairs)

            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
                result["score"] = float(score)

            results = sorted(
                results,
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

        results = results[: self.final_top_k]

        if use_cache:
            self.cache[cache_key] = results

        return results
