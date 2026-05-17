"""Retrieval helpers for semantic search, BM25, and fusion."""

from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def build_bm25_index(bm25_chunks):
    tokenized_chunks = [chunk.split() for chunk in bm25_chunks]
    return BM25Okapi(tokenized_chunks)


def reciprocal_rank_fusion(result_lists, rrf_k=60):
    fused_scores = defaultdict(float)

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            fused_scores[item["chunk_id"]] += 1.0 / (rrf_k + rank)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [{"chunk_id": cid, "score": fused_scores[cid]} for cid in ranked_ids]


def semantic_search(query, embedding_model, faiss_index, chunks, top_k):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = faiss_index.search(query_embedding, top_k)

    results = []
    for score, cid in zip(scores[0], ids[0]):
        if cid < 0:
            continue
        results.append({
            "chunk_id": int(cid),
            "chunk": chunks[int(cid)],
            "score": float(score),
            "source": "semantic",
        })

    return results


def bm25_search(query, bm25_index, bm25_chunks, embedding_chunks, top_k):
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    top_ids = np.argsort(scores)[::-1][:top_k]

    return [{
        "chunk_id": int(cid),
        "chunk": embedding_chunks[int(cid)],
        "score": float(scores[int(cid)]),
        "source": "bm25",
    } for cid in top_ids]

