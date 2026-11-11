"""Shared fixtures: a tiny corpus and fake models, so tests need no downloads."""

import zlib

import numpy as np
import pytest

from src.data_processing import clean_for_bm25, tokenize_for_bm25
from src.pipeline import AgriRAGPipeline
from src.retrieval import build_bm25_index, build_faiss_index

STOPWORDS = frozenset({"how", "do", "i", "to", "the", "in", "is", "a", "for", "what", "my", "on", "of"})

QA = [
    ("How do I control aphids on beans?", "Spray neem oil and encourage ladybirds to control aphids."),
    ("Which fertilizer is good for maize?", "Maize needs nitrogen, so apply urea or CAN as top dressing."),
    ("How can I improve clay soil?", "Add compost and organic matter to improve drainage in clay soil."),
    ("When should I plant cassava?", "Plant cassava at the start of the rainy season."),
    ("What causes brown spots on mango leaves?", "Anthracnose causes brown spots on mango leaves."),
]


class FakeEmbedder:
    """Deterministic bag-of-words embeddings."""

    dim = 64

    def encode(self, texts, normalize_embeddings=True, **_):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize_for_bm25(text, STOPWORDS):
                vectors[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return vectors


class FakeReranker:
    """Scores a pair by word overlap, on a logit-like scale."""

    def predict(self, pairs, **_):
        scores = []
        for query, passage in pairs:
            query_tokens = set(tokenize_for_bm25(query, STOPWORDS))
            overlap = len(query_tokens & set(tokenize_for_bm25(passage, STOPWORDS)))
            scores.append(8.0 * overlap / max(len(query_tokens), 1) - 4.0)
        return scores


@pytest.fixture
def corpus():
    chunks = [f"question: {q.lower()}\nanswer: {a.lower()}" for q, a in QA]
    bm25_chunks = [clean_for_bm25(chunk, STOPWORDS) for chunk in chunks]
    metadata = [
        {"chunk_id": i, "source_row": i, "original_question": q, "original_answer": a} for i, (q, a) in enumerate(QA)
    ]
    return chunks, bm25_chunks, metadata


@pytest.fixture
def pipeline(corpus):
    chunks, bm25_chunks, metadata = corpus
    embedder = FakeEmbedder()
    return AgriRAGPipeline(
        embedding_model=embedder,
        faiss_index=build_faiss_index(embedder.encode(chunks)),
        bm25_index=build_bm25_index(bm25_chunks),
        chunks=chunks,
        bm25_chunks=bm25_chunks,
        metadata=metadata,
        cross_encoder=FakeReranker(),
        retrieval_pool_k=5,
        final_top_k=3,
        bm25_stopwords=STOPWORDS,
    )
