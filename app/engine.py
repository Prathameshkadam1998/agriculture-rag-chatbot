"""The chatbot engine: loads the index and models once, then answers questions."""

import logging
import re
import threading
import time

from src import config
from src.data_processing import tokenize_for_bm25
from src.embeddings import OnnxEmbedder, OnnxReranker
from src.generation import ExtractiveGenerator, LLMGenerator, confidence_label
from src.index_store import load_index
from src.pipeline import AgriRAGPipeline
from src.retrieval import build_bm25_index, build_faiss_index

logger = logging.getLogger("agribot")

SMALL_TALK = re.compile(
    r"^\W*(hi|hello|hey|hola|namaste|good (morning|afternoon|evening)|thanks|thank you|"
    r"who are you|what can you do|help)\W*$",
    re.IGNORECASE,
)

GREETING = (
    "Hello! I'm AgriBot. Ask me about crops, pests, plant diseases, soil, "
    "fertilisers or irrigation and I'll answer from a knowledge base of farming "
    "questions and answers."
)

NO_MATCH = (
    "I couldn't find a reliable answer to that in my knowledge base, so I'd rather "
    'not guess. Try naming the crop and the problem (for example, "brown spots on '
    'mango leaves"), or pick one of the closest topics below.'
)


def expand_query(query):
    """Append agriculture synonyms for symptom/term phrases found in the query."""
    query_lower = query.lower()
    additions = [expansion for phrase, expansion in config.QUERY_EXPANSIONS.items() if phrase in query_lower]
    return f"{query} {' '.join(additions)}" if additions else query


def _normalise(text):
    return " ".join(tokenize_for_bm25(text))


def make_generator(settings):
    """Build the configured generator; a bad configuration falls back to extractive."""
    choice = settings.generator
    has_llm = bool(settings.llm_api_key and settings.llm_model)

    if has_llm and choice in ("auto", "llm"):
        return LLMGenerator(settings.llm_api_key, settings.llm_model, settings.llm_base_url)

    if choice == "llm":
        logger.warning("GENERATOR=llm needs both LLM_API_KEY and LLM_MODEL; using extractive answers.")
    elif choice not in ("auto", "extractive"):
        logger.warning("Unknown GENERATOR value %r; using extractive answers.", choice)

    return ExtractiveGenerator()


class ChatEngine:
    def __init__(self, pipeline, generator, manifest):
        self.pipeline = pipeline
        self.generator = generator
        self.manifest = manifest
        self._fallback = ExtractiveGenerator()
        # The instance has a fraction of one CPU; queueing requests is faster
        # than letting them fight over it.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings):
        started = time.perf_counter()

        embeddings, chunks, bm25_chunks, metadata, manifest = load_index(settings.index_dir)

        pipeline = AgriRAGPipeline(
            embedding_model=OnnxEmbedder(
                config.ONNX_EMBEDDING_REPO,
                config.ONNX_EMBEDDING_FILE,
                cache_dir=settings.model_cache_dir,
                threads=settings.onnx_threads,
            ),
            cross_encoder=OnnxReranker(
                config.ONNX_RERANKER_REPO,
                config.ONNX_RERANKER_FILE,
                cache_dir=settings.model_cache_dir,
                threads=settings.onnx_threads,
            ),
            faiss_index=build_faiss_index(embeddings),
            bm25_index=build_bm25_index(bm25_chunks),
            chunks=chunks,
            bm25_chunks=bm25_chunks,
            metadata=metadata,
            retrieval_pool_k=settings.rerank_pool,
            # Fetch extra so there are still enough sources after de-duplication.
            final_top_k=config.ANSWER_TOP_K + 5,
            rrf_k=config.RRF_K,
            query_expander=expand_query,
            expansion_target="bm25",
            bm25_stopwords=frozenset(manifest.get("bm25_stopwords", ())),
        )

        engine = cls(pipeline, make_generator(settings), manifest)
        logger.info(
            "Engine ready in %.1fs: %d chunks, generator=%s",
            time.perf_counter() - started,
            len(chunks),
            engine.generator.name,
        )
        return engine

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _unique_sources(results):
        """One source per distinct question/answer, best first."""
        sources, seen = [], set()

        for result in results:
            meta = result["metadata"]
            keys = {
                "q:" + _normalise(meta["original_question"]),
                "a:" + _normalise(meta["original_answer"]),
            }
            if keys & seen:
                continue
            seen |= keys

            sources.append(
                {
                    "id": result["chunk_id"],
                    "question": meta["original_question"],
                    "answer": meta["original_answer"],
                    "relevance": round(float(result.get("relevance", 0.0)), 4),
                }
            )

        return sources

    def info(self):
        return {
            "chunks": self.manifest.get("num_chunks"),
            "dataset": self.manifest.get("dataset"),
            "dataset_rows": self.manifest.get("dataset_rows"),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "reranker_model": config.CROSS_ENCODER_MODEL_NAME,
            "generator": self.generator.name,
        }

    # -- public API ----------------------------------------------------------

    def answer(self, question):
        started = time.perf_counter()
        question = " ".join(question.split())

        if SMALL_TALK.match(question):
            return self._response(GREETING, started, kind="greeting")

        with self._lock:
            results = self.pipeline.query(question)
        retrieval_ms = (time.perf_counter() - started) * 1000

        candidates = self._unique_sources(results)
        best = candidates[0]["relevance"] if candidates else 0.0

        if best < config.MIN_RELEVANCE:
            return self._response(
                NO_MATCH,
                started,
                kind="no_match",
                # Only suggest neighbours when the question was at least near
                # the knowledge base; for "who won the world cup" they are noise.
                related=[s["question"] for s in candidates[:3]] if best >= config.NEAR_MISS_RELEVANCE else [],
                retrieval_ms=retrieval_ms,
            )

        # Only sources in the same league as the best one are worth showing.
        floor = max(config.MIN_RELEVANCE, best * 0.25)
        sources = [s for s in candidates if s["relevance"] >= floor][: config.ANSWER_TOP_K]
        shown = {source["id"] for source in sources}
        related = [s["question"] for s in candidates if s["id"] not in shown][:3]

        generation_started = time.perf_counter()
        generator_name = self.generator.name
        try:
            text = self.generator.generate(question, sources)
        except Exception as error:  # noqa: BLE001 - never fail the request over generation
            logger.warning("Generator %s failed (%s); using extractive answer.", generator_name, error)
            text = self._fallback.generate(question, sources)
            generator_name = f"{self._fallback.name} (fallback)"

        return self._response(
            text,
            started,
            kind="answer",
            sources=sources,
            related=related,
            relevance=best,
            generator=generator_name,
            retrieval_ms=retrieval_ms,
            generation_ms=(time.perf_counter() - generation_started) * 1000,
        )

    @staticmethod
    def _response(
        text,
        started,
        *,
        kind,
        sources=(),
        related=(),
        relevance=None,
        generator=None,
        retrieval_ms=0.0,
        generation_ms=0.0,
    ):
        return {
            "answer": text,
            "kind": kind,
            "confidence": (
                {"label": confidence_label(relevance), "score": round(relevance, 4)} if relevance is not None else None
            ),
            "sources": list(sources),
            "related": list(related),
            "generator": generator,
            "timings_ms": {
                "retrieval": round(retrieval_ms),
                "generation": round(generation_ms),
                "total": round((time.perf_counter() - started) * 1000),
            },
        }
