import pandas as pd
import pytest

from src.data_processing import (
    build_chunks,
    chunk_sentence_based,
    clean_for_bm25,
    clean_text,
    repair_text,
    tokenize_for_bm25,
)


def test_clean_text_lowercases_and_strips_urls_and_symbols():
    assert clean_text("Visit https://example.com NOW!  50% off™") == "visit now! 50 off"
    assert clean_text(None) == ""


def test_bm25_tokens_drop_punctuation_and_stopwords():
    assert tokenize_for_bm25("How to control pests in cotton?", {"how", "to", "in"}) == [
        "control",
        "pests",
        "cotton",
    ]
    assert clean_for_bm25("Soil, soil. SOIL!", {"soil"}) == ""


def test_repair_text_restores_common_encoding_damage():
    assert repair_text("between 10-15�C") == "between 10-15°C"
    assert repair_text("the hen�s diet") == "the hen's diet"
    assert repair_text("ages 0�6") == "ages 0-6"
    assert "�" not in repair_text("� Organic �")


def test_short_text_is_one_chunk():
    assert chunk_sentence_based("One sentence. Two sentences.") == ["One sentence. Two sentences."]


def test_long_text_is_split_with_overlap_and_no_redundant_tail():
    text = " ".join(f"Sentence number {i}." for i in range(1, 8))
    chunks = chunk_sentence_based(text, max_sentences=4, overlap_sentences=1)

    assert chunks == [
        "Sentence number 1. Sentence number 2. Sentence number 3. Sentence number 4.",
        "Sentence number 4. Sentence number 5. Sentence number 6. Sentence number 7.",
    ]


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_sentence_based("A. B. C.", max_sentences=2, overlap_sentences=2)


def test_build_chunks_dedupes_and_keeps_lists_aligned():
    df = pd.DataFrame(
        {
            "question": ["What is crop rotation?", "What is crop rotation?", "Why mulch?", None],
            "answers": [
                "Growing different crops in sequence.",
                "Growing different crops in sequence.",
                "It keeps moisture.",
                "x",
            ],
        }
    )

    chunks, bm25_chunks, metadata = build_chunks(df, {"what", "is", "it", "in", "why"}, show_progress=False)

    assert len(chunks) == len(bm25_chunks) == len(metadata) == 2
    assert chunks[0] == "question: what is crop rotation?\nanswer: growing different crops in sequence."
    assert bm25_chunks[0] == "question crop rotation answer growing different crops sequence"
    assert metadata[1]["original_question"] == "Why mulch?"
    assert [meta["chunk_id"] for meta in metadata] == [0, 1]
