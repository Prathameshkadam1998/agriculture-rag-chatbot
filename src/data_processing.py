"""Data loading, cleaning, and chunking helpers."""

import hashlib
import re

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from .config import MAX_SENTENCES, OVERLAP_SENTENCES


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,?!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_bm25(text, stopwords):
    text = clean_text(text)
    return " ".join(word for word in text.split() if word not in stopwords)


def chunk_sentence_based(text, max_sentences=MAX_SENTENCES, overlap_sentences=OVERLAP_SENTENCES):
    if not isinstance(text, str) or not text.strip():
        return []

    if overlap_sentences >= max_sentences:
        raise ValueError("overlap_sentences must be smaller than max_sentences")

    sentences = sent_tokenize(text)

    if len(sentences) <= max_sentences:
        return [text]

    chunks = []
    start = 0

    while start < len(sentences):
        chunk = " ".join(sentences[start:start + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
        start += max_sentences - overlap_sentences

    return chunks


def build_chunks(df, bm25_stopwords):
    embedding_chunks = []
    bm25_chunks = []
    metadata = []
    seen = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building chunks"):
        question = clean_text(row["question"])
        answer = clean_text(row["answers"])

        for answer_chunk in chunk_sentence_based(answer):
            chunk_text = f"question: {question}\nanswer: {answer_chunk}"
            chunk_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()

            if chunk_hash in seen:
                continue

            seen.add(chunk_hash)
            chunk_id = len(embedding_chunks)

            embedding_chunks.append(chunk_text)
            bm25_chunks.append(clean_for_bm25(chunk_text, bm25_stopwords))

            metadata.append({
                "chunk_id": chunk_id,
                "source_row": idx,
                "original_question": row["question"],
                "original_answer": row["answers"],
                "answer_chunk": answer_chunk,
                "chunk_text": chunk_text,
            })

    return embedding_chunks, bm25_chunks, metadata
