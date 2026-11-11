"""Data loading, cleaning, and chunking helpers."""

import hashlib
import re

try:
    from .config import DOMAIN_STOPWORDS, MAX_SENTENCES, OVERLAP_SENTENCES
except ImportError:  # notebook style: ``src/`` itself is on sys.path
    from config import DOMAIN_STOPWORDS, MAX_SENTENCES, OVERLAP_SENTENCES

# Negations carry meaning in farming advice ("do not spray"), so they are kept
# even though NLTK lists them as stopwords.
KEEP_WORDS = {"no", "not", "nor", "against"}

_NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_data():
    """Download the NLTK resources the pipeline needs, if they are missing."""
    import nltk

    for package, resource in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def load_bm25_stopwords():
    """English stopwords plus the project's domain stopwords, minus negations."""
    ensure_nltk_data()
    from nltk.corpus import stopwords

    return (set(stopwords.words("english")) | set(DOMAIN_STOPWORDS)) - KEEP_WORDS


def repair_text(text):
    """Repair the encoding damage found in a few dataset rows (shown as U+FFFD).

    The original characters are lost, so this restores the obvious cases
    (degree signs, apostrophes, number ranges) and drops the rest.
    """
    if not isinstance(text, str):
        return ""
    if "\ufffd" not in text:
        return text.strip()

    text = re.sub(r"(?<=\d)\ufffd(?=[CF]\b)", "\u00b0", text)
    text = re.sub(r"(?<=\d)\ufffd(?=\d)", "-", text)
    text = re.sub(r"(?<=[A-Za-z])\ufffd(?=[A-Za-z])", "'", text)
    text = text.replace("\ufffd", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,?!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_bm25(text, stopwords=()):
    """Lowercase, strip punctuation, and drop stopwords.

    Chunks and queries must go through the same function, otherwise a query
    token such as ``cotton?`` never matches the indexed token ``cotton``.
    """
    text = re.sub(r"[^a-z0-9\s]", " ", clean_text(text))
    return [word for word in text.split() if word not in stopwords]


def clean_for_bm25(text, stopwords):
    return " ".join(tokenize_for_bm25(text, stopwords))


def chunk_sentence_based(text, max_sentences=MAX_SENTENCES, overlap_sentences=OVERLAP_SENTENCES):
    if not isinstance(text, str) or not text.strip():
        return []

    if overlap_sentences >= max_sentences:
        raise ValueError("overlap_sentences must be smaller than max_sentences")

    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)

    if len(sentences) <= max_sentences:
        return [text]

    chunks = []
    start = 0
    step = max_sentences - overlap_sentences

    while start < len(sentences):
        chunk = " ".join(sentences[start:start + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_sentences >= len(sentences):
            break
        start += step

    return chunks


def build_chunks(df, bm25_stopwords, show_progress=True):
    """Turn a question/answers dataframe into aligned, de-duplicated chunks.

    Returns ``(embedding_chunks, bm25_chunks, metadata)`` where position ``i``
    in each list describes the same chunk.
    """
    ensure_nltk_data()

    rows = df.iterrows()
    if show_progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(df), desc="Building chunks")

    embedding_chunks = []
    bm25_chunks = []
    metadata = []
    seen = set()

    for idx, row in rows:
        question = clean_text(row["question"])
        answer = clean_text(row["answers"])

        if not question or not answer:
            continue

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
                "source_row": int(idx),
                "original_question": repair_text(str(row["question"])),
                "original_answer": repair_text(str(row["answers"])),
                "answer_chunk": answer_chunk,
                "chunk_text": chunk_text,
            })

    return embedding_chunks, bm25_chunks, metadata
