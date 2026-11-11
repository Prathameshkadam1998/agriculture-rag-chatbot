"""Save and load the pre-built retrieval index.

The index is three small files so it can live in the repository and be loaded
without pandas, NLTK, or the dataset:

* ``embeddings.npy`` - float16 matrix, one L2-normalised row per chunk
* ``chunks.jsonl``   - one JSON object per chunk (text + original Q&A)
* ``manifest.json``  - how the index was built (dataset, models, stopwords)
"""

import json
from pathlib import Path

import numpy as np

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


def save_index(index_dir, embeddings, bm25_chunks, metadata, manifest):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if not (len(embeddings) == len(bm25_chunks) == len(metadata)):
        raise ValueError("embeddings, bm25_chunks and metadata must be aligned")

    np.save(index_dir / EMBEDDINGS_FILE, np.asarray(embeddings, dtype="float16"))

    with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as handle:
        for meta, bm25_text in zip(metadata, bm25_chunks):
            record = {
                "chunk_id": meta["chunk_id"],
                "source_row": meta["source_row"],
                "question": meta["original_question"],
                "answer": meta["original_answer"],
                "chunk_text": meta["chunk_text"],
                "bm25_text": bm25_text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = dict(manifest, num_chunks=len(metadata), dim=int(np.shape(embeddings)[1]))
    (index_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def index_exists(index_dir):
    index_dir = Path(index_dir)
    return all((index_dir / name).exists() for name in (EMBEDDINGS_FILE, CHUNKS_FILE, MANIFEST_FILE))


def load_index(index_dir):
    """Return ``(embeddings, chunks, bm25_chunks, metadata, manifest)``."""
    index_dir = Path(index_dir)

    if not index_exists(index_dir):
        raise FileNotFoundError(f"No index found in {index_dir}. Build it with: python -m scripts.build_index")

    embeddings = np.load(index_dir / EMBEDDINGS_FILE).astype("float32")
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))

    chunks, bm25_chunks, metadata = [], [], []
    with open(index_dir / CHUNKS_FILE, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(record["chunk_text"])
            bm25_chunks.append(record["bm25_text"])
            metadata.append(
                {
                    "chunk_id": record["chunk_id"],
                    "source_row": record["source_row"],
                    "original_question": record["question"],
                    "original_answer": record["answer"],
                }
            )

    if len(chunks) != len(embeddings):
        raise ValueError(f"Index is inconsistent: {len(chunks)} chunks but {len(embeddings)} embeddings")

    return embeddings, chunks, bm25_chunks, metadata, manifest
