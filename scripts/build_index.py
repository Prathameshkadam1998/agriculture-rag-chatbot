"""Build the retrieval index that the web app loads at start-up.

Downloads the dataset, cleans and chunks it, embeds every chunk with the ONNX
embedding model, and writes the result to ``data/index``.

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --output data/index --limit 500
"""

import argparse
import time
from datetime import UTC, datetime

from src import config
from src.data_processing import build_chunks, load_bm25_stopwords
from src.embeddings import OnnxEmbedder
from src.index_store import save_index


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=config.INDEX_DIR, help="directory to write the index to")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N rows (for quick tests)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads used for embedding")
    args = parser.parse_args()

    from datasets import load_dataset

    started = time.perf_counter()

    print(f"Loading dataset: {config.DATASET_NAME}")
    df = load_dataset(config.DATASET_NAME, split="train").to_pandas()
    df = df.dropna(subset=["question", "answers"])
    if args.limit:
        df = df.head(args.limit)
    print(f"Rows: {len(df):,}")

    stopwords = load_bm25_stopwords()
    embedding_chunks, bm25_chunks, metadata = build_chunks(df, stopwords)
    print(f"Unique chunks: {len(embedding_chunks):,}")

    print(f"Embedding with: {config.ONNX_EMBEDDING_REPO} ({config.ONNX_EMBEDDING_FILE})")
    embedder = OnnxEmbedder(config.ONNX_EMBEDDING_REPO, config.ONNX_EMBEDDING_FILE, threads=args.threads)
    embeddings = embedder.encode(embedding_chunks, batch_size=config.EMBEDDING_BATCH_SIZE)

    manifest = save_index(
        args.output,
        embeddings=embeddings,
        bm25_chunks=bm25_chunks,
        metadata=metadata,
        manifest={
            "dataset": config.DATASET_NAME,
            "dataset_rows": int(len(df)),
            "embedding_model": config.ONNX_EMBEDDING_REPO,
            "embedding_file": config.ONNX_EMBEDDING_FILE,
            "chunking": {
                "strategy": "sentence",
                "max_sentences": config.MAX_SENTENCES,
                "overlap_sentences": config.OVERLAP_SENTENCES,
            },
            "bm25_stopwords": sorted(stopwords),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )

    elapsed = time.perf_counter() - started
    print(f"Saved {manifest['num_chunks']:,} chunks ({manifest['dim']} dims) to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
