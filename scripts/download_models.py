"""Download the ONNX models into the model cache.

Run once at build time so the server never has to download anything when it
starts (free instances restart often, and start-up should not depend on the
network).

Usage:
    python -m scripts.download_models
"""

from app.settings import Settings
from src import config
from src.embeddings import fetch_model_files


def main():
    cache_dir = Settings().model_cache_dir

    for repo_id, model_file in (
        (config.ONNX_EMBEDDING_REPO, config.ONNX_EMBEDDING_FILE),
        (config.ONNX_RERANKER_REPO, config.ONNX_RERANKER_FILE),
    ):
        for path in fetch_model_files(repo_id, model_file, cache_dir):
            print(f"ok  {path}")


if __name__ == "__main__":
    main()
