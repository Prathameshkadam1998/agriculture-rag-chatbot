"""Runtime settings, read from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src import config

ROOT_DIR = Path(__file__).resolve().parents[1]


def _path(value):
    """Resolve a path from the environment against the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    index_dir: Path = field(default_factory=lambda: _path(os.environ.get("INDEX_DIR", config.INDEX_DIR)))
    # Where the ONNX models are stored. Unset = the shared Hugging Face cache.
    model_cache_dir: str | None = field(
        default_factory=lambda: str(_path(os.environ["MODEL_CACHE_DIR"])) if os.environ.get("MODEL_CACHE_DIR") else None
    )
    onnx_threads: int = field(default_factory=lambda: _int("ONNX_THREADS", 1))
    rerank_pool: int = field(default_factory=lambda: _int("RERANK_POOL", config.TOP_K_SEMANTIC))

    # "auto" picks the LLM when LLM_API_KEY and LLM_MODEL are set, otherwise
    # the extractive generator. Other values: extractive, llm.
    generator: str = field(default_factory=lambda: os.environ.get("GENERATOR", "auto").strip().lower())
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "").strip(), repr=False)
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "").strip())
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
    )

    rate_limit_per_minute: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MINUTE", 30))
