"""Lightweight ONNX models for the deployed app.

The research notebook runs the embedding model and the cross-encoder on
PyTorch through Sentence Transformers. A free hosting instance has 512 MB of
RAM, which PyTorch alone nearly fills, so the web app runs int8-quantised ONNX
exports of the *same* models on onnxruntime instead.

The two classes expose the small part of the Sentence Transformers interface
that ``pipeline.py`` relies on (``encode`` / ``predict``), so the pipelines do
not care which backend they are given.
"""

import numpy as np


def fetch_model_files(repo_id, model_file, cache_dir=None):
    """Return local paths to the model and its tokenizer, downloading if needed.

    The cache is tried first, so a server whose models were downloaded at build
    time starts without touching the network.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    paths = []
    for filename in (model_file, "tokenizer.json"):
        try:
            paths.append(hf_hub_download(repo_id, filename, cache_dir=cache_dir, local_files_only=True))
        except LocalEntryNotFoundError:
            paths.append(hf_hub_download(repo_id, filename, cache_dir=cache_dir))
    return paths


class _OnnxModel:
    """An ONNX session plus its tokenizer, run in batches."""

    max_length = 512

    def __init__(self, repo_id, model_file, cache_dir=None, threads=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path, tokenizer_path = fetch_model_files(repo_id, model_file, cache_dir)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or 1
        options.inter_op_num_threads = 1
        # The arena trades memory for speed by never returning what it allocates.
        # On a 512 MB instance, memory is the scarcer resource.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False

        self.model_name = repo_id
        self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self._input_names = {item.name for item in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=self.max_length)
        self._tokenizer.enable_padding()

    def _forward(self, items, batch_size):
        """Yield the model's first output for each batch of ``items``."""
        for start in range(0, len(items), batch_size):
            encodings = self._tokenizer.encode_batch(items[start : start + batch_size])
            feeds = {
                "input_ids": np.array([enc.ids for enc in encodings], dtype="int64"),
                "attention_mask": np.array([enc.attention_mask for enc in encodings], dtype="int64"),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array([enc.type_ids for enc in encodings], dtype="int64")
            yield self._session.run(None, feeds)[0]


class OnnxEmbedder(_OnnxModel):
    """Drop-in for ``SentenceTransformer.encode`` (CLS pooling, as BGE expects)."""

    def encode(self, texts, batch_size=32, normalize_embeddings=True, **_):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        vectors = np.concatenate([hidden[:, 0, :] for hidden in self._forward(texts, batch_size)]).astype("float32")

        if normalize_embeddings:
            vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

        return vectors


class OnnxReranker(_OnnxModel):
    """Drop-in for ``CrossEncoder.predict`` (returns raw relevance logits)."""

    # Passages are a few sentences long; a shorter window keeps reranking fast.
    max_length = 256

    def predict(self, pairs, batch_size=8, **_):
        pairs = [(str(query), str(passage)) for query, passage in pairs]
        return [float(score) for logits in self._forward(pairs, batch_size) for score in logits.reshape(-1)]
