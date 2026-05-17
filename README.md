# Agriculture RAG Chatbot

An agriculture question-answering chatbot built with Retrieval-Augmented Generation (RAG).

The project uses a hybrid retrieval pipeline:

1. Clean agriculture Q&A data
2. Create sentence-based chunks
3. Build semantic embeddings with Sentence Transformers
4. Search with FAISS
5. Search with BM25 keyword retrieval
6. Fuse results with Reciprocal Rank Fusion
7. Rerank using a cross-encoder
8. Generate grounded answers with Flan-T5

## Project Structure

```text
agriculture-rag-chatbot/
├── notebooks/
│   └── 00_agriculture_rag_from_scratch.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── retrieval.py
│   └── generation.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create and activate a clean environment:

```bash
conda create -n agri-rag python=3.11 -y
conda activate agri-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name agri-rag --display-name "Python (agri-rag)"
```

## Dataset

This project uses the Hugging Face dataset:

```text
KisanVaani/agriculture-qa-english-only
```

## Roadmap

- Build a clean RAG notebook from scratch
- Add hybrid FAISS + BM25 retrieval
- Add cross-encoder reranking
- Add source citations and confidence scores
- Add evaluation with Recall@K, MRR, NDCG, and ROUGE
- Optionally build a Streamlit or Gradio demo

