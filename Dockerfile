# Portable image for any container host (Render, Cloud Run, Fly, a VPS, ...).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_CACHE_DIR=/app/.models \
    PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend
COPY data/index ./data/index

# Bake the models into the image so a cold start downloads nothing.
RUN python -m scripts.download_models

RUN useradd --create-home --uid 1000 agribot && chown -R agribot /app
USER agribot

EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
