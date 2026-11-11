"""FastAPI app: JSON API under /api and the static frontend at /."""

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src import config

from .engine import ChatEngine
from .settings import ROOT_DIR, Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = ROOT_DIR / "frontend"

EXAMPLES = [
    {"topic": "Pests", "icon": "bug", "question": "How do I manage fall armyworm in maize?"},
    {"topic": "Disease", "icon": "leaf", "question": "What causes brown spots on mango leaves?"},
    {"topic": "Soil", "icon": "layers", "question": "How can I improve the fertility of clay soil?"},
    {"topic": "Fertiliser", "icon": "flask", "question": "Which fertilizer is good for maize?"},
    {"topic": "Planting", "icon": "sprout", "question": "When is the best time to plant cassava?"},
    {"topic": "Irrigation", "icon": "drop", "question": "How much water do potatoes need?"},
]


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=config.MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value):
        value = " ".join(value.split())
        if not value:
            raise ValueError("Please type a question.")
        return value


class RateLimiter:
    """Sliding one-minute window per client. Enough to protect a small demo."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client):
        if self.per_minute <= 0:
            return True

        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)

            if len(self._hits) > 10_000:  # forget idle clients
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] > 60]:
                    del self._hits[key]
            return True


def client_id(request):
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(engine=None, settings=None):
    settings = settings or Settings()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine or ChatEngine.load(settings)
        yield

    app = FastAPI(title="AgriBot", version="1.0.0", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return response

    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", **request.app.state.engine.info()}

    @app.get("/api/info")
    def info(request: Request):
        return {
            **request.app.state.engine.info(),
            "examples": EXAMPLES,
            "max_question_chars": config.MAX_QUESTION_CHARS,
        }

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        if not limiter.allow(client_id(request)):
            raise HTTPException(status_code=429, detail="Too many questions in a short time. Please wait a minute.")
        return request.app.state.engine.answer(body.question)

    # html=True serves index.html at "/". Mounted last so /api routes win.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
