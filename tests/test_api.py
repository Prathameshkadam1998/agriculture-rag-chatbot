import pytest
from fastapi.testclient import TestClient

from app.engine import ChatEngine, expand_query, make_generator
from app.main import create_app
from app.settings import Settings
from src.generation import ExtractiveGenerator


class BrokenGenerator:
    name = "llm:broken"

    def generate(self, query, sources):
        raise RuntimeError("provider is down")


@pytest.fixture
def engine(pipeline):
    pipeline.final_top_k = 5
    return ChatEngine(pipeline, ExtractiveGenerator(), {"num_chunks": 5, "dataset": "test", "dataset_rows": 5})


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine=engine)) as test_client:
        yield test_client


def test_expand_query_adds_synonyms_only_when_a_phrase_matches():
    assert expand_query("tractor prices") == "tractor prices"
    assert "chlorosis" in expand_query("Yellow leaves on tomato")


def test_health_and_info(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["chunks"] == 5
    assert health["generator"] == "extractive"

    info = client.get("/api/info").json()
    assert info["examples"]
    assert info["max_question_chars"] == 400


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "AgriBot" in page.text
    assert client.head("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert page.headers["x-content-type-options"] == "nosniff"


def test_chat_answers_from_the_knowledge_base(client):
    data = client.post("/api/chat", json={"question": "  Which fertilizer is good for maize?  "}).json()

    assert data["kind"] == "answer"
    assert data["answer"] == "Maize needs nitrogen, so apply urea or CAN as top dressing."
    assert data["confidence"]["label"] == "High"
    assert data["sources"][0]["question"] == "Which fertilizer is good for maize?"
    assert data["generator"] == "extractive"
    assert set(data["timings_ms"]) == {"retrieval", "generation", "total"}
    shown = {source["question"] for source in data["sources"]}
    assert not shown & set(data["related"])


def test_chat_declines_when_nothing_matches(client):
    data = client.post("/api/chat", json={"question": "best bitcoin wallet"}).json()

    assert data["kind"] == "no_match"
    assert data["sources"] == []
    assert data["confidence"] is None


def test_chat_greets_without_searching(client):
    data = client.post("/api/chat", json={"question": "Hello!"}).json()
    assert data["kind"] == "greeting"
    assert "AgriBot" in data["answer"]


@pytest.mark.parametrize("payload", [{"question": "   "}, {"question": "x" * 401}, {}])
def test_chat_rejects_bad_input(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_generator_failure_falls_back_to_quoting(pipeline):
    pipeline.final_top_k = 5
    engine = ChatEngine(pipeline, BrokenGenerator(), {})

    data = engine.answer("When should I plant cassava?")

    assert data["answer"] == "Plant cassava at the start of the rainy season."
    assert data["generator"] == "extractive (fallback)"


def test_rate_limit(engine, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    with TestClient(create_app(engine=engine, settings=Settings())) as limited:
        codes = [limited.post("/api/chat", json={"question": "hello"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_make_generator_choices(monkeypatch):
    for name in ("GENERATOR", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert make_generator(Settings()).name == "extractive"

    monkeypatch.setenv("GENERATOR", "llm")  # no key -> falls back instead of crashing
    assert make_generator(Settings()).name == "extractive"

    monkeypatch.setenv("GENERATOR", "auto")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    assert make_generator(Settings()).name == "llm:test-model"
    assert "test-key" not in repr(Settings())
