import json

import httpx
import pytest

from src.generation import ExtractiveGenerator, LLMGenerator, build_context, confidence_label

SOURCES = [
    {"question": "When should I plant cassava?", "answer": "plant at the start of the rains.", "relevance": 0.98},
    {"question": "How deep?", "answer": "About 10 cm.", "relevance": 0.61},
]


def test_confidence_labels():
    assert confidence_label(0.99) == "High"
    assert confidence_label(0.75) == "High"
    assert confidence_label(0.5) == "Medium"
    assert confidence_label(0.1) == "Low"


def test_context_is_numbered_and_trimmed():
    context = build_context(SOURCES)
    assert context.startswith("[1] Q: When should I plant cassava?\nA: plant at the start of the rains.")
    assert "[2] Q: How deep?" in context

    assert build_context(SOURCES, max_words=4).endswith("...")


def test_extractive_quotes_the_best_source():
    generator = ExtractiveGenerator()
    assert generator.generate("when to plant cassava", SOURCES) == "Plant at the start of the rains."
    assert generator.generate("anything", []) == ""


def test_llm_generator_sends_sources_and_reads_the_reply():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Plant when the rains begin [1]. "}}]})

    generator = LLMGenerator(
        "test-key",
        "test-model",
        "https://llm.example/v1/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert generator.name == "llm:test-model"
    assert generator.generate("When do I plant cassava?", SOURCES) == "Plant when the rains begin [1]."
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "test-model"
    assert "[2] Q: How deep?" in captured["body"]["messages"][1]["content"]
    assert "When do I plant cassava?" in captured["body"]["messages"][1]["content"]


def test_llm_generator_raises_on_http_error_and_empty_reply():
    def failing(request):
        return httpx.Response(401, json={"error": "bad key"})

    def empty(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    for handler, error in ((failing, httpx.HTTPStatusError), (empty, ValueError)):
        generator = LLMGenerator(
            "k", "m", "https://llm.example/v1", client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(error):
            generator.generate("q", SOURCES)
