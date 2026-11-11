"""Answer generation.

Two interchangeable generators turn retrieved sources into an answer:

* ``ExtractiveGenerator`` - returns the best-matching answer from the knowledge
  base. Needs no model and no API key, and cannot hallucinate. Default.
* ``LLMGenerator``        - asks any OpenAI-compatible chat API (Groq,
  OpenRouter, a local Ollama, ...) to write an answer from the sources.

Both have ``name`` and ``generate(query, sources) -> str`` where a source is
``{"question": str, "answer": str, "relevance": float}``.

The research notebook also evaluates local generation with Flan-T5. That needs
PyTorch and about 2 GB of RAM, which a free hosting instance does not have, so
it stays in the notebook.
"""

import httpx

try:
    from .config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
except ImportError:  # notebook style: ``src/`` itself is on sys.path
    from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM


SYSTEM_PROMPT = (
    "You are AgriBot, an agriculture advisor helping a farmer. "
    "Answer the farmer's question using ONLY the numbered sources provided. "
    "If the sources do not contain the answer, say what is missing instead of guessing. "
    "Never invent pesticide names, dosages, disease diagnoses, or government schemes. "
    "Be practical and concise: a short direct answer first, then up to five short "
    "bullet points if steps are useful. Cite the sources you used like [1] or [2]."
)


def confidence_label(relevance):
    """Map the best source's 0-1 relevance to High / Medium / Low."""
    if relevance >= CONFIDENCE_HIGH:
        return "High"
    if relevance >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def build_context(sources, max_words=450):
    """Number the sources and trim the block to ``max_words`` words."""
    parts = []
    for number, source in enumerate(sources, start=1):
        parts.append(f"[{number}] Q: {source['question']}\nA: {source['answer']}")

    context = "\n\n".join(parts)
    words = context.split(" ")

    if len(words) > max_words:
        context = " ".join(words[:max_words]) + "..."

    return context


def _sentence_case(text):
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


class ExtractiveGenerator:
    """Answer with the best-matching entry from the knowledge base, verbatim."""

    name = "extractive"

    def generate(self, query, sources):
        if not sources:
            return ""
        return _sentence_case(sources[0]["answer"])


class LLMGenerator:
    """Write the answer with an OpenAI-compatible chat completions API."""

    def __init__(self, api_key, model, base_url, timeout=30.0, client=None):
        self.model = model
        self.name = f"llm:{model}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, query, sources):
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Sources:\n{build_context(sources)}\n\nFarmer's question: {query}",
                },
            ],
        }

        response = self._client.post(self._url, json=payload, headers=self._headers)
        response.raise_for_status()

        answer = response.json()["choices"][0]["message"]["content"]
        if not answer or not answer.strip():
            raise ValueError("The language model returned an empty answer")

        return answer.strip()
