"""Answer generation helpers."""


PROMPT_TEMPLATE = """
You are an agriculture advisor helping a farmer.

Use ONLY the context below.
If the context is not enough, say what information is missing.
Do not invent pesticide names, dosage, disease diagnosis, or government schemes.

Context:
{context}

Farmer question:
{query}

Answer in this format:
1. Likely issue:
2. Recommended action:
3. Precautions:
4. When to ask a local agriculture expert:

Answer:
"""


def build_context(retrieved_results, max_words=450, top_k=4):
    selected = retrieved_results[:top_k]
    context = "\n---\n".join(result["chunk"] for result in selected)
    words = context.split()

    if len(words) > max_words:
        context = " ".join(words[:max_words])

    return context

