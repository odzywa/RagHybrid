import ast
import math
import re

import requests

from app.config import settings
from app.runtime_config import rerank_targets


def extract_indices(text):
    match = re.search(r"\[.*?\]", text)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except Exception:
            return None
    return None


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if not left_norm or not right_norm:
        return 0.0

    return dot / (left_norm * right_norm)


def embed_for_rerank(text, target):
    response = requests.post(
        f"{target['url']}/api/embed",
        json={
            "model": target["model"],
            "input": text[:settings.RERANK_MAX_CHARS]
        },
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"Ollama rerank embedding failed on {target['url']} with {target['model']}: {response.text}")

    payload = response.json()
    if "embeddings" in payload:
        return payload["embeddings"][0]
    return payload["embedding"]


def ollama_embedding_rerank(query, results, top_k=6):
    errors = []

    for target in rerank_targets():
        try:
            print(f"RERANK EMBEDDING BACKEND: {target['url']} model={target['model']}")
            query_embedding = embed_for_rerank(query, target)
            scored = []

            for result in results:
                content = result[0] or ""
                passage_embedding = embed_for_rerank(content, target)
                score = cosine_similarity(query_embedding, passage_embedding)
                scored.append((score, result))

            scored.sort(reverse=True, key=lambda item: item[0])
            score_spread = scored[0][0] - scored[-1][0] if len(scored) > 1 else 1.0
            print(
                "EMBEDDING RERANK SCORES:",
                [round(score, 4) for score, _ in scored[:top_k]]
            )

            if score_spread < settings.RERANK_MIN_SCORE_SPREAD:
                raise Exception(
                    f"embedding rerank score spread too small: {score_spread:.4f}"
                )

            return [result for _, result in scored[:top_k]]
        except Exception as exc:
            errors.append(str(exc))

    raise Exception(
        "Ollama embedding rerank failed: " + " | ".join(errors)
        if errors else "Ollama embedding rerank has no available targets"
    )


def llm_rerank(query, results, backend="auto", model="qwen2.5-coder:1.5b", top_k=6):
    if not results:
        return results

    # budujemy listę kandydatów
    candidates = []
    for i, r in enumerate(results):
        content = r[0]
        candidates.append(f"[{i}] {content}")

    joined = "\n\n".join(candidates)

    prompt = f"""
You are a ranking assistant.

Select the {top_k} most relevant passages for answering the question.

Question:
{query}

Passages:
{joined}

Return ONLY a JSON array of integers (e.g. [0,2,5]).
No explanation. No markdown. No text.
"""

    # użyj istniejącego generate()
    from app.rag.generate import generate

    response = generate(prompt, backend=backend, model=model)
    print("RERANK RAW:", response)

    indices = extract_indices(response)
    if not indices:
        return results[:top_k]

    filtered = []
    for i in indices:
        if isinstance(i, int) and i < len(results):
            filtered.append(results[i])

    if not filtered:
        return results[:top_k]

    return filtered[:top_k]


def rerank(query, results, backend="auto", model="qwen2.5-coder:1.5b", top_k=6):
    if not results:
        return results

    if settings.RERANK_BACKEND == "ollama_embedding":
        try:
            return ollama_embedding_rerank(query, results, top_k=top_k)
        except Exception as e:
            print("EMBEDDING RERANK FAILED, USING ORIGINAL ORDER:", e)
            return results[:top_k]

    return llm_rerank(query, results, backend=backend, model=model, top_k=top_k)
