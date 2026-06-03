import ast
import math
import re

from app.config import settings
from app.rag.backends import call_rerank_embed
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


def ollama_embedding_rerank(query, results, top_k=6):
    """
    Rerank using embedding cosine similarity.
    Works with any backend (Ollama /api/embed or OpenAI /v1/embeddings).
    Controlled by RERANK_BACKEND_TYPE in .env.
    """
    errors = []

    for target in rerank_targets():
        try:
            print(f"RERANK: {target['url']}  model={target['model']}")

            query_emb = call_rerank_embed(
                target["url"], target["model"], query,
                max_chars=settings.RERANK_MAX_CHARS,
            )
            scored = []

            for result in results:
                content = (result[0] or "")[:settings.RERANK_MAX_CHARS]
                passage_emb = call_rerank_embed(target["url"], target["model"], content)
                score = cosine_similarity(query_emb, passage_emb)
                scored.append((score, result))

            scored.sort(reverse=True, key=lambda item: item[0])
            score_spread = scored[0][0] - scored[-1][0] if len(scored) > 1 else 1.0

            print("RERANK SCORES:", [round(s, 4) for s, _ in scored[:top_k]])

            if score_spread < settings.RERANK_MIN_SCORE_SPREAD:
                raise Exception(f"score spread too small: {score_spread:.4f}")

            return [r for _, r in scored[:top_k]]

        except Exception as exc:
            errors.append(str(exc))

    raise Exception("Embedding rerank failed: " + " | ".join(errors))


def llm_rerank(query, results, backend="auto", model="qwen2.5-coder:1.5b", top_k=6):
    if not results:
        return results

    candidates = [f"[{i}] {r[0]}" for i, r in enumerate(results)]
    joined = "\n\n".join(candidates)

    prompt = f"""You are a ranking assistant.

Select the {top_k} most relevant passages for answering the question.

Question:
{query}

Passages:
{joined}

Return ONLY a JSON array of integers (e.g. [0,2,5]).
No explanation. No markdown. No text."""

    from app.rag.generate import generate
    response = generate(prompt, backend=backend, model=model)
    print("LLM RERANK RAW:", response)

    indices = extract_indices(response)
    if not indices:
        return results[:top_k]

    filtered = [results[i] for i in indices if isinstance(i, int) and i < len(results)]
    return (filtered or results)[:top_k]


def rerank(query, results, backend="auto", model="qwen2.5-coder:1.5b", top_k=6):
    if not results:
        return results

    if settings.RERANK_BACKEND == "ollama_embedding":
        try:
            return ollama_embedding_rerank(query, results, top_k=top_k)
        except Exception as exc:
            print(f"Embedding rerank failed, keeping original order: {exc}")
            return results[:top_k]

    return llm_rerank(query, results, backend=backend, model=model, top_k=top_k)
