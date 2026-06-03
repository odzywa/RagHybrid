"""
Multi-backend client for RAGHybrid.

Supports two backend families:
  ollama  — Ollama API (/api/embeddings, /api/generate, /api/embed)
  openai  — OpenAI-compatible API (/v1/embeddings, /v1/chat/completions)
            Covers: vLLM, llama.cpp server, LM Studio, OpenAI, Groq, etc.

Configure via .env:
  BACKEND_TYPE=ollama          # default for all components
  EMBED_BACKEND_TYPE=ollama    # override for embeddings only
  GEN_BACKEND_TYPE=openai      # override for generation only
  RERANK_BACKEND_TYPE=ollama   # override for reranker only
"""

from __future__ import annotations

import json
import os
from typing import Generator, List, Optional


import requests


# ── Backend type detection ────────────────────────────────────────────────────

def _env(name: str, fallback_name: str = "BACKEND_TYPE") -> str:
    """Return component-specific backend type, falling back to global BACKEND_TYPE."""
    value = os.getenv(name, "").strip().lower()
    if value in ("ollama", "openai"):
        return value
    global_val = os.getenv(fallback_name, "ollama").strip().lower()
    return global_val if global_val in ("ollama", "openai") else "ollama"


def embed_backend_type() -> str:
    try:
        from app.runtime_config import active_backend_type
        return active_backend_type("embed")
    except Exception:
        return _env("EMBED_BACKEND_TYPE")


def gen_backend_type() -> str:
    try:
        from app.runtime_config import active_backend_type
        return active_backend_type("gen")
    except Exception:
        return _env("GEN_BACKEND_TYPE")


def rerank_backend_type() -> str:
    try:
        from app.runtime_config import active_backend_type
        return active_backend_type("rerank")
    except Exception:
        return _env("RERANK_BACKEND_TYPE")


# ── Embeddings ────────────────────────────────────────────────────────────────

def call_embed(url: str, model: str, text: str, timeout: int = 60) -> List[float]:
    """
    Return an embedding vector for `text`.

    Tries the component-specific backend type first, then the global BACKEND_TYPE.
    Raises on failure.
    """
    backend = embed_backend_type()

    if backend == "openai":
        return _openai_embed(url, model, text, timeout)
    return _ollama_embed(url, model, text, timeout)


def _ollama_embed(url: str, model: str, text: str, timeout: int) -> List[float]:
    """Ollama /api/embeddings — returns {"embedding": [...]}"""
    resp = requests.post(
        f"{url.rstrip('/')}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _openai_embed(url: str, model: str, text: str, timeout: int) -> List[float]:
    """OpenAI /v1/embeddings — returns {"data": [{"embedding": [...]}]}"""
    resp = requests.post(
        f"{url.rstrip('/')}/v1/embeddings",
        json={"model": model, "input": text},
        timeout=timeout,
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'na')}"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


# ── Reranker embeddings ───────────────────────────────────────────────────────

def call_rerank_embed(url: str, model: str, text: str, max_chars: int = 1600,
                      timeout: int = 30) -> List[float]:
    """
    Return an embedding vector for reranking purposes.

    Ollama uses /api/embed (batch-aware).
    OpenAI-compatible backends use /v1/embeddings.
    """
    backend = rerank_backend_type()
    text = text[:max_chars]

    if backend == "openai":
        return _openai_embed(url, model, text, timeout)
    return _ollama_rerank_embed(url, model, text, timeout)


def _ollama_rerank_embed(url: str, model: str, text: str, timeout: int) -> List[float]:
    """Ollama /api/embed — returns {"embeddings": [[...]]} or {"embedding": [...]}"""
    resp = requests.post(
        f"{url.rstrip('/')}/api/embed",
        json={"model": model, "input": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "embeddings" in payload:
        return payload["embeddings"][0]
    return payload["embedding"]


# ── Generation ────────────────────────────────────────────────────────────────

def call_generate(url: str, model: str, prompt: str,
                  system: Optional[str] = None, timeout: int = 120) -> str:
    """
    Return a complete (non-streaming) text response.
    """
    backend = gen_backend_type()

    if backend == "openai":
        return _openai_generate(url, model, prompt, system, timeout)
    return _ollama_generate(url, model, prompt, timeout)


def stream_generate(url: str, model: str, prompt: str,
                    system: Optional[str] = None,
                    connect_timeout: int = 5,
                    read_timeout: int = 300) -> Generator[str, None, None]:
    """
    Yield raw JSON strings line-by-line for streaming.

    Ollama: yields JSONL objects  {"response": "token", "done": false}
    OpenAI: yields SSE data lines {"choices": [{"delta": {"content": "token"}}]}

    Callers that only need the token text should use stream_tokens() instead.
    """
    backend = gen_backend_type()

    if backend == "openai":
        yield from _openai_stream(url, model, prompt, system, connect_timeout, read_timeout)
    else:
        yield from _ollama_stream(url, model, prompt, connect_timeout, read_timeout)


def stream_tokens(url: str, model: str, prompt: str,
                  system: Optional[str] = None,
                  connect_timeout: int = 5,
                  read_timeout: int = 300) -> Generator[str, None, None]:
    """
    Yield plain text tokens (not raw JSON).
    Backend-agnostic wrapper over stream_generate().
    """
    backend = gen_backend_type()

    for raw in stream_generate(url, model, prompt, system, connect_timeout, read_timeout):
        token = _extract_token(raw, backend)
        if token:
            yield token


def _extract_token(raw: str, backend: str) -> str:
    """Extract the text token from a raw streaming line."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""

    if backend == "openai":
        choices = data.get("choices") or []
        if choices:
            return choices[0].get("delta", {}).get("content") or ""
        return ""

    # Ollama
    return data.get("response") or ""


# ── Ollama internals ──────────────────────────────────────────────────────────

def _ollama_generate(url: str, model: str, prompt: str, timeout: int) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _ollama_stream(url: str, model: str, prompt: str,
                   connect_timeout: int, read_timeout: int) -> Generator[str, None, None]:
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=(connect_timeout, read_timeout),
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            yield line.decode("utf-8")


# ── OpenAI-compatible internals ───────────────────────────────────────────────

def _build_messages(prompt: str, system: Optional[str]) -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _openai_headers() -> dict:
    key = os.getenv("OPENAI_API_KEY", "na")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _openai_generate(url: str, model: str, prompt: str,
                     system: Optional[str], timeout: int) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": _build_messages(prompt, system),
            "stream": False,
        },
        headers=_openai_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _openai_stream(url: str, model: str, prompt: str,
                   system: Optional[str],
                   connect_timeout: int, read_timeout: int) -> Generator[str, None, None]:
    resp = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": _build_messages(prompt, system),
            "stream": True,
        },
        headers=_openai_headers(),
        stream=True,
        timeout=(connect_timeout, read_timeout),
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            payload = decoded[6:].strip()
            if payload == "[DONE]":
                return
            yield payload
