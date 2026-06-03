import json
from pathlib import Path
import time

import requests

from app.config import settings


RUNTIME_CONFIG_PATH = Path("/tmp/raghybrid_runtime_config.json")
OLLAMA_STATUS_TIMEOUT = 2
OLLAMA_STATUS_TTL = 30
_availability_cache = {}
_models_cache = {}


def read_runtime_config():
    if not RUNTIME_CONFIG_PATH.exists():
        return {}

    try:
        with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def write_runtime_config(data):
    with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def embedding_backend_from_url(url):
    if url == settings.OLLAMA_GPU_URL:
        return "gpu"

    if url == settings.OLLAMA_CPU_URL:
        return "local"

    return "custom"


def embedding_backend_status():
    configured_url = settings.embedding_url
    runtime = read_runtime_config()
    configured_backend = runtime.get("embedding_backend") or "auto"
    gpu_available = ollama_available(settings.OLLAMA_GPU_URL)
    local_available = ollama_available(settings.OLLAMA_CPU_URL)

    if configured_backend == "auto":
        backend = "gpu" if gpu_available else "local"
    else:
        backend = configured_backend

    if backend == "gpu":
        url = settings.OLLAMA_GPU_URL
    elif backend == "local":
        url = settings.OLLAMA_CPU_URL
    else:
        url = configured_url

    return {
        "mode": configured_backend,
        "backend": backend,
        "url": url,
        "model": settings.OLLAMA_EMBED_MODEL,
        "configured_url": configured_url,
        "gpu_available": gpu_available,
        "local_available": local_available,
        "gpu_url": settings.OLLAMA_GPU_URL,
        "local_url": settings.OLLAMA_CPU_URL,
    }


def set_embedding_backend(backend):
    if backend not in ["auto", "gpu", "local"]:
        raise ValueError("backend must be auto, gpu or local")

    data = read_runtime_config()
    data["embedding_backend"] = backend
    write_runtime_config(data)

    return embedding_backend_status()


def current_embedding_url():
    return embedding_backend_status()["url"]


def embedding_urls():
    status = embedding_backend_status()
    urls = [status["url"]]

    if status["mode"] == "auto" and status["url"] != settings.OLLAMA_CPU_URL:
        urls.append(settings.OLLAMA_CPU_URL)

    return urls


def ollama_available(url):
    now = time.time()
    cached = _availability_cache.get(url)

    if cached and now - cached["checked_at"] < OLLAMA_STATUS_TTL:
        return cached["available"]

    try:
        response = requests.get(f"{url}/api/tags", timeout=OLLAMA_STATUS_TIMEOUT)
        available = response.status_code == 200
    except Exception:
        available = False

    _availability_cache[url] = {
        "available": available,
        "checked_at": now,
    }
    return available


def ollama_models(url):
    now = time.time()
    cached = _models_cache.get(url)

    if cached and now - cached["checked_at"] < OLLAMA_STATUS_TTL:
        return cached["models"]

    try:
        response = requests.get(f"{url}/api/tags", timeout=OLLAMA_STATUS_TIMEOUT)
        if response.status_code == 200:
            models = {
                model.get("name")
                for model in response.json().get("models", [])
                if model.get("name")
            }
        else:
            models = set()
    except Exception:
        models = set()

    _models_cache[url] = {
        "models": models,
        "checked_at": now,
    }
    return models


def ollama_has_model(url, model):
    names = ollama_models(url)
    if model in names:
        return True

    if ":" not in model and f"{model}:latest" in names:
        return True

    if model.endswith(":latest") and model[:-7] in names:
        return True

    return False


def rerank_targets():
    explicit_url = settings.OLLAMA_RERANK_URL.strip()
    preferred_urls = [explicit_url] if explicit_url else embedding_urls()
    targets = []

    for url in preferred_urls:
        if settings.OLLAMA_RERANK_MODEL and ollama_has_model(url, settings.OLLAMA_RERANK_MODEL):
            targets.append({
                "url": url,
                "model": settings.OLLAMA_RERANK_MODEL,
            })

        if settings.OLLAMA_EMBED_MODEL and ollama_has_model(url, settings.OLLAMA_EMBED_MODEL):
            targets.append({
                "url": url,
                "model": settings.OLLAMA_EMBED_MODEL,
            })

    if not targets and settings.OLLAMA_RERANK_MODEL:
        targets.append({
            "url": explicit_url or current_embedding_url(),
            "model": settings.OLLAMA_RERANK_MODEL,
        })

    return targets
