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


# ── Runtime config read/write ─────────────────────────────────────────────────

def read_runtime_config() -> dict:
    if not RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_runtime_config(data: dict) -> None:
    with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Backend type (live override) ──────────────────────────────────────────────

def get_backend_config() -> dict:
    """Return current backend configuration (runtime overrides + .env defaults)."""
    rc = read_runtime_config()
    return {
        "backend_type":        rc.get("backend_type") or settings.BACKEND_TYPE or "ollama",
        "embed_backend_type":  rc.get("embed_backend_type") or settings.EMBED_BACKEND_TYPE or "",
        "gen_backend_type":    rc.get("gen_backend_type") or settings.GEN_BACKEND_TYPE or "",
        "rerank_backend_type": rc.get("rerank_backend_type") or settings.RERANK_BACKEND_TYPE or "",
        "embed_url":           rc.get("embed_url") or settings.embedding_url,
        "embed_model":         rc.get("embed_model") or settings.OLLAMA_EMBED_MODEL,
        "gpu_url":             rc.get("gpu_url") or settings.OLLAMA_GPU_URL,
        "cpu_url":             rc.get("cpu_url") or settings.OLLAMA_CPU_URL,
        "laptop_url":          rc.get("laptop_url") or settings.OLLAMA_LAPTOP_URL,
        "rerank_url":          rc.get("rerank_url") or settings.rerank_url,
        "rerank_model":        rc.get("rerank_model") or settings.OLLAMA_RERANK_MODEL,
        "openai_api_key":      rc.get("openai_api_key") or settings.OPENAI_API_KEY,
    }


def set_backend_config(updates: dict) -> dict:
    """Merge updates into runtime config and return new effective config."""
    allowed = {
        "backend_type", "embed_backend_type", "gen_backend_type", "rerank_backend_type",
        "embed_url", "embed_model",
        "gpu_url", "cpu_url", "laptop_url",
        "rerank_url", "rerank_model",
        "openai_api_key",
    }
    data = read_runtime_config()
    for key, value in updates.items():
        if key in allowed:
            data[key] = value
    write_runtime_config(data)
    # Invalidate availability cache so status checks use new URLs
    _availability_cache.clear()
    _models_cache.clear()
    return get_backend_config()


def active_backend_type(component: str = "") -> str:
    """
    Return the effective backend type for a component.
    component: "embed" | "gen" | "rerank" | "" (global)
    """
    cfg = get_backend_config()
    key = f"{component}_backend_type" if component else "backend_type"
    specific = cfg.get(key, "").strip().lower()
    if specific in ("ollama", "openai"):
        return specific
    global_val = cfg.get("backend_type", "ollama").strip().lower()
    return global_val if global_val in ("ollama", "openai") else "ollama"


# ── Embedding backend ─────────────────────────────────────────────────────────

def embedding_backend_from_url(url):
    if url == settings.OLLAMA_GPU_URL:
        return "gpu"
    if url == settings.OLLAMA_CPU_URL:
        return "local"
    return "custom"


def embedding_backend_status():
    cfg = get_backend_config()
    embed_url = cfg["embed_url"]
    runtime = read_runtime_config()
    configured_backend = runtime.get("embedding_backend") or "auto"
    gpu_available = server_available(cfg["gpu_url"])
    local_available = server_available(cfg["cpu_url"])

    if configured_backend == "auto":
        backend = "gpu" if gpu_available else "local"
    else:
        backend = configured_backend

    url = cfg["gpu_url"] if backend == "gpu" else cfg["cpu_url"] if backend == "local" else embed_url

    return {
        "mode": configured_backend,
        "backend": backend,
        "url": url,
        "model": cfg["embed_model"],
        "configured_url": embed_url,
        "gpu_available": gpu_available,
        "local_available": local_available,
        "gpu_url": cfg["gpu_url"],
        "local_url": cfg["cpu_url"],
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
    cfg = get_backend_config()
    urls = [status["url"]]
    if status["mode"] == "auto" and status["url"] != cfg["cpu_url"]:
        urls.append(cfg["cpu_url"])
    return urls


# ── Server availability ───────────────────────────────────────────────────────

def server_available(url: str) -> bool:
    """Check if a server responds. Works for both Ollama and OpenAI-compatible."""
    now = time.time()
    cached = _availability_cache.get(url)
    if cached and now - cached["checked_at"] < OLLAMA_STATUS_TTL:
        return cached["available"]

    available = False
    backend = active_backend_type("gen")

    try:
        if backend == "openai":
            # OpenAI-compatible: try /v1/models
            resp = requests.get(f"{url.rstrip('/')}/v1/models", timeout=OLLAMA_STATUS_TIMEOUT)
            available = resp.status_code in (200, 401)  # 401 = running but needs key
        else:
            resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=OLLAMA_STATUS_TIMEOUT)
            available = resp.status_code == 200
    except Exception:
        available = False

    _availability_cache[url] = {"available": available, "checked_at": now}
    return available


# Keep backward-compatible alias
ollama_available = server_available


def ollama_models(url):
    now = time.time()
    cached = _models_cache.get(url)
    if cached and now - cached["checked_at"] < OLLAMA_STATUS_TTL:
        return cached["models"]

    models = set()
    backend = active_backend_type("gen")

    try:
        if backend == "openai":
            resp = requests.get(
                f"{url.rstrip('/')}/v1/models",
                timeout=OLLAMA_STATUS_TIMEOUT,
                headers={"Authorization": f"Bearer {get_backend_config()['openai_api_key']}"},
            )
            if resp.status_code == 200:
                models = {m.get("id") for m in resp.json().get("data", []) if m.get("id")}
        else:
            resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=OLLAMA_STATUS_TIMEOUT)
            if resp.status_code == 200:
                models = {m.get("name") for m in resp.json().get("models", []) if m.get("name")}
    except Exception:
        pass

    _models_cache[url] = {"models": models, "checked_at": now}
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
    cfg = get_backend_config()
    explicit_url = cfg["rerank_url"].strip()
    preferred_urls = [explicit_url] if explicit_url else embedding_urls()
    rerank_model = cfg["rerank_model"]
    embed_model = cfg["embed_model"]
    targets = []

    for url in preferred_urls:
        if rerank_model and ollama_has_model(url, rerank_model):
            targets.append({"url": url, "model": rerank_model})
        if embed_model and ollama_has_model(url, embed_model):
            targets.append({"url": url, "model": embed_model})

    if not targets and rerank_model:
        targets.append({"url": explicit_url or current_embedding_url(), "model": rerank_model})

    return targets
