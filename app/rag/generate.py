from app.config import settings
from app.rag.backends import call_generate, stream_generate


DEFAULT_MODEL = "qwen2.5-coder:1.5b"


def _urls_for_backend(backend: str) -> list:
    if backend == "gpu":
        return [settings.OLLAMA_GPU_URL]
    if backend == "cpu":
        return [settings.OLLAMA_CPU_URL]
    if backend == "laptop":
        return [settings.OLLAMA_LAPTOP_URL]
    # auto: try GPU first, then CPU, then laptop
    return [
        settings.OLLAMA_GPU_URL,
        settings.OLLAMA_CPU_URL,
        settings.OLLAMA_LAPTOP_URL,
    ]


def generate(prompt: str, backend: str = "auto", model: str = DEFAULT_MODEL) -> str:
    model = (model or DEFAULT_MODEL).strip()

    for url in _urls_for_backend(backend):
        try:
            print(f"generate: backend={url}  model={model}")
            return call_generate(url, model, prompt)
        except Exception as exc:
            print(f"generate FAILED {url}: {exc}")

    return "Brak dostępnego modelu"


def generate_stream(prompt: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    """
    Yield raw JSON lines (Ollama JSONL or OpenAI SSE payload).
    Consumers that need the text token should parse accordingly:
      - Ollama:  json.loads(line)["response"]
      - OpenAI:  json.loads(line)["choices"][0]["delta"]["content"]

    Use backends.stream_tokens() for backend-agnostic token iteration.
    """
    model = (model or DEFAULT_MODEL).strip()

    for url in _urls_for_backend(backend):
        try:
            for line in stream_generate(url, model, prompt):
                yield line
            return
        except Exception as exc:
            print(f"generate_stream FAILED {url}: {exc}")

    yield '{"response": "Brak dostępnego modelu", "done": true}'
