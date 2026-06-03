import requests
from app.config import settings


DEFAULT_MODEL = "qwen2.5-coder:1.5b"


def generate(prompt: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    model = model.strip() if model else DEFAULT_MODEL

    if backend == "gpu":
        urls = [settings.OLLAMA_GPU_URL]
    elif backend == "cpu":
        urls = [settings.OLLAMA_CPU_URL]
    elif backend == "laptop":
        urls = [settings.OLLAMA_LAPTOP_URL]
    else:
        urls = [
            settings.OLLAMA_GPU_URL,
            settings.OLLAMA_CPU_URL,
            settings.OLLAMA_LAPTOP_URL,
        ]

    for url in urls:
        try:
            print(f"Using backend: {url}")
            print(f"Using model: {model}")

            response = requests.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json()["response"]

        except Exception as e:
            print("FAILED:", url, e)

    return "Brak dostępnego modelu"


def generate_stream(prompt: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    model = model.strip() if model else DEFAULT_MODEL

    if backend == "gpu":
        urls = [settings.OLLAMA_GPU_URL]
    elif backend == "cpu":
        urls = [settings.OLLAMA_CPU_URL]
    elif backend == "laptop":
        urls = [settings.OLLAMA_LAPTOP_URL]
    else:
        urls = [
            settings.OLLAMA_GPU_URL,
            settings.OLLAMA_CPU_URL,
            settings.OLLAMA_LAPTOP_URL,
        ]

    for url in urls:
        try:
            response = requests.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=(5, 300)
            )

            if response.status_code != 200:
                continue

            # Ollama streams JSONL: one JSON object per line.
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = line.decode("utf-8")
                    yield data
                except Exception:
                    continue

            return

        except Exception as e:
            print("STREAM FAILED:", url, e)

    yield '{"response": "Brak dostępnego modelu", "done": true}'
