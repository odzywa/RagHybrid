import requests
from app.config import settings
from app.runtime_config import embedding_urls

def embed_text(text):
    errors = []

    for url in embedding_urls():
        try:
            response = requests.post(
                f"{url}/api/embeddings",
                json={
                    "model": settings.OLLAMA_EMBED_MODEL,
                    "prompt": text
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["embedding"]

            errors.append(f"{url}: {response.text}")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise Exception(f"Embedding failed: {' | '.join(errors)}")
