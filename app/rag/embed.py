from app.config import settings
from app.rag.backends import call_embed
from app.runtime_config import embedding_urls


def embed_text(text: str) -> list:
    errors = []

    for url in embedding_urls():
        try:
            return call_embed(url, settings.OLLAMA_EMBED_MODEL, text)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise Exception(f"Embedding failed: {' | '.join(errors)}")
