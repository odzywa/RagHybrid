from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://rag_user:changeme@raghybrid-db:5432/rag_db"

    # ── Backend type ──────────────────────────────────────────────────────────
    # Global default: "ollama" or "openai"
    # "openai" covers vLLM, llama.cpp server, LM Studio, any OpenAI-compatible API.
    # Override per-component with EMBED_BACKEND_TYPE / GEN_BACKEND_TYPE / RERANK_BACKEND_TYPE.
    BACKEND_TYPE: str = "ollama"

    # Optional per-component overrides (leave empty to inherit BACKEND_TYPE)
    EMBED_BACKEND_TYPE: str = ""
    GEN_BACKEND_TYPE: str = ""
    RERANK_BACKEND_TYPE: str = ""

    # API key for OpenAI-compatible backends (vLLM/llama.cpp usually accept any value)
    OPENAI_API_KEY: str = "na"

    # ── Ollama / OpenAI-compatible server URLs ────────────────────────────────
    EMBED_URL: str = "http://localhost:11434"
    GEN_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_URL: str = ""
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_CPU_URL: str = "http://localhost:11434"
    OLLAMA_GPU_URL: str = "http://localhost:11434"
    OLLAMA_LAPTOP_URL: str = "http://localhost:11434"

    # ── Reranker ──────────────────────────────────────────────────────────────
    # "ollama_embedding" — cosine similarity via embedding model (default, fast)
    # "llm"             — LLM-based reranking (slower, no extra model needed)
    RERANK_BACKEND: str = "ollama_embedding"
    OLLAMA_RERANK_URL: str = ""
    OLLAMA_RERANK_MODEL: str = "qllama/bce-reranker-base_v1"
    RERANK_MAX_CHARS: int = 1600
    RERANK_MIN_SCORE_SPREAD: float = 0.02

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI: str = "bolt://raghybrid-neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "changeme"

    # ── Relevance gate ────────────────────────────────────────────────────────
    RAG_MIN_RELEVANCE_SCORE: float = 0.45
    RAG_MIN_LEXICAL_OVERLAP: float = 0.10
    RAG_MIN_RESULTS: int = 1
    RAG_REQUIRE_EVIDENCE_FOR_GRAPH: bool = True
    RAG_GATE_DEBUG: bool = False

    # ── Telemetry ─────────────────────────────────────────────────────────────
    RAG_RETRIEVAL_TELEMETRY_PATH: str = "/runtime/raghybrid_retrieval_telemetry.jsonl"

    # ── OpenWebUI (optional) ──────────────────────────────────────────────────
    OPENWEBUI_DB_PATH: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def embedding_url(self):
        return self.OLLAMA_EMBED_URL or self.EMBED_URL

    @property
    def rerank_url(self):
        return self.OLLAMA_RERANK_URL or self.embedding_url


settings = Settings()
