# RAGHybrid
(docs/screenshot.png)
Lokalny system **Retrieval-Augmented Generation** z hybrydowym wyszukiwaniem wektorowo-grafowym.

RAGHybrid łączy:
- **pgvector** (PostgreSQL) — wyszukiwanie po embeddingach
- **Neo4j** — relacje encja→encja
- **Ollama** — embeddingi, reranking i generowanie odpowiedzi (wszystko lokalnie)
- **RRF fusion** — unifikacja sygnałów z obu baz przez Reciprocal Rank Fusion
- **Relevance gate** — automatyczna ocena jakości kontekstu przed zwróceniem wyników

Działa w pełni lokalnie. Żadne dane nie opuszczają Twojej infrastruktury.

---

## Architektura w 30 sekund

```
[dokument / notatka / URL / repo]
       │
       ▼
chunking + embedding (Ollama nomic-embed-text)
       │
       ├──► PostgreSQL / pgvector  (wektory)
       └──► Neo4j                  (relacje, opcjonalnie)

[pytanie użytkownika]
       │
       ▼ równolegle
   ┌───────────────────┐   ┌──────────────────────────────┐
   │ hybrid_search()   │   │ search_graph_scored()        │
   │ pgvector L2       │   │ Neo4j fulltext (Lucene BM25) │
   └─────────┬─────────┘   └──────────────┬───────────────┘
             │                            │
             └───────────┬────────────────┘
                         ▼
              source_evidence_chunks()
              (chunki ze źródeł relacji)
                         │
                         ▼
              fuse_retrieval_results()   ← app/rag/fusion.py
              RRF: vector 50% + graph 30% + evidence 20%
                         │
                         ▼
              calculate_relevance()      ← app/rag/relevance.py
              (bramka jakości, fused_score ≥ 0.45)
                         │
                         ▼
              JSON context → model generuje odpowiedź
```

---

## Wymagania

- Docker + Docker Compose v2
- **Ollama** uruchomiony lokalnie lub na serwerze w sieci
  - model embeddingowy: `nomic-embed-text`
  - model generacyjny: np. `qwen2.5-coder:1.5b` lub dowolny inny
  - model reranker (opcjonalny): `qllama/bce-reranker-base_v1`

```bash
# Pobierz modele przez Ollama
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:1.5b
ollama pull qllama/bce-reranker-base_v1
```

---

## Szybki start

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/TWOJ_LOGIN/RAGHybrid.git
cd RAGHybrid

# 2. Skopiuj konfigurację i ustaw adresy Ollama
cp .env.example .env
# edytuj .env — ustaw adresy OLLAMA_*_URL i hasła

# 3. Utwórz folder na wiedzę
mkdir knowledge

# 4. Uruchom
docker compose up -d

# 5. Sprawdź czy działa
curl http://localhost:8000/health
# → {"status":"ok"}
```

Aplikacja jest dostępna pod adresem: **http://localhost:8000**

---

## Konfiguracja (.env)

Skopiuj `.env.example` do `.env` i dostosuj:

| Zmienna | Opis | Domyślnie |
|---|---|---|
| `DATABASE_URL` | Connection string PostgreSQL | `postgresql://rag_user:changeme@raghybrid-db:5432/rag_db` |
| `OLLAMA_EMBED_URL` | Adres Ollama dla embeddingów | `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | Model embeddingowy | `nomic-embed-text` |
| `OLLAMA_GPU_URL` | Ollama dla generowania (GPU) | `http://localhost:11434` |
| `OLLAMA_CPU_URL` | Ollama dla generowania (CPU fallback) | `http://localhost:11434` |
| `OLLAMA_RERANK_MODEL` | Model reranker BCE | `qllama/bce-reranker-base_v1` |
| `NEO4J_URI` | Adres Neo4j Bolt | `bolt://raghybrid-neo4j:7687` |
| `NEO4J_PASSWORD` | Hasło Neo4j | `changeme` |
| `RAG_MIN_RELEVANCE_SCORE` | Minimalny fused_score bramki | `0.45` |
| `RAG_GATE_DEBUG` | Debug bramki w odpowiedzi API | `false` |

---

## Endpointy API

### Ingest

| Endpoint | Opis |
|---|---|
| `POST /upload` | Tekst formularzem |
| `POST /upload_note` | Notatka z tytułem i tagami |
| `POST /upload_file` | Plik PDF, DOCX, TXT, MD, CSV, JSON |
| `POST /upload_file_async` | Upload asynchroniczny z paskiem postępu |
| `POST /ingest_folder` | Import wszystkich `.md` z `/space` |
| `POST /import_website` | Crawl i import dokumentacji WWW |
| `POST /import_repo` | Import repozytorium kodu |
| `POST /import_chatgpt` | Import eksportu rozmów ChatGPT |

### Retrieval

| Endpoint | Opis |
|---|---|
| `POST /retrieve_json` | Główne API retrieval (JSON) |
| `POST /retrieve` | Retrieval formularzem |
| `POST /ask` | Pełna odpowiedź RAG (blokująca) |
| `POST /ask_stream` | Odpowiedź RAG strumieniowana (SSE) |

Przykład `/retrieve_json`:

```bash
curl -X POST http://localhost:8000/retrieve_json \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Jak działa Terraform remote state?",
    "max_vector": 5,
    "max_graph": 10,
    "max_evidence": 5,
    "top_k": 20
  }'
```

Odpowiedź zawiera:
- `results` — lista wyników typu `vector`, `graph`, `graph_evidence`
- `debug` — liczby wyników, fused_score, czas, status bramki
- `instruction` — instrukcja dla modelu (lub komunikat o braku kontekstu)

### Graf

| Endpoint | Opis |
|---|---|
| `GET /graph_data` | Dane grafu do wizualizacji |
| `GET /graph_explorer` | Interaktywna wizualizacja (UI) |
| `POST /graph_cleanup` | Czyszczenie relacji spoza schematu |

### Jakość

| Endpoint | Opis |
|---|---|
| `GET /retrieval_telemetry` | Ostatnie metryki retrievalu |
| `POST /retrieval_feedback` | Ocena jakości odpowiedzi |
| `POST /golden_tests/run` | Szybkie testy retrieval |

---

## Panel webowy

| URL | Opis |
|---|---|
| `http://localhost:8000/` | Główny panel czatu i importu |
| `http://localhost:8000/schemat` | Dokumentacja architektury |
| `http://localhost:8000/schemat_grafu` | Dokumentacja warstwy grafowej |
| `http://localhost:8000/graph_explorer` | Eksplorator grafu Neo4j |

---

## Import danych

### Folder z Markdown (Obsidian, Vault)

Umieść pliki `.md` w folderze `knowledge/` (montowanym do kontenera jako `/space`), następnie:

```bash
curl -X POST http://localhost:8000/ingest_folder
```

### Dokumentacja WWW

```bash
curl -X POST http://localhost:8000/import_website \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://docs.przykład.pl", "max_pages": 50, "graph": true}'
```

### Repozytorium kodu

```bash
curl -X POST http://localhost:8000/import_repo \
  -H 'Content-Type: application/json' \
  -d '{
    "path": "/path/to/repo",
    "collection": "moj-projekt",
    "extensions": [".py", ".md", ".yaml"],
    "max_files": 200
  }'
```

---

## Wyszukiwanie hybrydowe — jak działa scoring

RAGHybrid używa trójścieżkowego RRF fusion:

| Ścieżka | Źródło | Scoring | Dyskont |
|---|---|---|---|
| `vector` | pgvector embedding | `exp(−dystans/30)` | brak |
| `graph` relacja merytoryczna | Neo4j BM25 + priorytet | `0–1` | × 0.85 |
| `graph` kotwica (is_a, contains) | Neo4j BM25 | `0–1` | × 0.40 |
| `graph_evidence` | pgvector term-overlap | `0–1` | × 0.70 |
| RRF bonus (≥2 ścieżki) | — | `0–0.15` | brak |

```
fused_score = max(vector_score, graph_score × dyskont, evidence_score × 0.70)
            + rrf_norm × 0.15
```

Bramka jakości (`calculate_relevance`) akceptuje kontekst gdy `fused_score ≥ RAG_MIN_RELEVANCE_SCORE`.

---

## Integracja z OpenWebUI / Continue / Claude

RAGHybrid wystawia REST API. Rekomendowany przepływ dla klientów AI:

```
AI Client (OpenWebUI / Continue / Claude)
  → MCP Platform / managed runtime
  → POST http://raghybrid-app:8000/retrieve_json
  → kontekst źródłowy
  → model generuje odpowiedź
```

RAGHybrid **nie** wystawia endpointu `/mcp` — jest backendem RAG, nie MCP serverem.

---

## Struktura projektu

```
RAGHybrid/
├── app/
│   ├── main.py              # FastAPI app, endpointy, UI, routing
│   ├── config.py            # Konfiguracja z .env
│   ├── db.py                # Init PostgreSQL / pgvector
│   ├── openwebui_admin.py   # Zarządzanie profilami modeli OpenWebUI
│   ├── runtime_config.py    # Live config (modele, backendi)
│   ├── requirements.txt
│   └── rag/
│       ├── chunk.py         # Chunking, reguły tagowania
│       ├── embed.py         # Ollama embeddings
│       ├── fusion.py        # RRF fusion engine (FusedCandidate)
│       ├── generate.py      # Generowanie i streaming przez Ollama
│       ├── graph_cleanup.py # Czyszczenie relacji Neo4j
│       ├── graph_conflict_cleanup.py  # Scalanie konfliktów grafu
│       ├── graph_extract.py # Ekstrakcja relacji z tekstu (LLM)
│       ├── graph_schema.py  # Whitelist typów relacji
│       ├── graph_store.py   # Operacje Neo4j (upsert, search_graph_scored)
│       ├── import_chatgpt.py
│       ├── import_repo.py
│       ├── ingest.py        # Ingest chunków do PostgreSQL
│       ├── ingest_folder.py # Batch import folderów .md
│       ├── relevance.py     # Bramka jakości (calculate_relevance)
│       ├── rerank.py        # Reranker BCE + fallback LLM
│       ├── search.py        # hybrid_search(), source_evidence_chunks()
│       ├── smart_filter.py  # LLM helpery (tagi, rewrite query)
│       └── web_import.py    # Crawl i import dokumentacji WWW
├── tests/
│   └── test_relevance.py
├── runtime/                 # Telemetria (persystentna, .gitignored)
├── knowledge/               # Twoje pliki MD (.gitignored, montowany jako /space)
├── .env.example
├── .gitignore
├── ARCHITEKTURA_RAG.md      # Pełna dokumentacja architektury (PL)
├── docker-compose.yml
├── Dockerfile
└── Makefile
```

---

## Makefile

```bash
make build   # buduj obraz
make up      # uruchom w tle
make down    # zatrzymaj
make logs    # śledź logi aplikacji
```

---

## Testy

```bash
docker compose exec app python -m pytest tests/ -v
```

---

## Licencja

MIT
