# Architektura aplikacji RAG Hybrid

Ten dokument opisuje architekturę aplikacji RAG znajdującej się w tym repozytorium: z czego się składa, jak działa przepływ danych, jakie są zależności oraz za co odpowiadają najważniejsze pliki.

## 1. Cel aplikacji

Aplikacja jest lokalnym/hybrydowym systemem RAG, czyli Retrieval Augmented Generation. Jej zadaniem jest:

- przyjmowanie notatek, plików i dokumentów technicznych,
- dzielenie ich na mniejsze fragmenty,
- tworzenie embeddingów dla fragmentów,
- zapisywanie ich w PostgreSQL z rozszerzeniem pgvector,
- wyszukiwanie najlepszych fragmentów dla pytania użytkownika,
- opcjonalny reranking wyników przez lokalny scoring albo model,
- generowanie odpowiedzi z kontekstem źródłowym,
- udostępnianie istniejącego endpointu `/retrieve_json` jako wewnętrznego retrieval API,
- udostępnianie osobnego, bezpiecznego kontenera MCP jako read-only bramy dla OpenWebUI, Continue, LangGraph i usług zewnętrznych.

System nie trenuje własnego modelu. Modele są uruchamiane przez Ollama.

## 1a. Jak to działa w 30 sekund

Najprostszy obraz całego systemu wygląda tak:

1. wrzucasz notatkę, plik albo folder z wiedzą,
2. system dzieli treść na mniejsze fragmenty,
3. każdy fragment dostaje embedding i trafia do PostgreSQL,
4. opcjonalnie z dokumentu są wyciągane relacje do Neo4j,
5. gdy OpenWebUI, Continue albo zewnętrzny klient pyta o kontekst, trafia najpierw do osobnego kontenera MCP,
6. MCP waliduje request i jako read-only gateway woła istniejący HybridRAG `/retrieve_json`,
7. HybridRAG szuka najlepszych fragmentów w bazie RAG,
8. system dokłada relacje z Neo4j oraz tekstowe dowody dla tych relacji, jeśli są dostępne,
9. MCP zwraca gotowy kontekst źródłowy bez generowania odpowiedzi,
10. model w OpenWebUI albo innym kliencie generuje odpowiedź na podstawie tego kontekstu,
11. w Graph Explorerze możesz osobno obejrzeć relacje encja-encja.

To znaczy, że aplikacja ma dwie warstwy wiedzy:

- warstwę tekstową RAG w PostgreSQL,
- warstwę relacji w Neo4j.

Obie są przydatne, ale robią różne rzeczy:

- RAG dobrze odpowiada na pytania na podstawie fragmentów dokumentów,
- graf dobrze pokazuje powiązania typu `ODF -> uses -> Ceph`.

## 1b. Co tu naprawdę znaczy "hybrydowy"

To jest ważne, bo słowo "hybrydowy" można rozumieć na dwa sposoby.

W tym projekcie są dwa znaczenia:

### 1. Hybrydowy retrieval tekstowy

Funkcja `hybrid_search()` łączy:

- wyszukiwanie wektorowe po embeddingach,
- wyszukiwanie leksykalne po słowach i frazach.

To jest podstawowy pierwszy etap wyszukiwania w RAG.

### 2. Hybrydowy RAG + graf

Endpoint `/retrieve_json` łączy:

- kontekst tekstowy z PostgreSQL / pgvector,
- relacje z Neo4j,
- dodatkowe chunki tekstowe, które potwierdzają albo objaśniają relacje z grafu.

Graf jest traktowany jako mapa zależności, a nie jako jedyne źródło prawdy.

### Co generuje odpowiedź

Te endpointy same generują odpowiedź przez Ollama:

- `POST /ask`
- `POST /ask_stream`
- `POST /retrieve`

### Co zwraca kontekst dla platform i klientów integracyjnych

RAGHybrid wystawia retrieval jako REST API. MCP entrypoint należy do osobnego produktu MCP Platform, który może uruchomić generic runtime z template'em `RAGHybrid Assistant` i dopiero ten runtime woła RAGHybrid API.

Podstawowy endpoint backendu RAG:

- `POST /retrieve_json`

łączy oba źródła wiedzy:

- najpierw bierze najlepsze fragmenty z RAG,
- potem szuka relacji w grafie,
- potem dobiera tekstowe dowody dla relacji,
- zwraca wspólny wynik z rekordami typu `vector`, `graph` i `graph_evidence`.

Czyli uczciwie:

- panel `/ask` i `/ask_stream` jest nadal lokalnym czatem RAG,
- OpenWebUI/Continue/Cursor/Claude powinny docelowo łączyć się z MCP Platform, nie z RAGHybrid,
- pełny tryb RAG + graf jest implementowany w `/retrieve_json` jako backend API dla runtime'ów i integracji.

## 2. Główne komponenty

```text
 Użytkownik / UI / OpenWebUI / Continue / klient MCP
        |
        v
FastAPI app/main.py
        |
        +--> ingest plików/notatek/WWW/repo
        |       |
        |       v
        |   chunking + tagowanie + embedding
        |       |
        |       +--> PostgreSQL + pgvector
        |       |
        |       +--> opcjonalna ekstrakcja relacji
        |               |
        |               v
        |           Neo4j
        |
        +--> retrieval API /retrieve_json
                |
                v
            build_hybrid_context (5 faz)
                |
                +--> [równolegle] vector: hybrid_search() → pgvector
                +--> [równolegle] graph:  search_graph_scored() → Neo4j fulltext
                |        |
                |        v [zależne od graph]
                +--> evidence: source_evidence_chunks() → pgvector
                |
                v
            fuse_retrieval_results()   ← app/rag/fusion.py
            (RRF: vector 50% + graph 30% + evidence 20%)
                |
                v
            calculate_relevance()      ← app/rag/relevance.py
            (brama jakości z fusion-aware logiką)
                |
                v
            JSON context dla OpenWebUI

MCP Platform / managed MCP runtime
        |
        v
RAGHybrid REST API /retrieve_json
        |
        +--> PostgreSQL + pgvector
        +--> Neo4j graph DB
```

## 3. Runtime i usługi

Projekt jest uruchamiany przez Docker Compose.

### `app`

Kontener aplikacji FastAPI.

- obraz budowany z `Dockerfile`,
- startuje przez `uvicorn app.main:app --host 0.0.0.0 --port 8000`,
- czyta konfigurację z `.env`,
- wystawia API przez Traefik pod hostem `rag.dom`,
- ma podmontowany katalog `/var/docker/obsidian/data/knowledge:/space`, używany przez ingest folderu.

### `db`

PostgreSQL z pgvector:

- image: `pgvector/pgvector:pg16-trixie`,
- baza: `rag_db`,
- użytkownik: `user`,
- dane trzymane w wolumenie `pgdata`.

### Ollama

Ollama nie jest zdefiniowana w tym `docker-compose.yml`, ale aplikacja komunikuje się z nią przez HTTP.

Konfiguracja jest w `app/config.py`:

- `OLLAMA_EMBED_URL` / `EMBED_URL` - endpoint embeddingów,
- `OLLAMA_EMBED_MODEL` - model embeddingowy, domyślnie `nomic-embed-text`,
- `OLLAMA_CPU_URL` - backend CPU, domyślnie `http://localhost:11434`,
- `OLLAMA_GPU_URL` - backend GPU, domyślnie `http://localhost:11434`,
- `OLLAMA_LAPTOP_URL` - trzeci backend Ollama do czatu i importów, domyślnie `http://localhost:11434`,
- `RERANK_BACKEND` - backend rerankingu, domyślnie `ollama_embedding`,
- `OLLAMA_RERANK_URL` - endpoint Ollama dla rerankera; jeśli pusty, używany jest endpoint embeddingów,
- `OLLAMA_RERANK_MODEL` - model rerankera, domyślnie `qllama/bce-reranker-base_v1`,
- `RERANK_MAX_CHARS` - maksymalna długość tekstu wysyłanego do modelu rerankera,
- `RERANK_MIN_SCORE_SPREAD` - minimalna różnica scoringu, poniżej której system wraca do fallbacku LLM,
- `RAG_RELEVANCE_GATE_ENABLED` - włącza relevance gating dla `/retrieve_json`, domyślnie `true`,
- `RAG_MIN_RELEVANCE_SCORE` - minimalny końcowy wynik jakości kontekstu, domyślnie `0.45`,
- `RAG_MIN_LEXICAL_OVERLAP` - minimalne pokrycie sensownych słów z query w wynikach tekstowych, domyślnie `0.10`,
- `RAG_MIN_RESULTS` - minimalna liczba dobrych wyników `vector` albo `graph_evidence`, domyślnie `1`,
- `RAG_REQUIRE_EVIDENCE_FOR_GRAPH` - graf bez `vector` albo `graph_evidence` nie przechodzi gate, domyślnie `true`,
- `RAG_GATE_DEBUG` - przełącza szczegółowy debug gate w odpowiedzi, domyślnie `true`,
- `RAG_RETRIEVAL_TELEMETRY_PATH` - ścieżka pliku telemetryki retrievalu, domyślnie `/runtime/raghybrid_retrieval_telemetry.jsonl`,
- `GEN_URL` istnieje w konfiguracji, ale generowanie korzysta z `OLLAMA_GPU_URL`, `OLLAMA_CPU_URL` i `OLLAMA_LAPTOP_URL`.

## 4. Baza danych

Warstwa DB jest w `app/db.py`.

Przy starcie aplikacji `startup()` wywołuje `init_db()`, które:

- tworzy rozszerzenie `vector`,
- tworzy tabelę `documents`,
- dodaje kolumnę `hash`, jeśli jej nie ma,
- tworzy unikalny indeks `idx_documents_hash`.

Tabela:

```sql
documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(768),
    metadata JSONB,
    hash TEXT
)
```

Znaczenie pól:

- `content` - pojedynczy chunk dokumentu,
- `embedding` - wektor embeddingu o wymiarze 768,
- `metadata` - JSON z informacjami typu `source`, `tags`, `page`,
- `hash` - hash znormalizowanego chunka używany do deduplikacji.

### Historyczna tabela statusu grafu

Aplikacja może mieć też historyczną tabelę pomocniczą:

```sql
graph_index_status (
    document_id INTEGER PRIMARY KEY,
    processed_at TIMESTAMP,
    relations_count INTEGER,
    new_relations_count INTEGER,
    existing_relations_count INTEGER,
    status TEXT,
    error TEXT
)
```

Po co była używana w starszej wersji:

- pamiętała, które dokumenty z bazy RAG zostały już przetworzone do grafu,
- pozwalała policzyć postęp dawnego przetwarzania grafu,
- rozdziela dokumenty `done`, `failed` i `pending`,
- pozwala odróżnić relacje nowe od tych, które już istniały.

Aktualny przepływ nie używa już endpointów masowego przetwarzania grafu. Relacje trafiają do Neo4j bezpośrednio podczas importu z włączoną opcją graf. Tabela może nadal istnieć w bazie jako element zgodności ze starszą wersją aplikacji, ale nie jest główną częścią obecnego schematu działania.

Połączenia do DB dla ingestu i wyszukiwania obsługuje pula `SimpleConnectionPool` w `app/rag/ingest.py`.

## 5. Ingest danych

Za ingest odpowiadają głównie:

- `app/main.py`,
- `app/rag/ingest.py`,
- `app/rag/chunk.py`,
- `app/rag/embed.py`,
- `app/rag/ingest_folder.py`.

### Dostępne ścieżki ingestu

Endpointy:

- `POST /upload` - przyjmuje prosty tekst formularzem,
- `POST /upload_note` - tworzy notatkę z tytułem, tagami i treścią,
- `POST /upload_file` - przyjmuje pliki PDF, DOCX, TXT, MD, CSV, JSON, YAML, LOG,
- `POST /upload_file_async` - wariant używany przez UI; zapisuje upload jako job i pozwala odpytywać postęp,
- `GET /upload_file/status/{job_id}` - status importu pliku dla paska postępu w UI,
- `POST /ingest_folder` - importuje pliki `.md` z katalogu `/space`,
- `POST /import_chatgpt` - importuje eksport rozmów ChatGPT.

### Flow ingestu

```text
plik / tekst
   |
   v
ekstrakcja tekstu
   |
   v
split_markdown()
   |
   v
normalize() + walidacja chunka
   |
   v
tagi z dokumentu + tagi z komend
   |
   v
embed_text()
   |
   v
INSERT do documents
```

UI używa wariantu asynchronicznego dla plików. Po wysłaniu formularza pokazuje pasek postępu i opis aktualnego kroku:

- wysyłanie pliku z przeglądarki,
- odczyt tekstu z PDF/DOCX/TXT,
- sprawdzenie duplikatu i trybu reindex,
- opcjonalne tagowanie smart import,
- chunking, embeddingi i zapis do PostgreSQL,
- opcjonalna ekstrakcja relacji do Neo4j,
- końcowe statystyki: nowe chunki, duplikaty, invalid, relacje grafowe.

### Chunking

`app/rag/chunk.py` dzieli tekst na fragmenty:

- domyślny maksymalny rozmiar chunka: `1200` znaków,
- overlap: `200` znaków,
- preferowane granice cięcia: puste linie, nowe linie, zdania, spacje,
- `split_markdown()` zachowuje nagłówki `##` jako kontekst sekcji.

### Tagowanie

Tagi powstają z kilku źródeł:

- jawne tagi Markdown, np. `#docker`,
- tagi wywnioskowane z komend, np. `docker`, `kubectl`, `oc`, `ansible`, `systemctl`,
- opcjonalnie tagi nadane przez LLM przy smart imporcie.

Przykładowe reguły:

- `oc` -> `openshift`, `kubernetes`, `cli`,
- `kubectl` -> `kubernetes`, `cli`,
- `docker` -> `docker`, `containers`, `cli`,
- `systemctl` -> `linux`, `systemd`, `cli`.

### Embedding

`app/rag/embed.py` wysyła tekst do Ollama:

```text
POST {embedding_url}/api/embeddings
model: OLLAMA_EMBED_MODEL
prompt: content
```

Wynikowy embedding jest zapisywany w `documents.embedding`.

## 5a. Warstwa grafowa Neo4j

Oprócz klasycznego RAG system ma prostą bazę grafową Neo4j.

Graf służy do zapisu relacji między encjami, na przykład:

```text
ODF --uses--> Ceph
OpenShift --extends--> Kubernetes
```

Model danych:

```text
(:Entity {name})-[:RELATED {type, metadata, sources}]->(:Entity {name})
```

Znaczenie pól relacji:

- `type` - nazwa relacji pokazywana na krawędzi w UI,
- `metadata` - ostatnie zapisane metadane relacji,
- `sources` - lista źródeł, z których relacja pochodzi.

W praktyce:

- node = encja,
- edge = relacja,
- label na edge = typ relacji.

### Jak relacje trafiają do grafu

Relacje trafiają do grafu podczas importu z włączoną opcją graf, na przykład przy imporcie pliku albo dokumentacji WWW.

Flow jest prosty:

```text
dokument / chunk tekstu
   |
   v
LLM wyciąga relacje JSON
   |
   v
upsert_relation()
   |
   v
Neo4j
```

### Czy robią się duplikaty

System stara się ich nie robić.

Na poziomie relacji używany jest `MERGE`, więc identyczna relacja:

```text
source + relation + target
```

nie powinna utworzyć drugiej kopii tej samej krawędzi.

Dodatkowo:

- na relacji zapisywana jest lista `sources`,
- import liczy osobno relacje nowe i już istniejące,
- cleanup grafu pilnuje, żeby w Neo4j nie zostawały relacje spoza whitelisty,
- ekstrakcja grafu zapisuje tylko relacje, których obie encje występują w tekście źródłowym albo jako znane aliasy.

Każdy importowany dokument z włączonym grafem dostaje też uniwersalną kotwicę dokumentu:

```text
(:Entity {name: "<wykryty tytuł albo nazwa pliku>"})
  -[:RELATED {type: "is_a"}]->
(:Entity {name: "dokument"})

(:Entity {name: "<wykryty tytuł albo nazwa pliku>"})
  -[:RELATED {type: "contains"}]->
(:Entity {name: "<ważna encja znaleziona w tekście>"})
```

Tytuł jest wykrywany z nagłówków, pierwszych sensownych linii albo nazwy źródła. Encje są wybierane z nagłówków, nazw własnych i znanych aliasów technologicznych/prawnych, ale tylko wtedy, gdy faktycznie występują w tekście. Dzięki temu Graph Explorer nie jest pusty dla dokumentów, w których LLM nie znalazł relacji, a jednocześnie graf nie powinien łapać relacji z obcej domeny.

Dla importu WWW kotwica powstaje per strona/URL, bo `source` jest adresem konkretnej strony, a tekst zaczyna się tytułem strony. Dla importu repo kotwica powstaje tylko wtedy, gdy w UI/API włączysz `graph=true`; system indeksuje do grafu wybrane pliki dokumentacyjne i konfiguracyjne (`README`, `docs`, `*.md`, `*.adoc`, `*.tf`, `*.yaml`, `*.rsc` itd.), a pomija zwykły kod źródłowy, żeby nie zaśmiecać grafu tysiącem małych plików.

Trzeba jednak pamiętać o jednej rzeczy:

- jeśli model raz zwróci `uses`, a drugi raz `depends_on`, to dla systemu są to dwie różne relacje.

Dlatego retrieval nie wstrzykuje już wszystkich sprzecznych relacji dla tej samej pary encji. `search_graph()` pobiera kandydatów, filtruje typy relacji z whitelisty i wybiera jedną najlepszą relację dla pary `(source, target)` według:

- dopasowania encji do query,
- priorytetu typu relacji (`depends_on`, `requires`, `builds_on`, `runs_on`, `uses` wyżej niż ogólne `is_a`, `extends`, `has`),
- liczby źródeł jako czynnika pomocniczego.

Graf jest więc nadal zachowany w Neo4j, ale kontekst dla modelu jest mniej konfliktowy.

### Dozwolone typy relacji

Typ relacji jest normalizowany i sprawdzany względem whitelisty w `app/rag/graph_schema.py`.

Aktualne typy:

```text
accesses, allows, builds_on, contains, creates, depends_on,
exposes, extends, has, hosts, includes, is_a, is_managed_by,
manages, provides, requires, runs_on, stores, supports, uses
```

## 6. Wyszukiwanie hybrydowe

Wyszukiwanie jest w `app/rag/search.py`, funkcja `hybrid_search()`.

Flow:

```text
query
  |
  v
embed_text(query)
  |
  v
wyszukiwanie wektorowe pgvector
  |
  v
dodatkowe wyszukiwanie leksykalne ILIKE
  |
  v
deduplikacja
  |
  v
prosty rerank lokalny
```

### Wyszukiwanie wektorowe

DB sortuje wyniki po dystansie:

```sql
ORDER BY embedding <-> query_embedding
```

Jeśli podane są tagi, stosowany jest filtr:

```sql
metadata->'tags' ?| tags
```

### Wyszukiwanie leksykalne

Po wektorowym retrievalu aplikacja wyciąga z query słowa kluczowe i wykonuje dodatkowe zapytanie:

```sql
content ILIKE '%term%'
```

Dzięki temu można odzyskać wyniki zawierające konkretne komendy lub nazwy, które nie zawsze dobrze wychodzą przez same embeddingi.

### Rozwinięcie aliasów query

`search_terms()` rozszerza słowa kluczowe przez słownik `QUERY_ALIASES`:

```python
QUERY_ALIASES = {
    "raghybrid": ["rag", "hybrid", "raghybrid"],
    "hybridrag":  ["rag", "hybrid", "hybridrag"],
    "konstytucja": ["konstytucja", "konstytucji", "constitution"],
    "openwebui": ["openwebui", "open", "webui"],
    "sejmie":    ["sejm", "sejmie", "sejmu"],
}
```

Dzięki temu polskie pytanie ze słowem `sejmie` trafia też na dokumenty opisane jako `sejm`.

### Heurystyki rerankingu lokalnego

Funkcja `rerank()` w `search.py` nadaje punkty za:

- dokładne trafienie w nagłówek (`##`) — +10 pkt,
- trafienie w nazwę pliku / source — +8 pkt,
- trafienie w tagi — +6 pkt,
- trafienie w treść — +4 pkt,
- co najmniej 2 trafienia jednocześnie — +12 pkt bonus.

Specjalne przypadki:

- **pytania konceptualne** (`co to`, `jak działa`, `czym jest`): strony schematu RAG dostają +30 pkt, repozytoria kodu — −25 pkt,
- **polskie pytania prawne** (`ustawa`, `minister`, `konstytucja` itp.): historia ChatGPT dostaje −45 pkt, pliki PDF i tagi `polish_law` — boost,
- **pytania Ansible z rolami**: playbooki z sekcją `roles:` dostają +24 pkt, pliki `vars.yml` — −25 pkt,
- **pytania o dane bieżące** (`wczoraj`, `cena`, `wynik`): historia ChatGPT dostaje −35 pkt.

## 6a. Silnik fuzji RRF — `app/rag/fusion.py`

Plik `app/rag/fusion.py` jest odrębnym modułem scalającym wyniki trzech ścieżek retrieval w jeden ujednolicony ranking.

### Dlaczego RRF

Wyszukiwanie wektorowe i grafowe zwracają wyniki w różnych skalach i z różną semantyką:

- pgvector zwraca odległość L2 (niżej = lepiej),
- Neo4j BM25 zwraca scoring (wyżej = lepiej),
- evidence chunks mają prostą miarę term-overlap.

RRF (Reciprocal Rank Fusion) normalizuje te sygnały przez pozycję na liście, a nie przez wartość bezwzględną. Każda ścieżka dostaje wagę:

```text
vector   50 %
graph    30 %
evidence 20 %
```

### Klasa FusedCandidate

Każdy kandydat z dowolnej ścieżki jest konwertowany do `FusedCandidate` przed fuzją:

```python
@dataclass
class FusedCandidate:
    content: str
    source: str
    item_type: str          # "vector" | "graph" | "graph_evidence"
    vector_score: float
    graph_score:  float
    evidence_score: float
    rrf_score: float        # akumulowany wkład RRF
    fused_score: float      # końcowy wynik dla bramki jakości
    retrieval_sources: list # które ścieżki znalazły ten element
    traversal_path: str     # np. "ODF --uses--> Ceph"
    relation_type: str
    distance: float         # surowa odległość L2 z pgvector
```

### Formuła fused_score

```text
fused_score = base_score + rrf_bonus

base_score  = max(
    vector_score,              # embedding similarity
    graph_score  × 0.85,       # BM25 + rel + sources
    evidence_score × 0.70      # term overlap
)

rrf_bonus   = rrf_norm × 0.15  # nagroda za koroborację wielościeżkową
```

Dyskont trójek strukturalnych (`--contains-->`, `--is_a-->`):

```text
graph_score × 0.40   (zamiast 0.85)
```

Trójki strukturalne to kotwice dokumentów, które nie niosą treści. Ciężki dyskont sprawia, że właściwy tekst dokumentu zawsze wychodzi wyżej niż indeksowe wskaźniki `is_a` / `contains`.

### Normalizacja graph_score

Funkcja `normalize_graph_score()`:

```text
graph_score = 0.50 × (BM25 / 4.0)
            + 0.30 × (relation_priority / 80)
            + 0.20 × min(n_sources / 5, 1.0)
```

`_NEO4J_BM25_CAP = 4.0` — Neo4j Lucene daje ~4.4 dla idealnego trafienia jednym terminem.

### Koroboracja wielościeżkowa

Jeśli element pojawia się jednocześnie w ścieżce wektorowej i grafowej, `retrieval_sources` zawiera obie nazwy, a `rerank_reason = "multi_path(vector|graph)"`. Taki element dostaje bonus RRF z obu ścieżek.

### Pięciofazowy pipeline build_hybrid_context

Funkcja `build_hybrid_context()` w `app/main.py` wykonuje retrieval w 5 fazach:

```text
Faza 1 (równolegle):
  ├─ _vector_task()  → hybrid_search() + lexical_rerank() + clean_retrieve_results()
  └─ _graph_task()   → search_graph_scored()

Faza 2 (zależna od fazy 1):
  └─ source_evidence_chunks()
     (chunki z dokumentów, na które wskazują relacje grafowe)

Faza 3:
  └─ konwersja do FusedCandidate (vector / graph / evidence)

Faza 4:
  └─ fuse_retrieval_results()
     (RRF, dedup, obliczenie fused_score)

Faza 5:
  └─ context assembly (limit charów, potem trim per typ)
```

Fazy 1 i 2 korzystają z `ThreadPoolExecutor(max_workers=2)` — vector i graph są pobierane równolegle, evidence czeka na wynik grafu (potrzebuje listy źródeł relacji).

### search_graph_scored

Nowa funkcja `search_graph_scored()` w `app/rag/graph_store.py` zastępuje `search_graph()` w głównym pipeline:

```text
1. graph_search_terms()  → terminy Lucene (max 8)
2. lucene_query_string() → zapytanie OR dla Neo4j fulltext
3. CALL db.index.fulltext.queryNodes("entity_fulltext", q)
   → entity scores (BM25)
4. MATCH (s)-[r]->(t) WHERE s.name IN entities OR t.name IN entities
   → relacje sąsiadujące z trafieniami
5. normalize_graph_score() dla każdej relacji
6. choose_graph_relations() — dedup (source, target), wybranie najlepszej relacji na parę
```

Fallback: jeśli indeks `entity_fulltext` nie istnieje albo nie zwraca wyników, pipeline automatycznie spada do starego `search_graph()` opartego na CONTAINS.

## 7. Reranking

W projekcie są trzy warstwy rerankingu.

### Prosty reranking lokalny

W `app/rag/search.py` jest funkcja `rerank(query, results)`. Ona:

- dodaje punkty za słowa z query występujące w content,
- odejmuje punkty za bardzo długi tekst,
- sortuje wyniki.

Ten reranking działa wewnątrz `hybrid_search()`.

### Reranking przez Ollama BCE

W `app/rag/rerank.py` jest funkcja `rerank(query, results, backend, model, top_k=6)`.

Domyślnie korzysta z modelu:

```text
qllama/bce-reranker-base_v1
```

Model jest uruchamiany w Ollama, a aplikacja woła endpoint:

```text
POST /api/embed
```

Ten model w Ollama jest wystawiony jako model embeddingowy, więc obecna implementacja:

- tworzy embedding dla pytania,
- tworzy embedding dla każdego kandydata,
- liczy cosine similarity,
- sortuje wyniki według podobieństwa,
- zwraca maksymalnie `top_k` wyników.

Konfiguracja:

```env
RERANK_BACKEND=ollama_embedding
OLLAMA_RERANK_URL=http://localhost:11434
OLLAMA_RERANK_MODEL=qllama/bce-reranker-base_v1
RERANK_MAX_CHARS=1600
RERANK_MIN_SCORE_SPREAD=0.02
```

Ważne ograniczenie: to nie jest pełny endpoint cross-encoder `rerank/score`, tylko ranking oparty o embeddingi zwracane przez Ollama. Dlatego w kodzie jest bezpiecznik `RERANK_MIN_SCORE_SPREAD`. Jeśli wyniki modelu są zbyt blisko siebie albo Ollama/model nie odpowie, aplikacja wraca do fallbacku LLM.

### Fallback rerankingu przez LLM

Jeśli `RERANK_BACKEND` nie jest ustawiony na `ollama_embedding` albo reranker BCE zgłosi błąd, `app/rag/rerank.py` używa fallbacku `llm_rerank()`.

Fallback:

- buduje listę kandydatów z indeksami,
- pyta model generacyjny, które fragmenty są najbardziej trafne,
- oczekuje odpowiedzi jako JSON array indeksów, np. `[0,2,5]`,
- zwraca maksymalnie `top_k` wyników.

Ten reranking jest używany w trybie SMART, w `/retrieve_json` oraz w klasycznym `multilingual_search()`.

## 8. Generowanie odpowiedzi

Generowanie jest w `app/rag/generate.py`.

Są dwie funkcje:

- `generate()` - odpowiedź blokująca, bez streamingu,
- `generate_stream()` - odpowiedź strumieniowana z Ollama.

Backend można wybrać parametrem:

- `gpu` - tylko `OLLAMA_GPU_URL`,
- `cpu` - tylko `OLLAMA_CPU_URL`,
- `auto` - najpierw GPU, potem CPU.

Domyślny model generacyjny:

```text
qwen2.5-coder:1.5b
```

## 9. Endpointy aplikacji

Najważniejsze endpointy są w `app/main.py`.

### Zdrowie i modele

- `GET /health` - prosty healthcheck,
- `GET /models` - pobiera listę modeli z Ollama,
- `GET /backend_status` - sprawdza dostępność CPU/GPU Ollama.

### Ingest

- `POST /upload`,
- `POST /upload_note`,
- `POST /upload_file`,
- `POST /ingest_folder`,
- `POST /import_chatgpt`,
- `POST /import_website`,
- `POST /import_repo`.

### Retrieval

- `POST /retrieve` - klasyczny retrieval formularzem,
- `POST /retrieve_json` - JSON API retrievalu używane przez UI, testy, integracje i runtime'y MCP Platform.

`/retrieve_json` ma warstwę relevance gating w `app/rag/relevance.py`. Od momentu wprowadzenia fuzji bramka działa w trybie fusion-aware:

- Każdy wynik z pipeline niesie `fused_score` obliczony przez `fuse_retrieval_results()`.
- Bramka wykrywa tryb fuzji sprawdzając, czy jakikolwiek element ma pole `fused_score` (`fusion_active`).
- Gdy `fusion_active = True`:
  - wyniki grafowe z `fused_score >= threshold` są akceptowane jako pełnoprawny kontekst,
  - `fusion_overrides_gate` pomija sprawdzenie lexical overlap, gdy fused_score wyraźnie przekracza próg,
  - polska fleksja jest obsługiwana przez `term_variants()` (np. `sejmie` → `sejm`).
- Gdy `fusion_active = False` (stary path): `graph` bez `vector` albo `graph_evidence` jest blokowany.

Composited final_score:

```text
final_score = top_score × 0.35
            + lexical_overlap × 0.25
            + max_item_overlap × 0.25
            + min(text_supported / min_results, 1.0) × 0.15
```

Bramka blokuje retrieval dla pytań o dane bieżące (`wczoraj`, `wynik`, `pogoda`) gdy w bazie nie ma wyników z typem `source_type=current`.

`quality_penalty()` odejmuje od score:
- `secret_like` (hasła, tokeny) → −0.20
- `encoded_or_token_like` (base64, hashe) → −0.25
- `short_chunk` → −0.05

Limit `top_k` został podniesiony z 20 do 100. Schemat narzędzia w MCP Gateway ma zaktualizowane `"maximum": 100` dla `top_k` i `"maximum": 50` dla `max_vector`.

RAG jest używany, gdy przynajmniej jeden wynik `vector` albo `graph_evidence` przekracza próg jakości i query ma minimalne pokrycie słów w kontekście. RAG jest pomijany, gdy wyniki są puste, mają niski score, nie pokrywają query albo zawierają wyłącznie relacje grafowe bez tekstowego dowodu. Same relacje `graph` są wskazówką do nawigacji, nie pełnym kontekstem faktograficznym.

Przykład dobrego wyniku:

```json
{
  "debug": {
    "rag_used": true,
    "relevance_score": 0.72,
    "relevance_reason": "relevant",
    "top_score": 0.81,
    "threshold": 0.45
  },
  "results": [{"type": "vector", "source": "docs/docker.md"}]
}
```

Przykład braku trafnego kontekstu:

```json
{
  "instruction": "No relevant context was found in the knowledge base. Answer using general knowledge only, and do not claim the answer comes from RAGHybrid.",
  "debug": {
    "rag_used": false,
    "reason": "low_relevance",
    "results_before_gate": 4,
    "results_after_gate": 0
  },
  "results": []
}
```

### MCP Platform Separation

RAGHybrid nie jest MCP Platformą i nie zarządza runtime'ami MCP. MCP Platform / MCP Builder została wydzielona jako osobny projekt poza RAGHybrid:

```text
/var/docker/MCPPlatofrm
```

Ważne rozróżnienie:

- `RAGHybrid` - RAG engine, ingestion, retrieval, embeddings, vector search, graph evidence, AI pipelines,
- `MCP Platform` - osobny produkt runtime/orchestration działający w modelu `CONFIG -> GENERIC MCP RUNTIME -> RUNNING MCP SERVER`,
- `RAGHybrid Assistant` - template w MCP Platform, który konfiguruje generic runtime do wołania RAGHybrid REST API.

RAGHybrid nie zarządza:

- MCP runtime deployments,
- MCP lifecycle,
- MCP policies,
- MCP sandboxing,
- MCP runtime containers,
- MCP execution workers,
- MCP operators,
- Runtime Classes.

Docelowy flow:

```text
MCP Runtime Tool
  -> calls RAGHybrid APIs
  -> retrieval/search/AI
  -> result returns through MCP Runtime
```

### OpenWebUI Model Profiles UI

Do zarządzania zakładkami i profilami modeli OpenWebUI służy panel:

```text
http://rag.dom/admin/openwebui-models
```

Panel edytuje bazę OpenWebUI `webui.db`, która jest montowana do kontenera HybridRAG jako:

```text
/var/docker/openwebui -> /openwebui-data
OPENWEBUI_DB_PATH=/openwebui-data/webui.db
```

Co można robić:

- tworzyć własne profile modeli,
- przypisywać bazowy model przez `base_model_id`,
- nadawać tagi, które w OpenWebUI tworzą zakładki, np. `raghybrid`, `gpu`, `cpu`, `laptop`,
- przypinać tool serwery z MCP Platform, np. `server:1` dla runtime wystawionego przez MCP Platform,
- ustawiać system prompt,
- ustawiać `function_calling=native`,
- włączać i wyłączać profile.

Przykład profilu:

```text
id: qwen-rag-gpu
name: Qwen RAG GPU
base_model_id: qwen2.5-coder:1.5b
tags: raghybrid,gpu
tool_ids: server:1
function_calling: native
```

Zakładki w dropdownie OpenWebUI pochodzą z pola `meta.tags` w tabeli `model`. Przypięte MCP tools pochodzą z pola `meta.toolIds`, np. `["server:1"]`, ale docelowo ma to być serwer/runtime z MCP Platform, a nie endpoint z RAGHybrid. Sam model nadal pochodzi z jednej z konfiguracji Ollama w OpenWebUI.

## 9a. Jak używać bazy RAGHybrid i jak się połączyć

Są dwa praktyczne sposoby korzystania z RAGHybrid:

1. WebUI w przeglądarce,
2. API backendowe, np. `/retrieve_json`, `/ask`, `/ask_stream`, `/upload_file`.

Klienci AI typu OpenWebUI, Continue, Cursor albo Claude nie powinni łączyć się bezpośrednio do RAGHybrid przez MCP. Docelowy przepływ to:

```text
AI client
  -> MCP Platform
  -> managed generic MCP runtime
  -> RAGHybrid REST API
```

### Adres aplikacji

W obecnym `docker-compose.yml` aplikacja działa w kontenerze:

```text
raghybrid-app:8000
```

Przez Traefik jest wystawiona jako:

```text
http://rag.dom
```

Najprostsze sprawdzenie:

```bash
curl http://rag.dom/health
```

Oczekiwany wynik:

```json
{"status":"ok"}
```

Panel webowy:

```text
http://rag.dom/
```

Dokumentacja działania:

```text
http://rag.dom/schemat
http://rag.dom/schemat_grafu
http://rag.dom/graph_explorer
```

### Jak używać z OpenWebUI i klientami AI

OpenWebUI, Continue, Cursor, Claude i inne klienty AI nie powinny łączyć się bezpośrednio z RAGHybrid jako MCP serverem. Docelowy przepływ jest taki:

```text
OpenWebUI / Continue / Cursor / Claude
  -> MCP Platform
  -> Generic MCP Runtime
  -> RAGHybrid Assistant template
  -> POST http://raghybrid-app:8000/retrieve_json
```

RAGHybrid pozostaje backendem REST. MCP endpointy, lifecycle, policy, RBAC, audit, sandbox i runtime classes należą do projektu:

```text
/var/docker/MCPPlatofrm
```

Przykładowy backend request, który wykonuje runtime MCP Platform:

```bash
curl -X POST http://rag.dom/retrieve_json \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Jak działa Terraform remote state?",
    "max_vector": 5,
    "max_graph": 10,
    "max_evidence": 5
  }'
```

Co wraca:

- `instruction` - instrukcja dla modelu, jak używać kontekstu,
- `results` - lista fragmentów i relacji,
- `debug` - liczby wyników oraz czas retrievalu.

Typy rekordów w `results`:

- `vector` - tekstowy chunk z PostgreSQL / pgvector,
- `graph` - relacja z Neo4j,
- `graph_evidence` - tekstowy chunk powiązany ze źródłem relacji z grafu.

Najważniejsza zasada: `vector` i `graph_evidence` są treścią źródłową, a `graph` jest mapą relacji. Jeśli graf pokazuje relację bez tekstowego potwierdzenia, model powinien traktować ją jako wskazówkę.

### Jak importować wiedzę

Dokumentacja WWW:

```text
POST /import_website
```

W WebUI podajesz URL startowy, limit stron, opcję reindex i opcję grafu. System przechodzi po podstronach, czyści treść ze śmieci, zapisuje chunki do RAG i opcjonalnie wyciąga relacje do Neo4j.

Repozytorium albo katalog z kodem:

```text
POST /import_repo
```

Przykład:

```bash
curl -X POST http://rag.dom/import_repo \
  -H 'Content-Type: application/json' \
  -d '{
    "path": "/app/app",
    "collection": "raghybrid-code",
    "extensions": [".py", ".yaml", ".yml", ".sql", ".sh"],
    "max_files": 500,
    "reindex": false
  }'
```

Import kodu zapisuje metadane `collection`, `language`, `path` i `source_type`, dzięki czemu później łatwiej filtrować wyniki i robić testy jakości.

### Jak sprawdzać jakość

Telemetry:

```bash
curl http://rag.dom/retrieval_telemetry?limit=20
```

Endpoint zwraca obiekt:

- `summary` - liczba rekordów, średni czas, średni rozmiar kontekstu, typy wyników,
- `records` - ostatnie requesty retrievalowe z polami `rag_used`, `relevance_score`, `gate_reason`, `results_before_gate`, `results_after_gate`, `top_score`, `lexical_overlap`.

Telemetryka dotyczy retrievalu, nie importu. Jest zapisywana do `/runtime/raghybrid_retrieval_telemetry.jsonl`, a compose montuje `/var/docker/raghybrid/runtime:/runtime`, więc rekordy nie znikają po przebudowie kontenera. Postęp importu pliku jest w `/upload_file/status/{job_id}`.

Feedback:

```text
POST /retrieval_feedback
GET /retrieval_feedback
```

Golden tests:

```text
POST /golden_tests/run
```

Golden tests są tanie: sprawdzają retrieval i typy wyników, ale nie odpalają pełnego generowania odpowiedzi.

## 9b. Integracja przez MCP Platform

RAGHybrid nie wystawia `/mcp` i nie jest publicznym MCP serverem. Integracja MCP powinna być realizowana przez MCP Platform:

```text
AI Client
  -> MCP Platform / managed runtime /mcp
  -> execution adapter http_request
  -> RAGHybrid REST API /retrieve_json
```

W MCP Platform template `RAGHybrid Assistant` definiuje tool `hybridrag_search` jako zwykłe `http_request` do RAGHybrid API. Runtime nie zawiera specjalnego kodu dla RAGHybrid.

### Graf

- `POST /graph_cleanup` - dry-run albo apply czyszczenia relacji spoza schematu,
- `GET /graph_data` - zwraca dane grafu do wizualizacji,
- `GET /graph_explorer` - osobna podstrona z wizualizacją grafu,
- `GET /schemat_grafu` - dokumentacja warstwy grafowej.

### Jakość retrievalu

- `GET /retrieval_telemetry` - ostatnie metryki retrievalu,
- `POST /retrieval_feedback` - zapis oceny wyniku,
- `GET /retrieval_feedback` - odczyt ocen,
- `POST /golden_tests/run` - szybkie testy retrievalu bez generowania odpowiedzi.

### Odpowiadanie

- `POST /ask` - odpowiedź pełna JSON,
- `POST /ask_stream` - odpowiedź strumieniowana SSE,
- `GET /` - wbudowany panel HTML/JS.

## 10. Flow `/ask`

`/ask` służy do pełnej odpowiedzi generowanej po stronie API.

To jest endpoint klasyczny, blokujący. Klient wysyła pytanie i czeka, aż backend wykona cały retrieval, zbuduje prompt, wywoła model i zwróci gotową odpowiedź jako JSON.

Przyjmuje dane formularzem:

- `query` - pytanie użytkownika,
- `tags` - opcjonalne tagi zawężające wyszukiwanie,
- `backend` - `auto`, `gpu` albo `cpu`,
- `model` - nazwa modelu Ollama używanego do generowania i fallbacku rerankingu LLM,
- `session_id` - identyfikator rozmowy dla krótkiej pamięci konwersacji,
- `mode` - `auto`, `fast` albo `smart`.

Flow:

```text
query
  |
  v
auto_tags_for_query()
  |
  v
answer_search()
  |
  +--> search_query_for_rag()
  +--> command_hints_for_query()
  +--> hybrid_search()
  +--> dedupe_results()
  +--> prioritize_results()
  +--> FAST/SMART/AUTO routing
  +--> opcjonalny rerank z app/rag/rerank.py
  |
  v
budowa context max 4000 znaków
  |
  v
prompt operacyjny
  |
  v
generate()
  |
  v
JSON: answer + mode
```

Co robi `/ask` krok po kroku:

- normalizuje nazwę modelu i `session_id`,
- jeśli użytkownik nie podał tagów, uruchamia `auto_tags_for_query()` i próbuje dobrać tagi przez LLM,
- wywołuje `answer_search()`, czyli główny flow retrievalu dla odpowiedzi,
- generuje alternatywne query przez `search_query_for_rag()`, żeby polskie pytanie mogło znaleźć angielskie notatki,
- dodaje podpowiedzi komend przez `command_hints_for_query()`, np. dla OpenShift, Kubernetes albo zasobów,
- wykonuje `hybrid_search()` dla wariantów zapytania,
- usuwa duplikaty przez `dedupe_results()`,
- sortuje wyniki heurystycznie przez `prioritize_results()`,
- wybiera tryb FAST/SMART/AUTO przez `choose_mode()`,
- w trybie SMART uruchamia reranking z `app/rag/rerank.py`,
- w trybie FAST pomija ten dodatkowy reranking,
- buduje kontekst ze źródłami w formacie `[Źródło: ...]`,
- ucina kontekst do `MAX_CONTEXT`, czyli 4000 znaków,
- dodaje krótką historię rozmowy z `chat_memory`,
- buduje prompt operacyjny,
- wywołuje `generate()`,
- zapisuje pytanie i odpowiedź do pamięci sesji,
- zwraca JSON z odpowiedzią i wybranym trybem.

Format odpowiedzi:

```json
{
  "answer": "wygenerowana odpowiedź",
  "mode": "fast"
}
```

Najważniejsza cecha `/ask`: odpowiedź pojawia się dopiero po zakończeniu generowania. To jest wygodne dla integracji, które chcą prosty JSON, ale użytkownik w UI czeka dłużej bez widzenia tokenów na żywo.

### Tryby FAST / SMART / AUTO

Funkcja `choose_mode(query, mode)`:

- `fast` - pomija dodatkowy reranking z `app/rag/rerank.py`,
- `smart` - używa dodatkowego rerankingu z `app/rag/rerank.py`,
- `auto` - wybiera tryb heurystycznie.

Heurystyki AUTO:

- krótkie pytania poniżej 40 znaków -> FAST,
- pytania zawierające `jak`, `co to`, `komenda`, `command` -> FAST,
- reszta -> SMART.

## 11. Flow `/ask_stream`

`/ask_stream` działa podobnie do `/ask`, ale zwraca odpowiedź jako Server-Sent Events.

To jest endpoint streamingowy. Klient dostaje najpierw statusy pracy backendu, a potem kolejne fragmenty odpowiedzi generowane przez Ollama. Dzięki temu użytkownik szybciej widzi, że system pracuje, i nie musi czekać na pełną odpowiedź.

Przyjmuje te same pola formularza co `/ask`:

- `query`,
- `tags`,
- `backend`,
- `model`,
- `session_id`,
- `mode`.

Flow:

```text
query
  |
  v
status: Dobieram tagi
  |
  v
retrieval + FAST/SMART/AUTO
  |
  v
status: Buduję kontekst
  |
  v
generate_stream()
  |
  v
SSE token po tokenie
```

Co robi `/ask_stream` krok po kroku:

- normalizuje model i `session_id`,
- tworzy generator SSE `event_generator()`,
- wysyła status `Dobieram tagi`,
- dobiera tagi tak samo jak `/ask`,
- wysyła status `Szukam w bazie`,
- uruchamia `answer_search()` z tym samym routingiem FAST/SMART/AUTO,
- jeśli wybrano SMART, informuje o rerankingu,
- jeśli wybrano FAST, informuje, że reranking jest pomijany,
- buduje kontekst do 4000 znaków,
- buduje prompt z historią, kontekstem, pytaniem i użytym query,
- wysyła status, że model generuje odpowiedź,
- wywołuje `generate_stream()`,
- każdy token z Ollama opakowuje jako `data: ...`,
- po zakończeniu wysyła status `Gotowy`,
- zapisuje pełną odpowiedź do pamięci sesji.

Format strumienia:

```text
event: status
data: Dobieram tagi

event: status
data: Szukam w bazie

data: fragment odpowiedzi

data: kolejny fragment odpowiedzi

event: status
data: Gotowy
```

To jest domyślny tryb używany przez UI. Przycisk `Ask` w panelu webowym wysyła żądanie do `/ask_stream`, a JavaScript odczytuje strumień przez `response.body.getReader()`.

Najważniejsza różnica względem `/ask`:

- `/ask` zwraca jeden gotowy JSON po całym generowaniu,
- `/ask_stream` zwraca statusy i tokeny na żywo przez SSE,
- oba endpointy używają tego samego retrievalu i tego samego wyboru FAST/SMART/AUTO,
- `/ask_stream` daje lepsze UX w panelu webowym, bo odpowiedź zaczyna pojawiać się szybciej.

## 12. Flow MCP retrieval dla OpenWebUI

OpenWebUI powinien używać MCP Platform jako jedynego wejścia MCP. RAGHybrid nie wystawia `/mcp`, nie publikuje OpenAPI tool spec dla OpenWebUI i nie zarządza runtime'ami MCP. RAGHybrid jest backendem RAG/API, a MCP Platform wystawia tool przez generic runtime oraz template `RAGHybrid Assistant`.

Flow:

```text
OpenWebUI
  |
  v
MCP Platform Runtime /mcp
  |
  v
generic runtime
  |
  v
execution adapter: http_request
  |
  v
POST http://raghybrid-app:8000/retrieve_json
  |
  v
build_hybrid_context()
  |
  +--> hybrid_search() w PostgreSQL / pgvector
  |        |
  |        v
  |    wyniki typu vector
  |
  +--> search_graph() w Neo4j
  |        |
  |        v
  |    wyniki typu graph
  |
  +--> source_evidence_chunks()
           |
           v
       wyniki typu graph_evidence
  |
  v
JSON z instruction + debug + results
  |
  v
OpenWebUI generuje odpowiedź swoim modelem
```

Warstwa MCP Platform kontroluje:

- walidację schematów wejścia i wyjścia,
- limity requestów, timeouty i rozmiary payloadów,
- polityki capability/RBAC,
- lifecycle runtime'u,
- delegowanie do istniejącego pipeline RAG bez zmiany jego logiki.

Istniejący endpoint `/retrieve_json` nadal kontroluje:

- puste contenty,
- duplikaty contentu,
- limity `max_vector`, `max_graph`, `max_evidence`,
- limit długości kontekstu `max_context_chars`,
- telemetrykę czasu i liczby wyników.

Format odpowiedzi:

```json
{
  "query": "pytanie użytkownika",
  "instruction": "Use only the provided results as context. If the answer is not in the context, say you do not know.",
  "results": [
    {
      "rank": 1,
      "type": "vector",
      "source": "plik.md",
      "tags": ["docker"],
      "metadata": {},
      "content": "..."
    }
  ]
}
```

Typy rekordów:

- rekordy typu `vector` dla wyników tekstowych,
- rekordy typu `graph` dla relacji z Neo4j,
- rekordy typu `graph_evidence` dla tekstowych chunków powiązanych ze źródłami relacji.

To jest ważne:

- `graph` pokazuje relacje,
- `graph_evidence` daje tekstowe potwierdzenie relacji, jeśli system je znajdzie,
- jeśli relacja z grafu nie ma wsparcia w `vector` albo `graph_evidence`, model powinien traktować ją jako wskazówkę, nie jako twardy fakt.

OpenWebUI używa MCP jako narzędzia retrievalowego: pobiera wyniki, a następnie własny model odpowiada na podstawie przekazanego kontekstu.

## 13. Auto tagi i rewrite zapytań

`app/rag/smart_filter.py` zawiera funkcje pomocnicze oparte o LLM:

- `classify_document()` - nadaje tagi dokumentowi przy smart imporcie,
- `classify_query()` - dobiera tagi do pytania,
- `rewrite_query_for_search()` - przepisuje pytanie na angielski query pod wyszukiwanie,
- `classify_qa()` - ocenia przydatność par Q/A przy imporcie historii.

Te funkcje proszą model o odpowiedź JSON i próbują ją parsować. Jeśli parsing się nie uda, aplikacja ma fallbacki, np. brak tagów albo oryginalne query.

## 14. UI

UI jest osadzone bezpośrednio w `app/main.py` jako HTML zwracany przez `GET /`.

Panel umożliwia:

- zadanie pytania,
- wybór backendu CPU/GPU/AUTO,
- wybór modelu,
- wybór trybu AUTO/FAST/SMART,
- podgląd odpowiedzi streamowanej,
- upload notatki,
- upload pliku,
- import ChatGPT JSON.

Przycisk `Ask` używa obecnie `/ask_stream`, czyli odpowiedź pojawia się stopniowo bez czekania na pełny JSON.

Import plików w UI nie powinien przechodzić zwykłym submittem formularza. JavaScript przechwytuje formularz, wysyła plik do `/upload_file_async`, a potem odpytuje `/upload_file/status/{job_id}`. Dzięki temu użytkownik widzi:

- pasek uploadu,
- etap pracy backendu,
- aktualny komunikat,
- statystyki po zakończeniu.

Jeśli w przeglądarce widać surowy JSON z `/upload_file`, oznacza to, że frontend JS nie został załadowany albo ma błąd składni. W takim przypadku formularz działa jako awaryjny HTML submit, ale nie pokazuje panelowego postępu.

Dodatkowe widoki:

- `Schemat działania` - opis architektury całej aplikacji,
- `Schemat bazy grafowej` - prosty opis Neo4j, relacji i Graph Explorera,
- `Graph Explorer` - osobna podstrona do oglądania relacji.

## 14a. Jak działa Graph Explorer

Graph Explorer jest tylko warstwą podglądu. Sam niczego nie liczy i niczego nie zapisuje.

Jego zadanie:

- pobrać dane z `GET /graph_data`,
- narysować nodes i edges w przeglądarce przez `vis-network`,
- pozwolić filtrować po nazwie encji,
- ograniczać liczbę relacji przez `limit`.

Najważniejsze zasady:

- puste `Search entity` = widok całej bazy grafowej,
- nadal działa limit bezpieczeństwa,
- maksymalny limit to `500`,
- widok służy do eksploracji, nie do edycji danych.

## 14b. Jak działa czyszczenie grafu

Graph Cleanup sprawdza relacje zapisane w Neo4j i pilnuje, żeby graf był zgodny ze schematem.

Endpoint:

- `POST /graph_cleanup`

Tryby:

- dry-run - pokazuje, co zostałoby usunięte albo poprawione,
- apply - wykonuje czyszczenie.

Najważniejsze zasady:

- relacje spoza whitelisty są traktowane jako śmieci,
- encje z pustymi nazwami albo technicznym szumem mogą zostać pominięte,
- cleanup nie dotyka tekstowych chunków w PostgreSQL.

## 15. Zależności Python

Z `app/requirements.txt`:

- `fastapi` - API HTTP,
- `uvicorn` - serwer ASGI,
- `psycopg2-binary` - połączenie z PostgreSQL,
- `python-multipart` - formularze i upload plików,
- `pydantic-settings` - konfiguracja z `.env`,
- `requests` - komunikacja z Ollama,
- `pypdf` - odczyt PDF,
- `python-docx` - odczyt DOCX.

## 16. Najważniejsze pliki

```text
app/main.py
```

Główna aplikacja FastAPI, endpointy, UI, routing FAST/SMART/AUTO, budowanie promptu.

```text
app/config.py
```

Konfiguracja z `.env`: DB, Ollama embedding, Ollama CPU/GPU, Neo4j oraz reranker.

```text
app/db.py
```

Inicjalizacja tabeli `documents` i rozszerzenia pgvector.

```text
app/rag/ingest.py
```

Ingest chunków do DB, walidacja, deduplikacja, metadata, embedding.

```text
app/rag/chunk.py
```

Dzielenie tekstu na chunki oraz reguły tagowania technicznego.

```text
app/rag/embed.py
```

Wywołanie Ollama `/api/embeddings`.

```text
app/rag/search.py
```

Hybrid search: pgvector + lexical search + prosty rerank.

```text
app/rag/rerank.py
```

Reranking przez lokalny model Ollama `qllama/bce-reranker-base_v1` z fallbackiem do LLM.

```text
app/rag/generate.py
```

Generowanie odpowiedzi i streaming z Ollama.

```text
app/rag/fusion.py
```

Silnik RRF: `FusedCandidate`, `fuse_retrieval_results()`, `normalize_graph_score()`, dyskont trójek strukturalnych.

```text
app/rag/smart_filter.py
```

LLM helpery: tagowanie, rewrite query, klasyfikacja dokumentów.

```text
docker-compose.yml
```

Definicja kontenera aplikacji, PostgreSQL/pgvector, wolumenów i Traefika.

## 17. Najważniejsze zależności logiczne

```text
main.py
  -> config.py
  -> db.py
  -> ingest.py
       -> chunk.py
       -> embed.py
       -> config.py
  -> search.py
       -> embed.py
       -> ingest.py connection_pool
  -> graph_store.py
       -> graph_schema.py
       -> fusion.py (normalize_graph_score, lucene_query_string)
  -> fusion.py
       (FusedCandidate, fuse_retrieval_results, fused_to_dict)
  -> relevance.py
       (calculate_relevance, RelevanceDecision)
  -> rerank.py
       -> generate.py
  -> generate.py
       -> config.py
  -> smart_filter.py
       -> generate.py
```

Najbardziej krytyczne zależności runtime:

- FastAPI musi mieć dostęp do PostgreSQL,
- PostgreSQL musi mieć pgvector,
- embedding Ollama musi odpowiadać dla ingestu i searcha,
- Ollama z modelem `qllama/bce-reranker-base_v1` musi odpowiadać dla domyślnego rerankingu,
- generacyjna Ollama musi odpowiadać dla `/ask`, `/ask_stream`, fallbacku rerankingu i smart tagów.

## 18. Typowe scenariusze działania

### Import pliku PDF

```text
/upload_file
  -> extract_file_text()
  -> ingest()
  -> split_markdown()
  -> embed_text()
  -> INSERT documents
```

Wariant UI:

```text
/upload_file_async
  -> zapis pliku do /tmp
  -> background job
  -> /upload_file/status/{job_id}
  -> process_uploaded_file()
  -> progress: odczyt, reindex, smart tagi, ingest, graf, done
```

### Pytanie z UI

```text
GET /
  -> użytkownik klika Ask
  -> JS wysyła POST /ask_stream
  -> retrieval
  -> kontekst
  -> Ollama stream
  -> odpowiedź w UI
```

### OpenWebUI jako tool

```text
OpenWebUI
  -> MCP Platform / managed MCP runtime
  -> tool template: RAGHybrid Assistant
  -> execution adapter: http_request
  -> POST http://raghybrid-app:8000/retrieve_json
  -> vector + graph + graph_evidence
  -> OpenWebUI generuje odpowiedź swoim modelem
```

### Zapis relacji do grafu podczas importu

```text
POST /upload_file albo POST /import_website z opcją graph=true
  -> extract_relations_from_text()
  -> walidacja: encje relacji muszą występować w tekście źródłowym
  -> upsert relacji do Neo4j
  -> wynik importu zawiera liczbę nowych i istniejących relacji
```

Jeśli dokument jest nietechniczny albo model zwróci relacje spoza tekstu, relacje mogą zostać pominięte. To jest celowe: lepiej mieć mniej relacji niż halucynacje typu relacja z obcej domeny przypięta do PDF-a.

### Cleanup konfliktów grafu

Do czyszczenia wielu typów relacji między tą samą parą encji służy:

```bash
python -m app.rag.graph_conflict_cleanup
python -m app.rag.graph_conflict_cleanup --apply
```

Dry-run pokazuje konfliktowe pary i relacje, które zostałyby scalone. Tryb `--apply` zostawia jedną najlepszą relację według priorytetu semantycznego z `app/rag/graph_store.py`, scala `sources` i zapisuje poprzednie typy w `merged_relation_types`. Dzięki temu graf nie pokazuje kilku sprzecznych krawędzi `OpenShift -> Kubernetes`, ale nadal zostaje informacja, jakie typy relacji były wcześniej widziane.

### Podgląd całej bazy grafowej

```text
GET /graph_explorer
  -> klik Load graph
  -> GET /graph_data?limit=500
  -> vis-network rysuje graf w przeglądarce
```

## 19a. Historia zmian algorytmu fuzji i retrievalu

### 2026-05-18: wprowadzenie silnika RRF i fuzji wielościeżkowej

Przed tą zmianą `build_hybrid_context()` budował kontekst przez proste łączenie wyników z trzech ścieżek (vector, graph, evidence) bez wspólnego rankingu. Każda ścieżka zwracała niezależną listę, a wyniki były scalane w kolejności: najpierw vector, potem graph, potem evidence — bez normalizacji score'ów między ścieżkami.

**Wprowadzone zmiany:**

1. Nowy moduł `app/rag/fusion.py` z silnikiem RRF.
2. Każdy kandydat z każdej ścieżki jest konwertowany do `FusedCandidate` przed scalaniem.
3. `fuse_retrieval_results()` liczy `fused_score` jako `base_score + rrf_bonus`.
4. `build_hybrid_context()` działa teraz w 5 fazach (patrz sekcja 6a).
5. Fazy 1 i 2 są równoległe przez `ThreadPoolExecutor(max_workers=2)`.
6. `search_graph_scored()` zastępuje `search_graph()` w głównym pipeline — używa indeksu fulltext Neo4j zamiast CONTAINS.

### Dyskont trójek strukturalnych

Problem: trójki `--contains-->` i `--is_a-->` (kotwice dokumentów) dominowały nad tekstem właściwym, bo BM25 dawał im wysoki wynik dla zapytań zawierających nazwę encji:

```
Dz.u. 2022 Poz. 655 --is_a--> USTAWA          score: 0.72
Dz.u. 2022 Poz. 655 --contains--> Minister...  score: 0.67
D20220655L.pdf  (tekst str. 13)                 score: 0.65  ← przegrywał
```

Rozwiązanie: dyskont `0.40` zamiast `0.85` dla trójek zawierających `--contains-->` lub `--is_a-->` w polu `content`.

Trójki z relacjami niosącymi treść (`--uses-->`, `--depends_on-->`, `--runs_on-->` itd.) nadal dostają `0.85`.

### Podniesienie limitu top_k

Stary cap: `min(top_k, 20)`. Nowy: `min(top_k, 100)`.
Schemat MCP Gateway zaktualizowany: `"top_k": {"maximum": 100}`, `"max_vector": {"maximum": 50}`.

### Fusion-aware relevance gate

Po wprowadzeniu `fused_score` bramka jakości w `relevance.py` działa inaczej niż wcześniej:

- Wykrywa tryb fuzji (`fusion_active`).
- Wyniki grafowe z `fused_score >= threshold` mogą samodzielnie otworzyć bramkę.
- `fusion_overrides_gate` wyłącza sprawdzenie lexical overlap dla klientów z fuzją.
- Dodano obsługę polskiej fleksji (`term_variants()`) i wykrywanie pytań o dane bieżące.

---

## 19. Co warto wiedzieć przy rozwoju

- Zmiana modelu embeddingowego może wymagać przebudowy embeddingów w DB, szczególnie jeśli zmienia się wymiar wektora.
- Tabela oczekuje `VECTOR(768)`, więc model embeddingowy musi zwracać wektor zgodny z tym wymiarem.
- `/ask` i `/ask_stream` same generują odpowiedź, a `/retrieve_json` tylko zwraca kontekst.
- OpenWebUI, Continue i usługi zewnętrzne powinny używać osobnego MCP kontenera, nie bezpośredniego `/retrieve_json`.
- MCP jest read-only: nie wykonuje inference, nie zna modeli, nie wybiera backendu CPU/GPU i nie komunikuje się z Ollamą.
- `hybrid_search()` już robi prosty rerank lokalny; SMART i endpointy toolowe dodają reranking z `app/rag/rerank.py`.
- Domyślny reranker `qllama/bce-reranker-base_v1` w Ollama nie wymaga przebudowy embeddingów w tabeli `documents`, bo jest używany tylko do sortowania kandydatów po retrievalu.
- Deduplikacja ingestu działa przez hash znormalizowanego contentu.
- Metadata są w JSONB, więc można rozwijać tagi, źródła i page bez migracji kolumn.
- Warstwa grafowa nie zastępuje RAG. Ona uzupełnia RAG o relacje encja-encja.
- Jeśli chcesz większy porządek w grafie, największy wpływ ma jakość nazw encji i spójność typów relacji.
