import re

from app.rag.embed import embed_text
from app.rag.ingest import connection_pool, get_conn


STOPWORDS = {
    "and",
    "are",
    "czy",
    "dla",
    "for",
    "jak",
    "jest",
    "kto",
    "mowi",
    "mówi",
    "oraz",
    "the",
    "with",
    "wczoraj",
}


QUERY_ALIASES = {
    "raghybrid": ["rag", "hybrid", "raghybrid"],
    "hybridrag": ["rag", "hybrid", "hybridrag"],
    "konstytucja": ["konstytucja", "konstytucji", "constitution"],
    "openwebui": ["openwebui", "open", "webui"],
    "sejmie": ["sejm", "sejmie", "sejmu"],
}

CURRENT_QUERY_WORDS = {
    "aktualnie",
    "cena",
    "dzisiaj",
    "dziś",
    "kurs",
    "mecz",
    "najnowsze",
    "ostatni",
    "pogoda",
    "teraz",
    "wczoraj",
    "wygrał",
    "wygral",
    "wynik",
}


def search_terms(query):
    raw_terms = [
        word.strip(".,:;!?()[]{}'\"`").lower()
        for word in re.split(r"\s+", query or "")
    ]
    terms = []

    for word in raw_terms:
        if not word or len(word) < 3 or word in STOPWORDS:
            continue

        aliases = QUERY_ALIASES.get(word, [word])

        for alias in aliases:
            if alias and alias not in STOPWORDS and alias not in terms:
                terms.append(alias)

    lowered = (query or "").lower()

    if "rag hybrid" in lowered or "hybrid rag" in lowered:
        for alias in ["rag", "hybrid", "raghybrid"]:
            if alias not in terms:
                terms.append(alias)

    return terms


def low_quality_text_score(content):
    text = content or ""

    if not text.strip():
        return 10

    tokens = re.findall(r"[A-Za-z0-9+/=]{24,}", text)
    encoded_chars = sum(len(token) for token in tokens)
    ratio = encoded_chars / max(len(text), 1)

    if ratio > 0.35:
        return 20

    if len(re.findall(r"[A-Za-z]{3,}", text)) < 5:
        return 8

    return 0


def metadata_quality_penalty(metadata):
    flags = set((metadata or {}).get("rag_quality_flags") or [])
    penalty = 0

    if "secret_like" in flags:
        penalty += 12
    if "encoded_or_token_like" in flags:
        penalty += 18
    if "short_chunk" in flags:
        penalty += 3

    return penalty


def is_current_query(query):
    lowered = (query or "").lower()
    return any(word in lowered for word in CURRENT_QUERY_WORDS)


def is_conceptual_query(query, terms):
    lowered = (query or "").lower()
    concept_words = ["co to", "czym jest", "jak działa", "wytłumacz", "wyjaśnij", "explain", "what is", "how does"]
    has_concept = any(word in lowered for word in concept_words)
    has_raghybrid_topic = any(term in {"rag", "hybrid", "raghybrid"} for term in terms)
    code_words = {"kod", "code", "python", "plik", "file", "funkcja", "function", "class", "debug", "błąd", "error"}
    asks_code = any(word in lowered for word in code_words)

    return has_concept and has_raghybrid_topic and not asks_code


def is_ansible_playbook_query(query, terms):
    lowered = (query or "").lower()
    has_ansible = "ansible" in lowered or "playbook" in lowered or "playbook" in terms
    asks_roles = "roles" in terms or "role" in terms or "rolami" in lowered or "rola" in lowered

    return has_ansible and asks_roles


def is_polish_legal_query(query, terms):
    lowered = (query or "").lower()
    legal_terms = {
        "konstytucja",
        "konstytucji",
        "minister",
        "ministra",
        "obrony",
        "ustawa",
        "ustawy",
        "ustawach",
        "bezpieczeństwa",
        "bezpieczenstwa",
    }
    return bool(set(terms) & legal_terms) or any(term in lowered for term in legal_terms)


def rerank(query, results):
    scored = []
    terms = search_terms(query)
    conceptual_query = is_conceptual_query(query, terms)
    ansible_playbook_query = is_ansible_playbook_query(query, terms)
    polish_legal_query = is_polish_legal_query(query, terms)

    for r in results:
        content = r[0]
        metadata = r[1] or {}
        source = str(metadata.get("source", "")).lower()
        tags = " ".join(metadata.get("tags") or []).lower()
        distance = float(r[2] or 0.0) if len(r) > 2 else 1.0

        score = -min(distance, 50) * 0.25

        lowered = content.lower()
        heading = "\n".join([
            line for line in lowered.splitlines()[:8]
            if line.startswith("#")
        ])

        exact_hits = 0

        for word in terms:
            if word in heading:
                score += 10
            if word in source:
                score += 2 if source.startswith("repo:") else 8
            if word in tags:
                score += 6
            if word in lowered:
                score += 4
                exact_hits += 1

        if exact_hits >= 2:
            score += 12

        if source.endswith("/schemat") or source.endswith("/schemat_grafu"):
            if any(term in {"rag", "hybrid", "raghybrid", "openwebui", "mcp"} for term in terms):
                score += 35

        if conceptual_query:
            if source.endswith("/schemat") or source.endswith("/schemat_grafu"):
                score += 30
            if source.startswith("repo:"):
                score -= 25

        if source == "chatgpt" and is_current_query(query):
            score -= 35

        if polish_legal_query:
            if "chatgpt" in source or "chatgpt-history" in str(metadata.get("collection", "")).lower():
                score -= 45
            if any(tag in tags for tag in ["constitution", "polish_law", "law", "legislation"]):
                score += 28
            if source.lower().endswith(".pdf"):
                score += 12

        if ansible_playbook_query:
            if source.startswith("repo:") and source.endswith(("/site.yml", "/site.yaml", "/playbook.yml", "/playbook.yaml")):
                score += 30
            if source.startswith("repo:") and "rolling_update" in source:
                score += 20
            if re.search(r"(?m)^\s*roles\s*:", content):
                score += 24
            if re.search(r"(?m)^\s*-\s*common\s*$", content):
                score += 10
            if "wordpress-nginx" in source and "nginx" in terms:
                score += 10
            if source.endswith(("/vars.yml", "/vars.yaml")):
                score -= 25

        score -= low_quality_text_score(content)
        score -= metadata_quality_penalty(metadata)

        score -= min(len(content), 3000) / 6000

        scored.append((score, r))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [r[1] for r in scored]


def hybrid_search(query, tags=None, limit=25):
    print(f"Searching for query: {query}")
    print("SEARCH TAGS:", tags)

    conn = get_conn()
    cur = conn.cursor()
    query_embedding = embed_text(query)

    sql = """
    SELECT content, metadata,
           embedding <-> %s::vector AS distance
    FROM documents
    """

    params = [query_embedding]

    # filtr po tagach
    if tags:
        sql += " WHERE metadata->'tags' ?| %s::text[]"
        params.append(tags)

    try:
        sql += " ORDER BY embedding <-> %s::vector LIMIT %s"
        params.append(query_embedding)
        params.append(limit)

        cur.execute(sql, params)
        results = cur.fetchall()
        seen = {row[0] for row in results}

        lexical_terms = search_terms(query)[:10]

        if lexical_terms:
            where_parts = []
            lexical_score_parts = []
            score_params = []
            where_params = []

            for term in lexical_terms:
                pattern = f"%{term}%"
                where_parts.extend([
                    "content ILIKE %s",
                    "metadata->>'source' ILIKE %s",
                    "metadata::text ILIKE %s",
                ])
                where_params.extend([pattern, pattern, pattern])

                lexical_score_parts.extend([
                    "(CASE WHEN content ILIKE %s THEN 2 ELSE 0 END)",
                    "(CASE WHEN metadata->>'source' ILIKE %s THEN 4 ELSE 0 END)",
                    "(CASE WHEN metadata::text ILIKE %s THEN 1 ELSE 0 END)",
                ])
                score_params.extend([pattern, pattern, pattern])

            lexical_sql = f"""
            SELECT content, metadata,
                   embedding <-> %s::vector AS distance,
                   ({' + '.join(lexical_score_parts)}) AS lexical_score
            FROM documents
            WHERE (
            {' OR '.join(where_parts)}
            """
            lexical_params = [query_embedding] + score_params + where_params

            if tags:
                lexical_sql += ") AND metadata->'tags' ?| %s::text[]"
                lexical_params.append(tags)
            else:
                lexical_sql += ")"

            lexical_sql += " ORDER BY lexical_score DESC, embedding <-> %s::vector LIMIT %s"
            lexical_params.append(query_embedding)
            lexical_params.append(max(limit * 10, 200))

            cur.execute(lexical_sql, lexical_params)

            for row in cur.fetchall():
                if row[0] not in seen:
                    seen.add(row[0])
                    results.append(row)

        results = rerank(query, results)

        print("RESULT COUNT:", len(results))

        return results

    finally:
        cur.close()
        connection_pool.putconn(conn)


def source_evidence_chunks(query, sources, relation_texts=None, exclude_contents=None, limit=6):
    sources = [source for source in sources if source]
    relation_texts = relation_texts or []
    exclude_contents = set(exclude_contents or [])

    if not sources:
        return []

    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT content, metadata
            FROM documents
            WHERE metadata->>'source' = ANY(%s)
            LIMIT 500
            """,
            (sources,)
        )
        terms = evidence_terms(query, relation_texts)
        rows = []
        seen_sources = set()

        for row in cur.fetchall():
            if row[0] in exclude_contents:
                continue

            score = evidence_score(row[0], terms)
            if score <= 0:
                continue

            source = (row[1] or {}).get("source")
            rows.append((score, source, row[0], row[1]))

        rows.sort(reverse=True, key=lambda item: item[0])
        selected = []

        for score, source, content, metadata in rows:
            if source in seen_sources:
                continue

            seen_sources.add(source)
            selected.append((content, metadata, score))

            if len(selected) >= limit:
                break

        return selected

    finally:
        cur.close()
        connection_pool.putconn(conn)


def evidence_terms(query, relation_texts=None):
    relation_texts = relation_texts or []
    raw = " ".join([query] + relation_texts[:12]).lower()
    terms = []

    for token in raw.replace("_", " ").split():
        token = token.strip(".,:;!?()[]{}'\"`-/")

        if len(token) < 3 or token in STOPWORDS:
            continue

        if token not in terms:
            terms.append(token)

    return terms[:30]


def evidence_score(content, terms):
    lowered = (content or "").lower()
    score = 0

    for term in terms:
        if term in lowered:
            score += 1

    return score
