import json
import re
import time

from neo4j import GraphDatabase

from app.config import settings
from app.rag.graph_schema import ALLOWED_RELATIONS


driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)

_FULLTEXT_INDEX_READY: bool = False


def _ensure_fulltext_index() -> bool:
    """Create entity_fulltext index if not already present. Returns True on success."""
    global _FULLTEXT_INDEX_READY
    if _FULLTEXT_INDEX_READY:
        return True
    try:
        with driver.session() as session:
            session.run(
                "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                "FOR (n:Entity) ON EACH [n.name]"
            )
        _FULLTEXT_INDEX_READY = True
        return True
    except Exception as exc:
        print(f"WARNING: entity_fulltext index unavailable: {exc}")
        return False


RELATION_PRIORITY = {
    "is_a": 68,
    "depends_on": 80,
    "requires": 78,
    "builds_on": 76,
    "runs_on": 74,
    "uses": 70,
    "provides": 66,
    "stores": 64,
    "hosts": 62,
    "manages": 60,
    "supports": 58,
    "exposes": 56,
    "contains": 54,
    "includes": 52,
    "creates": 50,
    "accesses": 48,
    "allows": 46,
    "extends": 42,
    "has": 30,
    "is_managed_by": 28,
}


def close_graph():
    driver.close()


def upsert_relation(source: str, relation: str, target: str, metadata=None):
    metadata = metadata or {}
    source_name = str(metadata.get("source", "")).strip()
    merge_token = f"{source}:{relation}:{target}:{time.time_ns()}"

    query = """
    MERGE (s:Entity {name: $source})
    MERGE (t:Entity {name: $target})
    MERGE (s)-[r:RELATED {type: $relation}]->(t)
    ON CREATE SET r.created_token = $merge_token
    WITH r, r.created_token = $merge_token AS created_now, $metadata AS metadata, $source_name AS source_name
    SET r.metadata = $metadata,
        r.sources = CASE
            WHEN source_name = "" THEN coalesce(r.sources, [])
            WHEN source_name IN coalesce(r.sources, []) THEN coalesce(r.sources, [])
            ELSE coalesce(r.sources, []) + source_name
        END
    RETURN created_now AS created_now,
           size(coalesce(r.sources, [])) AS source_count
    """

    with driver.session() as session:
        row = session.run(
            query,
            source=source,
            relation=relation,
            target=target,
            metadata=json.dumps(metadata),
            source_name=source_name,
            merge_token=merge_token
        ).single()

        return {
            "created": bool(row["created_now"]) if row else False,
            "source_count": int(row["source_count"]) if row else 0
        }


GRAPH_STOPWORDS = {
    "about",
    "auto",
    "backend",
    "czy",
    "dla",
    "from",
    "jak",
    "jest",
    "oraz",
    "pod",
    "the",
    "with",
    "what",
    "when",
    "where",
    "który",
}


def graph_search_terms(text: str, extra_terms=None, max_terms: int = 8):
    extra_terms = extra_terms or []
    terms = []

    for value in [text] + list(extra_terms):
        value = str(value or "").strip()
        if not value:
            continue

        terms.append(value)

        for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b", value):
            terms.append(match.group(0))

        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", value.lower()):
            if token not in GRAPH_STOPWORDS and len(token) >= 4:
                terms.append(token)

    clean_terms = []
    seen = set()

    for term in terms:
        term = re.sub(r"\s+", " ", str(term)).strip(" `\"'.,:;!?()[]{}")
        key = term.lower()

        if len(term) < 3 or key in seen:
            continue

        seen.add(key)
        clean_terms.append(term)

        if len(clean_terms) >= max_terms:
            break

    return clean_terms


def search_graph(entity: str, limit: int = 10, extra_terms=None):
    terms = graph_search_terms(entity, extra_terms=extra_terms)

    if not terms:
        return []

    query = """
    MATCH (s:Entity)-[r:RELATED]->(t:Entity)
    WHERE r.type IN $allowed_relations
      AND any(term IN $terms WHERE
        toLower(s.name) CONTAINS toLower(term)
        OR toLower(t.name) CONTAINS toLower(term)
      )
    WITH s, r, t,
      reduce(score = 0, term IN $terms |
        score
        + CASE WHEN toLower(s.name) = toLower(term) THEN 10 ELSE 0 END
        + CASE WHEN toLower(t.name) = toLower(term) THEN 10 ELSE 0 END
        + CASE WHEN toLower(s.name) CONTAINS toLower(term) THEN 2 ELSE 0 END
        + CASE WHEN toLower(t.name) CONTAINS toLower(term) THEN 2 ELSE 0 END
      ) AS score
    RETURN s.name AS source,
           r.type AS relation,
           t.name AS target,
           coalesce(r.sources, []) AS sources,
           score
    ORDER BY score DESC, size(coalesce(r.sources, [])) DESC
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            query,
            terms=terms,
            allowed_relations=sorted(ALLOWED_RELATIONS),
            limit=max(limit * 20, 100)
        )

        candidates = [
            {
                "source": row["source"],
                "relation": row["relation"],
                "target": row["target"],
                "sources": row["sources"],
                "score": row["score"]
            }
            for row in rows
        ]
        return choose_graph_relations(candidates, limit=limit)


def choose_graph_relations(rows, limit: int = 10):
    best_by_pair = {}

    for row in rows:
        sources = row.get("sources") or []
        relation = row.get("relation")
        source = row.get("source")
        target = row.get("target")

        if not sources or relation not in ALLOWED_RELATIONS or not source or not target:
            continue

        pair = (source.lower(), target.lower())
        rank = (
            int(row.get("score") or 0),
            RELATION_PRIORITY.get(relation, 0),
            min(len(sources), 10),
        )

        previous = best_by_pair.get(pair)
        if not previous or rank > previous[0]:
            best_by_pair[pair] = (rank, row)

    selected = [item[1] for item in best_by_pair.values()]
    selected.sort(
        key=lambda row: (
            int(row.get("score") or 0),
            RELATION_PRIORITY.get(row.get("relation"), 0),
            min(len(row.get("sources") or []), 10),
        ),
        reverse=True
    )
    return selected[:limit]


def get_graph_data(entity: str = "", limit: int = 100):
    query = """
    MATCH (s:Entity)-[r:RELATED]->(t:Entity)
    WHERE $entity = ""
       OR toLower(s.name) CONTAINS toLower($entity)
       OR toLower(t.name) CONTAINS toLower($entity)
    RETURN s.name AS source, r.type AS relation, t.name AS target
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(query, entity=entity, limit=limit)

        nodes = {}
        edges = []

        for row in rows:
            source = row["source"]
            target = row["target"]
            relation = row["relation"]

            nodes[source] = {
                "id": source,
                "label": source,
                "title": source
            }

            nodes[target] = {
                "id": target,
                "label": target,
                "title": target
            }

            edges.append({
                "from": source,
                "to": target,
                "label": relation,
                "arrows": "to",
                "title": relation
            })

        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }


def get_graph_counts():
    with driver.session() as session:
        node_count = session.run(
            "MATCH (n:Entity) RETURN count(n) AS count"
        ).single()["count"]
        edge_row = session.run(
            """
            MATCH ()-[r:RELATED]->()
            RETURN count(r) AS edge_count, count(DISTINCT r.type) AS relation_types
            """
        ).single()

        return {
            "nodes": int(node_count),
            "edges": int(edge_row["edge_count"]),
            "relation_types": int(edge_row["relation_types"])
        }


# ── Scored graph retrieval (hybrid fusion path) ───────────────────────────────

def search_graph_scored(query: str, limit: int = 12, extra_terms=None) -> list:
    """
    Full-retriever graph search with normalized graph_score.

    Pipeline
    --------
    1. Build Lucene query from search terms (graph_search_terms + extra_terms)
    2. Run Neo4j fulltext index query → matched entities + BM25 scores
    3. Traverse matched entities → adjacent relations (both directions)
    4. Compute graph_score = normalize(BM25, relation_priority, n_sources)
    5. Dedup by (source, target) pair; return top `limit` by graph_score

    Falls back to term-matching search_graph() if fulltext index unavailable.

    Returns
    -------
    List of dicts with keys: source, relation, target, sources,
    score (raw BM25), graph_score (normalised 0–1), ft_score.
    """
    from app.rag.fusion import normalize_graph_score, lucene_query_string

    _ensure_fulltext_index()

    terms = graph_search_terms(query, extra_terms=extra_terms or [])
    ft_query = lucene_query_string(terms)
    if not ft_query:
        return []

    try:
        with driver.session() as session:
            # Phase 1: fulltext entity search
            ft_rows = list(session.run(
                """
                CALL db.index.fulltext.queryNodes("entity_fulltext", $q, {limit: 20})
                YIELD node, score AS ft_score
                RETURN node.name AS entity, ft_score
                ORDER BY ft_score DESC
                """,
                q=ft_query,
            ))

            if not ft_rows:
                raise ValueError("fulltext returned no entity candidates")

            entity_scores = {row["entity"]: float(row["ft_score"]) for row in ft_rows}
            entity_names = list(entity_scores.keys())

            # Phase 2: traverse from matched entities (both directions)
            rel_rows = list(session.run(
                """
                MATCH (s:Entity)-[r:RELATED]->(t:Entity)
                WHERE (s.name IN $entities OR t.name IN $entities)
                  AND r.type IN $allowed
                RETURN s.name AS source,
                       r.type AS relation,
                       t.name AS target,
                       coalesce(r.sources, []) AS sources
                LIMIT $cap
                """,
                entities=entity_names,
                allowed=sorted(ALLOWED_RELATIONS),
                cap=limit * 6,
            ))

        candidates = []
        for row in rel_rows:
            src = row["source"]
            tgt = row["target"]
            rel = row["relation"]
            srcs = list(row["sources"] or [])

            # BM25 score from whichever endpoint matched (take max)
            ft = max(entity_scores.get(src, 0.0), entity_scores.get(tgt, 0.0))
            gs = normalize_graph_score(ft, rel, len(srcs), RELATION_PRIORITY)

            candidates.append({
                "source": src,
                "relation": rel,
                "target": tgt,
                "sources": srcs,
                "score": ft,       # raw BM25 (backward compat)
                "ft_score": ft,
                "graph_score": gs,
            })

        return _choose_scored(candidates, limit=limit)

    except Exception as exc:
        print(f"INFO: graph fulltext search fell back to term matching: {exc}")
        # Fallback: use existing term search, inject normalised graph_score
        results = search_graph(query, limit=limit, extra_terms=extra_terms)
        n_terms = max(len(terms), 1)
        for r in results:
            raw = float(r.get("score") or 0)
            r.setdefault("ft_score", 0.0)
            r.setdefault(
                "graph_score",
                normalize_graph_score(
                    min(raw / (10.0 * n_terms) * _NEO4J_BM25_CAP, _NEO4J_BM25_CAP),
                    r.get("relation", ""),
                    len(r.get("sources") or []),
                    RELATION_PRIORITY,
                ),
            )
        return results


_NEO4J_BM25_CAP = 4.0


def _choose_scored(rows: list, limit: int = 10) -> list:
    """Dedup by (source, target) pair, keep highest graph_score."""
    best: dict = {}
    for row in rows:
        if not row.get("source") or not row.get("target"):
            continue
        pair = (row["source"].lower(), row["target"].lower())
        gs = row.get("graph_score", 0.0)
        if pair not in best or gs > best[pair][0]:
            best[pair] = (gs, row)

    selected = [item[1] for item in best.values()]
    selected.sort(key=lambda r: r.get("graph_score", 0.0), reverse=True)
    return selected[:limit]
