import math
import os
import re
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_NO_CONTEXT_INSTRUCTION = (
    "No relevant context was found in the knowledge base. "
    "Answer using general knowledge only, and do not claim the answer comes from RAGHybrid."
)

STOPWORDS = {
    "and", "are", "czy", "dla", "for", "jak", "jest", "jaki", "jaka", "jakie",
    "kto", "mowi", "mówi", "oraz", "się", "sie", "the", "with", "wczoraj",
}

CURRENT_QUERY_PATTERNS = [
    r"\b(wczoraj|dzisiaj|dziś|teraz|aktualnie|najnowsz\w*|ostatni\w*)\b",
    r"\b(kto|co|jaki|jaka|jakie)\s+(wygrał|wygral|wygrała|wygrala)\b",
    r"\b(mecz|wynik|score|liga|turniej|wybory|kurs|cena|pogoda)\b",
]

CURRENT_SOURCE_TYPES = {
    "current",
    "live",
    "news",
    "sports",
    "weather",
    "market",
    "finance",
}


@dataclass
class RelevanceDecision:
    rag_used: bool
    score: float
    reason: str
    top_score: Optional[float]
    lexical_overlap: float
    accepted_results: list[dict[str, Any]]
    rejected_results: list[dict[str, Any]]
    threshold: float
    results_before_gate: int

    @property
    def results_after_gate(self) -> int:
        return len(self.accepted_results)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def query_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ-]+", text.lower()):
        token = token.strip("-_")
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def term_variants(term: str) -> set[str]:
    variants = {term}

    replacements = {
        "bezpieczeństwa": "bezpieczeństwo",
        "bezpieczenstwa": "bezpieczenstwo",
        "sejmie": "sejm",
        "sejmu": "sejm",
        "sejmem": "sejm",
        "konstytucji": "konstytucja",
        "konstytucją": "konstytucja",
        "ministra": "minister",
        "ministrem": "minister",
        "obrony": "obrona",
        "narodowej": "narodowa",
        "prezydencie": "prezydent",
        "prezydenta": "prezydent",
        "ustawach": "ustawa",
        "ustawie": "ustawa",
        "ustawy": "ustawa",
    }

    if term in replacements:
        variants.add(replacements[term])

    for suffix in ["owie", "ami", "ach", "ego", "emu", "owi", "iej", "ie", "em", "om", "ą", "ę", "u", "a", "y"]:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            variants.add(term[:-len(suffix)])

    return {variant for variant in variants if len(variant) >= 3}


def term_in_text(term: str, text: str) -> bool:
    return any(variant in text for variant in term_variants(term))


def lexical_overlap(query: str, results: list[dict[str, Any]]) -> float:
    terms = query_terms(query)
    if not terms:
        return 0.0

    searchable_text = " ".join(
        [
            str(item.get("content") or "")
            + " "
            + str(item.get("source") or "")
            + " "
            + " ".join(item.get("tags") or [])
            for item in results
            if item.get("type") in {"vector", "graph_evidence"}
        ]
    ).lower()
    if not searchable_text:
        return 0.0

    matched = sum(1 for term in terms if term_in_text(term, searchable_text))
    return matched / max(len(terms), 1)


def result_lexical_overlap(query: str, item: dict[str, Any]) -> float:
    terms = query_terms(query)
    if not terms or item.get("type") not in {"vector", "graph_evidence"}:
        return 0.0

    searchable_text = (
        str(item.get("content") or "")
        + " "
        + str(item.get("source") or "")
        + " "
        + " ".join(item.get("tags") or [])
    ).lower()
    matched = sum(1 for term in terms if term_in_text(term, searchable_text))
    return matched / max(len(terms), 1)


def is_current_or_temporal_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in CURRENT_QUERY_PATTERNS)


def has_current_source(results: list[dict[str, Any]]) -> bool:
    for item in results:
        metadata = item.get("metadata") or {}
        source_type = str(metadata.get("source_type") or "").lower()
        collection = str(metadata.get("collection") or "").lower()
        tags = {str(tag).lower() for tag in (metadata.get("tags") or item.get("tags") or [])}

        if source_type in CURRENT_SOURCE_TYPES or collection in CURRENT_SOURCE_TYPES or tags & CURRENT_SOURCE_TYPES:
            return True

    return False


def quality_penalty(item: dict[str, Any]) -> float:
    metadata = item.get("metadata") or {}
    flags = set(metadata.get("rag_quality_flags") or [])
    penalty = 0.0

    if "secret_like" in flags:
        penalty += 0.20
    if "encoded_or_token_like" in flags:
        penalty += 0.25
    if "short_chunk" in flags:
        penalty += 0.05

    return penalty


def normalized_result_score(item: dict[str, Any]) -> float:
    metadata = item.get("metadata") or {}
    item_type = item.get("type")

    # ── Fused score from hybrid pipeline (highest priority) ──────────────────
    # When the fusion engine has run, every item carries fused_score which
    # already incorporates vector_score + graph_score + evidence_score + RRF.
    # Use it directly so the gate reflects the true cross-path relevance.
    fused = item.get("fused_score")
    if fused is not None:
        try:
            return max(0.0, min(float(fused), 1.0))
        except (TypeError, ValueError):
            pass

    # ── Legacy path: graph items without fusion ───────────────────────────────
    if item_type == "graph":
        # If the item has a graph_score from scored retrieval, use it;
        # otherwise fall back to the conservative constant so pure-graph
        # items without fusion scores don't accidentally open the gate.
        gs = item.get("graph_score")
        if gs is not None:
            try:
                return max(0.0, min(float(gs), 1.0))
            except (TypeError, ValueError):
                pass
        return 0.15

    raw_score = item.get("score")
    if raw_score is None:
        raw_score = metadata.get("relevance_score") or metadata.get("score")
    if raw_score is not None:
        try:
            score = float(raw_score)
            return max(0.0, min(score, 1.0))
        except (TypeError, ValueError):
            pass

    distance = item.get("distance")
    if distance is None:
        distance = metadata.get("retrieval_distance") or metadata.get("distance")
    if distance is not None:
        try:
            # exp(-d/30) calibrated for nomic-embed-text L2 distances ~18-25.
            return math.exp(-max(float(distance), 0.0) / 30.0)
        except (TypeError, ValueError):
            pass

    if item_type == "graph_evidence":
        return 0.55

    return 0.35 if item_type == "vector" else 0.0


def calculate_relevance(query: str, results: list[dict[str, Any]]) -> RelevanceDecision:
    threshold = env_float("RAG_MIN_RELEVANCE_SCORE", 0.45)
    min_overlap = env_float("RAG_MIN_LEXICAL_OVERLAP", 0.10)
    min_results = env_int("RAG_MIN_RESULTS", 1)
    require_graph_evidence = env_bool("RAG_REQUIRE_EVIDENCE_FOR_GRAPH", True)

    results = list(results or [])
    scored = [
        (max(0.0, normalized_result_score(item) - quality_penalty(item)), item)
        for item in results
    ]
    top_score = max((score for score, _ in scored), default=None)
    item_overlaps = {
        id(item): result_lexical_overlap(query, item)
        for _, item in scored
    }
    strong_item_overlap = max(0.75, min_overlap)

    # ── Determine whether fusion is active ───────────────────────────────────
    # When fused_score is present on items, graph can act as a primary retriever.
    fusion_active = any(item.get("fused_score") is not None for item in results)

    # ── High-confidence graph: items with fused_score >= threshold ────────────
    # These items count as "text support" even though they are graph triples,
    # because fusion already validated them against vector/evidence evidence.
    def _is_high_conf_graph(item: dict[str, Any], score: float) -> bool:
        return (
            item.get("type") == "graph"
            and fusion_active
            and score >= threshold
        )

    text_supported = [
        item for score, item in scored
        if (
            item.get("type") in {"vector", "graph_evidence"}
            or _is_high_conf_graph(item, score)
        )
        and (score >= threshold or item_overlaps.get(id(item), 0.0) >= strong_item_overlap)
    ]

    # text_support_count: how many non-passive items exist
    # With fusion, high-confidence graph triples count.
    text_support_count = sum(
        1 for score, item in scored
        if item.get("type") in {"vector", "graph_evidence"}
        or _is_high_conf_graph(item, score)
    )

    vector_count = sum(1 for item in results if item.get("type") == "vector")
    graph_count = sum(1 for item in results if item.get("type") == "graph")
    evidence_count = sum(1 for item in results if item.get("type") == "graph_evidence")
    overlap = lexical_overlap(query, results)
    max_item_overlap = max(item_overlaps.values(), default=0.0)

    score_parts = [
        top_score or 0.0,
        overlap,
        max_item_overlap,
        min(len(text_supported) / max(min_results, 1), 1.0),
    ]
    final_score = round(
        (score_parts[0] * 0.35)
        + (score_parts[1] * 0.25)
        + (score_parts[2] * 0.25)
        + (score_parts[3] * 0.15),
        4
    )

    # ── Build accepted list ───────────────────────────────────────────────────
    # High-confidence graph items (fused_score >= threshold) are accepted
    # unconditionally.  Low-confidence graph items still need vector/evidence.
    if fusion_active:
        high_conf_graph = [
            item for score, item in scored
            if item.get("type") == "graph" and score >= threshold
        ]
        low_conf_graph = [
            item for score, item in scored
            if item.get("type") == "graph" and score < threshold
        ]
        non_graph = [item for item in results if item.get("type") != "graph"]
        accepted = non_graph + high_conf_graph
        if low_conf_graph and (not require_graph_evidence or vector_count or evidence_count):
            accepted.extend(low_conf_graph)
    else:
        accepted = [item for item in results if item.get("type") != "graph"]
        if graph_count and (not require_graph_evidence or vector_count or evidence_count):
            accepted.extend([item for item in results if item.get("type") == "graph"])

    reason = "relevant"
    rag_used = True

    # When fusion is active, top_score IS the fused score (vector + graph + RRF).
    # If top_score >= threshold, fusion has already validated relevance; the gate
    # should respect that judgment even when lexical overlap is near-zero
    # (e.g. Polish queries, entity-only graph matches).
    fusion_overrides_gate = (
        fusion_active
        and top_score is not None
        and top_score >= threshold
        and text_support_count > 0
    )

    if is_current_or_temporal_query(query) and not has_current_source(results):
        rag_used = False
        reason = "current_query_without_current_source"
    elif not results:
        rag_used = False
        reason = "empty_results"
    elif text_support_count == 0:
        # Graph-only without fusion evidence: gate blocks unless fusion gave strong scores.
        # With fusion active this path is only reached when ALL graph scores < threshold.
        rag_used = False
        reason = "graph_only_without_evidence"
    elif len(text_supported) < min_results and overlap < 0.70:
        if not fusion_overrides_gate:
            rag_used = False
            reason = "not_enough_relevant_results"
    elif overlap < min_overlap and not fusion_active:
        # When fusion is active, lexical overlap is a weak signal — skip this block.
        rag_used = False
        reason = "low_lexical_overlap"
    elif final_score < threshold and overlap < 0.70:
        # Fusion override: if the fused top_score clearly exceeds threshold, allow
        # retrieval even when the composite final_score (which mixes in lexical
        # components) falls short.  The fused score already encodes multi-path
        # validation; requiring lexical overlap on top is redundant.
        if not fusion_overrides_gate:
            rag_used = False
            reason = "low_relevance"
    elif (
        graph_count
        and require_graph_evidence
        and evidence_count == 0
        and vector_count == 0
        and not (fusion_active and text_support_count > 0)
    ):
        rag_used = False
        reason = "graph_only_without_evidence"

    if not rag_used:
        accepted = []

    rejected = [item for item in results if item not in accepted]

    return RelevanceDecision(
        rag_used=rag_used,
        score=final_score,
        reason=reason,
        top_score=top_score,
        lexical_overlap=round(overlap, 4),
        accepted_results=accepted,
        rejected_results=rejected,
        threshold=threshold,
        results_before_gate=len(results),
    )
