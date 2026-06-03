"""
Hybrid retrieval fusion engine.

Combines results from three independent retrieval paths:
  vector   — pgvector embedding similarity (exp(-d/30) scoring)
  graph    — Neo4j fulltext entity search + relation traversal (BM25 + structural)
  evidence — pgvector chunks cited by graph relations (term-overlap scoring)

Uses Reciprocal Rank Fusion (RRF) with path weights to produce a unified
fused_score that drives the relevance gate.  Graph items can now independently
open the gate when their fused_score >= threshold.
"""

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ── BM25 normalisation constant ───────────────────────────────────────────────
# Neo4j fulltext (Lucene BM25) returns ~4.4 for a perfect single-term match.
# Dividing by this cap maps the range to [0, 1] for most real queries.
_NEO4J_BM25_CAP = 4.0


@dataclass
class FusedCandidate:
    """Unified representation of a retrieval candidate from any path."""

    content: str
    source: str
    item_type: str  # "vector" | "graph" | "graph_evidence"

    # Per-path raw scores (0–1 after normalisation)
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    evidence_score: Optional[float] = None

    # Fusion outputs
    rrf_score: float = 0.0      # accumulated RRF contribution
    fused_score: float = 0.0    # final gate-facing score

    # Provenance
    retrieval_sources: list = field(default_factory=list)  # ["vector","graph",…]
    rank_vector: Optional[int] = None
    rank_graph: Optional[int] = None
    rank_evidence: Optional[int] = None

    # Graph-specific fields
    traversal_path: Optional[str] = None   # "A --rel--> B"
    relation_type: Optional[str] = None
    entity_overlap: float = 0.0            # fraction of query entities in graph
    structural_proximity: Optional[float] = None  # reserved for hop-distance

    # Vector-specific
    distance: Optional[float] = None      # raw L2 distance

    # Common
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    rerank_reason: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def content_key(content: str, source: str) -> str:
    """Stable hash key for deduplication across retrieval paths."""
    return hashlib.md5(f"{source}|{(content or '')[:200]}".encode()).hexdigest()[:16]


def rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a single item at position `rank`."""
    return 1.0 / (k + rank + 1)


def normalize_graph_score(
    ft_score: float,
    relation: str,
    n_sources: int,
    relation_priority_map: Optional[dict] = None,
) -> float:
    """
    Map raw Neo4j fulltext + structural signals → [0, 1].

    Components
    ----------
    ft_score : BM25 score from Neo4j fulltext index
    relation : relation type string (looked up in RELATION_PRIORITY)
    n_sources: number of source documents attesting this relation
    """
    rp = relation_priority_map or {}
    ft_part = min(ft_score / _NEO4J_BM25_CAP, 1.0)
    rel_part = rp.get(relation, 20) / 80.0          # RELATION_PRIORITY max = 80
    src_part = min(n_sources / 5.0, 1.0)            # saturates at 5 sources

    # Weights: fulltext quality > relation type > source confidence
    return round(0.50 * ft_part + 0.30 * rel_part + 0.20 * src_part, 4)


def normalize_evidence_score(term_hits: float, max_hits: float) -> float:
    """Map raw term-overlap count → [0, 1]."""
    if max_hits <= 0:
        return 0.0
    return round(min(term_hits / max_hits, 1.0), 4)


def lucene_query_string(terms: list) -> str:
    """Convert search terms to a Lucene OR query for Neo4j fulltext."""
    parts = []
    for term in terms[:8]:
        term = str(term).strip()
        if not term:
            continue
        # Escape Lucene special characters
        escaped = re.sub(r'([+\-&|!(){}[\]^"~*?:\\/])', r'\\\1', term)
        if " " in escaped:
            parts.append(f'"{escaped}"')
        else:
            parts.append(escaped)
    return " OR ".join(parts) if parts else ""


# ── Core fusion ───────────────────────────────────────────────────────────────

def fuse_retrieval_results(
    vector_candidates: list,
    graph_candidates: list,
    evidence_candidates: list,
    rrf_k: int = 60,
    weights: tuple = (0.50, 0.30, 0.20),
) -> list:
    """
    RRF fusion of three retrieval paths.

    Items appearing in multiple paths accumulate RRF score from each path
    (weighted by path weight), then receive a final fused_score:

        fused = 0.40·vector + 0.25·graph + 0.15·evidence + 0.20·rrf_norm

    The 0.20 RRF term rewards items found by multiple paths (cross-path
    corroboration).  Items found only by graph can still reach fused ≥ 0.45
    when graph_score is strong (e.g. 0.80 → 0.25·0.80 + 0.20·rrf ≈ 0.40+,
    plus entity_overlap boost).

    Returns
    -------
    Sorted list of FusedCandidate, highest fused_score first.
    """
    merged: dict = {}
    path_names = ["vector", "graph", "evidence"]
    all_lists = [vector_candidates, graph_candidates, evidence_candidates]

    for path_idx, (candidates, weight) in enumerate(zip(all_lists, weights)):
        path_name = path_names[path_idx]

        for rank, cand in enumerate(candidates):
            key = content_key(cand.content, cand.source)

            if key not in merged:
                # First encounter: copy full candidate
                merged[key] = FusedCandidate(
                    content=cand.content,
                    source=cand.source,
                    item_type=cand.item_type,
                    vector_score=cand.vector_score,
                    graph_score=cand.graph_score,
                    evidence_score=cand.evidence_score,
                    traversal_path=cand.traversal_path,
                    relation_type=cand.relation_type,
                    entity_overlap=cand.entity_overlap,
                    distance=cand.distance,
                    metadata=cand.metadata,
                    tags=cand.tags,
                )
            else:
                # Seen before: merge best scores across paths
                m = merged[key]
                if cand.vector_score is not None:
                    m.vector_score = max(m.vector_score or 0.0, cand.vector_score)
                if cand.graph_score is not None:
                    m.graph_score = max(m.graph_score or 0.0, cand.graph_score)
                if cand.evidence_score is not None:
                    m.evidence_score = max(m.evidence_score or 0.0, cand.evidence_score)
                # Prefer richer traversal metadata
                if cand.traversal_path and not m.traversal_path:
                    m.traversal_path = cand.traversal_path
                    m.relation_type = cand.relation_type

            m = merged[key]
            # Accumulate weighted RRF
            m.rrf_score += weight * rrf(rank, k=rrf_k)

            # Track which paths contributed
            if path_name not in m.retrieval_sources:
                m.retrieval_sources.append(path_name)

            # Record best rank per path
            if path_name == "vector" and m.rank_vector is None:
                m.rank_vector = rank
            elif path_name == "graph" and m.rank_graph is None:
                m.rank_graph = rank
            elif path_name == "evidence" and m.rank_evidence is None:
                m.rank_evidence = rank

    # ── Compute fused_score for each candidate ─────────────────────────────
    #
    # Formula:  fused = base_score + rrf_bonus
    #
    # base_score = max of path-specific scores (preserves the strongest signal;
    #              prevents dilution of a strong single-path match):
    #   vector:   raw embedding similarity (calibrated exp(-d/30))
    #   graph:    BM25 + relation + source confidence (×0.85 discount — entity-
    #             level matching is semantically coarser than full-chunk embed)
    #   evidence: term-overlap fraction (×0.70 discount — weakest signal)
    #
    # rrf_bonus = 0–0.15 awarded to items found by 2+ paths simultaneously
    #             (cross-path corroboration). Max bonus = 0.15 × rrf_norm
    #             when ranked #1 across all three paths.
    #
    # Properties
    # ----------
    # • A strong vector item (vs=0.48) → fused ≈ 0.48 + small_rrf  (gate passes)
    # • A strong graph item  (gs=0.70) → fused ≈ 0.595 + small_rrf (gate passes)
    # • Multi-path item             → up to +0.15 bonus over single-path
    max_rrf = sum(w * rrf(0, rrf_k) for w in weights)

    for m in merged.values():
        vs = m.vector_score or 0.0
        gs = m.graph_score or 0.0
        es = m.evidence_score or 0.0
        rrf_norm = min(m.rrf_score / max(max_rrf, 1e-9), 1.0)

        # Best individual path score (with path-specific discounts)
        # Structural pointer triples (--contains-->, --is_a-->) carry no text — penalise
        # them heavily so actual document chunks are ranked above entity-index noise.
        _structural = any(t in (m.content or "") for t in ("--contains-->", "--is_a-->"))
        base_score = max(
            vs,                                       # embedding similarity — used as-is
            gs * (0.40 if _structural else 0.85),     # graph BM25; pointer triples discounted
            es * 0.70,                                # term overlap — weaker signal
        )
        # Cross-path corroboration bonus (max 0.15, only meaningful when 2+ paths agree)
        rrf_bonus = rrf_norm * 0.15

        m.fused_score = round(min(base_score + rrf_bonus, 1.0), 4)

        # Human-readable rerank explanation
        paths = m.retrieval_sources
        if len(paths) >= 2:
            m.rerank_reason = f"multi_path({'|'.join(paths)})"
        elif "graph" in paths:
            m.rerank_reason = "graph_primary"
        elif "evidence" in paths:
            m.rerank_reason = "evidence_primary"
        else:
            m.rerank_reason = "vector_primary"

    return sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)


def fused_to_dict(cand: FusedCandidate, rank: int) -> dict:
    """Convert FusedCandidate to the dict format consumed by the pipeline."""
    return {
        "rank": rank,
        "type": cand.item_type,
        "content": cand.content,
        "source": cand.source,
        # score = fused_score so normalized_result_score() in relevance.py picks it up
        "score": cand.fused_score,
        "fused_score": cand.fused_score,
        "vector_score": cand.vector_score,
        "graph_score": cand.graph_score,
        "evidence_score": cand.evidence_score,
        "retrieval_sources": cand.retrieval_sources,
        "traversal_path": cand.traversal_path,
        "relation_type": cand.relation_type,
        "entity_overlap": cand.entity_overlap,
        "rerank_reason": cand.rerank_reason,
        "rrf_score": round(cand.rrf_score, 6),
        "rank_vector": cand.rank_vector,
        "rank_graph": cand.rank_graph,
        "rank_evidence": cand.rank_evidence,
        "distance": cand.distance,
        "metadata": cand.metadata,
        "tags": cand.tags,
    }
