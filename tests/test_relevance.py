from app.rag.relevance import calculate_relevance


def test_technical_question_with_matching_context_uses_rag(monkeypatch):
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", "0.45")
    decision = calculate_relevance(
        "jak działa Docker",
        [
            {
                "type": "vector",
                "content": "Docker uses images and containers to package and run applications.",
                "source": "docs/docker.md",
                "metadata": {"retrieval_distance": 0.35},
            }
        ],
    )

    assert decision.rag_used is True
    assert decision.accepted_results


def test_unrelated_question_is_gated_out(monkeypatch):
    monkeypatch.setenv("RAG_MIN_LEXICAL_OVERLAP", "0.10")
    decision = calculate_relevance(
        "ile się rozkłada kupa psa",
        [
            {
                "type": "vector",
                "content": "Terraform remote state stores shared infrastructure state.",
                "source": "docs/terraform.md",
                "metadata": {"retrieval_distance": 0.2},
            }
        ],
    )

    assert decision.rag_used is False
    assert decision.reason == "low_lexical_overlap"
    assert decision.accepted_results == []


def test_empty_results_return_no_context():
    decision = calculate_relevance("jaki jest przepis na sernik", [])

    assert decision.rag_used is False
    assert decision.reason == "empty_results"
    assert decision.accepted_results == []


def test_graph_only_without_evidence_does_not_pass(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_EVIDENCE_FOR_GRAPH", "true")
    decision = calculate_relevance(
        "jak działa OpenShift ingress",
        [
            {
                "type": "graph",
                "content": "OpenShift --uses--> Ingress",
                "source": "knowledge_graph",
                "metadata": {"source": "OpenShift", "relation": "uses", "target": "Ingress"},
            }
        ],
    )

    assert decision.rag_used is False
    assert decision.reason == "graph_only_without_evidence"


def test_debug_values_do_not_include_secret_material():
    decision = calculate_relevance(
        "co to jest Terraform remote state",
        [
            {
                "type": "vector",
                "content": "Terraform remote state shares state between runs.",
                "source": "docs/terraform.md",
                "metadata": {"retrieval_distance": 0.4},
            }
        ],
    )

    debug = {
        "rag_used": decision.rag_used,
        "relevance_score": decision.score,
        "gate_reason": decision.reason,
        "top_score": decision.top_score,
        "lexical_overlap": decision.lexical_overlap,
    }
    text = str(debug).lower()

    assert "password" not in text
    assert "postgresql://" not in text
    assert "secret" not in text
