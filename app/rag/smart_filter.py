import json
import re

from app.rag.generate import generate
from app.rag.generate import DEFAULT_MODEL


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_json_array(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


TECH_TAG_KEYWORDS = {
    "ansible": ["ansible", "playbook"],
    "awk": ["awk"],
    "bash": ["bash", "shell", "skrypt", "script"],
    "ceph": ["ceph"],
    "docker": ["docker", "compose", "container", "kontener"],
    "fastapi": ["fastapi"],
    "git": ["git", "github", "gitlab"],
    "kubernetes": ["kubernetes", "kubectl", "pod", "pods", "deployment", "service", "ingress"],
    "linux": ["linux", "systemctl", "journalctl", "ssh", "sudo", "grep", "sed", "chmod", "chown"],
    "networking": ["dns", "tcp", "udp", "ip route", "firewall", "vlan", "network", "sieci", "ssh"],
    "openshift": ["openshift", "oc ", "odf", "operator", "route"],
    "postgres": ["postgres", "postgresql", "pgvector"],
    "python": ["python", "pip", "venv", "django", "flask"],
    "rag": ["rag", "embedding", "vector", "llm", "ollama"],
    "security": ["tls", "ssl", "cert", "oauth", "keycloak", "security"],
    "sql": ["sql", "select", "insert", "update", "delete", "join", "database", "baza danych"],
    "storage": ["storage", "disk", "lvm", "nfs", "iscsi", "s3"],
}

NON_TECH_KEYWORDS = [
    "angielski", "gramatyka", "wakacje", "film", "piosenka", "jedzenie",
    "przepis", "samochód", "auto", "rower", "zakupy", "telefon", "zdjęcie",
    "miłość", "randka", "pogoda", "hotel", "lot", "bilet",
]


def quick_classify_qa(question: str, answer: str):
    text = f"{question}\n{answer}".lower()

    if len(question) < 30 or len(answer) < 60:
        return {
            "decision": "reject",
            "tags": [],
            "reason": "too_short",
        }

    tags = []
    for tag, keywords in TECH_TAG_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    has_code_signal = bool(re.search(r"(```|\b(select|kubectl|docker|oc|ssh|awk|grep|systemctl|curl|ansible|python|pip)\b|[{};=|])", text))
    non_tech_hits = sum(1 for keyword in NON_TECH_KEYWORDS if keyword in text)

    if tags and (has_code_signal or len(tags) >= 2):
        return {
            "decision": "keep",
            "tags": tags[:10],
            "reason": "quick_technical_match",
        }

    if non_tech_hits and not tags and not has_code_signal:
        return {
            "decision": "reject",
            "tags": [],
            "reason": "quick_non_technical_match",
        }

    if not tags and not has_code_signal:
        return {
            "decision": "reject",
            "tags": [],
            "reason": "no_technical_signal",
        }

    return {
        "decision": "llm",
        "tags": tags[:10],
        "reason": "needs_llm",
    }


def classify_qa(question: str, answer: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    prompt = f"""
You are a strict knowledge-base curator.

Decide if this Q/A pair is useful for a technical RAG knowledge base.

Useful examples:
- DevOps, Docker, Kubernetes, OpenShift, Ansible, Linux, AI infra, RAG, security
- troubleshooting steps
- commands
- architecture decisions
- configuration explanations

Not useful:
- greetings
- casual chat
- very short answers
- duplicate or low-value content
- unclear conversations

Return ONLY valid JSON:
{{
  "keep": true,
  "tags": ["docker", "networking"],
  "reason": "short reason"
}}

Question:
{question}

Answer:
{answer}
"""

    raw = generate(prompt, backend=backend, model=model)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "keep": False,
            "tags": [],
            "reason": "failed_to_parse_llm_response"
        }

    keep = bool(parsed.get("keep", False))
    tags = parsed.get("tags", [])

    if not isinstance(tags, list):
        tags = []

    tags = [
        str(t).lower().replace("#", "").strip()
        for t in tags
        if str(t).strip()
    ]

    return {
        "keep": keep,
        "tags": tags[:10],
        "reason": parsed.get("reason", "")
    }


def classify_qa_batch(items, backend: str = "auto", model: str = DEFAULT_MODEL):
    if not items:
        return []

    compact_items = []
    for item in items:
        compact_items.append({
            "id": item["id"],
            "question": item["question"][:1200],
            "answer": item["answer"][:2400],
            "hint_tags": item.get("hint_tags", []),
        })

    prompt = f"""
You are a strict knowledge-base curator.

Classify each Q/A pair for a technical RAG knowledge base.

Keep useful technical content:
- DevOps, Docker, Kubernetes, OpenShift, Ansible, Linux, AI infra, RAG, security
- troubleshooting steps
- commands, scripts, SQL, configuration examples
- architecture or operational explanations

Reject:
- greetings, casual chat, language learning, shopping, general life advice
- unclear or low-value content

Return ONLY valid JSON array. One object per input item:
[
  {{"id": 1, "keep": true, "tags": ["docker", "networking"], "reason": "short reason"}}
]

Input items:
{json.dumps(compact_items, ensure_ascii=False)}
"""

    raw = generate(prompt, backend=backend, model=model)
    parsed = extract_json_array(raw)

    if not isinstance(parsed, list):
        return [
            {
                "id": item["id"],
                "keep": False,
                "tags": [],
                "reason": "failed_to_parse_llm_batch_response",
            }
            for item in items
        ]

    by_id = {}
    for decision in parsed:
        if not isinstance(decision, dict):
            continue

        try:
            item_id = int(decision.get("id"))
        except Exception:
            continue

        tags = decision.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        tags = [
            str(t).lower().replace("#", "").strip()
            for t in tags
            if str(t).strip()
        ]

        by_id[item_id] = {
            "id": item_id,
            "keep": bool(decision.get("keep", False)),
            "tags": tags[:10],
            "reason": decision.get("reason", ""),
        }

    return [
        by_id.get(item["id"], {
            "id": item["id"],
            "keep": False,
            "tags": [],
            "reason": "missing_llm_batch_decision",
        })
        for item in items
    ]


def classify_document(text: str, filename: str = "", backend: str = "auto", model: str = DEFAULT_MODEL):
    sample = text[:6000]

    prompt = f"""
You are a strict technical knowledge-base curator.

Analyze this document and assign useful tags for a RAG knowledge base.

Prefer short, lowercase tags. Examples:
docker, kubernetes, openshift, ansible, rag, security, linux, python,
networking, storage, monitoring, postgres, fastapi, ollama, openwebui.

Return ONLY valid JSON:
{{
  "tags": ["openshift", "kubernetes"],
  "reason": "short reason"
}}

Filename:
{filename}

Document sample:
{sample}
"""

    raw = generate(prompt, backend=backend, model=model)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "tags": [],
            "reason": "failed_to_parse_llm_response"
        }

    tags = parsed.get("tags", [])

    if not isinstance(tags, list):
        tags = []

    tags = [
        str(t).lower().replace("#", "").strip()
        for t in tags
        if str(t).strip()
    ]

    return {
        "tags": tags[:10],
        "reason": parsed.get("reason", "")
    }


def classify_query(query: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    prompt = f"""
You are a technical RAG query tagger.

Assign useful search tags for this question.

Prefer short, lowercase tags. Examples:
docker, kubernetes, openshift, ansible, rag, security, linux, python,
networking, storage, monitoring, postgres, fastapi, ollama, openwebui.

Return ONLY valid JSON:
{{
  "tags": ["openshift", "kubernetes"],
  "reason": "short reason"
}}

Question:
{query}
"""

    raw = generate(prompt, backend=backend, model=model)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "tags": [],
            "reason": "failed_to_parse_llm_response"
        }

    tags = parsed.get("tags", [])

    if not isinstance(tags, list):
        tags = []

    tags = [
        str(t).lower().replace("#", "").strip()
        for t in tags
        if str(t).strip()
    ]

    return {
        "tags": tags[:8],
        "reason": parsed.get("reason", "")
    }


def rewrite_query_for_search(query: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    prompt = f"""
You rewrite user questions into English search queries for a technical RAG system.

The knowledge base may be in English even if the user asks in Polish.
Keep product names and commands. Add likely CLI terms when helpful.

Return ONLY valid JSON:
{{
  "query": "how to list pods in OpenShift using oc get pods",
  "reason": "short reason"
}}

User question:
{query}
"""

    raw = generate(prompt, backend=backend, model=model)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "query": query,
            "reason": "failed_to_parse_llm_response"
        }

    rewritten = str(parsed.get("query", "")).strip()

    return {
        "query": rewritten or query,
        "reason": parsed.get("reason", "")
    }
