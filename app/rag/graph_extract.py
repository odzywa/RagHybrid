import json
from pathlib import Path
import re

from app.rag.generate import generate
from app.rag.graph_schema import ALLOWED_RELATIONS
from app.rag.graph_store import upsert_relation


RELATION_ALIASES = {
    "built_on": "builds_on",
    "create": "creates",
    "created_by": "creates",
    "depends": "depends_on",
    "depends upon": "depends_on",
    "is": "is_a",
    "is_an": "is_a",
    "is_type_of": "is_a",
    "managed_by": "is_managed_by",
    "provide": "provides",
    "require": "requires",
    "run_on": "runs_on",
    "used_by": "uses",
    "utilizes": "uses",
}

ENTITY_ALIASES = {
    "ceph": "Ceph",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "mysql": "MySQL",
    "mikrotik": "MikroTik",
    "neo4j": "Neo4j",
    "nginx": "NGINX",
    "ocp": "OpenShift",
    "openshift": "OpenShift",
    "openshift container platform": "OpenShift",
    "operatorhub": "OperatorHub",
    "odf": "ODF",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "red hat": "Red Hat",
    "red_hat": "Red Hat",
    "rhel": "RHEL",
    "routeros": "RouterOS",
    "scc": "SCC",
    "terraform": "Terraform",
    "wireguard": "WireGuard",
    "yaml": "YAML",
}

ENTITY_STOPWORDS = {
    "admin",
    "admin_user",
    "application",
    "applications",
    "container",
    "containers",
    "cpu",
    "deployment",
    "developer_user",
    "groups",
    "hello_pod",
    "memory",
    "network",
    "oc",
    "oc_adm",
    "oc_describe_node",
    "oc_login",
    "pod",
    "pods",
    "project",
    "route",
    "secret",
    "secrets",
    "service",
    "storage",
    "traffic",
    "user",
    "users",
}


CONSTITUTION_ANCHOR = "Konstytucja Rzeczypospolitej Polskiej"

CONSTITUTION_ENTITIES = [
    "Rzeczpospolita Polska",
    "Naród Polski",
    "Sejm",
    "Senat",
    "Prezydent Rzeczypospolitej",
    "Rada Ministrów",
    "Prezes Rady Ministrów",
    "Trybunał Konstytucyjny",
    "Trybunał Stanu",
    "Sąd Najwyższy",
    "Naczelny Sąd Administracyjny",
    "Krajowa Rada Sądownictwa",
    "Najwyższa Izba Kontroli",
    "Rzecznik Praw Obywatelskich",
    "Narodowy Bank Polski",
    "Krajowa Rada Radiofonii i Telewizji",
    "jednostki samorządu terytorialnego",
    "stany nadzwyczajne",
    "prawa i wolności",
    "wolności i prawa człowieka i obywatela",
]


ANCHOR_ENTITY_STOPWORDS = {
    "imported document",
    "page",
    "copyright",
    "kancelaria sejmu",
    "dziennik ustaw",
    "unknown",
}


def extract_json_array(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        return json.loads(match.group(0))
    except Exception:
        return []


def normalize_relation(relation: str):
    relation = str(relation or "").strip().lower()
    relation = re.sub(r"[^a-z0-9]+", "_", relation)
    relation = re.sub(r"_+", "_", relation).strip("_")
    relation = RELATION_ALIASES.get(relation, relation)

    if relation not in ALLOWED_RELATIONS:
        return None

    return relation


def normalize_entity(entity: str):
    entity = str(entity or "").strip()
    entity = re.sub(r"\s+", " ", entity)
    entity = entity.strip("`\"'.,:;()[]{}")

    if not entity:
        return None

    lookup = entity.lower().replace("-", " ").replace("_", " ")
    lookup = re.sub(r"\s+", " ", lookup).strip()
    alias_key = lookup.replace(" ", "_")

    if lookup in ENTITY_ALIASES:
        return ENTITY_ALIASES[lookup]

    if alias_key in ENTITY_ALIASES:
        return ENTITY_ALIASES[alias_key]

    if is_noise_entity(entity, lookup):
        return None

    if entity.isupper() and len(entity) <= 8:
        return entity

    words = []
    for word in lookup.split():
        if word in {"of", "and", "for", "the", "to"}:
            words.append(word)
        elif word in {"api", "dns", "http", "https", "ip", "rbac", "scc", "tls"}:
            words.append(word.upper())
        else:
            words.append(word.capitalize())

    return " ".join(words)


def is_noise_entity(original: str, lookup: str):
    compact = lookup.replace(" ", "_")

    if lookup in ENTITY_STOPWORDS or compact in ENTITY_STOPWORDS:
        return True

    if re.fullmatch(r"[0-9.]+", lookup):
        return True

    if re.fullmatch(r"\d{2,5}", lookup):
        return True

    if len(lookup) < 2:
        return True

    if "/" in original and not re.search(r"\.(io|com|org|net|local|dom)\b", original):
        return True

    if lookup.startswith(("oc ", "kubectl ", "systemctl ", "docker ", "podman ")):
        return True

    if compact.endswith(("_user", "_pod", "_deployment", "_service", "_project")):
        return True

    return False


def clean_relation(raw_relation):
    source = normalize_entity(raw_relation.get("source", ""))
    relation = normalize_relation(raw_relation.get("relation", ""))
    target = normalize_entity(raw_relation.get("target", ""))

    if not source or not relation or not target:
        return None

    if source == target:
        return None

    return source, relation, target


def entity_supported_by_text(entity: str, text: str):
    if not entity:
        return False

    lowered = (text or "").lower()
    entity_lower = entity.lower()

    if entity_lower in lowered:
        return True

    aliases = [
        alias
        for alias, canonical in ENTITY_ALIASES.items()
        if canonical.lower() == entity_lower
    ]

    return any(alias.replace("_", " ") in lowered for alias in aliases)


def relation_supported_by_text(source: str, target: str, text: str):
    return (
        entity_supported_by_text(source, text)
        and entity_supported_by_text(target, text)
    )


def is_constitution_document(text: str, source: str = ""):
    lowered = (text or "").lower()
    source_lower = (source or "").lower()

    return (
        "konstytucja rzeczypospolitej polskiej" in lowered
        or ("konstytucja" in lowered and "rzeczpospolitej polskiej" in lowered)
        or "d19970483" in source_lower
    )


def clean_document_title(value: str):
    value = str(value or "").strip()
    value = re.sub(r"^#+\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" `\"'.,:;()[]{}")

    if not value:
        return None

    lowered = value.lower()

    if "#" in value or "http://" in lowered or "https://" in lowered:
        return None

    if lowered in ANCHOR_ENTITY_STOPWORDS:
        return None

    if lowered.startswith(("page ", "strona ", "©", "#")):
        return None

    if re.fullmatch(r"\d{4}[-./]\d{2}[-./]\d{2}", lowered):
        return None

    if len(value) < 4 or len(value) > 140:
        return None

    if not re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]", value):
        return None

    return value


def title_from_source(source: str):
    stem = Path(str(source or "document")).stem
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()

    if re.fullmatch(r"[A-Z]?\d+[A-Za-z]{0,4}", stem):
        return None

    return clean_document_title(stem)


def infer_document_anchor(text: str, source: str = "unknown"):
    if is_constitution_document(text, source):
        return CONSTITUTION_ANCHOR

    lines = (text or "").splitlines()

    for line in lines[:160]:
        candidate = clean_document_title(line)

        if not candidate:
            continue

        lowered = candidate.lower()

        if lowered.startswith("art."):
            continue

        if candidate.startswith("#"):
            continue

        if line.lstrip().startswith("#") or candidate.isupper() or len(candidate.split()) >= 2:
            return normalize_entity(candidate) or candidate

    return title_from_source(source) or str(source or "document")


def candidate_supported_count(candidate: str, text: str):
    if not candidate:
        return 0

    pattern = re.escape(candidate)
    return len(re.findall(pattern, text or "", flags=re.IGNORECASE))


def clean_anchor_candidate(candidate: str, document_anchor: str):
    candidate = clean_document_title(candidate)

    if not candidate:
        return None

    normalized = normalize_entity(candidate)

    if not normalized or normalized == document_anchor:
        return None

    lowered = normalized.lower()

    if lowered in ANCHOR_ENTITY_STOPWORDS:
        return None

    if lowered.startswith(("art ", "art.", "rozdział", "chapter", "section")):
        return None

    if len(normalized.split()) > 8:
        return None

    return normalized


def extract_document_entities(text: str, document_anchor: str, limit: int = 20):
    candidates = []

    def add(candidate):
        cleaned = clean_anchor_candidate(candidate, document_anchor)

        if cleaned:
            candidates.append(cleaned)

    for alias, canonical in ENTITY_ALIASES.items():
        if alias.replace("_", " ") in (text or "").lower():
            add(canonical)

    for line in (text or "").splitlines()[:220]:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            add(stripped.lstrip("#").strip())
            continue

        if stripped.isupper() and 4 <= len(stripped) <= 100:
            add(stripped)

    capitalized = (
        r"\b[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9-]+"
        r"(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9-]+){1,6}\b"
    )

    for match in re.finditer(capitalized, text or ""):
        add(match.group(0))

    scored = []
    seen = set()

    for candidate in candidates:
        key = candidate.lower()

        if key in seen:
            continue

        seen.add(key)
        count = candidate_supported_count(candidate, text)

        if count <= 0:
            continue

        scored.append((count, len(candidate), candidate))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [candidate for _, _, candidate in scored[:limit]]


def extract_document_anchor_relations(text: str, source: str = "unknown"):
    relations = []
    anchor = infer_document_anchor(text, source)

    if anchor:
        relations.append((anchor, "is_a", "dokument"))

        for entity in extract_document_entities(text, anchor):
            relations.append((anchor, "contains", entity))

    if is_constitution_document(text, source):
        relations.append((CONSTITUTION_ANCHOR, "is_a", "akt normatywny"))

        for entity in CONSTITUTION_ENTITIES:
            if entity_supported_by_text(entity, text):
                relations.append((CONSTITUTION_ANCHOR, "contains", entity))

    return relations


def save_graph_relation(s, r, t, source, unique_relations):
    relation_key = (s, r, t)

    if relation_key in unique_relations:
        return "duplicate"

    unique_relations.add(relation_key)

    result = upsert_relation(
        s,
        r,
        t,
        metadata={
            "source": source
        }
    )

    return "created" if result.get("created") else "existing"


def extract_relations_from_text(
    text: str,
    source: str = "unknown",
    backend: str = "auto",
    model: str = "qwen2.5-coder:7b"
):
    sample = text[:8000]

    prompt = f"""
You extract knowledge graph relations from source text.

Return ONLY valid JSON array.

Schema:
[
  {{
    "source": "OpenShift",
    "relation": "extends",
    "target": "Kubernetes"
  }}
]

Rules:
- Extract only clear relations that are explicitly supported by the text.
- Do not invent facts.
- Prefer DevOps, Kubernetes, OpenShift, storage, networking and security relations when the text is technical.
- For non-technical documents, extract only strong document/domain relations that fit the allowed relation names.
- Relation must be one of:
  accesses, allows, builds_on, contains, creates, depends_on, exposes,
  extends, has, hosts, includes, is_a, is_managed_by, manages, provides,
  requires, runs_on, stores, supports, uses.
- Use named products, platforms, components, organizations, documents, legal bodies,
  systems, standards or important domain concepts as entities.
- Do not use ports, versions, usernames, commands, generic nouns, metrics,
  example object names, page numbers, dates or copyright boilerplate as entities.
- If the text does not contain clear relations, return [].

Text:
{sample}
"""

    raw = generate(prompt, backend=backend, model=model)
    relations = extract_json_array(raw)

    unique_relations = set()
    created_count = 0
    existing_count = 0
    valid_count = 0
    skipped_count = 0

    for s, r, t in extract_document_anchor_relations(text, source):
        if r == "is_a" and t == "dokument":
            supported = True
        elif s == CONSTITUTION_ANCHOR and t == "akt normatywny":
            supported = True
        else:
            supported = entity_supported_by_text(t, text)

        if not supported:
            skipped_count += 1
            continue

        valid_count += 1
        result = save_graph_relation(s, r, t, source, unique_relations)

        if result == "created":
            created_count += 1
        elif result == "existing":
            existing_count += 1

    for rel in relations:
        if not isinstance(rel, dict):
            skipped_count += 1
            continue

        cleaned = clean_relation(rel)
        if not cleaned:
            skipped_count += 1
            continue

        s, r, t = cleaned

        if not relation_supported_by_text(s, t, sample):
            skipped_count += 1
            continue

        valid_count += 1
        relation_key = (s, r, t)

        if relation_key in unique_relations:
            continue

        result = save_graph_relation(s, r, t, source, unique_relations)

        if result == "created":
            created_count += 1
        elif result == "existing":
            existing_count += 1

    stats = {
        "valid": valid_count,
        "unique": len(unique_relations),
        "created": created_count,
        "existing": existing_count,
        "skipped": skipped_count
    }
    print(f"GRAPH RELATIONS SAVED: {stats}")
    return stats
