import re


def extract_tags(text):
    tags = re.findall(r"(?<!\w)#([A-Za-z][A-Za-z0-9_-]*)", text)
    return unique_tags(tags)


def unique_tags(tags):
    unique = []

    for tag in tags:
        tag = str(tag).lower().replace("#", "").strip()
        if tag and tag not in unique:
            unique.append(tag)

    return unique


def merge_tags(*tag_groups):
    merged = []

    for tags in tag_groups:
        for tag in tags or []:
            tag = str(tag).lower().replace("#", "").strip()
            if tag and tag not in merged:
                merged.append(tag)

    return merged


def infer_command_tags(text):
    lower = text.lower()
    tags = []

    rules = [
        (r"\boc\s+", ["openshift", "kubernetes", "cli"]),
        (r"\bkubectl\s+", ["kubernetes", "cli"]),
        (r"\bdocker\s+", ["docker", "containers", "cli"]),
        (r"\bpodman\s+", ["podman", "containers", "cli"]),
        (r"\bansible-playbook\b|\bansible\s+", ["ansible", "automation", "cli"]),
        (r"\bsystemctl\s+", ["linux", "systemd", "cli"]),
        (r"\bhelm\s+", ["helm", "kubernetes", "cli"]),
        (r"\bterraform\s+", ["terraform", "iac", "cli"]),
    ]

    for pattern, pattern_tags in rules:
        if re.search(pattern, lower):
            tags.extend(pattern_tags)

    if "openshift" in lower or "rhocp" in lower or "ocp" in lower:
        tags.extend(["openshift", "kubernetes"])

    if "kubernetes" in lower:
        tags.append("kubernetes")

    return unique_tags(tags)


def chunk_text(text, max_length=1200, overlap=200):
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_length, len(text))

        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )

            if boundary > start + int(max_length * 0.55):
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def split_markdown(text, max_length=1200, overlap=200):
    # Zachowaj nagłówki sekcji jako część chunka.
    sections = re.split(r"(?m)(?=^##\s+)", text)
    chunks = []

    for section in sections:
        section_chunks = chunk_text(section, max_length=max_length, overlap=overlap)
        heading_match = re.match(r"(?m)^(##\s+[^\n]+)", section.strip())
        heading = heading_match.group(1) if heading_match else None

        for index, chunk in enumerate(section_chunks):
            if heading and index > 0 and not chunk.startswith(heading):
                chunk = f"{heading}\n\n{chunk}"

            chunks.append(chunk)

    return chunks


def code_language_from_filename(filename: str):
    lower = (filename or "").lower()
    mapping = {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".ps1": "powershell",
        ".bat": "batch",
        ".cmd": "batch",
        ".tf": "terraform",
        ".md": "markdown",
    }

    for suffix, language in mapping.items():
        if lower.endswith(suffix):
            return language

    return "text"


def split_code_file(text, filename="", max_length=1800, overlap=120):
    language = code_language_from_filename(filename)
    path_header = f"## File: {filename}\n\n" if filename else ""

    if language == "python":
        return split_python_code(text, path_header, max_length=max_length)

    if language == "yaml":
        return split_yaml_code(text, path_header, max_length=max_length)

    if language == "sql":
        return split_sql_code(text, path_header, max_length=max_length)

    fenced = f"{path_header}```{language}\n{text.strip()}\n```"
    return chunk_text(fenced, max_length=max_length, overlap=overlap)


def split_python_code(text, path_header="", max_length=1800):
    lines = text.splitlines()
    starts = []

    for idx, line in enumerate(lines):
        if re.match(r"^(class|def|async def)\s+\w+", line):
            starts.append(idx)

    if not starts:
        fenced = f"{path_header}```python\n{text.strip()}\n```"
        return chunk_text(fenced, max_length=max_length, overlap=120)

    starts.append(len(lines))
    chunks = []
    prelude = "\n".join(lines[:starts[0]]).strip()

    for pos in range(len(starts) - 1):
        start = starts[pos]
        end = starts[pos + 1]
        block = "\n".join(lines[start:end]).strip()

        if prelude and pos == 0:
            block = f"{prelude}\n\n{block}"

        title = lines[start].strip()
        fenced = f"{path_header}### {title}\n\n```python\n{block}\n```"
        chunks.extend(chunk_text(fenced, max_length=max_length, overlap=120))

    return chunks


def split_yaml_code(text, path_header="", max_length=1800):
    docs = re.split(r"(?m)^---\s*$", text)
    chunks = []

    for index, doc in enumerate(docs, start=1):
        doc = doc.strip()

        if not doc:
            continue

        kind_match = re.search(r"(?m)^kind:\s*([A-Za-z0-9_-]+)", doc)
        name_match = re.search(r"(?m)^\s*name:\s*([A-Za-z0-9_.-]+)", doc)
        title_parts = []

        if kind_match:
            title_parts.append(kind_match.group(1))
        if name_match:
            title_parts.append(name_match.group(1))

        title = " ".join(title_parts) or f"YAML document {index}"
        fenced = f"{path_header}### {title}\n\n```yaml\n{doc}\n```"
        chunks.extend(chunk_text(fenced, max_length=max_length, overlap=120))

    return chunks


def split_sql_code(text, path_header="", max_length=1800):
    statements = []
    current = []

    for line in text.splitlines():
        current.append(line)

        if line.rstrip().endswith(";"):
            statements.append("\n".join(current).strip())
            current = []

    if current:
        statements.append("\n".join(current).strip())

    chunks = []

    for index, statement in enumerate(statements, start=1):
        if not statement:
            continue

        first_line = statement.splitlines()[0].strip()[:80]
        fenced = f"{path_header}### SQL statement {index}: {first_line}\n\n```sql\n{statement}\n```"
        chunks.extend(chunk_text(fenced, max_length=max_length, overlap=120))

    return chunks
