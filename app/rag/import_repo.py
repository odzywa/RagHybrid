from pathlib import Path

from app.rag.chunk import code_language_from_filename
from app.rag.ingest import delete_source, ingest, source_count


DEFAULT_GRAPH_MODEL = "qwen2.5-coder:1.5b"

DEFAULT_EXTENSIONS = {
    ".adoc",
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".sql",
    ".sh",
    ".bash",
    ".ps1",
    ".bat",
    ".cmd",
    ".tf",
    ".md",
    ".toml",
    ".ini",
    ".conf",
    ".env",
    ".txt",
    ".j2",
    ".rsc",
}

DEFAULT_FILENAMES = {
    "Containerfile",
    "Dockerfile",
    "Makefile",
}

GRAPH_EXTENSIONS = {
    ".adoc",
    ".conf",
    ".ini",
    ".json",
    ".md",
    ".rsc",
    ".rst",
    ".tf",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

GRAPH_FILENAMES = {
    "Containerfile",
    "Dockerfile",
    "Makefile",
    "README",
}

GRAPH_PATH_HINTS = {
    "charts",
    "config",
    "deploy",
    "deployments",
    "doc",
    "docs",
    "examples",
    "install",
    "manifests",
    "operator",
    "runbooks",
    "samples",
    "troubleshooting",
}

IGNORE_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".terraform",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}


def import_code_repository(
    root_path: str,
    collection: str = "code",
    extensions=None,
    max_files: int = 500,
    max_file_bytes: int = 250_000,
    include_paths=None,
    reindex: bool = False,
    graph: bool = False,
    backend: str = "auto",
    model: str = DEFAULT_GRAPH_MODEL,
):
    root = Path(root_path).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError("Podaj istniejący katalog repozytorium.")

    extensions = normalize_extensions(extensions)
    max_files = max(1, min(int(max_files or 500), 5000))
    stats = {
        "root": str(root),
        "collection": collection,
        "files_seen": 0,
        "files_imported": 0,
        "files_skipped": 0,
        "files_too_large": 0,
        "files_outside_paths": 0,
        "deleted_chunks": 0,
        "chunks": 0,
        "inserted": 0,
        "duplicates": 0,
        "invalid": 0,
        "errors": 0,
        "graph": bool(graph),
        "graph_files": 0,
        "graph_skipped_files": 0,
        "graph_valid": 0,
        "graph_created": 0,
        "graph_existing": 0,
        "graph_skipped": 0,
        "examples": [],
    }

    include_paths = normalize_include_paths(include_paths)

    for path in iter_repo_files(root, extensions, include_paths):
        if stats["files_seen"] >= max_files:
            break

        stats["files_seen"] += 1
        relative = path.relative_to(root).as_posix()
        source = f"repo:{collection}:{relative}"

        try:
            file_size = path.stat().st_size
        except OSError:
            stats["files_skipped"] += 1
            continue

        if max_file_bytes and file_size > int(max_file_bytes):
            stats["files_too_large"] += 1
            stats["files_skipped"] += 1
            continue

        if source_count(source) and not reindex:
            stats["files_skipped"] += 1
            continue

        if reindex:
            stats["deleted_chunks"] += delete_source(source)

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            stats["files_skipped"] += 1
            continue

        if not text.strip():
            stats["files_skipped"] += 1
            continue

        language = code_language_from_filename(relative)
        metadata = {
            "collection": collection,
            "path": relative,
            "language": language,
            "source_type": "repo",
            "repo_root": str(root),
        }
        ingest_stats = ingest(
            text,
            source=source,
            metadata=metadata,
            chunk_mode="code"
        )

        stats["files_imported"] += 1
        stats["chunks"] += ingest_stats.get("chunks", 0)
        stats["inserted"] += ingest_stats.get("inserted", 0)
        stats["duplicates"] += ingest_stats.get("duplicates", 0)
        stats["invalid"] += ingest_stats.get("invalid", 0)
        stats["errors"] += ingest_stats.get("errors", 0)

        graph_stats = None

        if graph:
            if should_graph_index_repo_file(relative, path):
                from app.rag.graph_extract import extract_relations_from_text

                graph_text = repo_graph_text(collection, relative, text)
                graph_stats = extract_relations_from_text(
                    graph_text,
                    source=source,
                    backend=backend,
                    model=model or DEFAULT_GRAPH_MODEL,
                )
                stats["graph_files"] += 1
                stats["graph_valid"] += graph_stats.get("valid", 0)
                stats["graph_created"] += graph_stats.get("created", 0)
                stats["graph_existing"] += graph_stats.get("existing", 0)
                stats["graph_skipped"] += graph_stats.get("skipped", 0)
            else:
                stats["graph_skipped_files"] += 1

        if len(stats["examples"]) < 20:
            stats["examples"].append({
                "path": relative,
                "language": language,
                "inserted": ingest_stats.get("inserted", 0),
                "duplicates": ingest_stats.get("duplicates", 0),
                "graph": graph_stats,
            })

    return stats


def should_graph_index_repo_file(relative: str, path: Path):
    suffix = path.suffix.lower()
    name = path.name
    parts = set(Path(relative).parts[:-1])

    if name in GRAPH_FILENAMES:
        return True

    if suffix not in GRAPH_EXTENSIONS:
        return False

    if suffix in {".env"}:
        return False

    if parts & GRAPH_PATH_HINTS:
        return True

    if name.lower().startswith(("readme", "architecture", "design", "runbook", "troubleshooting")):
        return True

    return suffix in {".md", ".adoc", ".rst", ".tf", ".yaml", ".yml", ".rsc"}


def repo_graph_text(collection: str, relative: str, text: str):
    title = Path(relative).stem.replace("_", " ").replace("-", " ").strip()

    if Path(relative).name.lower().startswith("readme"):
        title = f"{collection} README"
    elif not title:
        title = relative

    return f"# {title}\n\nRepository: {collection}\nPath: {relative}\n\n{text}"


def normalize_extensions(extensions):
    if not extensions:
        return DEFAULT_EXTENSIONS

    if isinstance(extensions, str):
        extensions = [item.strip() for item in extensions.split(",")]

    normalized = set()

    for extension in extensions:
        extension = str(extension or "").strip().lower()

        if not extension:
            continue

        if not extension.startswith("."):
            extension = f".{extension}"

        normalized.add(extension)

    return normalized or DEFAULT_EXTENSIONS


def normalize_include_paths(include_paths):
    if not include_paths:
        return []

    normalized = []

    for item in include_paths:
        item = str(item or "").strip()
        if item == ".":
            normalized.append(item)
            continue

        item = item.strip("/")

        if item:
            normalized.append(item.lower())

    return normalized


def matches_include_paths(relative: str, include_paths):
    if not include_paths:
        return True

    lowered = relative.lower()
    parts = lowered.split("/")

    for include_path in include_paths:
        if include_path == "." and "/" not in lowered:
            return True

        if lowered == include_path or lowered.startswith(f"{include_path}/"):
            return True

        if include_path in parts:
            return True

    return False


def is_allowed_repo_file(path: Path, extensions):
    if path.name in DEFAULT_FILENAMES:
        return True

    return path.suffix.lower() in extensions


def iter_repo_files(root: Path, extensions, include_paths=None):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts[:-1])
        if parts & IGNORE_DIRS:
            continue

        if not matches_include_paths(relative, include_paths):
            continue

        if not is_allowed_repo_file(path, extensions):
            continue

        yield path
