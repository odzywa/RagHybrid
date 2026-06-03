from collections import defaultdict
from io import BytesIO
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import os
from urllib.parse import urlparse
from typing import List, Optional
import zipfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import requests
from docx import Document
from pypdf import PdfReader

from app.config import settings
from app.rag.ingest import delete_source, ingest, source_count
from app.rag.ingest_folder import ingest_folder
from app.rag.import_chatgpt import import_chatgpt_json
from app.rag.import_repo import import_code_repository
from app.rag.web_import import crawl_documentation
from app.rag.search import hybrid_search, rerank as lexical_rerank, source_evidence_chunks
from app.rag.generate import generate, generate_stream
from app.rag.rerank import rerank as smart_rerank
from app.rag.relevance import DEFAULT_NO_CONTEXT_INSTRUCTION, calculate_relevance, env_bool
from app.rag.smart_filter import classify_document, classify_query, rewrite_query_for_search
from app.runtime_config import (
    embedding_backend_status, rerank_targets, set_embedding_backend,
    get_backend_config, set_backend_config, server_available,
)
from app.db import init_db
from app.db import get_db_connection
from app.openwebui_admin import router as openwebui_admin_router

app = FastAPI()
app.include_router(openwebui_admin_router)
DEFAULT_MODEL = "qwen2.5-coder:1.5b"
MAX_CONTEXT = 4000
SCHEMA_DOC_PATH = Path(__file__).resolve().parent.parent / "ARCHITEKTURA_RAG.md"
RETRIEVAL_TELEMETRY_PATH = Path(os.getenv("RAG_RETRIEVAL_TELEMETRY_PATH", "/runtime/raghybrid_retrieval_telemetry.jsonl"))
MAX_TELEMETRY_LINES = 1000

chat_memory = defaultdict(list)
MAX_HISTORY = 6
chatgpt_import_jobs = {}
upload_file_jobs = {}


class RetrieveRequest(BaseModel):
    query: str
    tags: Optional[List[str]] = None
    backend: str = "auto"
    model: str = DEFAULT_MODEL
    top_k: Optional[int] = None
    max_vector: int = 5
    max_graph: int = 10
    max_evidence: int = 5
    max_context_chars: int = 12000
    telemetry: bool = True


class TagQueryRequest(BaseModel):
    query: str
    backend: str = "auto"
    model: str = DEFAULT_MODEL


class EmbeddingBackendRequest(BaseModel):
    backend: str


class RepoImportRequest(BaseModel):
    path: str
    collection: str = "code"
    extensions: Optional[List[str]] = None
    max_files: int = 500
    max_file_bytes: int = 250_000
    include_paths: Optional[List[str]] = None
    reindex: bool = False
    graph: bool = False
    backend: str = "auto"
    model: str = DEFAULT_MODEL


class GitImportRequest(BaseModel):
    url: str
    collection: str = "code"
    extensions: Optional[List[str]] = None
    max_files: int = 500
    max_file_bytes: int = 250_000
    include_paths: Optional[List[str]] = None
    reindex: bool = False
    ref: Optional[str] = None
    graph: bool = False
    backend: str = "auto"
    model: str = DEFAULT_MODEL


class RetrievalFeedbackRequest(BaseModel):
    query: str
    rating: str = "good"
    missing_source: Optional[str] = None
    comment: Optional[str] = None
    metadata: dict = {}


class GoldenTestItem(BaseModel):
    query: str
    expected_sources: List[str] = []
    expected_types: List[str] = ["vector"]
    collection: Optional[str] = None


class GoldenTestRunRequest(BaseModel):
    tests: List[GoldenTestItem]
    backend: str = "cpu"
    max_vector: int = 4
    max_graph: int = 8
    max_evidence: int = 4
    max_context_chars: int = 8000


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(text: str = Form(...)):
    ingest(text)
    return {"status": "ok"}


@app.post("/upload_note")
def upload_note(
    title: str = Form("manual-note"),
    text: str = Form(...),
    tags: str = Form(None)
):
    note_tags = [tag.strip().lstrip("#") for tag in tags.split(",") if tag.strip()] if tags else []
    tag_line = " ".join([f"#{tag}" for tag in note_tags])
    content = f"# {title}\n\n{tag_line}\n\n{text}" if tag_line else f"# {title}\n\n{text}"

    ingest(content, source=title)

    return {
        "status": "note ingested",
        "source": title
    }


def extract_file_text(filename: str, data: bytes):
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = [
            f"## Page {index}\n\n{page.extract_text() or ''}"
            for index, page in enumerate(reader.pages, start=1)
        ]
        return "\n\n".join(pages).strip()

    if lower_name.endswith(".docx"):
        document = Document(BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(paragraphs).strip()

    if lower_name.endswith((".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".log")):
        return data.decode("utf-8", errors="ignore").strip()

    return None


def smart_enabled(value: str):
    return value.lower() in ["true", "1", "yes", "on"]


def add_document_tags(text: str, filename: str, tags):
    clean_tags = []
    for tag in tags:
        tag = str(tag).lower().replace("#", "").strip()
        if tag and tag not in clean_tags:
            clean_tags.append(tag)

    if not clean_tags:
        return text

    tag_line = " ".join([f"#{tag}" for tag in clean_tags])
    tagged_text = f"# Imported Document\n\n{tag_line}\n\n{text}"
    tagged_text = re.sub(
        r"(^|\n)(##\s+[^\n]+)",
        rf"\1\2\n\n{tag_line}",
        tagged_text
    )
    print(f"SMART DOCUMENT TAGS for {filename}:", clean_tags)
    return tagged_text


@app.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    smart: str = Form("false"),
    reindex: str = Form("false"),
    graph: str = Form("false"),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL)
):
    source = file.filename
    data = await file.read()

    return process_uploaded_file(
        source=source,
        data=data,
        smart=smart,
        reindex=reindex,
        graph=graph,
        backend=backend,
        model=model,
    )


def process_uploaded_file(
    source: str,
    data: bytes,
    smart: str = "false",
    reindex: str = "false",
    graph: str = "false",
    backend: str = "auto",
    model: str = DEFAULT_MODEL,
    progress_callback=None
):
    source = source or "upload"

    def progress(percent: int, message: str, **fields):
        if progress_callback:
            progress_callback(percent, message, **fields)

    progress(8, "Sprawdzam, czy dokument już istnieje", source=source)
    existing_chunks = source_count(source)
    reindex_enabled = smart_enabled(reindex)

    if existing_chunks and not reindex_enabled:
        progress(100, "Dokument już jest w bazie", existing_chunks=existing_chunks)
        return {
            "status": "duplicate",
            "message": "Ten dokument już jest w bazie. Zaznacz Reindex, żeby zaimportować go od nowa.",
            "source": source,
            "existing_chunks": existing_chunks,
            "smart": smart_enabled(smart),
            "tags": []
        }

    progress(15, "Odczytuję tekst z pliku")
    text = extract_file_text(source, data)

    if not text:
        progress(0, "Nie udało się odczytać tekstu z pliku")
        return {
            "status": "error",
            "message": "Nie udało się odczytać tekstu z pliku."
        }

    progress(25, "Sprawdzam tryb reindex")
    deleted_chunks = delete_source(source) if existing_chunks and reindex_enabled else 0

    smart_import = smart_enabled(smart)
    document_tags = []

    if smart_import:
        progress(38, "Smart import: klasyfikuję dokument i dobieram tagi")
        decision = classify_document(text, filename=source, backend=backend, model=model)
        print("SMART DOCUMENT DECISION:", decision)
        document_tags = decision.get("tags", [])
        text = add_document_tags(text, source, document_tags)

    progress(55, "Tworzę chunki, embeddingi i zapisuję do PostgreSQL")
    stats = ingest(text, source=source)
    graph_enabled = smart_enabled(graph)
    graph_stats = None

    if graph_enabled:
        from app.rag.graph_extract import extract_relations_from_text

        progress(78, "Wyciągam relacje i zapisuję graf w Neo4j")
        graph_stats = extract_relations_from_text(
            text,
            source=source,
            backend=backend,
            model=model
        )

    progress(100, "Import pliku zakończony", chunks=stats.get("chunks", 0), inserted=stats.get("inserted", 0), graph=graph_stats)
    return {
        "status": "file ingested",
        "source": source,
        "smart": smart_import,
        "graph": graph_enabled,
        "reindex": reindex_enabled,
        "existing_chunks": existing_chunks,
        "deleted_chunks": deleted_chunks,
        "tags": document_tags,
        "ingest": stats,
        "graph_stats": graph_stats,
    }


def update_upload_file_job(job_id: str, **fields):
    job = upload_file_jobs.get(job_id)

    if not job:
        return

    job.update(fields)


def run_upload_file_job(job_id: str, upload_path: str, source: str, smart: str, reindex: str, graph: str, backend: str, model: str):
    def set_progress(percent: int, message: str, **fields):
        update_upload_file_job(
            job_id,
            progress=max(0, min(100, int(percent))),
            message=message,
            **fields
        )

    try:
        set_progress(4, "Czytam zapisany plik tymczasowy", source=source)
        data = Path(upload_path).read_bytes()
        result = process_uploaded_file(
            source=source,
            data=data,
            smart=smart,
            reindex=reindex,
            graph=graph,
            backend=backend,
            model=model,
            progress_callback=set_progress,
        )

        status = "done" if result.get("status") != "error" else "error"
        set_progress(100 if status == "done" else 0, result.get("message") or "Import pliku zakończony", result=result)
        update_upload_file_job(job_id, status=status)
    except Exception as exc:
        update_upload_file_job(
            job_id,
            status="error",
            progress=0,
            message=str(exc),
        )
    finally:
        Path(upload_path).unlink(missing_ok=True)


@app.post("/upload_file_async")
async def upload_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    smart: str = Form("false"),
    reindex: str = Form("false"),
    graph: str = Form("false"),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL)
):
    source = file.filename or "upload"
    suffix = Path(source).suffix or ".upload"

    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            tmp.write(chunk)

        upload_path = tmp.name

    job_id = str(uuid.uuid4())
    upload_file_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 1,
        "message": "Import pliku dodany do kolejki",
        "source": source,
        "result": None,
    }
    background_tasks.add_task(
        run_upload_file_job,
        job_id,
        upload_path,
        source,
        smart,
        reindex,
        graph,
        backend,
        model,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "source": source,
        "message": "Import pliku dodany do kolejki",
    }


@app.get("/upload_file/status/{job_id}")
def upload_file_status(job_id: str):
    job = upload_file_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Nie znaleziono importu pliku.")

    return job


@app.post("/ingest_folder")
def ingest_all():
    ingest_folder()
    return {"status": "folder ingested"}


@app.post("/import_repo")
def import_repo(req: RepoImportRequest):
    try:
        result = import_code_repository(
            req.path,
            collection=req.collection,
            extensions=req.extensions,
            max_files=req.max_files,
            max_file_bytes=req.max_file_bytes,
            include_paths=req.include_paths,
            reindex=req.reindex,
            graph=req.graph,
            backend=req.backend,
            model=req.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "repo ingested",
        **result,
    }


def safe_git_url(url: str):
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Podaj URL git zaczynający się od http:// albo https://.")

    if parsed.username or parsed.password:
        raise ValueError("Nie podawaj tokenów ani haseł w Git URL.")

    return parsed.geturl()


def collection_from_git_url(url: str):
    name = Path(urlparse(url).path).name or "git-repo"

    if name.endswith(".git"):
        name = name[:-4]

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-").lower()
    return cleaned or "git-repo"


@app.post("/import_git")
def import_git(req: GitImportRequest):
    try:
        git_url = safe_git_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    collection = (req.collection or "").strip() or collection_from_git_url(git_url)
    max_files = max(1, min(int(req.max_files or 500), 5000))

    with tempfile.TemporaryDirectory(prefix="raghybrid-git-") as tmpdir:
        repo_path = Path(tmpdir) / "repo"
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            git_url,
            str(repo_path),
        ]

        if req.ref:
            clone_cmd = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                req.ref.strip(),
                git_url,
                str(repo_path),
            ]

        try:
            clone = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Git clone timeout.")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Brak komendy git w kontenerze.")

        if clone.returncode != 0:
            message = (clone.stderr or clone.stdout or "git clone failed").strip()
            raise HTTPException(status_code=400, detail=message[-1000:])

        try:
            result = import_code_repository(
                str(repo_path),
                collection=collection,
                extensions=req.extensions,
                max_files=max_files,
                max_file_bytes=req.max_file_bytes,
                include_paths=req.include_paths,
                reindex=req.reindex,
                graph=req.graph,
                backend=req.backend,
                model=req.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "git repo ingested",
        "url": git_url,
        "ref": req.ref,
        "collection": collection,
        **result,
        "root": git_url,
        "cloned_to": "temporary directory",
    }


@app.post("/import_website")
def import_website(
    url: str = Form(...),
    max_pages: int = Form(50),
    smart: str = Form("false"),
    reindex: str = Form("false"),
    graph: str = Form("false"),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL)
):
    smart_import = smart_enabled(smart)
    reindex_enabled = smart_enabled(reindex)
    graph_enabled = smart_enabled(graph)

    try:
        crawl = crawl_documentation(url, max_pages=max_pages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    totals = {
        "chunks": 0,
        "valid": 0,
        "inserted": 0,
        "duplicates": 0,
        "invalid": 0,
        "errors": 0,
        "deleted_chunks": 0,
        "graph_valid": 0,
        "graph_created": 0,
        "graph_existing": 0,
    }
    imported_pages = []
    duplicate_pages = 0

    for page in crawl["pages"]:
        source = page.url
        existing_chunks = source_count(source)

        if existing_chunks and not reindex_enabled:
            duplicate_pages += 1
            continue

        if existing_chunks and reindex_enabled:
            totals["deleted_chunks"] += delete_source(source)

        text = f"# {page.title}\n\nSource: {page.url}\n\n{page.text}"
        tags = []

        if smart_import:
            decision = classify_document(text, filename=source, backend=backend, model=model)
            print("SMART WEBSITE DOCUMENT DECISION:", decision)
            tags = decision.get("tags", [])
            text = add_document_tags(text, source, tags)

        ingest_stats = ingest(text, source=source)

        for key in ["chunks", "valid", "inserted", "duplicates", "invalid", "errors"]:
            totals[key] += ingest_stats.get(key, 0)

        graph_stats = None

        if graph_enabled:
            from app.rag.graph_extract import extract_relations_from_text

            graph_stats = extract_relations_from_text(
                text,
                source=source,
                backend=backend,
                model=model
            )
            totals["graph_valid"] += graph_stats.get("valid", 0)
            totals["graph_created"] += graph_stats.get("created", 0)
            totals["graph_existing"] += graph_stats.get("existing", 0)

        imported_pages.append({
            "url": source,
            "title": page.title,
            "chars": len(page.text),
            "tags": tags,
            "ingest": ingest_stats,
            "graph": graph_stats,
        })

    return {
        "status": "website ingested",
        "url": crawl["start_url"],
        "scope_prefix": crawl["scope_prefix"],
        "visited": crawl["visited"],
        "pages_found": len(crawl["pages"]),
        "pages_imported": len(imported_pages),
        "duplicate_pages": duplicate_pages,
        "skipped_pages": crawl["skipped"],
        "errors": crawl["errors"],
        "smart": smart_import,
        "graph": graph_enabled,
        "reindex": reindex_enabled,
        "totals": totals,
        "pages": imported_pages[:20],
    }


@app.post("/graph_cleanup")
def graph_cleanup(
    apply: str = Form("false"),
    strict_relations: str = Form("true"),
    sample_limit: int = Form(20)
):
    apply_enabled = apply.lower() in ["true", "1", "yes", "on"]
    strict = strict_relations.lower() in ["true", "1", "yes", "on"]

    if sample_limit < 1:
        sample_limit = 1

    if sample_limit > 50:
        sample_limit = 50

    from app.rag.graph_cleanup import cleanup_graph

    result = cleanup_graph(
        dry_run=not apply_enabled,
        sample_limit=sample_limit,
        strict_relations=strict
    )

    return {
        "status": "ok",
        "applied": apply_enabled,
        "strict_relations": strict,
        **result
    }


@app.get("/graph_data")
def graph_data(entity: str = "", limit: int = 100):
    if limit > 500:
        limit = 500

    from app.rag.graph_store import get_graph_data

    return get_graph_data(entity=entity, limit=limit)


def update_chatgpt_import_job(job_id: str, **fields):
    job = chatgpt_import_jobs.get(job_id)

    if not job:
        return

    job.update(fields)

    total = max(int(job.get("total_files") or 0), 1)
    processed = int(job.get("files") or 0) + int(job.get("skipped_files") or 0)

    if job.get("status") == "done":
        job["progress"] = 100
    elif job.get("status") == "error":
        job["progress"] = 0
    else:
        job["progress"] = max(1, min(99, int((processed / total) * 100)))


def run_chatgpt_import_job(job_id: str, uploads, smart_import: bool, backend: str, model: str):
    update_chatgpt_import_job(
        job_id,
        status="running",
        message="Import historii i tagowanie",
        files=0,
        skipped_files=0,
        pairs=0,
        errors=[],
    )

    def add_error(message: str):
        job = chatgpt_import_jobs[job_id]
        errors = job.get("errors") or []
        errors.append(message)
        job["errors"] = errors[-20:]

    def import_json_path(path: str, name: str):
        job = chatgpt_import_jobs[job_id]
        update_chatgpt_import_job(job_id, current_file=name)

        try:
            imported = import_chatgpt_json(
                path,
                smart=smart_import,
                backend=backend,
                model=model
            )
            update_chatgpt_import_job(
                job_id,
                files=job.get("files", 0) + 1,
                pairs=job.get("pairs", 0) + imported,
            )
        except json.JSONDecodeError:
            add_error(f"Niepoprawny JSON: {name}")
            update_chatgpt_import_job(job_id, skipped_files=job.get("skipped_files", 0) + 1)

    try:
        for upload in uploads:
            path = upload["path"]
            name = upload["name"]
            suffix = upload["suffix"]

            if suffix == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        items = archive.infolist()
                        job = chatgpt_import_jobs[job_id]
                        update_chatgpt_import_job(
                            job_id,
                            total_files=job.get("total_files", 0) + len(items) - 1,
                        )

                        for item in items:
                            item_path = Path(item.filename)

                            if item.is_dir() or item_path.suffix.lower() != ".json":
                                job = chatgpt_import_jobs[job_id]
                                update_chatgpt_import_job(job_id, skipped_files=job.get("skipped_files", 0) + 1)
                                continue

                            with archive.open(item) as source, tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as tmp:
                                shutil.copyfileobj(source, tmp)
                                json_path = tmp.name

                            try:
                                import_json_path(json_path, item.filename)
                            finally:
                                Path(json_path).unlink(missing_ok=True)
                except zipfile.BadZipFile:
                    add_error(f"Niepoprawny ZIP: {name}")
                    job = chatgpt_import_jobs[job_id]
                    update_chatgpt_import_job(job_id, skipped_files=job.get("skipped_files", 0) + 1)
                continue

            if suffix == ".json":
                import_json_path(path, name)
                continue

            job = chatgpt_import_jobs[job_id]
            update_chatgpt_import_job(job_id, skipped_files=job.get("skipped_files", 0) + 1)

        job = chatgpt_import_jobs[job_id]

        if job.get("files", 0) == 0:
            update_chatgpt_import_job(
                job_id,
                status="error",
                message="Nie zaimportowano żadnego JSON-a historii.",
            )
        else:
            update_chatgpt_import_job(
                job_id,
                status="done",
                message="Historia zaimportowana",
                current_file="",
            )
    except Exception as exc:
        update_chatgpt_import_job(
            job_id,
            status="error",
            message=str(exc),
        )
    finally:
        for upload in uploads:
            Path(upload["path"]).unlink(missing_ok=True)


@app.post("/import_chatgpt")
async def import_chat(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., alias="file"),
    smart: str = Form("false"),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL)
):
    smart_import = smart_enabled(smart)
    saved_uploads = []

    async def save_upload(upload: UploadFile, suffix: str):
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            while True:
                chunk = await upload.read(1024 * 1024)

                if not chunk:
                    break

                tmp.write(chunk)

            return tmp.name

    for upload in files:
        filename = upload.filename or "upload"
        suffix = Path(filename).suffix.lower()
        upload_path = await save_upload(upload, suffix or ".upload")
        saved_uploads.append({
            "path": upload_path,
            "name": filename,
            "suffix": suffix,
        })

    if not saved_uploads:
        raise HTTPException(status_code=400, detail="Dodaj plik JSON albo ZIP z eksportem historii.")

    job_id = str(uuid.uuid4())
    chatgpt_import_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Import dodany do kolejki",
        "smart": smart_import,
        "files": 0,
        "skipped_files": 0,
        "pairs": 0,
        "total_files": len(saved_uploads),
        "progress": 1,
        "current_file": "",
        "errors": [],
    }
    background_tasks.add_task(
        run_chatgpt_import_job,
        job_id,
        saved_uploads,
        smart_import,
        backend,
        model,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "smart": smart_import,
    }


@app.get("/import_chatgpt/status/{job_id}")
def import_chat_status(job_id: str):
    job = chatgpt_import_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Nie znaleziono importu.")

    return job


def backend_urls(backend: str):
    if backend == "gpu":
        return [settings.OLLAMA_GPU_URL]
    if backend == "cpu":
        return [settings.OLLAMA_CPU_URL]
    if backend == "laptop":
        return [settings.OLLAMA_LAPTOP_URL]
    return [settings.OLLAMA_GPU_URL, settings.OLLAMA_CPU_URL, settings.OLLAMA_LAPTOP_URL]


def ollama_available(url: str):
    try:
        response = requests.get(f"{url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception as e:
        print("OLLAMA STATUS FAILED:", url, e)
        return False


def build_history(session_id):
    history = chat_memory.get(session_id, [])

    if not history:
        return ""

    formatted = ""
    for msg in history[-MAX_HISTORY:]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            formatted += f"User: {content}\n"
        else:
            formatted += f"Assistant: {content}\n"

    return formatted


def parse_tags(tags):
    return [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None


def context_source(metadata):
    metadata = metadata or {}
    source = metadata.get("source", "unknown")
    page = metadata.get("page")

    if page:
        return f"{source}, page {page}"

    return source


def auto_tags_for_query(query: str, tags: str = None, backend: str = "auto", model: str = DEFAULT_MODEL):
    tag_list = parse_tags(tags)

    if tag_list:
        return tag_list

    decision = classify_query(query, backend=backend, model=model)
    print("AUTO QUERY TAGS:", decision)
    return decision.get("tags", []) or None


def search_query_for_rag(query: str, backend: str = "auto", model: str = DEFAULT_MODEL):
    decision = rewrite_query_for_search(query, backend=backend, model=model)
    print("SEARCH QUERY:", decision)
    return decision.get("query") or query


def dedupe_results(results):
    seen = set()
    unique = []

    for result in results:
        key = result[0].strip().lower()
        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

    return unique


def clean_retrieve_results(results, max_results=3):
    cleaned = []
    seen = set()

    for r in results:
        content = (r[0] or "").strip()

        if len(content) < 30:
            continue

        normalized = content.lower().strip()
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(r)

        if len(cleaned) >= max_results:
            break

    return cleaned


def build_hybrid_context(
    query: str,
    tags=None,
    backend: str = "auto",
    graph_entity: str = None,
    max_vector: int = 5,
    max_graph: int = 10,
    max_evidence: int = 5,
    max_context_chars: int = 12000
):
    max_vector = 5 if max_vector is None else max_vector
    max_graph = 10 if max_graph is None else max_graph
    max_evidence = 5 if max_evidence is None else max_evidence
    max_context_chars = 12000 if max_context_chars is None else max_context_chars

    max_vector = max(1, min(int(max_vector), 10))
    max_graph = max(0, min(int(max_graph), 30))
    max_evidence = max(0, min(int(max_evidence), 12))
    max_context_chars = max(2000, min(int(max_context_chars), 50000))

    results = hybrid_search(query, tags=tags)
    results = lexical_rerank(query, results)
    results = clean_retrieve_results(results, max_results=max_vector)

    entity = graph_entity or query
    graph_terms = []
    vector_sources = set()
    vector_contents = set()

    for result in results:
        metadata = result[1] or {}
        graph_terms.extend(metadata.get("tags") or [])
        graph_terms.extend(extract_context_terms(result[0]))
        vector_sources.add(metadata.get("source"))
        vector_contents.add(result[0])

    from app.rag.graph_store import search_graph

    graph_results = search_graph(entity, limit=max_graph, extra_terms=graph_terms)
    graph_sources = []
    relation_texts = []

    for relation in graph_results:
        relation_texts.append(
            f"{relation.get('source', '')} {relation.get('relation', '')} {relation.get('target', '')}"
        )

        for source in relation.get("sources") or []:
            if source and source not in vector_sources:
                graph_sources.append(source)

    evidence_results = source_evidence_chunks(
        query,
        filter_evidence_sources(graph_sources, vector_sources),
        relation_texts=relation_texts,
        exclude_contents=vector_contents,
        limit=max_evidence
    )
    combined = []
    used_chars = 0

    def append_item(item):
        nonlocal used_chars

        content = item.get("content") or ""
        item_chars = len(content)

        if item["type"] in {"vector", "graph_evidence"} and used_chars + item_chars > max_context_chars:
            remaining = max_context_chars - used_chars

            if remaining < 500:
                return False

            item = {
                **item,
                "content": content[:remaining].rstrip() + "\n[truncated]",
                "metadata": {
                    **(item.get("metadata") or {}),
                    "truncated": True,
                }
            }
            item_chars = len(item["content"])

        combined.append(item)
        used_chars += item_chars
        return True

    for idx, r in enumerate(results):
        metadata = dict(r[1] or {})
        if len(r) > 2 and r[2] is not None:
            metadata["retrieval_distance"] = float(r[2])
            metadata["relevance_score"] = 1.0 / (1.0 + max(float(r[2]), 0.0))

        append_item({
            "rank": idx + 1,
            "type": "vector",
            "content": r[0],
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", []),
            "metadata": metadata,
            "distance": metadata.get("retrieval_distance"),
            "score": metadata.get("relevance_score"),
        })

    for idx, g in enumerate(graph_results):
        append_item({
            "rank": idx + 1,
            "type": "graph",
            "content": f"{g['source']} --{g['relation']}--> {g['target']}",
            "source": "knowledge_graph",
            "tags": ["graph"],
            "metadata": g
        })

    for idx, r in enumerate(evidence_results):
        metadata = dict(r[1] or {})
        if len(r) > 2 and r[2] is not None:
            metadata["evidence_score"] = r[2]
        append_item({
            "rank": idx + 1,
            "type": "graph_evidence",
            "content": r[0],
            "source": metadata.get("source", "unknown"),
            "tags": metadata.get("tags", []),
            "metadata": metadata,
            "score": 0.55,
        })

    return combined, {
        "vector_results": len(results),
        "graph_results": len(graph_results),
        "graph_evidence_results": len(evidence_results),
        "context_chars": used_chars,
        "max_context_chars": max_context_chars,
    }


def extract_context_terms(text: str, limit: int = 10):
    terms = []

    for line in (text or "").splitlines()[:20]:
        stripped = line.strip()

        if stripped.startswith("#"):
            terms.append(stripped.lstrip("#").strip())

    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b", text or ""):
        terms.append(match.group(0))

    clean_terms = []
    seen = set()

    for term in terms:
        term = re.sub(r"\s+", " ", term).strip(" `\"'.,:;!?()[]{}")
        key = term.lower()

        if len(term) < 3 or key in seen:
            continue

        seen.add(key)
        clean_terms.append(term)

        if len(clean_terms) >= limit:
            break

    return clean_terms


def source_scope_key(source: str):
    source = str(source or "").strip()

    if source.startswith("http://") or source.startswith("https://"):
        match = re.match(r"https?://([^/]+)(/[^?#]*)?", source)

        if not match:
            return source

        host = match.group(1).lower()
        parts = [part for part in (match.group(2) or "").split("/") if part]
        prefix = "/".join(parts[:2])
        return f"{host}/{prefix}" if prefix else host

    return source.rsplit("/", 1)[0] if "/" in source else source


def filter_evidence_sources(graph_sources, vector_sources):
    vector_scopes = {
        source_scope_key(source)
        for source in vector_sources
        if source
    }

    if not vector_scopes:
        return graph_sources

    return [
        source
        for source in graph_sources
        if source_scope_key(source) in vector_scopes
    ]


HYBRID_CONTEXT_INSTRUCTION = (
    "Use vector context as the primary source of factual truth. "
    "Use graph context as a relationship map that can connect entities and dependencies. "
    "Use graph_evidence as textual support for graph relations when available. "
    "Do not treat graph-only relations as proven facts unless vector or graph_evidence context supports them. "
    "If vector and graph context conflict, mention the conflict. "
    "If the answer is not present in context, say you do not know."
)


def retrieval_summary(results):
    by_type = defaultdict(int)
    sources = []
    source_set = set()

    for item in results:
        by_type[item.get("type", "unknown")] += 1
        source = item.get("source") or "unknown"

        if source != "knowledge_graph" and source not in source_set:
            source_set.add(source)
            sources.append(source)

    return {
        "result_count": len(results),
        "types": dict(by_type),
        "sources": sources[:20],
    }


def log_retrieval_telemetry(endpoint: str, query: str, debug: dict, results, elapsed_ms: int):
    summary = retrieval_summary(results)
    record = {
        "ts": int(time.time()),
        "endpoint": endpoint,
        "query": query[:500],
        "elapsed_ms": elapsed_ms,
        "rag_used": debug.get("rag_used"),
        "relevance_score": debug.get("relevance_score"),
        "gate_reason": debug.get("gate_reason") or debug.get("relevance_reason"),
        "results_before_gate": debug.get("results_before_gate"),
        "results_after_gate": debug.get("results_after_gate", len(results)),
        "top_score": debug.get("top_score"),
        "lexical_overlap": debug.get("lexical_overlap"),
        "debug": debug,
        **summary,
    }

    try:
        RETRIEVAL_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)

        with RETRIEVAL_TELEMETRY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        rotate_retrieval_telemetry()
    except Exception as exc:
        print("RETRIEVAL TELEMETRY WRITE FAILED:", exc)

    print(
        "RETRIEVAL TELEMETRY:",
        json.dumps({
            "endpoint": endpoint,
            "elapsed_ms": elapsed_ms,
            "debug": debug,
            "types": summary["types"],
            "sources": summary["sources"][:5],
        }, ensure_ascii=False)
    )


def rotate_retrieval_telemetry():
    try:
        lines = RETRIEVAL_TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    if len(lines) <= MAX_TELEMETRY_LINES:
        return

    RETRIEVAL_TELEMETRY_PATH.write_text(
        "\n".join(lines[-MAX_TELEMETRY_LINES:]) + "\n",
        encoding="utf-8"
    )


def read_retrieval_telemetry(limit: int = 50):
    limit = max(1, min(int(limit or 50), 500))

    try:
        lines = RETRIEVAL_TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

    records = []

    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except Exception:
            continue

    return records


def _md_inline(text: str) -> str:
    """Render inline Markdown: bold and inline code (operates on html-escaped text)."""
    # inline code  `foo`  — wrap in <code> (content already escaped)
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{html.escape(m.group(1))}</code>", text)
    # bold  **foo**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text


def render_markdown_document(markdown: str) -> str:
    lines = markdown.splitlines()
    html_lines = []
    in_code = False
    in_ul = False
    in_ol = False

    def _close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            _close_lists()
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue

        if in_code:
            html_lines.append(html.escape(line))
            continue

        # blank line
        if not stripped:
            _close_lists()
            continue

        # heading
        if stripped.startswith("#"):
            _close_lists()
            level = min(max(len(stripped) - len(stripped.lstrip("#")), 1), 4)
            text = _md_inline(html.escape(stripped[level:].strip()))
            html_lines.append(f"<h{level}>{text}</h{level}>")
            continue

        # horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            _close_lists()
            html_lines.append("<hr>")
            continue

        # unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{_md_inline(html.escape(stripped[2:]))}</li>")
            continue

        # ordered list  1. …
        m = re.match(r"^\d+\.\s+(.*)", stripped)
        if m:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{_md_inline(html.escape(m.group(1)))}</li>")
            continue

        # Markdown table row
        if stripped.startswith("|") and "|" in stripped[1:]:
            _close_lists()
            if re.match(r"^\|[-| :]+\|$", stripped):
                continue  # separator row
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            is_first = not html_lines or not html_lines[-1].startswith("<tr>") and "</table>" not in html_lines[-1]
            if not html_lines or "</table>" in html_lines[-1] or html_lines[-1] == "<hr>":
                html_lines.append("<table>")
            tag = "th" if (not html_lines or "<table>" == html_lines[-1]) else "td"
            row = "".join(f"<{tag}>{_md_inline(html.escape(c))}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
            # auto-close table on next non-table line handled below
            continue

        # close open table
        if html_lines and html_lines[-1].startswith("<tr>"):
            html_lines.append("</table>")

        _close_lists()
        html_lines.append(f"<p>{_md_inline(html.escape(stripped))}</p>")

    _close_lists()
    if in_code:
        html_lines.append("</code></pre>")

    # close any unclosed table
    result = "\n".join(html_lines)
    result = re.sub(r"(<tr>.*</tr>)\n(?!<tr>|</table>)", r"\1\n</table>\n", result)
    return result


def command_hints_for_query(query: str):
    lowered = query.lower()
    hints = []

    if any(word in lowered for word in ["klastr", "cluster"]):
        if any(word in lowered for word in ["informac", "info", "sprawdz", "show", "check"]):
            hints.append("OpenShift cluster information oc get all oc describe oc api-resources oc get nodes oc get clusterversion oc cluster-info")

    if any(word in lowered for word in ["pod", "pody", "pods"]):
        hints.append("OpenShift pods oc get pods oc describe pod kubectl get pods")

    if any(word in lowered for word in ["zasob", "resource", "resources"]):
        hints.append("OpenShift resources oc get all oc api-resources oc describe")

    return hints


def score_result_for_query(query: str, result):
    lowered_query = query.lower()
    content = result[0].lower()
    score = 0

    cluster_info_intent = any(word in lowered_query for word in ["klastr", "cluster"])
    auth_intent = any(word in lowered_query for word in ["auth", "oauth", "login", "użytkownik", "user"])

    if cluster_info_intent:
        preferred_phrases = [
            "oc cluster-info",
            "oc get clusterversion",
            "oc get clusteroperators",
            "oc get nodes",
            "oc get all",
            "cluster version",
            "cluster operators",
            "health of nodes",
            "status and health of nodes",
        ]

        for phrase in preferred_phrases:
            if phrase in content:
                score += 8

        if "oauth" in content and not auth_intent:
            score -= 12

    if any(word in lowered_query for word in ["pod", "pody", "pods"]):
        for phrase in ["oc get pods", "oc describe pod", "pods in this project"]:
            if phrase in content:
                score += 8

    for word in lowered_query.split():
        clean_word = word.strip(".,:;!?()[]{}'\"`")
        if len(clean_word) >= 4 and clean_word in content:
            score += 1

    return score


def prioritize_results(query: str, results):
    return sorted(
        results,
        key=lambda result: score_result_for_query(query, result),
        reverse=True
    )


def choose_mode(query: str, mode: str) -> str:
    mode = (mode or "auto").lower()
    if mode in ["fast", "smart"]:
        return mode

    q = query.lower()

    if len(q) < 40:
        return "fast"

    if any(x in q for x in ["jak", "co to", "komenda", "command"]):
        return "fast"

    return "smart"


def multilingual_search(query: str, tags=None, backend: str = "auto", model: str = DEFAULT_MODEL):
    search_query = search_query_for_rag(query, backend=backend, model=model)
    search_variants = [query, search_query] + command_hints_for_query(query)
    unique_variants = []

    for variant in search_variants:
        variant = variant.strip()
        if variant and variant.lower() not in [item.lower() for item in unique_variants]:
            unique_variants.append(variant)

    combined_query = "\n".join(unique_variants)
    results = []

    for variant in unique_variants:
        results.extend(hybrid_search(variant, tags=tags))

        if tags:
            results.extend(hybrid_search(variant, tags=None))

    results = dedupe_results(results)
    results = prioritize_results(combined_query, results)
    print("BEFORE RERANK:", len(results))
    results = smart_rerank(combined_query, results[:20], backend=backend, model=model)

    return results, search_query


def answer_search(query: str, tags=None, backend: str = "auto", model: str = DEFAULT_MODEL, mode: str = "auto"):
    search_query = search_query_for_rag(query, backend=backend, model=model)
    search_variants = [query, search_query] + command_hints_for_query(query)
    unique_variants = []

    for variant in search_variants:
        variant = variant.strip()
        if variant and variant.lower() not in [item.lower() for item in unique_variants]:
            unique_variants.append(variant)

    combined_query = "\n".join(unique_variants)
    results = []

    for variant in unique_variants:
        results.extend(hybrid_search(variant, tags=tags))

        if tags:
            results.extend(hybrid_search(variant, tags=None))

    results = dedupe_results(results)
    results = prioritize_results(combined_query, results)

    selected_mode = choose_mode(query, mode)
    print("MODE:", selected_mode)
    print("RESULT COUNT:", len(results))

    if selected_mode == "smart":
        print("BEFORE RERANK:", len(results))
        results = smart_rerank(combined_query, results[:20], backend=backend, model=model)
        print("RESULT COUNT:", len(results))
    else:
        print("SKIP RERANK (FAST MODE)")

    return results, search_query, selected_mode


@app.get("/models")
def models(backend: str = "auto"):
    names = []

    for url in backend_urls(backend):
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code != 200:
                continue

            for model in response.json().get("models", []):
                name = model.get("name")
                families = model.get("details", {}).get("families", [])
                if "embed" in (name or "") or "nomic-bert" in families:
                    continue

                if name and name not in names:
                    names.append(name)

        except Exception as e:
            print("MODELS FAILED:", url, e)

    preferred = [
        DEFAULT_MODEL,
        "gemma3:1b",
        "qwen2.5-coder:0.5b",
        "qwen2.5-coder:3b",
        "llama3.2:latest",
        "gemma3:4b",
        "gemma3-i5:latest",
        "qwen2.5-coder:7b",
    ]
    names.sort(key=lambda name: preferred.index(name) if name in preferred else len(preferred))

    return {"models": names}


@app.post("/tag_query")
def tag_query(req: TagQueryRequest):
    decision = classify_query(req.query, backend=req.backend, model=req.model)
    return decision


@app.get("/backend_status")
def backend_status():
    cpu = ollama_available(settings.OLLAMA_CPU_URL)
    gpu = ollama_available(settings.OLLAMA_GPU_URL)
    laptop = ollama_available(settings.OLLAMA_LAPTOP_URL)

    return {
        "cpu": {
            "available": cpu,
            "url": settings.OLLAMA_CPU_URL
        },
        "gpu": {
            "available": gpu,
            "url": settings.OLLAMA_GPU_URL
        },
        "laptop": {
            "available": laptop,
            "url": settings.OLLAMA_LAPTOP_URL
        },
        "auto": {
            "available": cpu or gpu or laptop
        },
        "embedding": embedding_backend_status(),
        "rerank_targets": rerank_targets(),
    }


@app.get("/embedding_backend")
def get_embedding_backend():
    return embedding_backend_status()


@app.post("/embedding_backend")
def update_embedding_backend(req: EmbeddingBackendRequest):
    return set_embedding_backend(req.backend)


# ── Backend settings API ──────────────────────────────────────────────────────

@app.get("/api/settings")
def api_get_settings():
    cfg = get_backend_config()
    return {
        "backend_type": cfg["backend_type"],
        "embed_backend_type": cfg["embed_backend_type"],
        "gen_backend_type": cfg["gen_backend_type"],
        "rerank_backend_type": cfg["rerank_backend_type"],
        "embed_url": cfg["embed_url"],
        "embed_model": cfg["embed_model"],
        "gpu_url": cfg["gpu_url"],
        "cpu_url": cfg["cpu_url"],
        "laptop_url": cfg["laptop_url"],
        "rerank_url": cfg["rerank_url"],
        "rerank_model": cfg["rerank_model"],
        "openai_api_key": "***" if cfg.get("openai_api_key") and cfg["openai_api_key"] != "na" else cfg.get("openai_api_key", "na"),
        "status": {
            "gpu": server_available(cfg["gpu_url"]),
            "cpu": server_available(cfg["cpu_url"]),
            "laptop": server_available(cfg["laptop_url"]),
        },
    }


@app.post("/api/settings")
def api_post_settings(data: dict):
    # Don't persist masked key
    if data.get("openai_api_key") == "***":
        data.pop("openai_api_key", None)
    cfg = set_backend_config(data)
    return {"ok": True, "config": cfg}


@app.get("/settings")
def settings_page():
    return HTMLResponse("""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAGHybrid — Ustawienia</title>
<style>
:root{--bg:#f0f4f8;--panel:#fff;--text:#151922;--muted:#667085;--line:#d9e1ea;
      --accent:#0f766e;--asoft:#d9f3ee;--ok:#16a34a;--err:#dc2626;--warn:#d97706;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);
     font-family:Inter,ui-sans-serif,sans-serif;line-height:1.6;}
main{width:min(900px,calc(100% - 32px));margin:0 auto;padding:32px 0 60px;}
header{display:flex;justify-content:space-between;align-items:center;
       padding-bottom:18px;border-bottom:1px solid var(--line);margin-bottom:24px;}
header h1{margin:0;font-size:1.35rem;}
.nav a{color:var(--accent);text-decoration:none;font-weight:600;font-size:.9rem;margin-left:16px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-bottom:18px;}
.card-title{font-weight:700;font-size:.95rem;color:var(--accent);
            border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:16px;}

/* Backend selector cards */
.backend-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:0;}
.bcard{border:2px solid var(--line);border-radius:10px;padding:16px 18px;
       cursor:pointer;background:#fff;transition:all .15s;position:relative;}
.bcard:hover{border-color:#94a3b8;}
.bcard.on{border-color:var(--accent);background:var(--asoft);}
.bcard .bc-icon{font-size:1.5rem;margin-bottom:6px;}
.bcard .bc-name{font-weight:700;font-size:.95rem;}
.bcard .bc-desc{font-size:.78rem;color:var(--muted);margin-top:2px;}
.bcard .bc-port{font-size:.75rem;font-family:monospace;color:#0f766e;
                background:#e0f2f1;border-radius:4px;padding:1px 6px;
                display:inline-block;margin-top:4px;}

/* Quick URL section that appears after selecting a backend */
.url-wizard{display:none;background:linear-gradient(135deg,#f0fdf4,#eff6ff);
            border:1px solid #c7d8e8;border-radius:10px;padding:20px 24px;
            margin-bottom:18px;animation:fadein .2s;}
.url-wizard.show{display:block;}
@keyframes fadein{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
.url-wizard-title{font-weight:700;color:var(--accent);margin-bottom:14px;font-size:.95rem;}

.field{margin-bottom:12px;}
.field label{display:block;font-weight:600;font-size:.8rem;color:var(--muted);
             margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;}
.field input,.field select{width:100%;padding:9px 12px;border:1px solid var(--line);
  border-radius:6px;font:inherit;font-size:.92rem;background:#fff;}
.field input:focus,.field select:focus{outline:none;border-color:var(--accent);}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}

/* Quick fill chips */
.quick-chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;}
.qchip{padding:3px 10px;border-radius:12px;border:1px solid var(--line);
       cursor:pointer;font-size:.78rem;background:#f8fafc;font-family:monospace;}
.qchip:hover{background:var(--asoft);border-color:var(--accent);}

/* Status */
.status-row{display:flex;gap:8px;flex-wrap:wrap;}
.schip{display:flex;align-items:center;gap:6px;background:#f8fafc;
       border:1px solid var(--line);border-radius:20px;padding:5px 14px;font-size:.82rem;}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.ok-d{background:var(--ok);} .err-d{background:var(--err);}
.warn-d{background:var(--warn);animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Advanced section toggle */
.toggle-adv{cursor:pointer;font-size:.82rem;color:var(--accent);font-weight:600;
            display:inline-flex;align-items:center;gap:4px;margin-bottom:12px;
            user-select:none;}
.adv-section{display:none;}
.adv-section.open{display:block;}

.btn{padding:10px 22px;border:none;border-radius:7px;cursor:pointer;
     font:inherit;font-size:.92rem;font-weight:600;transition:background .15s;}
.btn-save{background:var(--accent);color:#fff;} .btn-save:hover{background:#115e59;}
.btn-test{background:#f0f4f8;color:var(--text);border:1px solid var(--line);}
.btn-test:hover{background:#e2e8f0;}
.notice{padding:10px 14px;border-radius:7px;font-size:.88rem;margin-top:12px;}
.n-ok{background:#dcfce7;color:#166534;} .n-err{background:#fee2e2;color:#991b1b;}
.check-mark{position:absolute;top:10px;right:12px;color:var(--accent);
            font-size:1.1rem;display:none;}
.bcard.on .check-mark{display:block;}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>⚙ Ustawienia backendu LLM</h1>
      <div style="color:var(--muted);font-size:.85rem">Zmień backend bez restartu kontenera</div>
    </div>
    <div class="nav"><a href="/">Panel RAG</a><a href="/schemat">Architektura</a></div>
  </header>

  <!-- Status -->
  <div class="card">
    <div class="card-title">Status serwerów</div>
    <div class="status-row" id="srv-status">
      <div class="schip"><div class="dot warn-d"></div>Sprawdzam...</div>
    </div>
  </div>

  <!-- Backend type chooser -->
  <div class="card">
    <div class="card-title">Wybierz backend LLM</div>
    <div class="backend-grid">
      <div class="bcard" id="bc-ollama" onclick="selectBackend('ollama')">
        <div class="check-mark">✓</div>
        <div class="bc-icon">🦙</div>
        <div class="bc-name">Ollama</div>
        <div class="bc-desc">Lokalny serwer Ollama<br>Najprostsze ustawienie</div>
        <span class="bc-port">:11434</span>
      </div>
      <div class="bcard" id="bc-openai-vllm" onclick="selectBackend('openai','vllm')">
        <div class="check-mark">✓</div>
        <div class="bc-icon">⚡</div>
        <div class="bc-name">vLLM</div>
        <div class="bc-desc">Szybki inference GPU<br>OpenAI-compatible API</div>
        <span class="bc-port">:8000</span>
      </div>
      <div class="bcard" id="bc-openai-llama" onclick="selectBackend('openai','llama')">
        <div class="check-mark">✓</div>
        <div class="bc-icon">🔥</div>
        <div class="bc-name">llama.cpp</div>
        <div class="bc-desc">llama-server<br>OpenAI-compatible API</div>
        <span class="bc-port">:8080</span>
      </div>
    </div>
  </div>

  <!-- Wizard: URL input (appears after selecting backend) -->
  <div class="url-wizard" id="url-wizard">
    <div class="url-wizard-title" id="wizard-title">Podaj adres serwera</div>
    <div id="wizard-body"></div>
    <div style="display:flex;gap:10px;margin-top:6px">
      <button class="btn btn-save" onclick="save()">💾 Zapisz</button>
      <button class="btn btn-test" onclick="test()">🔍 Testuj</button>
    </div>
    <div id="notice"></div>
  </div>

  <!-- Advanced settings -->
  <div class="card">
    <div class="toggle-adv" onclick="toggleAdv()">
      <span id="adv-arrow">▶</span> Ustawienia zaawansowane (osobne backendy per komponent, reranker)
    </div>
    <div class="adv-section" id="adv-section">
      <div style="margin-bottom:14px">
        <div class="field"><label>Embedding URL (osobny, jeśli różni się od generowania)</label>
          <input id="embed_url" type="url" oninput="dirty=true" placeholder="zostaw puste = użyj głównego URL"></div>
        <div class="row2">
          <div class="field"><label>Embedding model</label>
            <input id="embed_model" type="text" oninput="dirty=true"></div>
          <div class="field"><label>Reranker model</label>
            <input id="rerank_model" type="text" oninput="dirty=true"></div>
        </div>
      </div>
      <div style="margin-bottom:14px">
        <div style="font-weight:600;font-size:.82rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px">
          Osobny backend per komponent (puste = dziedzicz globalny)
        </div>
        <div class="row3">
          <div class="field"><label>Embeddingi</label>
            <select id="embed_backend_type" onchange="dirty=true">
              <option value="">— globalny —</option>
              <option value="ollama">ollama</option>
              <option value="openai">openai</option>
            </select></div>
          <div class="field"><label>Generowanie</label>
            <select id="gen_backend_type" onchange="dirty=true">
              <option value="">— globalny —</option>
              <option value="ollama">ollama</option>
              <option value="openai">openai</option>
            </select></div>
          <div class="field"><label>Reranker</label>
            <select id="rerank_backend_type" onchange="dirty=true">
              <option value="">— globalny —</option>
              <option value="ollama">ollama</option>
              <option value="openai">openai</option>
            </select></div>
        </div>
      </div>
      <div class="field"><label>OpenAI API key (dla vLLM/llama.cpp wpisz dowolne, np. "na")</label>
        <input id="openai_api_key" type="password" placeholder="na" oninput="dirty=true"></div>
    </div>
  </div>

</main>
<script>
let dirty=false;
let currentBackend='ollama';
let currentVariant='';

const BACKEND_DEFAULTS={
  ollama: {port:'11434', placeholder:'http://192.168.1.X:11434', embed_model:'nomic-embed-text'},
  vllm:   {port:'8000',  placeholder:'http://192.168.1.X:8000',  embed_model:'text-embedding-ada-002'},
  llama:  {port:'8080',  placeholder:'http://192.168.1.X:8080',  embed_model:'nomic-embed-text'},
};

function selectBackend(type, variant=''){
  currentBackend=type;
  currentVariant=variant;
  // highlight card
  document.querySelectorAll('.bcard').forEach(c=>c.classList.remove('on'));
  const cardId = type==='ollama' ? 'bc-ollama' : variant==='vllm' ? 'bc-openai-vllm' : 'bc-openai-llama';
  document.getElementById(cardId).classList.add('on');

  const key = variant||type;
  const def = BACKEND_DEFAULTS[key] || BACKEND_DEFAULTS.ollama;
  const isOpenAI = type==='openai';
  const name = type==='ollama' ? 'Ollama' : variant==='vllm' ? 'vLLM' : 'llama.cpp server';

  document.getElementById('wizard-title').textContent = `Adresy serwerów — ${name}`;

  const urlVal = (id) => document.getElementById(id)?.value || '';

  document.getElementById('wizard-body').innerHTML = `
    <div style="margin-bottom:12px">
      <div style="font-size:.82rem;color:var(--muted);margin-bottom:8px">
        Szybkie wypełnienie (kliknij żeby wstawić):
      </div>
      <div class="quick-chips">
        <span class="qchip" onclick="fillAll('http://localhost:${def.port}')">localhost:${def.port}</span>
        <span class="qchip" onclick="fillAll('http://127.0.0.1:${def.port}')">127.0.0.1:${def.port}</span>
        <span class="qchip" onclick="fillAll('http://192.168.1.1:${def.port}')">192.168.1.1:${def.port}</span>
        <span class="qchip" onclick="fillAll('http://192.168.18.34:${def.port}')">192.168.18.34:${def.port}</span>
      </div>
    </div>
    <div class="row3">
      <div class="field"><label>GPU / główny serwer</label>
        <input id="gpu_url" type="url" placeholder="${def.placeholder}" oninput="dirty=true"
               value="${urlVal('gpu_url')}"></div>
      <div class="field"><label>CPU / fallback</label>
        <input id="cpu_url" type="url" placeholder="${def.placeholder}" oninput="dirty=true"
               value="${urlVal('cpu_url')}"></div>
      <div class="field"><label>Laptop / trzeci</label>
        <input id="laptop_url" type="url" placeholder="${def.placeholder}" oninput="dirty=true"
               value="${urlVal('laptop_url')}"></div>
    </div>
    ${isOpenAI ? `<div class="field" style="max-width:360px"><label>API key (np. "na" dla vLLM)</label>
      <input id="openai_api_key_w" type="text" placeholder="na" oninput="dirty=true"
             value="${document.getElementById('openai_api_key')?.value||'na'}"></div>` : ''}
  `;

  document.getElementById('url-wizard').classList.add('show');
  dirty=true;
}

function fillAll(url){
  ['gpu_url','cpu_url','laptop_url'].forEach(id=>{
    const e=document.getElementById(id);if(e)e.value=url;
  });
  dirty=true;
}

function toggleAdv(){
  const s=document.getElementById('adv-section');
  const a=document.getElementById('adv-arrow');
  s.classList.toggle('open');
  a.textContent=s.classList.contains('open')?'▼':'▶';
}

function set(id,val){const e=document.getElementById(id);if(e)e.value=val||'';}
function setSelect(id,val){const e=document.getElementById(id);if(e)e.value=val||'';}

async function load(){
  const d=await(await fetch('/api/settings')).json();

  // Highlight correct card
  const bt=d.backend_type||'ollama';
  if(bt==='openai'){
    // guess variant from port
    const gurl=d.gpu_url||'';
    const v=gurl.includes('8080')?'llama':'vllm';
    currentBackend='openai';currentVariant=v;
    const cardId=v==='vllm'?'bc-openai-vllm':'bc-openai-llama';
    document.getElementById(cardId).classList.add('on');
  } else {
    currentBackend='ollama';currentVariant='';
    document.getElementById('bc-ollama').classList.add('on');
  }

  // Fill advanced fields
  setSelect('embed_backend_type',d.embed_backend_type||'');
  setSelect('gen_backend_type',d.gen_backend_type||'');
  setSelect('rerank_backend_type',d.rerank_backend_type||'');
  set('embed_url',d.embed_url);
  set('embed_model',d.embed_model);
  set('rerank_model',d.rerank_model);
  set('openai_api_key',d.openai_api_key==='***'?'':d.openai_api_key);

  // Trigger wizard display with current URLs prefilled
  const key=currentVariant||currentBackend;
  const def=BACKEND_DEFAULTS[key]||BACKEND_DEFAULTS.ollama;
  const name=currentBackend==='ollama'?'Ollama':currentVariant==='vllm'?'vLLM':'llama.cpp server';
  document.getElementById('wizard-title').textContent=`Adresy serwerów — ${name}`;
  document.getElementById('wizard-body').innerHTML=`
    <div style="margin-bottom:12px">
      <div style="font-size:.82rem;color:var(--muted);margin-bottom:8px">Szybkie wypełnienie:</div>
      <div class="quick-chips">
        <span class="qchip" onclick="fillAll('http://localhost:${def.port}')">localhost:${def.port}</span>
        <span class="qchip" onclick="fillAll('http://127.0.0.1:${def.port}')">127.0.0.1:${def.port}</span>
        <span class="qchip" onclick="fillAll('http://192.168.1.1:${def.port}')">192.168.1.1:${def.port}</span>
        <span class="qchip" onclick="fillAll('http://192.168.18.34:${def.port}')">192.168.18.34:${def.port}</span>
      </div>
    </div>
    <div class="row3">
      <div class="field"><label>GPU / główny serwer</label>
        <input id="gpu_url" type="url" placeholder="${def.placeholder}" oninput="dirty=true" value="${d.gpu_url||''}"></div>
      <div class="field"><label>CPU / fallback</label>
        <input id="cpu_url" type="url" placeholder="${def.placeholder}" oninput="dirty=true" value="${d.cpu_url||''}"></div>
      <div class="field"><label>Laptop / trzeci</label>
        <input id="laptop_url" type="url" placeholder="${def.placeholder}" oninput="dirty=true" value="${d.laptop_url||''}"></div>
    </div>
    ${currentBackend==='openai'?`<div class="field" style="max-width:360px"><label>API key (np. "na")</label>
      <input id="openai_api_key_w" type="text" placeholder="na" oninput="dirty=true" value="${d.openai_api_key==='***'?'na':d.openai_api_key||'na'}"></div>`:''}
  `;
  document.getElementById('url-wizard').classList.add('show');

  renderStatus(d.status||{},d);
  dirty=false;
}

function renderStatus(status,d){
  const labels={gpu:'GPU',cpu:'CPU',laptop:'Laptop'};
  const urls={gpu:d.gpu_url||'',cpu:d.cpu_url||'',laptop:d.laptop_url||''};
  document.getElementById('srv-status').innerHTML=
    Object.entries(status).map(([k,ok])=>`
      <div class="schip" title="${urls[k]}">
        <div class="dot ${ok?'ok-d':'err-d'}"></div>
        ${labels[k]||k}: <strong>${ok?'online':'offline'}</strong>
        <span style="color:var(--muted);font-size:.75rem;margin-left:4px">${(urls[k]||'').replace(/https?:\\/\\//,'')}</span>
      </div>`).join('');
}

async function save(){
  const btn=document.querySelector('.btn-save');
  btn.disabled=true;btn.textContent='Zapisuję...';
  // Sync wizard API key to adv field if present
  const wkey=document.getElementById('openai_api_key_w');
  if(wkey) document.getElementById('openai_api_key').value=wkey.value;

  const p={
    backend_type:currentBackend,
    embed_backend_type:document.getElementById('embed_backend_type').value,
    gen_backend_type:document.getElementById('gen_backend_type').value,
    rerank_backend_type:document.getElementById('rerank_backend_type').value,
    embed_url:document.getElementById('embed_url')?.value||'',
    embed_model:document.getElementById('embed_model')?.value||'',
    gpu_url:document.getElementById('gpu_url')?.value||'',
    cpu_url:document.getElementById('cpu_url')?.value||'',
    laptop_url:document.getElementById('laptop_url')?.value||'',
    rerank_url:'',
    rerank_model:document.getElementById('rerank_model')?.value||'',
    openai_api_key:document.getElementById('openai_api_key')?.value||'na',
  };
  try{
    const r=await(await fetch('/api/settings',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(p)})).json();
    if(r.ok){notice('✓ Zapisano — backend aktywny','ok');dirty=false;
             const d=await(await fetch('/api/settings')).json();renderStatus(d.status||{},d);}
    else notice('Błąd: '+JSON.stringify(r),'err');
  }catch(e){notice('Błąd: '+e,'err');}
  btn.disabled=false;btn.textContent='💾 Zapisz';
}

async function test(){
  const btn=document.querySelectorAll('.btn-test')[0];
  btn.disabled=true;btn.textContent='Sprawdzam...';
  if(dirty)await save();
  const d=await(await fetch('/api/settings')).json();
  renderStatus(d.status||{},d);
  btn.disabled=false;btn.textContent='🔍 Testuj';
}

function notice(msg,type){
  const e=document.getElementById('notice');
  e.innerHTML=`<div class="notice n-${type}">${msg}</div>`;
  if(type==='ok')setTimeout(()=>e.innerHTML='',4000);
}
load();
</script>
</body>
</html>""")




@app.post("/retrieve")
def retrieve(
    query: str = Form(...),
    tags: str = Form(None),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL)
):
    tag_list = parse_tags(tags)
    results, search_query = multilingual_search(query, tags=tag_list, backend=backend, model=model)

    return {
        "query": query,
        "search_query": search_query,
        "results": [
            {
                "content": r[0],
                "metadata": r[1] or {},
                "source": (r[1] or {}).get("source", "unknown"),
                "tags": (r[1] or {}).get("tags", [])
            }
            for r in results
        ]
    }


def run_retrieve_json(req: RetrieveRequest, endpoint: str = "retrieve_json"):
    started_at = time.time()
    result_limit = None

    if req.top_k is not None:
        result_limit = max(1, min(int(req.top_k), 20))

    results, debug = build_hybrid_context(
        req.query,
        tags=req.tags,
        backend=req.backend,
        max_vector=min(req.max_vector, result_limit) if result_limit else req.max_vector,
        max_graph=req.max_graph,
        max_evidence=req.max_evidence,
        max_context_chars=req.max_context_chars
    )
    elapsed_ms = int((time.time() - started_at) * 1000)
    debug["elapsed_ms"] = elapsed_ms
    debug["results_before_gate"] = len(results)

    gate_enabled = env_bool("RAG_RELEVANCE_GATE_ENABLED", True)
    decision = calculate_relevance(req.query, results)

    if gate_enabled:
        results = decision.accepted_results

    if result_limit is not None:
        results = results[:result_limit]

    debug.update({
        "rag_used": bool(decision.rag_used if gate_enabled else True),
        "relevance_score": decision.score,
        "relevance_reason": decision.reason,
        "gate_reason": decision.reason,
        "top_score": decision.top_score,
        "threshold": decision.threshold,
        "lexical_overlap": decision.lexical_overlap,
        "accepted_results": len(decision.accepted_results),
        "rejected_results": len(decision.rejected_results),
        "results_after_gate": len(results),
        "top_k": result_limit,
        "gate_enabled": gate_enabled,
        "gate_debug": env_bool("RAG_GATE_DEBUG", True),
    })

    print("RETRIEVE_JSON RESULTS:", len(results))
    for idx, r in enumerate(results):
        print("RETRIEVE_JSON RESULT", idx + 1, r.get("source", "unknown"), r.get("content", "")[:100])

    if req.telemetry:
        log_retrieval_telemetry(endpoint, req.query, debug, results, elapsed_ms)

    if gate_enabled and not decision.rag_used:
        return {
            "query": req.query,
            "instruction": DEFAULT_NO_CONTEXT_INSTRUCTION,
            "results": [],
            "debug": {
                **debug,
                "rag_used": False,
                "reason": decision.reason,
            },
        }

    return {
        "query": req.query,
        "instruction": HYBRID_CONTEXT_INSTRUCTION,
        "debug": debug,
        "results": results
    }


@app.post("/retrieve_json")
def retrieve_json(req: RetrieveRequest):
    return run_retrieve_json(req, endpoint="retrieve_json")


@app.get("/retrieval_telemetry")
def get_retrieval_telemetry(limit: int = 50):
    records = read_retrieval_telemetry(limit=limit)
    totals = {
        "count": len(records),
        "avg_elapsed_ms": 0,
        "avg_context_chars": 0,
        "types": defaultdict(int),
        "path": str(RETRIEVAL_TELEMETRY_PATH),
        "message": "Brak rekordów. Telemetryka pojawi się po wywołaniu /retrieve_json albo toola MCP hybridrag_search.",
    }

    if records:
        totals["message"] = "OK"
        totals["avg_elapsed_ms"] = int(sum(r.get("elapsed_ms", 0) for r in records) / len(records))
        totals["avg_context_chars"] = int(
            sum((r.get("debug") or {}).get("context_chars", 0) for r in records) / len(records)
        )

        for record in records:
            for key, value in (record.get("types") or {}).items():
                totals["types"][key] += value

    totals["types"] = dict(totals["types"])

    return {
        "summary": totals,
        "records": records,
    }


@app.post("/retrieval_feedback")
def retrieval_feedback(req: RetrievalFeedbackRequest):
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO retrieval_feedback
                (query, rating, missing_source, comment, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                req.query,
                req.rating,
                req.missing_source,
                req.comment,
                json.dumps(req.metadata or {}),
            )
        )
        feedback_id = cur.fetchone()[0]
        conn.commit()
        return {"status": "ok", "id": feedback_id}

    finally:
        cur.close()
        conn.close()


@app.get("/retrieval_feedback")
def get_retrieval_feedback(limit: int = 50):
    limit = max(1, min(int(limit or 50), 500))
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT id, created_at, query, rating, missing_source, comment, metadata
            FROM retrieval_feedback
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )
        return {
            "records": [
                {
                    "id": row[0],
                    "created_at": row[1].isoformat() if row[1] else None,
                    "query": row[2],
                    "rating": row[3],
                    "missing_source": row[4],
                    "comment": row[5],
                    "metadata": row[6] or {},
                }
                for row in cur.fetchall()
            ]
        }

    finally:
        cur.close()
        conn.close()


@app.post("/golden_tests/run")
def run_golden_tests(req: GoldenTestRunRequest):
    results = []

    for test in req.tests:
        tags = [test.collection] if test.collection else None
        retrieved, debug = build_hybrid_context(
            test.query,
            tags=tags,
            backend=req.backend,
            max_vector=req.max_vector,
            max_graph=req.max_graph,
            max_evidence=req.max_evidence,
            max_context_chars=req.max_context_chars,
        )
        sources = [item.get("source", "") for item in retrieved]
        types = {item.get("type") for item in retrieved}
        source_hits = [
            expected
            for expected in test.expected_sources
            if any(expected.lower() in source.lower() for source in sources)
        ]
        type_hits = [
            expected
            for expected in test.expected_types
            if expected in types
        ]
        passed = (
            len(source_hits) == len(test.expected_sources)
            and len(type_hits) == len(test.expected_types)
        )
        results.append({
            "query": test.query,
            "passed": passed,
            "expected_sources": test.expected_sources,
            "source_hits": source_hits,
            "expected_types": test.expected_types,
            "type_hits": type_hits,
            "debug": debug,
            "sources": sources[:20],
            "types": sorted([item for item in types if item]),
        })

    passed_count = sum(1 for item in results if item["passed"])

    return {
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
        },
        "results": results,
    }


@app.post("/ask")
def ask(
    query: str = Form(...),
    tags: str = Form(None),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL),
    session_id: str = Form("default"),
    mode: str = Form("auto")
):
    model = model.strip() if model else DEFAULT_MODEL
    session_id = session_id.strip() if session_id else "default"
    tag_list = auto_tags_for_query(query, tags=tags, backend=backend, model=model)
    results, search_query, selected_mode = answer_search(
        query,
        tags=tag_list,
        backend=backend,
        model=model,
        mode=mode
    )

    if not results:
        return {"answer": "Brak danych w bazie."}

    context = ""
    for r in results:
        part = f"[Źródło: {context_source(r[1])}]\n{r[0]}\n\n"
        if len(context) + len(part) > MAX_CONTEXT:
            break
        context += part
    context = context[:MAX_CONTEXT]

    history = build_history(session_id)

    print("FINAL CONTEXT SIZE:", len(context))
    print("CONTEXT:\n", context)

    prompt = f"""
Conversation history:
{history}

You are an operations-focused RAG assistant.

Use the context below as the source of truth, but synthesize a practical answer.
If the context is broad or training-oriented, say that explicitly and still give the best operational checklist supported by the context.
Prefer concrete commands, checks, and short steps when the question asks how to do something.
Do not invent exact commands unless they are standard and relevant; mark them as examples when they are not directly present in the context.

{context}

Question:
{query}

Search query used for retrieval:
{search_query}

Answer in Polish if the question is in Polish.
Always cite the source next to the claims you derive from context.
"""

    answer = generate(prompt, backend=backend, model=model)

    chat_memory[session_id].append({"role": "user", "content": query})
    chat_memory[session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer, "mode": selected_mode}


@app.post("/ask_stream")
def ask_stream(
    query: str = Form(...),
    tags: str = Form(None),
    backend: str = Form("auto"),
    model: str = Form(DEFAULT_MODEL),
    session_id: str = Form("default"),
    mode: str = Form("auto")
):
    model = model.strip() if model else DEFAULT_MODEL
    session_id = session_id.strip() if session_id else "default"

    full_answer = ""

    def event_generator():
        nonlocal full_answer

        yield "event: status\ndata: Dobieram tagi\n\n"
        tag_list = auto_tags_for_query(query, tags=tags, backend=backend, model=model)

        yield "event: status\ndata: Szukam w bazie\n\n"
        results, search_query, selected_mode = answer_search(
            query,
            tags=tag_list,
            backend=backend,
            model=model,
            mode=mode
        )

        if selected_mode == "smart":
            yield "event: status\ndata: Reranking kontekstu\n\n"
        else:
            yield "event: status\ndata: FAST mode bez rerankingu\n\n"

        yield "event: status\ndata: Buduję kontekst\n\n"
        context = ""
        for r in results:
            part = f"[Źródło: {context_source(r[1])}]\n{r[0]}\n\n"
            if len(context) + len(part) > MAX_CONTEXT:
                break
            context += part
        context = context[:MAX_CONTEXT]

        history = build_history(session_id)

        prompt = f"""
Conversation history:
{history}

You are an operations-focused RAG assistant.

Use the context below as the source of truth, but synthesize a practical answer.
If the context is broad or training-oriented, say that explicitly and still give the best operational checklist supported by the context.
Prefer concrete commands, checks, and short steps when the question asks how to do something.
Do not invent exact commands unless they are standard and relevant; mark them as examples when they are not directly present in the context.

{context}

Question:
{query}

Search query used for retrieval:
{search_query}

Answer in Polish if the question is in Polish.
Always cite the source next to the claims you derive from context.
"""

        yield f"event: status\ndata: Model generuje odpowiedź ({model})\n\n"

        for raw in generate_stream(prompt, backend=backend, model=model):
            try:
                obj = json.loads(raw)
                token = obj.get("response", "")
                done = obj.get("done", False)

                full_answer += token

                yield f"data: {token}\n\n"

                if done:
                    break

            except Exception:
                continue

        yield "event: status\ndata: Gotowy\n\n"

        chat_memory[session_id].append({"role": "user", "content": query})
        chat_memory[session_id].append({"role": "assistant", "content": full_answer})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/schemat")
def schemat():
    if not SCHEMA_DOC_PATH.exists():
        return HTMLResponse(
            "<h1>Brak dokumentacji</h1><p>Nie znaleziono pliku ARCHITEKTURA_RAG.md.</p>",
            status_code=404
        )

    markdown = SCHEMA_DOC_PATH.read_text(encoding="utf-8")
    body = render_markdown_document(markdown)

    pipeline_diagram = """
    <div class="pipeline-card">
        <div class="pipeline-title">Jak działa /retrieve_json — 5 faz</div>
        <div class="pipeline-row">
            <div class="phase phase-query">
                <div class="phase-label">Pytanie</div>
                <div class="phase-detail">query + opcjonalne tagi</div>
            </div>
            <div class="phase-arrow">→</div>
            <div class="phase phase-parallel">
                <div class="phase-label">Faza 1 — równolegle</div>
                <div class="phase-cols">
                    <div class="phase-box">
                        <strong>Vector</strong><br>
                        <span class="mono">hybrid_search()</span><br>
                        pgvector L2 + ILIKE
                    </div>
                    <div class="phase-box">
                        <strong>Graph</strong><br>
                        <span class="mono">search_graph_scored()</span><br>
                        Neo4j fulltext BM25
                    </div>
                </div>
            </div>
            <div class="phase-arrow">→</div>
            <div class="phase phase-evidence">
                <div class="phase-label">Faza 2</div>
                <div class="phase-detail"><span class="mono">source_evidence_chunks()</span><br>chunki ze źródeł relacji</div>
            </div>
        </div>
        <div class="pipeline-row pipeline-row-bottom">
            <div class="phase phase-fusion">
                <div class="phase-label">Faza 4 — RRF Fusion</div>
                <div class="phase-detail">
                    <span class="mono">fuse_retrieval_results()</span><br>
                    vector 50% + graph 30% + evidence 20%<br>
                    → <strong>fused_score</strong> dla każdego kandydata
                </div>
            </div>
            <div class="phase-arrow">←</div>
            <div class="phase phase-convert">
                <div class="phase-label">Faza 3</div>
                <div class="phase-detail">konwersja do <span class="mono">FusedCandidate</span></div>
            </div>
            <div class="phase-arrow">←</div>
            <div class="phase-spacer"></div>
        </div>
        <div class="pipeline-row">
            <div class="phase phase-gate">
                <div class="phase-label">Faza 5 — Bramka jakości</div>
                <div class="phase-detail">
                    <span class="mono">calculate_relevance()</span><br>
                    próg: fused_score ≥ 0.45<br>
                    → JSON context lub "brak wiedzy"
                </div>
            </div>
            <div class="phase-arrow">→</div>
            <div class="phase phase-result">
                <div class="phase-label">Wynik</div>
                <div class="phase-detail">
                    <span class="chip chip-v">vector</span>
                    <span class="chip chip-g">graph</span>
                    <span class="chip chip-e">graph_evidence</span>
                </div>
            </div>
        </div>
    </div>
    """

    return HTMLResponse(f"""
    <!doctype html>
    <html lang="pl">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Schemat RAG Hybrid</title>
        <style>
            :root {{
                color-scheme: light;
                --bg: #f0f4f8;
                --panel: #ffffff;
                --text: #151922;
                --muted: #667085;
                --line: #d9e1ea;
                --accent: #0f766e;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                line-height: 1.6;
            }}
            main {{
                width: min(1020px, calc(100% - 32px));
                margin: 0 auto;
                padding: 36px 0 56px;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--line);
                margin-bottom: 28px;
                gap: 20px;
            }}
            header h1 {{ margin: 0; font-size: 1.5rem; }}
            .nav-links {{ display: flex; gap: 16px; }}
            .nav-links a {{ color: var(--accent); text-decoration: none; font-weight: 600; font-size: 0.9rem; }}
            h1,h2,h3,h4 {{ line-height: 1.2; margin: 1.5em 0 0.5em; }}
            a {{ color: var(--accent); text-decoration: none; font-weight: 700; }}
            p {{ margin: 0 0 12px; }}
            ul,ol {{ margin: 0 0 16px 22px; padding: 0; }}
            li {{ margin-bottom: 3px; }}
            hr {{ border: none; border-top: 1px solid var(--line); margin: 24px 0; }}
            pre {{
                overflow-x: auto;
                background: #111827;
                color: #f9fafb;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #1f2937;
                margin: 0 0 16px;
            }}
            code {{
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
                font-size: 0.9rem;
            }}
            p code, li code, td code {{
                background: #e8f0fa;
                color: #1a3150;
                border-radius: 4px;
                padding: 1px 5px;
                font-size: 0.86rem;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 0 0 16px;
                font-size: 0.9rem;
            }}
            th, td {{ border: 1px solid var(--line); padding: 7px 12px; text-align: left; }}
            th {{ background: #f0f4f8; font-weight: 600; }}
            tr:nth-child(even) td {{ background: #f8fafc; }}
            .doc {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 32px;
            }}
            .doc h2 {{ border-top: 1px solid var(--line); padding-top: 20px; margin-top: 32px; }}
            .doc h2:first-child {{ border-top: none; padding-top: 0; margin-top: 0; }}
            .muted {{ color: var(--muted); font-size: 0.9rem; }}

            /* Pipeline diagram */
            .pipeline-card {{
                background: linear-gradient(135deg, #f0fdf4, #eff6ff);
                border: 1px solid #c7d8e8;
                border-radius: 12px;
                padding: 24px;
                margin: 0 0 32px;
            }}
            .pipeline-title {{
                font-weight: 700;
                font-size: 1rem;
                color: #0f766e;
                margin-bottom: 16px;
                letter-spacing: 0.02em;
            }}
            .pipeline-row {{
                display: flex;
                align-items: stretch;
                gap: 8px;
                flex-wrap: wrap;
                margin-bottom: 8px;
            }}
            .pipeline-row-bottom {{
                flex-direction: row-reverse;
            }}
            .phase {{
                background: #fff;
                border: 1px solid #d9e1ea;
                border-radius: 8px;
                padding: 12px 14px;
                flex: 1;
                min-width: 160px;
            }}
            .phase-label {{ font-weight: 700; font-size: 0.82rem; color: #334155; margin-bottom: 4px; }}
            .phase-detail {{ font-size: 0.8rem; color: #475569; }}
            .phase-cols {{ display: flex; gap: 8px; margin-top: 6px; }}
            .phase-box {{
                flex: 1;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 0.78rem;
                color: #334155;
            }}
            .phase-query  {{ border-left: 3px solid #6366f1; }}
            .phase-parallel {{ border-left: 3px solid #0ea5e9; flex: 2; }}
            .phase-evidence {{ border-left: 3px solid #0f766e; }}
            .phase-convert  {{ border-left: 3px solid #f59e0b; }}
            .phase-fusion   {{ border-left: 3px solid #8b5cf6; flex: 2; }}
            .phase-gate     {{ border-left: 3px solid #ef4444; flex: 2; }}
            .phase-result   {{ border-left: 3px solid #22c55e; }}
            .phase-spacer   {{ flex: 1; }}
            .phase-arrow {{
                display: flex;
                align-items: center;
                font-size: 1.3rem;
                color: #94a3b8;
                padding: 0 2px;
                flex-shrink: 0;
            }}
            .mono {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.8rem; color: #1a3150; background: #e8f0fa; border-radius: 3px; padding: 1px 4px; }}
            .chip {{ display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600; margin: 2px; }}
            .chip-v {{ background: #dbeafe; color: #1e40af; }}
            .chip-g {{ background: #fce7f3; color: #9d174d; }}
            .chip-e {{ background: #d1fae5; color: #065f46; }}
        </style>
    </head>
    <body>
        <main>
            <header>
                <div>
                    <h1>Schemat RAG Hybrid</h1>
                    <div class="muted">Dokumentacja architektury · pełny opis systemu</div>
                </div>
                <div class="nav-links">
                    <a href="/schemat_grafu">Schemat grafu</a>
                    <a href="/graph_explorer">Graph Explorer</a>
                    <a href="/">Panel RAG</a>
                </div>
            </header>
            {pipeline_diagram}
            <article class="doc">
                {body}
            </article>
        </main>
    </body>
    </html>
    """)


@app.get("/schemat_grafu")
def schemat_grafu():
    body = """
    <div class="flow-diagram">
        <div class="flow-title">Jak graf trafia do /retrieve_json</div>
        <div class="flow-row">
            <div class="flow-box flow-query">Pytanie użytkownika</div>
            <div class="flow-arrow">→</div>
            <div class="flow-box flow-scored"><strong>search_graph_scored()</strong><br><small>Neo4j fulltext (Lucene BM25) → normalizacja → graph_score</small></div>
            <div class="flow-arrow">→</div>
            <div class="flow-box flow-evidence"><strong>source_evidence_chunks()</strong><br><small>chunki ze źródeł relacji z PostgreSQL</small></div>
        </div>
        <div class="flow-row">
            <div class="flow-box flow-fusion"><strong>fuse_retrieval_results()</strong><br><small>RRF: vector 50% + graph 30% + evidence 20% → fused_score</small></div>
            <div class="flow-arrow">→</div>
            <div class="flow-box flow-gate"><strong>calculate_relevance()</strong><br><small>bramka jakości · fused_score ≥ 0.45</small></div>
            <div class="flow-arrow">→</div>
            <div class="flow-box flow-out">
                <span class="chip-v">vector</span>
                <span class="chip-g">graph</span>
                <span class="chip-e">graph_evidence</span>
            </div>
        </div>
    </div>

    <h2>Dwie warstwy wiedzy</h2>
    <table>
        <tr><th>Warstwa</th><th>Baza</th><th>Co trzyma</th><th>Kiedy pomaga</th></tr>
        <tr><td><strong>RAG</strong></td><td>PostgreSQL / pgvector</td><td>fragmenty tekstu z dokumentów</td><td>pytania o treść, wyjaśnienia, komendy</td></tr>
        <tr><td><strong>Graf</strong></td><td>Neo4j</td><td>relacje encja → encja</td><td>pytania o zależności, architekturę, „co używa czego"</td></tr>
    </table>

    <h2>Jak działa search_graph_scored</h2>
    <ol>
        <li>Buduje zapytanie Lucene z terminów query: np. <code>ceph OR odf OR storage</code></li>
        <li>Odpytuje indeks fulltext Neo4j <code>entity_fulltext</code> → encje + BM25 score</li>
        <li>Traversal: pobiera relacje sąsiadujące z trafionymi encjami (obie strony)</li>
        <li>Normalizuje: <code>graph_score = 0.50×BM25 + 0.30×relation_priority + 0.20×n_sources</code></li>
        <li>Dedup: jedna najlepsza relacja na parę <code>(source, target)</code></li>
    </ol>
    <p>Fallback: jeśli indeks fulltext jest niedostępny — stary <code>search_graph()</code> z CONTAINS.</p>

    <h2>Jak liczy się fused_score</h2>
    <table>
        <tr><th>Sygnał</th><th>Formuła</th><th>Dyskont</th></tr>
        <tr><td>pgvector embedding distance</td><td><code>exp(−d / 30)</code></td><td>brak</td></tr>
        <tr><td>Neo4j relacja merytoryczna</td><td>BM25 + priorytet + źródła → 0–1</td><td>× 0.85</td></tr>
        <tr><td>Neo4j kotwica (is_a, contains)</td><td>j.w.</td><td>× 0.40</td></tr>
        <tr><td>Evidence term-overlap</td><td>trafienia / wszystkich terminów</td><td>× 0.70</td></tr>
        <tr><td>RRF bonus (koroboracja ≥2 ścieżek)</td><td>rrf_norm × 0.15</td><td>brak</td></tr>
    </table>
    <pre><code>fused_score = max(vector_score, graph_score × dyskont, evidence_score × 0.70)
            + rrf_norm × 0.15</code></pre>

    <h2>Trzy typy wyników</h2>
    <table>
        <tr><th>Typ</th><th>Źródło</th><th>Co niesie</th></tr>
        <tr><td><span class="chip-v">vector</span></td><td>PostgreSQL / pgvector</td><td>tekst właściwy — treść dokumentów</td></tr>
        <tr><td><span class="chip-g">graph</span></td><td>Neo4j</td><td>relacja: <code>ODF --uses--&gt; Ceph</code> — mapa zależności</td></tr>
        <tr><td><span class="chip-e">graph_evidence</span></td><td>PostgreSQL (ze źródeł relacji)</td><td>chunk potwierdzający relację — tekstowy dowód</td></tr>
    </table>
    <p><strong>Zasada:</strong> <code>vector</code> i <code>graph_evidence</code> to treść. <code>graph</code> bez evidence to tylko wskazówka — model nie powinien traktować jej jak twardy fakt.</p>

    <h2>Model danych Neo4j</h2>
    <pre><code>(:Entity {name: "ODF"})-[:RELATED {type: "uses", sources: [...]}]-&gt;(:Entity {name: "Ceph"})</code></pre>
    <ul>
        <li><code>type</code> — typ relacji: <code>uses</code>, <code>depends_on</code>, <code>runs_on</code>, <code>extends</code>…</li>
        <li><code>sources</code> — lista plików/URL potwierdzających relację</li>
        <li><code>metadata</code> — ostatnie metadane z importu</li>
    </ul>

    <h3>Priorytety typów relacji (wyżej = lepszy przy dedup)</h3>
    <pre><code>depends_on (80) &gt; requires (78) &gt; builds_on (76) &gt; runs_on (74) &gt; uses (70)
&gt; is_a (68) &gt; provides (66) &gt; stores (64) &gt; hosts (62) &gt; manages (60)
&gt; supports (58) &gt; exposes (56) &gt; contains (54) &gt; includes (52) &gt; creates (50)
&gt; accesses (48) &gt; allows (46) &gt; extends (42) &gt; has (30) &gt; is_managed_by (28)</code></pre>

    <h2>Jak import tworzy relacje</h2>
    <ol>
        <li>Dokument trafia do RAG jako chunki tekstu (PostgreSQL).</li>
        <li>Import z opcją <code>graph=true</code> wysyła chunk do LLM.</li>
        <li>LLM zwraca JSON z relacjami — tylko encje obecne w tekście.</li>
        <li><code>upsert_relation()</code> zapisuje do Neo4j przez MERGE (bez duplikatów krawędzi).</li>
        <li>Każdy dokument dostaje kotwicę: <code>tytuł --is_a--&gt; dokument</code> i <code>tytuł --contains--&gt; encja</code>.</li>
    </ol>

    <h2>Czyszczenie grafu</h2>
    <ul>
        <li><code>POST /graph_cleanup</code> — dry-run / apply, usuwa relacje spoza whitelisty.</li>
        <li><code>python -m app.rag.graph_conflict_cleanup --apply</code> — scala konflikty (kilka typów dla tej samej pary).</li>
    </ul>

    <h2>Jak czytać Graph Explorer</h2>
    <ul>
        <li>kółko = encja</li>
        <li>strzałka = relacja</li>
        <li>label na krawędzi = <code>type</code> relacji</li>
        <li>puste Search entity = cała baza (limit 500 relacji)</li>
    </ul>

    <p style="margin-top:28px">
        <a href="/graph_explorer">Otwórz Graph Explorer</a> &nbsp;·&nbsp;
        <a href="/schemat">Pełna dokumentacja architektury</a> &nbsp;·&nbsp;
        <a href="/">Panel RAG</a>
    </p>
    """

    return HTMLResponse(f"""
    <!doctype html>
    <html lang="pl">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Schemat bazy grafowej</title>
        <style>
            :root {{
                color-scheme: light;
                --bg: #f0f4f8;
                --panel: #ffffff;
                --text: #151922;
                --muted: #667085;
                --line: #d9e1ea;
                --accent: #0f766e;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                line-height: 1.6;
            }}
            main {{
                width: min(1020px, calc(100% - 32px));
                margin: 0 auto;
                padding: 36px 0 56px;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                gap: 20px;
                align-items: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--line);
                margin-bottom: 28px;
            }}
            header h1 {{ margin: 0; font-size: 1.5rem; }}
            .nav-links {{ display: flex; gap: 16px; }}
            .nav-links a {{ color: var(--accent); text-decoration: none; font-weight: 600; font-size: 0.9rem; }}
            h1,h2,h3,h4 {{ line-height: 1.2; margin: 1.5em 0 0.5em; }}
            a {{ color: var(--accent); text-decoration: none; font-weight: 700; }}
            p {{ margin: 0 0 12px; }}
            ul,ol {{ margin: 0 0 16px 22px; padding: 0; }}
            li {{ margin-bottom: 3px; }}
            pre {{
                overflow-x: auto;
                background: #111827;
                color: #f9fafb;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #1f2937;
                margin: 0 0 16px;
            }}
            code {{
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
                font-size: 0.9rem;
            }}
            p code, li code, td code {{
                background: #e8f0fa;
                color: #1a3150;
                border-radius: 4px;
                padding: 1px 5px;
                font-size: 0.86rem;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 0 0 16px;
                font-size: 0.9rem;
            }}
            th, td {{ border: 1px solid var(--line); padding: 7px 12px; text-align: left; }}
            th {{ background: #f0f4f8; font-weight: 600; }}
            tr:nth-child(even) td {{ background: #f8fafc; }}
            .doc {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 32px;
            }}
            .doc h2 {{ border-top: 1px solid var(--line); padding-top: 20px; margin-top: 32px; }}
            .doc h2:first-child {{ border-top: none; padding-top: 0; margin-top: 0; }}
            .muted {{ color: var(--muted); font-size: 0.9rem; }}
            /* flow diagram */
            .flow-diagram {{
                background: linear-gradient(135deg, #f0fdf4, #eff6ff);
                border: 1px solid #c7d8e8;
                border-radius: 12px;
                padding: 20px 24px;
                margin-bottom: 28px;
            }}
            .flow-title {{ font-weight: 700; color: #0f766e; margin-bottom: 14px; font-size: 0.95rem; }}
            .flow-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
            .flow-box {{
                background: #fff;
                border: 1px solid #d9e1ea;
                border-radius: 8px;
                padding: 10px 14px;
                flex: 1;
                min-width: 160px;
                font-size: 0.82rem;
                color: #334155;
            }}
            .flow-box small {{ display: block; color: #64748b; margin-top: 2px; }}
            .flow-query  {{ border-left: 3px solid #6366f1; }}
            .flow-scored {{ border-left: 3px solid #0ea5e9; }}
            .flow-evidence {{ border-left: 3px solid #0f766e; }}
            .flow-fusion {{ border-left: 3px solid #8b5cf6; flex: 2; }}
            .flow-gate   {{ border-left: 3px solid #ef4444; }}
            .flow-out    {{ border-left: 3px solid #22c55e; text-align: center; }}
            .flow-arrow {{ font-size: 1.3rem; color: #94a3b8; flex-shrink: 0; }}
            .chip-v {{ display:inline-block; background:#dbeafe; color:#1e40af; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; margin:2px; }}
            .chip-g {{ display:inline-block; background:#fce7f3; color:#9d174d; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; margin:2px; }}
            .chip-e {{ display:inline-block; background:#d1fae5; color:#065f46; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; margin:2px; }}
        </style>
    </head>
    <body>
        <main>
            <header>
                <div>
                    <h1>Schemat bazy grafowej</h1>
                    <div class="muted">Neo4j · RRF fuzja · scoring · Graph Explorer</div>
                </div>
                <div class="nav-links">
                    <a href="/schemat">Pełna architektura</a>
                    <a href="/graph_explorer">Graph Explorer</a>
                    <a href="/">Panel RAG</a>
                </div>
            </header>
            <article class="doc">
                {body}
            </article>
        </main>
    </body>
    </html>
    """)


@app.get("/graph_explorer")
def graph_explorer():
    return HTMLResponse("""
    <!doctype html>
    <html lang="pl">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Graph Explorer</title>
        <style>
            :root {
                color-scheme: light;
                --bg: #f6f7f9;
                --panel: #ffffff;
                --panel-soft: #f0f4f8;
                --text: #151922;
                --muted: #667085;
                --line: #d9e1ea;
                --accent: #0f766e;
                --accent-strong: #115e59;
                --accent-soft: #d9f3ee;
                --ink: #111827;
                --shadow: 0 16px 48px rgba(17, 24, 39, 0.08);
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                background:
                    linear-gradient(135deg, rgba(15, 118, 110, 0.10), transparent 34%),
                    linear-gradient(225deg, rgba(37, 99, 235, 0.08), transparent 30%),
                    var(--bg);
                color: var(--text);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            button,
            input {
                font: inherit;
            }

            .shell {
                width: min(1180px, calc(100% - 32px));
                margin: 0 auto;
                padding: 32px 0;
            }

            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 20px;
                margin-bottom: 24px;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 14px;
            }

            .mark {
                display: grid;
                width: 42px;
                height: 42px;
                place-items: center;
                border: 1px solid rgba(15, 118, 110, 0.22);
                border-radius: 8px;
                background: var(--accent-soft);
                color: var(--accent-strong);
                font-weight: 800;
            }

            h1,
            h2 {
                margin: 0;
                letter-spacing: 0;
            }

            h1 {
                font-size: clamp(28px, 4vw, 44px);
                line-height: 1.05;
            }

            h2 {
                font-size: 17px;
            }

            .hint {
                color: var(--muted);
                font-size: 13px;
            }

            .topbar-actions {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                flex-wrap: wrap;
                justify-content: flex-end;
            }

            .nav-link {
                display: inline-flex;
                align-items: center;
                min-height: 34px;
                padding: 0 12px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #ffffff;
                color: var(--accent-strong);
                font-size: 13px;
                font-weight: 750;
                text-decoration: none;
                white-space: nowrap;
            }

            .nav-link:hover {
                border-color: rgba(15, 118, 110, 0.35);
                background: var(--accent-soft);
            }

            .panel {
                border: 1px solid rgba(217, 225, 234, 0.9);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.88);
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
            }

            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 18px 20px;
                border-bottom: 1px solid var(--line);
            }

            .panel-body {
                padding: 20px;
            }

            .stack {
                display: grid;
                gap: 14px;
            }

            label {
                display: grid;
                gap: 7px;
                color: var(--muted);
                font-size: 13px;
                font-weight: 650;
            }

            input {
                width: 100%;
                min-height: 44px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #ffffff;
                color: var(--ink);
                outline: none;
                padding: 10px 12px;
                transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
            }

            input:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
            }

            .graph-controls {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 120px auto;
                gap: 12px;
                align-items: end;
            }

            .quick-actions,
            .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
            }

            .secondary {
                min-height: 42px;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0 15px;
                cursor: pointer;
                font-weight: 750;
                background: #ffffff;
                color: var(--ink);
                transition: transform 140ms ease, border-color 140ms ease, background 140ms ease, color 140ms ease;
            }

            .secondary:hover {
                transform: translateY(-1px);
                border-color: #a7b4c2;
                background: var(--panel-soft);
            }

            .secondary:disabled {
                cursor: wait;
                opacity: 0.7;
                transform: none;
            }

            .graph-status {
                min-height: 24px;
                color: var(--muted);
                font-size: 13px;
                font-weight: 650;
            }

            .graph-canvas {
                width: 100%;
                height: 720px;
                border: 1px solid #444;
                border-radius: 8px;
                background: #fbfdff;
            }

            @media (max-width: 860px) {
                .shell {
                    width: min(100% - 22px, 1180px);
                    padding: 18px 0;
                }

                .topbar {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .topbar-actions {
                    justify-content: flex-start;
                }

                .graph-controls {
                    grid-template-columns: 1fr;
                }

                .panel-header,
                .panel-body {
                    padding: 16px;
                }

                .graph-canvas {
                    height: 620px;
                }
            }
        </style>
    </head>
    <body>
        <main class="shell">
            <header class="topbar">
                <div class="brand">
                    <div class="mark">G</div>
                    <div>
                        <h1>Graph Explorer</h1>
                        <div class="hint">Neo4j relations explorer. Puste search = cała baza, max 500 relacji.</div>
                    </div>
                </div>
                <div class="topbar-actions">
                    <a class="nav-link" href="/schemat_grafu">Schemat bazy grafowej</a>
                    <a class="nav-link" href="/">Panel RAG</a>
                </div>
            </header>

            <section class="panel" id="graph-explorer">
                <div class="panel-header">
                    <h2>Graph Explorer</h2>
                    <span class="hint">Nodes = encje, edges = relacje</span>
                </div>
                <div class="panel-body">
                    <div class="stack">
                        <div class="graph-controls">
                            <label>
                                Search entity
                                <input id="graphEntity" placeholder="entity np. ODF, Ceph, OpenShift">
                            </label>
                            <label>
                                Limit
                                <input id="graphLimit" value="100" size="5">
                            </label>
                            <div class="actions">
                                <button class="secondary" type="button" id="loadGraphButton" onclick="loadGraph()">Load graph</button>
                            </div>
                        </div>
                        <div class="quick-actions">
                            <button class="secondary" type="button" onclick="document.getElementById('graphEntity').value='ODF'; loadGraph()">ODF</button>
                            <button class="secondary" type="button" onclick="document.getElementById('graphEntity').value='Ceph'; loadGraph()">Ceph</button>
                            <button class="secondary" type="button" onclick="document.getElementById('graphEntity').value='OpenShift'; loadGraph()">OpenShift</button>
                            <button class="secondary" type="button" onclick="document.getElementById('graphEntity').value=''; document.getElementById('graphLimit').value='500'; loadGraph()">Cała baza (max 500)</button>
                        </div>
                        <div id="graphStatus" class="graph-status"></div>
                        <div id="graph" class="graph-canvas"></div>
                    </div>
                </div>
            </section>
        </main>

        <script src="https://cdn.jsdelivr.net/npm/vis-network@10.0.2/standalone/umd/vis-network.min.js"></script>
        <script>
        const graphStatus = document.getElementById("graphStatus");
        const loadGraphButton = document.getElementById("loadGraphButton");
        let graphNetwork = null;

        async function loadGraph() {
            const entity = document.getElementById("graphEntity").value || "";
            const limitValue = parseInt(document.getElementById("graphLimit").value || "100", 10);
            const limit = Number.isFinite(limitValue) ? Math.min(Math.max(limitValue, 1), 500) : 100;
            const container = document.getElementById("graph");

            document.getElementById("graphLimit").value = String(limit);
            graphStatus.textContent = "Loading graph...";
            loadGraphButton.disabled = true;

            try {
                const resp = await fetch(`/graph_data?entity=${encodeURIComponent(entity)}&limit=${encodeURIComponent(limit)}`);
                const data = await resp.json();

                graphStatus.textContent = `Nodes: ${data.nodes.length}, Edges: ${data.edges.length}`;

                const nodes = new vis.DataSet(data.nodes);
                const edges = new vis.DataSet(data.edges);

                const options = {
                    nodes: {
                        shape: "dot",
                        size: 16,
                        font: {
                            size: 14
                        }
                    },
                    edges: {
                        arrows: {
                            to: { enabled: true, scaleFactor: 0.8 }
                        },
                        font: {
                            align: "middle"
                        },
                        smooth: {
                            type: "dynamic"
                        }
                    },
                    physics: {
                        enabled: true,
                        stabilization: true,
                        barnesHut: {
                            gravitationalConstant: -30000,
                            springLength: 140,
                            springConstant: 0.04
                        }
                    },
                    interaction: {
                        hover: true,
                        tooltipDelay: 100,
                        navigationButtons: true,
                        keyboard: true
                    }
                };

                if (graphNetwork) {
                    graphNetwork.destroy();
                }

                graphNetwork = new vis.Network(container, { nodes, edges }, options);
            } catch (error) {
                graphStatus.textContent = "Nie udało się załadować grafu.";
            } finally {
                loadGraphButton.disabled = false;
            }
        }
        </script>
    </body>
    </html>
    """)


@app.get("/")
def home():
    return HTMLResponse("""
    <!doctype html>
    <html lang="pl">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Hybrid RAG</title>
        <style>
            :root {
                color-scheme: light;
                --bg: #f6f7f9;
                --panel: #ffffff;
                --panel-soft: #f0f4f8;
                --text: #151922;
                --muted: #667085;
                --line: #d9e1ea;
                --accent: #0f766e;
                --accent-strong: #115e59;
                --accent-soft: #d9f3ee;
                --ink: #111827;
                --danger: #b42318;
                --shadow: 0 16px 48px rgba(17, 24, 39, 0.08);
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                background:
                    linear-gradient(135deg, rgba(15, 118, 110, 0.10), transparent 34%),
                    linear-gradient(225deg, rgba(37, 99, 235, 0.08), transparent 30%),
                    var(--bg);
                color: var(--text);
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            button,
            input,
            select,
            textarea {
                font: inherit;
            }

            .shell {
                width: min(1180px, calc(100% - 32px));
                margin: 0 auto;
                padding: 32px 0;
            }

            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 20px;
                margin-bottom: 24px;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 14px;
            }

            .mark {
                display: grid;
                width: 42px;
                height: 42px;
                place-items: center;
                border: 1px solid rgba(15, 118, 110, 0.22);
                border-radius: 8px;
                background: var(--accent-soft);
                color: var(--accent-strong);
                font-weight: 800;
            }

            h1,
            h2 {
                margin: 0;
                letter-spacing: 0;
            }

            h1 {
                font-size: clamp(28px, 4vw, 44px);
                line-height: 1.05;
            }

            h2 {
                font-size: 17px;
            }

            .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                min-height: 34px;
                padding: 0 12px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.72);
                color: var(--muted);
                font-size: 13px;
                white-space: nowrap;
            }

            .topbar-actions {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                flex-wrap: wrap;
                justify-content: flex-end;
            }

            .schema-link {
                display: inline-flex;
                align-items: center;
                min-height: 34px;
                padding: 0 12px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #ffffff;
                color: var(--accent-strong);
                font-size: 13px;
                font-weight: 750;
                text-decoration: none;
                white-space: nowrap;
            }

            .schema-link:hover {
                border-color: rgba(15, 118, 110, 0.35);
                background: var(--accent-soft);
            }

            .dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--accent);
            }

            .layout {
                display: grid;
                grid-template-columns: 1fr;
                gap: 18px;
                align-items: start;
                justify-items: center;
            }

            .layout > .panel {
                width: min(100%, 920px);
            }

            .panel {
                border: 1px solid rgba(217, 225, 234, 0.9);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.88);
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
            }

            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 18px 20px;
                border-bottom: 1px solid var(--line);
            }

            .panel-body {
                padding: 20px;
            }

            .stack {
                display: grid;
                gap: 14px;
            }

            .tools {
                width: 100%;
                display: grid;
                grid-template-columns: repeat(3, minmax(280px, 1fr));
                gap: 18px;
                align-items: start;
            }

            .tools > .panel {
                height: fit-content;
            }

            .tools > .panel:first-child {
                grid-column: 1 / -1;
            }

            label {
                display: grid;
                gap: 7px;
                color: var(--muted);
                font-size: 13px;
                font-weight: 650;
            }

            input,
            select,
            textarea {
                width: 100%;
                min-height: 44px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #ffffff;
                color: var(--ink);
                outline: none;
                padding: 10px 12px;
                transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
            }

            input[type="checkbox"] {
                width: auto;
                min-height: 0;
            }

            textarea {
                min-height: 150px;
                resize: vertical;
                line-height: 1.5;
            }

            input:focus,
            select:focus,
            textarea:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
            }

            .row {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
            }

            .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
                margin-top: 2px;
            }

            .field-with-action {
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 10px;
                align-items: end;
            }

            .field-with-action button {
                min-height: 44px;
                white-space: nowrap;
            }

            button {
                min-height: 42px;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 0 15px;
                cursor: pointer;
                font-weight: 750;
                transition: transform 140ms ease, border-color 140ms ease, background 140ms ease, color 140ms ease;
            }

            button:hover {
                transform: translateY(-1px);
            }

            button:disabled {
                cursor: wait;
                opacity: 0.7;
                transform: none;
            }

            .primary {
                background: var(--accent);
                color: #ffffff;
            }

            .primary:hover {
                background: var(--accent-strong);
            }

            .secondary {
                border-color: var(--line);
                background: #ffffff;
                color: var(--ink);
            }

            .secondary:hover {
                border-color: #a7b4c2;
                background: var(--panel-soft);
            }

            .file-input {
                padding: 8px;
            }

            .progress-wrap {
                display: none;
                gap: 8px;
            }

            .progress-wrap.active {
                display: grid;
            }

            .progress-meta {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                color: var(--muted);
                font-size: 13px;
                font-weight: 650;
            }

            .progress-track {
                height: 10px;
                overflow: hidden;
                border: 1px solid var(--line);
                border-radius: 999px;
                background: #eef2f6;
            }

            .progress-fill {
                width: 0%;
                height: 100%;
                border-radius: inherit;
                background: var(--accent);
                transition: width 220ms ease;
            }

            .answer {
                min-height: 340px;
                margin: 0;
                padding: 18px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: #0d1117;
                color: #eef6f4;
                white-space: pre-wrap;
                overflow: auto;
                line-height: 1.58;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size: 14px;
            }

            .answer:empty::before {
                content: "Odpowiedź pojawi się tutaj.";
                color: #8b949e;
            }

            .hint {
                color: var(--muted);
                font-size: 13px;
            }

            .loading .dot {
                animation: pulse 900ms ease-in-out infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 0.35; transform: scale(0.85); }
                50% { opacity: 1; transform: scale(1); }
            }

            @media (max-width: 860px) {
                .shell {
                    width: min(100% - 22px, 1180px);
                    padding: 18px 0;
                }

                .topbar {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .topbar-actions {
                    justify-content: flex-start;
                }

                .tools {
                    grid-template-columns: 1fr;
                }

                .row {
                    grid-template-columns: 1fr;
                }

                .panel-header,
                .panel-body {
                    padding: 16px;
                }
            }

            @media (min-width: 861px) and (max-width: 1100px) {
                .tools {
                    grid-template-columns: repeat(2, minmax(280px, 1fr));
                }
            }
        </style>
    </head>
    <body>
        <main class="shell">
            <header class="topbar">
                <div class="brand">
                    <div class="mark">R</div>
                    <div>
                        <h1>Hybrid RAG</h1>
                        <div class="hint">RAG, reranking i streaming w jednym panelu.</div>
                    </div>
                </div>
                <div class="topbar-actions">
                    <a class="schema-link" href="/schemat">Schemat działania</a>
                    <a class="schema-link" href="/schemat_grafu">Schemat bazy grafowej</a>
                    <a class="schema-link" href="/graph_explorer">Graph Explorer</a>
                    <div class="status" id="status"><span class="dot"></span><span id="statusText">Gotowy</span></div>
                </div>
            </header>

            <section class="layout">
                <div class="panel" hidden>
                    <div class="panel-header">
                        <h2>Zapytaj</h2>
                        <span class="hint">CPU / GPU / auto</span>
                    </div>
                    <div class="panel-body">
                        <form action="/ask_stream" method="post" id="form" class="stack">
                            <label>
                                Pytanie
                                <textarea name="query" id="query" placeholder="Zadaj pytanie"></textarea>
                            </label>

                            <label>
                                Tagi
                                <div class="field-with-action">
                                    <input name="tags" id="tags" placeholder="auto albo docker,k8s">
                                    <button class="secondary" type="button" id="tagQueryButton">Auto tagi</button>
                                </div>
                            </label>

                            <label>
                                Sesja
                                <input name="session_id" id="session_id" value="default">
                            </label>

                            <div class="row">
                                <label>
                                    Embedding
                                    <select id="embeddingBackend">
                                        <option value="auto">AUTO embedding (GPU → Local)</option>
                                        <option value="gpu">GPU embedding</option>
                                        <option value="local">Local Ollama</option>
                                    </select>
                                </label>

                                <label>
                                    Backend
                                    <select name="backend" id="backend">
                                        <option value="auto">AUTO (GPU → CPU → laptop)</option>
                                        <option value="gpu">GPU (OLLAMA_GPU_URL)</option>
                                        <option value="cpu">CPU (OLLAMA_CPU_URL)</option>
                                        <option value="laptop">Laptop (OLLAMA_LAPTOP_URL)</option>
                                    </select>
                                </label>

                                <label>
                                    Model
                                    <select name="model" id="model">
                                        <option value="qwen2.5-coder:1.5b">qwen2.5-coder:1.5b</option>
                                    </select>
                                </label>

                                <label>
                                    Tryb
                                    <select name="mode" id="mode">
                                        <option value="auto">AUTO</option>
                                        <option value="fast">FAST</option>
                                        <option value="smart">SMART</option>
                                    </select>
                                </label>
                            </div>

                            <div class="hint" id="embeddingBackendStatus">Embedding: ładowanie...</div>

                            <div class="actions">
                                <button class="primary" type="submit" id="askButton">Ask</button>
                            </div>
                        </form>
                    </div>
                </div>

                <aside class="tools">
                    <div class="panel" hidden>
                        <div class="panel-header">
                            <h2>Odpowiedź</h2>
                            <span class="hint" id="modeLabel">SSE</span>
                        </div>
                        <div class="panel-body">
                            <pre id="output" class="answer"></pre>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Dodaj notatkę</h2>
                        </div>
                        <div class="panel-body">
                            <form action="/upload_note" method="post" class="stack" id="uploadNoteForm">
                                <label>
                                    Tytuł
                                    <input name="title" placeholder="Tytuł notatki">
                                </label>
                                <label>
                                    Tagi
                                    <input name="tags" placeholder="docker,k8s">
                                </label>
                                <label>
                                    Treść
                                    <textarea name="text" placeholder="Treść notatki"></textarea>
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="submit" id="uploadNoteButton">Dodaj notatkę</button>
                                </div>
                                <div class="progress-wrap" id="noteProgress">
                                    <div class="progress-meta">
                                        <span id="noteProgressText">Gotowy</span>
                                        <span id="noteProgressPercent">0%</span>
                                    </div>
                                    <div class="progress-track">
                                        <div class="progress-fill" id="noteProgressFill"></div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Dodaj plik</h2>
                        </div>
                        <div class="panel-body">
                            <form action="/upload_file" method="post" enctype="multipart/form-data" class="stack" id="uploadFileForm">
                                <label>
                                    Plik
                                    <input class="file-input" type="file" name="file" accept=".pdf,.txt,.md,.markdown,.docx,.csv,.json,.yaml,.yml,.log">
                                </label>
                                <label>
                                    <input type="checkbox" name="smart" value="true">
                                    Smart import: LLM auto tagowanie
                                </label>
                                <label>
                                    <input type="checkbox" name="reindex" value="true">
                                    Reindex: usuń stare chunki tego pliku i zaimportuj od nowa
                                </label>
                                <label>
                                    <input type="checkbox" name="graph" value="true">
                                    Graf: wyciągnij relacje do Neo4j
                                </label>
                                <label>
                                    Backend
                                    <select name="backend" id="uploadFileBackend">
                                        <option value="auto">AUTO</option>
                                        <option value="gpu">GPU</option>
                                        <option value="cpu">CPU</option>
                                        <option value="laptop">Laptop</option>
                                    </select>
                                </label>
                                <label>
                                    Model
                                    <select name="model" id="uploadFileModel">
                                        <option value="qwen2.5-coder:1.5b">qwen2.5-coder:1.5b</option>
                                    </select>
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="submit" id="uploadFileButton">Dodaj plik</button>
                                </div>
                                <div class="progress-wrap" id="uploadProgress">
                                    <div class="progress-meta">
                                        <span id="uploadProgressText">Gotowy</span>
                                        <span id="uploadProgressPercent">0%</span>
                                    </div>
                                    <div class="progress-track">
                                        <div class="progress-fill" id="uploadProgressFill"></div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Import dokumentacji WWW</h2>
                        </div>
                        <div class="panel-body">
                            <form action="/import_website" method="post" class="stack" id="websiteImportForm">
                                <label>
                                    URL startowy
                                    <input name="url" type="url" placeholder="https://docs.example.com/product/">
                                </label>
                                <label>
                                    Limit stron
                                    <input name="max_pages" value="50">
                                </label>
                                <label>
                                    <input type="checkbox" name="smart" value="true">
                                    Smart import: LLM auto tagowanie
                                </label>
                                <label>
                                    <input type="checkbox" name="reindex" value="true">
                                    Reindex: usuń stare chunki tych URL-i i zaimportuj od nowa
                                </label>
                                <label>
                                    <input type="checkbox" name="graph" value="true">
                                    Graf: wyciągnij relacje do Neo4j
                                </label>
                                <label>
                                    Backend
                                    <select name="backend" id="websiteImportBackend">
                                        <option value="auto">AUTO</option>
                                        <option value="gpu">GPU</option>
                                        <option value="cpu">CPU</option>
                                        <option value="laptop">Laptop</option>
                                    </select>
                                </label>
                                <label>
                                    Model
                                    <select name="model" id="websiteImportModel">
                                        <option value="qwen2.5-coder:1.5b">qwen2.5-coder:1.5b</option>
                                    </select>
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="submit" id="websiteImportButton">Import WWW</button>
                                </div>
                                <div class="progress-wrap" id="websiteProgress">
                                    <div class="progress-meta">
                                        <span id="websiteProgressText">Gotowy</span>
                                        <span id="websiteProgressPercent">0%</span>
                                    </div>
                                    <div class="progress-track">
                                        <div class="progress-fill" id="websiteProgressFill"></div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Import repo kodu</h2>
                        </div>
                        <div class="panel-body">
                            <form class="stack" id="repoImportForm">
                                <label>
                                    Git URL
                                    <input name="git_url" placeholder="https://github.com/ansible/ansible-examples.git">
                                </label>
                                <label>
                                    Ścieżka lokalna
                                    <input name="path" placeholder="/space/repos/ansible-examples">
                                </label>
                                <label>
                                    Branch/tag
                                    <input name="ref" placeholder="main">
                                </label>
                                <label>
                                    Kolekcja
                                    <input name="collection" placeholder="auto z Git URL">
                                </label>
                                <label>
                                    Profil
                                    <select name="profile" id="repoImportProfile">
                                        <option value="smart">Smart DevOps / Platform / Networking</option>
                                        <option value="docs">Docs only</option>
                                        <option value="wide">Wide code + config</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                </label>
                                <label>
                                    Rozszerzenia
                                    <input name="extensions" id="repoImportExtensions" value=".md,.adoc,.txt,.yaml,.yml,.json,.tf,.sh,.py,.go,.j2,.rsc">
                                </label>
                                <label>
                                    Ścieżki
                                    <textarea name="include_paths" id="repoImportPaths" rows="4">.
docs
doc
Documentation
examples
charts
manifests
deploy
deployments
operator
runbooks
troubleshooting
hack
contrib
scripts
tools
cmd
config
install
man
samples</textarea>
                                </label>
                                <label>
                                    Limit plików
                                    <input name="max_files" id="repoImportMaxFiles" value="75">
                                </label>
                                <label>
                                    Limit rozmiaru pliku w bajtach
                                    <input name="max_file_bytes" id="repoImportMaxFileBytes" value="120000">
                                </label>
                                <label>
                                    <input type="checkbox" name="reindex" value="true">
                                    Reindex kolekcji repo
                                </label>
                                <label>
                                    <input type="checkbox" name="graph" value="true">
                                    Graf: indeksuj README/docs/configi do Neo4j
                                </label>
                                <label>
                                    Backend grafu
                                    <select name="backend" id="repoImportBackend">
                                        <option value="auto">AUTO</option>
                                        <option value="gpu">GPU</option>
                                        <option value="cpu">CPU</option>
                                        <option value="laptop">Laptop</option>
                                    </select>
                                </label>
                                <label>
                                    Model grafu
                                    <select name="model" id="repoImportModel">
                                        <option value="qwen2.5-coder:1.5b">qwen2.5-coder:1.5b</option>
                                    </select>
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="submit" id="repoImportButton">Import repo</button>
                                </div>
                                <pre id="repoImportOutput" class="answer"></pre>
                            </form>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Jakość retrievalu</h2>
                        </div>
                        <div class="panel-body">
                            <form class="stack" id="qualityForm">
                                <label>
                                    Test query
                                    <input name="query" value="Terraform provider backend state">
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="button" id="telemetryButton">Telemetry</button>
                                    <button class="secondary" type="button" id="goldenTestButton">Golden smoke</button>
                                </div>
                                <pre id="qualityOutput" class="answer"></pre>
                            </form>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Import historii AI</h2>
                        </div>
                        <div class="panel-body">
                            <form action="/import_chatgpt" method="post" enctype="multipart/form-data" class="stack" id="chatgptImportForm">
                                <label>
                                    JSON/ZIP eksportu
                                    <input class="file-input" type="file" name="file" accept=".json,.zip,application/json,application/zip" multiple>
                                </label>
                                <label>
                                    <input type="checkbox" name="smart" value="true">
                                    Smart import: LLM filtr + auto tagowanie
                                </label>
                                <label>
                                    Backend
                                    <select name="backend" id="chatgptImportBackend">
                                        <option value="auto">AUTO</option>
                                        <option value="gpu">GPU</option>
                                        <option value="cpu">CPU</option>
                                        <option value="laptop">Laptop</option>
                                    </select>
                                </label>
                                <label>
                                    Model
                                    <select name="model" id="chatgptImportModel">
                                        <option value="qwen2.5-coder:1.5b">qwen2.5-coder:1.5b</option>
                                    </select>
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="submit" id="chatgptImportButton">Import</button>
                                </div>
                                <div class="progress-wrap" id="chatgptProgress">
                                    <div class="progress-meta">
                                        <span id="chatgptProgressText">Gotowy</span>
                                        <span id="chatgptProgressPercent">0%</span>
                                    </div>
                                    <div class="progress-track">
                                        <div class="progress-fill" id="chatgptProgressFill"></div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-header">
                            <h2>Graph Cleanup</h2>
                        </div>
                        <div class="panel-body">
                            <form action="/graph_cleanup" method="post" class="stack" id="graphCleanupForm">
                                <label>
                                    Sample limit
                                    <input name="sample_limit" value="20">
                                </label>
                                <label>
                                    <input type="checkbox" name="strict_relations" value="true" checked>
                                    Strict relations: usuń także relacje spoza whitelisty
                                </label>
                                <div class="actions">
                                    <button class="secondary" type="button" id="graphCleanupDryRunButton">Dry run</button>
                                    <button class="secondary" type="button" id="graphCleanupApplyButton">Apply cleanup</button>
                                </div>
                                <p class="muted">Dry run niczego nie kasuje. Apply usuwa śmieciowe relacje, przepisuje aliasy i kasuje osierocone encje.</p>
                                <div class="progress-wrap" id="graphCleanupProgress">
                                    <div class="progress-meta">
                                        <span id="graphCleanupProgressText">Gotowy</span>
                                        <span id="graphCleanupProgressPercent">0%</span>
                                    </div>
                                    <div class="progress-track">
                                        <div class="progress-fill" id="graphCleanupProgressFill"></div>
                                    </div>
                                </div>
                                <pre id="graphCleanupOutput" class="answer"></pre>
                            </form>
                        </div>
                    </div>
                </aside>
            </section>
        </main>

        <script>
        const fallbackModels = ["qwen2.5-coder:1.5b", "gemma3:1b", "qwen2.5-coder:0.5b", "qwen2.5-coder:3b", "llama3.2:latest", "qwen2.5-coder:7b", "mistral", "codellama"];
        const form = document.getElementById("form");
        const output = document.getElementById("output");
        const statusBox = document.getElementById("status");
        const statusText = document.getElementById("statusText");
        const modeLabel = document.getElementById("modeLabel");
        const askButton = document.getElementById("askButton");
        const tagQueryButton = document.getElementById("tagQueryButton");
        const embeddingBackendSelect = document.getElementById("embeddingBackend");
        const embeddingBackendStatus = document.getElementById("embeddingBackendStatus");
        const uploadNoteForm = document.getElementById("uploadNoteForm");
        const uploadNoteButton = document.getElementById("uploadNoteButton");
        const uploadFileForm = document.getElementById("uploadFileForm");
        const uploadFileButton = document.getElementById("uploadFileButton");
        const websiteImportForm = document.getElementById("websiteImportForm");
        const websiteImportButton = document.getElementById("websiteImportButton");
        const repoImportForm = document.getElementById("repoImportForm");
        const repoImportButton = document.getElementById("repoImportButton");
        const repoImportOutput = document.getElementById("repoImportOutput");
        const repoImportProfile = document.getElementById("repoImportProfile");
        const repoImportExtensions = document.getElementById("repoImportExtensions");
        const repoImportPaths = document.getElementById("repoImportPaths");
        const repoImportMaxFiles = document.getElementById("repoImportMaxFiles");
        const repoImportMaxFileBytes = document.getElementById("repoImportMaxFileBytes");
        const repoImportBackend = document.getElementById("repoImportBackend");
        const repoImportModel = document.getElementById("repoImportModel");
        const qualityForm = document.getElementById("qualityForm");
        const qualityOutput = document.getElementById("qualityOutput");
        const telemetryButton = document.getElementById("telemetryButton");
        const goldenTestButton = document.getElementById("goldenTestButton");
        const graphCleanupForm = document.getElementById("graphCleanupForm");
        const graphCleanupDryRunButton = document.getElementById("graphCleanupDryRunButton");
        const graphCleanupApplyButton = document.getElementById("graphCleanupApplyButton");
        const graphCleanupOutput = document.getElementById("graphCleanupOutput");
        const chatgptImportForm = document.getElementById("chatgptImportForm");
        const chatgptImportButton = document.getElementById("chatgptImportButton");
        const noteProgress = document.getElementById("noteProgress");
        const noteProgressText = document.getElementById("noteProgressText");
        const noteProgressPercent = document.getElementById("noteProgressPercent");
        const noteProgressFill = document.getElementById("noteProgressFill");
        const uploadProgress = document.getElementById("uploadProgress");
        const uploadProgressText = document.getElementById("uploadProgressText");
        const uploadProgressPercent = document.getElementById("uploadProgressPercent");
        const uploadProgressFill = document.getElementById("uploadProgressFill");
        const websiteProgress = document.getElementById("websiteProgress");
        const websiteProgressText = document.getElementById("websiteProgressText");
        const websiteProgressPercent = document.getElementById("websiteProgressPercent");
        const websiteProgressFill = document.getElementById("websiteProgressFill");
        const graphCleanupProgress = document.getElementById("graphCleanupProgress");
        const graphCleanupProgressText = document.getElementById("graphCleanupProgressText");
        const graphCleanupProgressPercent = document.getElementById("graphCleanupProgressPercent");
        const graphCleanupProgressFill = document.getElementById("graphCleanupProgressFill");
        const chatgptProgress = document.getElementById("chatgptProgress");
        const chatgptProgressText = document.getElementById("chatgptProgressText");
        const chatgptProgressPercent = document.getElementById("chatgptProgressPercent");
        const chatgptProgressFill = document.getElementById("chatgptProgressFill");
        let backendStatus = null;

        function setStatus(text, loading = false) {
            statusText.textContent = text;
            statusBox.classList.toggle("loading", loading);
            askButton.disabled = loading;
        }

        async function tagCurrentQuery() {
            const query = document.getElementById("query").value.trim();
            const backend = document.getElementById("backend").value;
            const model = document.getElementById("model").value;
            const tagsInput = document.getElementById("tags");

            if (!query) {
                return [];
            }

            tagQueryButton.disabled = true;
            tagQueryButton.textContent = "Taguję";

            try {
                const response = await fetch("/tag_query", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        query: query,
                        backend: backend,
                        model: model
                    })
                });
                const data = await response.json();
                const tags = data.tags || [];

                if (tags.length) {
                    tagsInput.value = tags.join(",");
                }

                return tags;
            } catch (error) {
                return [];
            } finally {
                tagQueryButton.disabled = false;
                tagQueryButton.textContent = "Auto tagi";
            }
        }

        function applyBackendStatus(select) {
            if (!backendStatus) {
                return;
            }

            const gpuOption = select.querySelector('option[value="gpu"]');
            const cpuOption = select.querySelector('option[value="cpu"]');
            const laptopOption = select.querySelector('option[value="laptop"]');

            if (gpuOption) {
                gpuOption.disabled = false;
                gpuOption.textContent = backendStatus.gpu.available ? "GPU" : "GPU (offline)";
            }

            if (cpuOption) {
                cpuOption.disabled = false;
                cpuOption.textContent = backendStatus.cpu.available ? "CPU" : "CPU (offline)";
            }

            if (laptopOption) {
                laptopOption.disabled = false;
                laptopOption.textContent = backendStatus.laptop.available ? "Laptop" : "Laptop (offline)";
            }
        }

        async function loadBackendStatus() {
            try {
                const response = await fetch("/backend_status");
                backendStatus = await response.json();

                document.querySelectorAll("select[name='backend']").forEach(applyBackendStatus);
            } catch (error) {
                backendStatus = null;
            }
        }

        function renderEmbeddingBackendStatus(data) {
            if (!data) {
                embeddingBackendStatus.textContent = "Embedding: status niedostępny";
                return;
            }

            embeddingBackendSelect.value = data.mode || data.backend || "auto";
            const backendLabel = data.backend === "local" ? "Local Ollama" : "GPU";
            const modeLabel = data.mode === "auto" ? "AUTO" : backendLabel;
            const gpuLabel = data.gpu_available ? "GPU online" : "GPU offline";
            const localLabel = data.local_available ? "Local online" : "Local offline";
            embeddingBackendStatus.textContent = `Embedding: ${modeLabel} → ${backendLabel} (${data.model}) ${data.url} · ${gpuLabel}, ${localLabel}`;
        }

        async function loadEmbeddingBackend() {
            try {
                const response = await fetch("/embedding_backend");
                const data = await response.json();
                renderEmbeddingBackendStatus(data);
            } catch (error) {
                renderEmbeddingBackendStatus(null);
            }
        }

        embeddingBackendSelect.addEventListener("change", async function() {
            embeddingBackendSelect.disabled = true;
            embeddingBackendStatus.textContent = "Embedding: zapisuję...";

            try {
                const response = await fetch("/embedding_backend", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        backend: embeddingBackendSelect.value
                    })
                });
                const data = await response.json();
                renderEmbeddingBackendStatus(data);
            } catch (error) {
                embeddingBackendStatus.textContent = "Embedding: błąd zapisu";
            } finally {
                embeddingBackendSelect.disabled = false;
            }
        });

        async function populateModels(backend, modelSelect) {
            const current = modelSelect.value;
            const backendLabels = {
                auto: "AUTO",
                gpu: "GPU",
                cpu: "CPU",
                laptop: "Laptop"
            };

            modelSelect.innerHTML = '<option value="">Ładowanie...</option>';

            try {
                const response = await fetch(`/models?backend=${encodeURIComponent(backend)}`);
                const data = await response.json();
                const models = data.models && data.models.length ? data.models : [];

                modelSelect.innerHTML = "";

                if (!models.length) {
                    const option = document.createElement("option");
                    option.value = "";
                    option.textContent = `${backendLabels[backend] || backend} offline albo brak modeli`;
                    option.disabled = true;
                    option.selected = true;
                    modelSelect.appendChild(option);
                    return;
                }

                for (const model of models) {
                    const option = document.createElement("option");
                    option.value = model;
                    option.textContent = model;
                    option.selected = model === current;
                    modelSelect.appendChild(option);
                }
            } catch (error) {
                modelSelect.innerHTML = "";

                const option = document.createElement("option");
                option.value = "";
                option.textContent = "Nie udało się pobrać modeli";
                option.disabled = true;
                option.selected = true;
                modelSelect.appendChild(option);
            }
        }

        async function loadModels() {
            const backend = document.getElementById("backend").value;
            const modelSelect = document.getElementById("model");
            await populateModels(backend, modelSelect);
        }

        async function loadSmartUploadModels() {
            const backend = document.getElementById("uploadFileBackend").value;
            const modelSelect = document.getElementById("uploadFileModel");
            await populateModels(backend, modelSelect);
        }

        async function loadWebsiteImportModels() {
            const backend = document.getElementById("websiteImportBackend").value;
            const modelSelect = document.getElementById("websiteImportModel");
            await populateModels(backend, modelSelect);
        }

        async function loadChatgptImportModels() {
            const backend = document.getElementById("chatgptImportBackend").value;
            const modelSelect = document.getElementById("chatgptImportModel");
            await populateModels(backend, modelSelect);
        }

        async function loadRepoImportModels() {
            const backend = document.getElementById("repoImportBackend").value;
            const modelSelect = document.getElementById("repoImportModel");
            await populateModels(backend, modelSelect);
        }

        document.getElementById("backend").addEventListener("change", async function() {
            applyBackendStatus(this);
            await loadModels();
        });
        document.getElementById("uploadFileBackend").addEventListener("change", async function() {
            applyBackendStatus(this);
            await loadSmartUploadModels();
        });
        document.getElementById("websiteImportBackend").addEventListener("change", async function() {
            applyBackendStatus(this);
            await loadWebsiteImportModels();
        });
        document.getElementById("chatgptImportBackend").addEventListener("change", async function() {
            applyBackendStatus(this);
            await loadChatgptImportModels();
        });
        document.getElementById("repoImportBackend").addEventListener("change", async function() {
            applyBackendStatus(this);
            await loadRepoImportModels();
        });
        loadBackendStatus().then(() => {
            loadModels();
            loadSmartUploadModels();
            loadWebsiteImportModels();
            loadChatgptImportModels();
            loadRepoImportModels();
        });
        loadEmbeddingBackend();

        tagQueryButton.addEventListener("click", tagCurrentQuery);

        function setProgress(progress, fill, percentLabel, textLabel, percent, text) {
            const clamped = Math.max(0, Math.min(100, Math.round(percent)));
            progress.classList.add("active");
            fill.style.width = `${clamped}%`;
            percentLabel.textContent = `${clamped}%`;
            textLabel.textContent = text;
        }

        function decodeZipName(bytes, utf8) {
            if (utf8) {
                return new TextDecoder("utf-8").decode(bytes);
            }

            return Array.from(bytes, function(byte) {
                return String.fromCharCode(byte);
            }).join("");
        }

        async function inflateRawZipEntry(bytes) {
            if (!("DecompressionStream" in window)) {
                throw new Error("Przeglądarka nie obsługuje lokalnego rozpakowania ZIP. Wybierz pojedyncze JSON-y z eksportu.");
            }

            const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
            return new Response(stream).blob();
        }

        async function extractJsonFilesFromZip(file) {
            const buffer = await file.arrayBuffer();
            const view = new DataView(buffer);
            const bytes = new Uint8Array(buffer);
            const minEocd = 22;
            const maxComment = 65535;
            const start = Math.max(0, bytes.length - minEocd - maxComment);
            let eocd = -1;

            for (let i = bytes.length - minEocd; i >= start; i -= 1) {
                if (view.getUint32(i, true) === 0x06054b50) {
                    eocd = i;
                    break;
                }
            }

            if (eocd === -1) {
                throw new Error(`Niepoprawny ZIP: ${file.name}`);
            }

            const entries = view.getUint16(eocd + 10, true);
            let offset = view.getUint32(eocd + 16, true);
            const jsonFiles = [];

            for (let i = 0; i < entries; i += 1) {
                if (view.getUint32(offset, true) !== 0x02014b50) {
                    throw new Error(`Niepoprawna struktura ZIP: ${file.name}`);
                }

                const flags = view.getUint16(offset + 8, true);
                const method = view.getUint16(offset + 10, true);
                const compressedSize = view.getUint32(offset + 20, true);
                const nameLength = view.getUint16(offset + 28, true);
                const extraLength = view.getUint16(offset + 30, true);
                const commentLength = view.getUint16(offset + 32, true);
                const localOffset = view.getUint32(offset + 42, true);
                const rawName = bytes.slice(offset + 46, offset + 46 + nameLength);
                const name = decodeZipName(rawName, Boolean(flags & 0x0800));

                if (name.toLowerCase().endsWith(".json")) {
                    if (view.getUint32(localOffset, true) !== 0x04034b50) {
                        throw new Error(`Niepoprawny wpis ZIP: ${name}`);
                    }

                    const localNameLength = view.getUint16(localOffset + 26, true);
                    const localExtraLength = view.getUint16(localOffset + 28, true);
                    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
                    const data = bytes.slice(dataStart, dataStart + compressedSize);
                    let blob = null;

                    if (method === 0) {
                        blob = new Blob([data], {type: "application/json"});
                    } else if (method === 8) {
                        blob = await inflateRawZipEntry(data);
                    } else {
                        throw new Error(`Nieobsługiwana kompresja ZIP dla ${name}`);
                    }

                    jsonFiles.push(new File([blob], name.replace(/^.*[\\/]/, "") || "export.json", {type: "application/json"}));
                }

                offset += 46 + nameLength + extraLength + commentLength;
            }

            return jsonFiles;
        }

        async function prepareChatgptImportData(formData) {
            const prepared = new FormData();
            const files = formData.getAll("file");
            let jsonCount = 0;

            for (const [key, value] of formData.entries()) {
                if (key !== "file") {
                    prepared.append(key, value);
                }
            }

            for (const file of files) {
                if (!file || !file.name) {
                    continue;
                }

                if (file.name.toLowerCase().endsWith(".zip")) {
                    const jsonFiles = await extractJsonFilesFromZip(file);

                    for (const jsonFile of jsonFiles) {
                        prepared.append("file", jsonFile);
                        jsonCount += 1;
                    }
                } else {
                    prepared.append("file", file);
                    jsonCount += 1;
                }
            }

            if (jsonCount === 0) {
                throw new Error("ZIP nie zawiera plików JSON historii.");
            }

            return prepared;
        }

        async function submitPanelForm(config) {
            const {
                form,
                button,
                progress,
                fill,
                percentLabel,
                textLabel,
                url,
                sendingText,
                processingText,
                doneText,
                requireFile,
                prepareFormData,
                preparingText,
                pollStatusUrl,
                onDone
            } = config;

            let formData = new FormData(form);

            if (requireFile) {
                const file = formData.get("file");

                if (!file || !file.name) {
                    setProgress(progress, fill, percentLabel, textLabel, 0, "Wybierz plik");
                    return;
                }
            }

            const request = new XMLHttpRequest();
            let simulated = 35;
            let timer = null;

            button.disabled = true;

            if (prepareFormData) {
                setProgress(progress, fill, percentLabel, textLabel, 0, preparingText || "Przygotowuję pliki");

                try {
                    formData = await prepareFormData(formData);
                } catch (error) {
                    button.disabled = false;
                    setProgress(progress, fill, percentLabel, textLabel, 0, error.message || "Błąd przygotowania plików");
                    return;
                }
            }

            setProgress(progress, fill, percentLabel, textLabel, 0, sendingText);

            request.upload.addEventListener("progress", function(event) {
                if (!event.lengthComputable) {
                    return;
                }

                const uploadPercent = (event.loaded / event.total) * 35;
                simulated = Math.max(simulated, uploadPercent);
                setProgress(progress, fill, percentLabel, textLabel, uploadPercent, sendingText);
            });

            request.upload.addEventListener("load", function() {
                setProgress(progress, fill, percentLabel, textLabel, 35, processingText);

                timer = window.setInterval(function() {
                    if (simulated < 92) {
                        simulated += simulated < 70 ? 3 : 1;
                        setProgress(progress, fill, percentLabel, textLabel, simulated, processingText);
                    }
                }, 900);
            });

            request.addEventListener("load", function() {
                if (timer) {
                    window.clearInterval(timer);
                }

                if (request.status >= 200 && request.status < 300) {
                    let data = {};

                    try {
                        data = JSON.parse(request.responseText);
                    } catch (error) {
                        data = {};
                    }

                    if (pollStatusUrl && data.job_id) {
                        setProgress(progress, fill, percentLabel, textLabel, 40, data.message || processingText);
                        pollPanelJob({
                            jobId: data.job_id,
                            button,
                            progress,
                            fill,
                            percentLabel,
                            textLabel,
                            pollStatusUrl,
                            doneText,
                            onDone
                        });
                        return;
                    }

                    button.disabled = false;
                    const message = onDone ? onDone(data) : doneText;
                    setProgress(progress, fill, percentLabel, textLabel, 100, message || doneText);
                } else {
                    button.disabled = false;
                    let message = `Błąd importu (${request.status})`;

                    try {
                        const errorData = JSON.parse(request.responseText);

                        if (errorData.detail) {
                            message = errorData.detail;
                        }
                    } catch (error) {
                        if (request.responseText) {
                            message = request.responseText.slice(0, 120);
                        }
                    }

                    setProgress(progress, fill, percentLabel, textLabel, 0, message);
                }
            });

            request.addEventListener("error", function() {
                if (timer) {
                    window.clearInterval(timer);
                }

                button.disabled = false;
                setProgress(progress, fill, percentLabel, textLabel, 0, "Błąd połączenia");
            });

            request.open("POST", url);
            request.send(formData);
        }

        async function pollPanelJob(config) {
            const {
                jobId,
                button,
                progress,
                fill,
                percentLabel,
                textLabel,
                pollStatusUrl,
                doneText,
                onDone
            } = config;

            try {
                const response = await fetch(`${pollStatusUrl}/${jobId}`);

                if (!response.ok) {
                    throw new Error(`Błąd statusu (${response.status})`);
                }

                const data = await response.json();
                const fileInfo = data.total_files ? ` ${data.files || 0}/${data.total_files} JSON` : "";
                const pairInfo = Number.isFinite(data.pairs) ? ` Q/A: ${data.pairs}` : "";
                const skippedInfo = data.skipped_files ? ` Pominięto: ${data.skipped_files}` : "";
                const currentInfo = data.current_file ? ` (${data.current_file})` : "";
                const chunkInfo = Number.isFinite(data.chunks) ? ` Chunks: ${data.chunks}` : "";
                const insertedInfo = Number.isFinite(data.inserted) ? ` Nowe: ${data.inserted}` : "";
                const graphInfo = data.graph && Number.isFinite(data.graph.created) ? ` Graf: ${data.graph.created} nowych/${data.graph.skipped || 0} pominiętych` : "";
                const message = `${data.message || "Import"}${fileInfo}${pairInfo}${skippedInfo}${currentInfo}${chunkInfo}${insertedInfo}${graphInfo}`;

                if (data.status === "done") {
                    button.disabled = false;
                    const doneMessage = onDone ? onDone(data.result || data) : doneText;
                    setProgress(progress, fill, percentLabel, textLabel, 100, doneMessage || message || doneText);
                    return;
                }

                if (data.status === "error") {
                    button.disabled = false;
                    setProgress(progress, fill, percentLabel, textLabel, 0, data.message || "Błąd importu");
                    return;
                }

                setProgress(progress, fill, percentLabel, textLabel, data.progress || 5, message);
                window.setTimeout(function() {
                    pollPanelJob(config);
                }, 2500);
            } catch (error) {
                button.disabled = false;
                setProgress(progress, fill, percentLabel, textLabel, 0, error.message || "Błąd statusu importu");
            }
        }

        uploadNoteForm.addEventListener("submit", function(event) {
            event.preventDefault();

            submitPanelForm({
                form: uploadNoteForm,
                button: uploadNoteButton,
                progress: noteProgress,
                fill: noteProgressFill,
                percentLabel: noteProgressPercent,
                textLabel: noteProgressText,
                url: "/upload_note",
                sendingText: "Wysyłam notatkę",
                processingText: "Ingest notatki",
                doneText: "Notatka dodana",
                requireFile: false,
                onDone: function(data) {
                    return data.source ? `Notatka dodana: ${data.source}` : "Notatka dodana";
                }
            });
        });

        uploadFileForm.addEventListener("submit", function(event) {
            event.preventDefault();

            submitPanelForm({
                form: uploadFileForm,
                button: uploadFileButton,
                progress: uploadProgress,
                fill: uploadProgressFill,
                percentLabel: uploadProgressPercent,
                textLabel: uploadProgressText,
                url: "/upload_file_async",
                sendingText: "Wysyłam plik",
                processingText: "Czekam na start importu",
                doneText: "Plik dodany",
                requireFile: true,
                pollStatusUrl: "/upload_file/status",
                onDone: function(data) {
                    if (data.status === "duplicate") {
                        return `Już jest: ${data.source} (${data.existing_chunks} chunków). Zaznacz Reindex, żeby odświeżyć.`;
                    }

                    const inserted = data.ingest ? data.ingest.inserted : 0;
                    const duplicates = data.ingest ? data.ingest.duplicates : 0;
                    const invalid = data.ingest ? data.ingest.invalid : 0;
                    const deleted = data.deleted_chunks || 0;
                    const tags = data.tags && data.tags.length ? ` Tagi: ${data.tags.join(", ")}` : "";
                    const reindex = data.reindex ? ` Usunięto stare: ${deleted}.` : "";
                    const graph = data.graph_stats ? ` Graf: ${data.graph_stats.created} nowych, ${data.graph_stats.existing} istniejących, ${data.graph_stats.skipped} pominiętych.` : "";

                    return `Plik dodany. Nowe chunki: ${inserted}, duplikaty: ${duplicates}, nieważne: ${invalid}.${graph}${reindex}${tags}`;
                }
            });
        });

        websiteImportForm.addEventListener("submit", function(event) {
            event.preventDefault();

            const urlInput = websiteImportForm.querySelector('input[name="url"]');
            if (!urlInput.value.trim()) {
                setProgress(websiteProgress, websiteProgressFill, websiteProgressPercent, websiteProgressText, 0, "Podaj URL dokumentacji");
                return;
            }

            submitPanelForm({
                form: websiteImportForm,
                button: websiteImportButton,
                progress: websiteProgress,
                fill: websiteProgressFill,
                percentLabel: websiteProgressPercent,
                textLabel: websiteProgressText,
                url: "/import_website",
                sendingText: "Wysyłam URL",
                processingText: "Crawler czyta dokumentację",
                doneText: "Dokumentacja zaimportowana",
                requireFile: false,
                onDone: function(data) {
                    const totals = data.totals || {};
                    const inserted = totals.inserted || 0;
                    const duplicates = totals.duplicates || 0;
                    const graphCreated = totals.graph_created || 0;
                    const graphText = data.graph ? ` Graf: ${graphCreated} nowych relacji.` : "";
                    const duplicateText = data.duplicate_pages ? ` Pominięte duplikaty URL: ${data.duplicate_pages}.` : "";
                    const errorText = data.errors && data.errors.length ? ` Błędy stron: ${data.errors.length}.` : "";

                    return `WWW gotowe. Odwiedzone: ${data.visited || 0}, zaimportowane strony: ${data.pages_imported || 0}, nowe chunki: ${inserted}, duplikaty chunków: ${duplicates}.${graphText}${duplicateText}${errorText}`;
                }
            });
        });

        const repoImportProfiles = {
            smart: {
                extensions: ".md,.adoc,.txt,.yaml,.yml,.json,.tf,.sh,.py,.go,.j2,.rsc",
                includePaths: [
                    ".",
                    "docs",
                    "doc",
                    "Documentation",
                    "examples",
                    "charts",
                    "manifests",
                    "deploy",
                    "deployments",
                    "operator",
                    "runbooks",
                    "troubleshooting",
                    "hack",
                    "contrib",
                    "scripts",
                    "tools",
                    "cmd",
                    "config",
                    "install",
                    "man",
                    "samples"
                ],
                maxFiles: 75,
                maxFileBytes: 120000
            },
            docs: {
                extensions: ".md,.adoc,.txt,.yaml,.yml,.j2,.rsc",
                includePaths: [
                    ".",
                    "docs",
                    "doc",
                    "Documentation",
                    "examples",
                    "runbooks",
                    "troubleshooting",
                    "man",
                    "samples"
                ],
                maxFiles: 100,
                maxFileBytes: 180000
            },
            wide: {
                extensions: ".md,.adoc,.txt,.yaml,.yml,.json,.tf,.sh,.bash,.py,.go,.sql,.toml,.ini,.conf,.env,.j2,.rsc,.ps1,.bat,.cmd",
                includePaths: [],
                maxFiles: 250,
                maxFileBytes: 250000
            }
        };

        function repoLines(value) {
            return String(value || "")
                .split(/[\\n,]+/)
                .map(function(item) { return item.trim(); })
                .filter(Boolean);
        }

        function applyRepoImportProfile(profileName) {
            const profile = repoImportProfiles[profileName];

            if (!profile) {
                return;
            }

            repoImportExtensions.value = profile.extensions;
            repoImportPaths.value = profile.includePaths.join("\\n");
            repoImportMaxFiles.value = String(profile.maxFiles);
            repoImportMaxFileBytes.value = String(profile.maxFileBytes);
        }

        repoImportProfile.addEventListener("change", function() {
            applyRepoImportProfile(repoImportProfile.value);
        });

        [repoImportExtensions, repoImportPaths, repoImportMaxFiles, repoImportMaxFileBytes].forEach(function(input) {
            input.addEventListener("input", function() {
                repoImportProfile.value = "custom";
            });
        });

        repoImportForm.addEventListener("submit", async function(event) {
            event.preventDefault();

            const formData = new FormData(repoImportForm);
            const extensions = String(formData.get("extensions") || "")
                .split(",")
                .map(function(item) { return item.trim(); })
                .filter(Boolean);
            const includePaths = repoLines(formData.get("include_paths"));
            const rawCollection = String(formData.get("collection") || "").trim();
            const maxFiles = parseInt(formData.get("max_files") || "75", 10) || 75;
            const maxFileBytes = parseInt(formData.get("max_file_bytes") || "120000", 10) || 120000;
            const payload = {
                url: String(formData.get("git_url") || "").trim(),
                path: String(formData.get("path") || "").trim(),
                collection: rawCollection,
                extensions: extensions,
                include_paths: includePaths,
                max_files: Math.max(1, Math.min(maxFiles, 5000)),
                max_file_bytes: Math.max(20000, Math.min(maxFileBytes, 2000000)),
                reindex: formData.get("reindex") === "true",
                graph: formData.get("graph") === "true",
                backend: String(formData.get("backend") || "auto"),
                model: String(formData.get("model") || "qwen2.5-coder:1.5b"),
                ref: String(formData.get("ref") || "").trim() || null
            };

            if (!payload.url && !payload.path) {
                repoImportOutput.textContent = "Podaj Git URL albo lokalną ścieżkę repo.";
                return;
            }

            repoImportButton.disabled = true;
            repoImportOutput.textContent = payload.url ? "Klonuję i importuję repo..." : "Importuję repo...";

            try {
                const endpoint = payload.url ? "/import_git" : "/import_repo";
                const body = payload.url
                    ? {
                        url: payload.url,
                        collection: payload.collection,
                        extensions: payload.extensions,
                        include_paths: payload.include_paths,
                        max_files: payload.max_files,
                        max_file_bytes: payload.max_file_bytes,
                        reindex: payload.reindex,
                        graph: payload.graph,
                        backend: payload.backend,
                        model: payload.model,
                        ref: payload.ref
                    }
                    : {
                        path: payload.path,
                        collection: payload.collection || "code",
                        extensions: payload.extensions,
                        include_paths: payload.include_paths,
                        max_files: payload.max_files,
                        max_file_bytes: payload.max_file_bytes,
                        reindex: payload.reindex,
                        graph: payload.graph,
                        backend: payload.backend,
                        model: payload.model
                    };
                const response = await fetch(endpoint, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(body)
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || "Import repo failed");
                }

                repoImportOutput.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                repoImportOutput.textContent = String(error);
            } finally {
                repoImportButton.disabled = false;
            }
        });

        telemetryButton.addEventListener("click", async function() {
            telemetryButton.disabled = true;
            qualityOutput.textContent = "Ładuję telemetrykę...";

            try {
                const response = await fetch("/retrieval_telemetry?limit=20");
                const data = await response.json();
                qualityOutput.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                qualityOutput.textContent = String(error);
            } finally {
                telemetryButton.disabled = false;
            }
        });

        goldenTestButton.addEventListener("click", async function() {
            goldenTestButton.disabled = true;
            qualityOutput.textContent = "Odpalam golden smoke...";

            const query = qualityForm.querySelector('input[name="query"]').value || "Terraform provider backend state";
            const payload = {
                backend: "cpu",
                max_vector: 4,
                max_graph: 8,
                max_evidence: 4,
                max_context_chars: 8000,
                tests: [
                    {
                        query: query,
                        expected_sources: [],
                        expected_types: ["vector", "graph"]
                    }
                ]
            };

            try {
                const response = await fetch("/golden_tests/run", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                qualityOutput.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                qualityOutput.textContent = String(error);
            } finally {
                goldenTestButton.disabled = false;
            }
        });

        chatgptImportForm.addEventListener("submit", function(event) {
            event.preventDefault();

            submitPanelForm({
                form: chatgptImportForm,
                button: chatgptImportButton,
                progress: chatgptProgress,
                fill: chatgptProgressFill,
                percentLabel: chatgptProgressPercent,
                textLabel: chatgptProgressText,
                url: "/import_chatgpt",
                sendingText: "Wysyłam eksport",
                processingText: "Import historii i tagowanie",
                doneText: "Historia zaimportowana",
                requireFile: true,
                prepareFormData: prepareChatgptImportData,
                preparingText: "Czytam ZIP lokalnie",
                pollStatusUrl: "/import_chatgpt/status",
                onDone: function(data) {
                    const files = data.files ? ` Pliki JSON: ${data.files}.` : "";
                    const pairs = Number.isFinite(data.pairs) ? ` Q/A: ${data.pairs}.` : "";
                    const skipped = data.skipped_files ? ` Pominięto: ${data.skipped_files}.` : "";
                    const mode = data.smart ? "Historia zaimportowana smart." : "Historia zaimportowana.";

                    return `${mode}${files}${pairs}${skipped}`;
                }
            });
        });

        function formatGraphCleanupResult(data) {
            const lines = [
                `mode: ${data.dry_run ? "dry-run" : "apply"}`,
                `strict_relations: ${data.strict_relations ? "true" : "false"}`,
                `total_relations: ${data.total_relations || 0}`,
                `keep_relations: ${data.keep_relations || 0}`,
                `delete_relations: ${data.delete_relations || 0}`,
                `rewrite_relations: ${data.rewrite_relations || 0}`,
                `invalid_relation_types: ${data.invalid_relation_types || 0}`,
                `isolated_nodes_deleted: ${data.isolated_nodes_deleted || 0}`
            ];

            const invalidRelationSamples = data.invalid_relation_samples || [];
            const deleteSamples = data.delete_samples || [];
            const rewriteSamples = data.rewrite_samples || [];

            if (invalidRelationSamples.length) {
                lines.push("", "invalid_relation_samples:");
                invalidRelationSamples.forEach(function(item) {
                    lines.push(`- ${item}`);
                });
            }

            if (deleteSamples.length) {
                lines.push("", "delete_samples:");
                deleteSamples.forEach(function(item) {
                    lines.push(`- ${item}`);
                });
            }

            if (rewriteSamples.length) {
                lines.push("", "rewrite_samples:");
                rewriteSamples.forEach(function(item) {
                    lines.push(`- ${item}`);
                });
            }

            return lines.join("\\n");
        }

        async function runGraphCleanup(applyCleanup) {
            if (applyCleanup && !window.confirm("Uruchomić cleanup grafu? To usunie relacje oznaczone jako śmieciowe.")) {
                return;
            }

            const formData = new FormData(graphCleanupForm);
            formData.set("apply", applyCleanup ? "true" : "false");

            graphCleanupDryRunButton.disabled = true;
            graphCleanupApplyButton.disabled = true;
            graphCleanupOutput.textContent = "";
            setProgress(
                graphCleanupProgress,
                graphCleanupProgressFill,
                graphCleanupProgressPercent,
                graphCleanupProgressText,
                applyCleanup ? 45 : 25,
                applyCleanup ? "Czyszczę graf" : "Liczenie dry-run"
            );

            try {
                const response = await fetch("/graph_cleanup", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || "graph cleanup failed");
                }

                graphCleanupOutput.textContent = formatGraphCleanupResult(data);
                setProgress(
                    graphCleanupProgress,
                    graphCleanupProgressFill,
                    graphCleanupProgressPercent,
                    graphCleanupProgressText,
                    100,
                    applyCleanup
                        ? `Cleanup gotowy. Usunięte: ${data.delete_relations || 0}, przepisane: ${data.rewrite_relations || 0}`
                        : `Dry run gotowy. Do usunięcia: ${data.delete_relations || 0}, do przepisania: ${data.rewrite_relations || 0}`
                );
            } catch (error) {
                graphCleanupOutput.textContent = String(error);
                setProgress(
                    graphCleanupProgress,
                    graphCleanupProgressFill,
                    graphCleanupProgressPercent,
                    graphCleanupProgressText,
                    0,
                    "Błąd cleanupu"
                );
            } finally {
                graphCleanupDryRunButton.disabled = false;
                graphCleanupApplyButton.disabled = false;
            }
        }

        graphCleanupDryRunButton.addEventListener("click", function() {
            runGraphCleanup(false);
        });

        graphCleanupApplyButton.addEventListener("click", function() {
            runGraphCleanup(true);
        });


        form.addEventListener("submit", async function(event) {
            event.preventDefault();
            await askStream();
        });

        async function askStream() {
            const query = document.getElementById("query").value;
            const backend = document.getElementById("backend").value;
            const model = document.getElementById("model").value;
            const sessionId = document.getElementById("session_id").value;
            const mode = document.getElementById("mode").value;

            if (!document.getElementById("tags").value.trim()) {
                await tagCurrentQuery();
            }

            const tags = document.getElementById("tags").value;

            output.textContent = "";
            modeLabel.textContent = "SSE";
            setStatus("Streamuję", true);

            const formData = new FormData();
            formData.append("query", query);
            formData.append("tags", tags);
            formData.append("backend", backend);
            formData.append("model", model);
            formData.append("session_id", sessionId);
            formData.append("mode", mode);

            fetch("/ask_stream", {
                method: "POST",
                body: formData
            }).then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                function read() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            setStatus("Gotowy");
                            return;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split("\\n\\n");
                        buffer = lines.pop();

                        lines.forEach(block => {
                            const eventType = block.includes("event: status") ? "status" : "message";
                            const data = block
                                .split("\\n")
                                .filter(line => line.startsWith("data: "))
                                .map(line => line.replace("data: ", ""))
                                .join("\\n");

                            if (!data) {
                                return;
                            }

                            if (eventType === "status") {
                                setStatus(data, data !== "Gotowy");
                            } else {
                                output.textContent += data;
                            }
                        });

                        read();
                    });
                }

                read();
            }).catch(error => {
                output.textContent = "Błąd streamingu.";
                setStatus("Błąd");
            });
        }
        </script>
    </body>
    </html>
    """)
