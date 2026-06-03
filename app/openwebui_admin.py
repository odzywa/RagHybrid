import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests


router = APIRouter()

OPENWEBUI_DB_PATH = Path(os.getenv("OPENWEBUI_DB_PATH", "/openwebui-data/webui.db"))
DEFAULT_RAG_PROMPT = """You are a retrieval-first infrastructure assistant.

Before answering questions about Linux, Kubernetes, OpenShift, DevOps, SRE, networking, MikroTik, storage, observability, security, AI platforms, homelab, troubleshooting, imported repositories, local documentation, code, Terraform, Ansible, Helm, PostgreSQL, GitOps, service mesh, eBPF, GPU platforms, or platform engineering, use HybridRAG MCP retrieval first.

Use retrieved vector and graph_evidence results as the primary source of truth. Graph-only results are relationship hints, not standalone proof.

If RAGHybrid returns no relevant context, say that the imported knowledge base has no relevant context, then answer separately as general knowledge.
"""


class OpenWebUIProfileRequest(BaseModel):
    id: str
    name: str
    base_model_id: Optional[str] = None
    tags: List[str] = []
    tool_ids: List[str] = []
    system: str = ""
    function_calling: str = "native"
    is_active: bool = True


class OpenWebUIProfilePatch(BaseModel):
    name: Optional[str] = None
    base_model_id: Optional[str] = None
    tags: Optional[List[str]] = None
    tool_ids: Optional[List[str]] = None
    system: Optional[str] = None
    function_calling: Optional[str] = None
    is_active: Optional[bool] = None


def openwebui_db() -> sqlite3.Connection:
    if not OPENWEBUI_DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"OpenWebUI database not found at {OPENWEBUI_DB_PATH}. Mount /var/docker/openwebui to /openwebui-data.",
        )
    conn = sqlite3.connect(str(OPENWEBUI_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_json(value: Any, default: Any) -> Any:
    if not value:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def clean_list(values: Optional[List[str]]) -> List[str]:
    result = []
    for value in values or []:
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def admin_user_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT id FROM user WHERE email = ? LIMIT 1", ("admin@gmail.com",)).fetchone()
    if row:
        return row["id"]
    row = conn.execute("SELECT id FROM user LIMIT 1").fetchone()
    if row:
        return row["id"]
    raise HTTPException(status_code=503, detail="OpenWebUI user table has no users")


def model_to_public(row: sqlite3.Row) -> Dict[str, Any]:
    meta = load_json(row["meta"], {})
    params = load_json(row["params"], {})
    return {
        "id": row["id"],
        "name": row["name"],
        "base_model_id": row["base_model_id"],
        "tags": meta.get("tags") or [],
        "tool_ids": meta.get("toolIds") or [],
        "capabilities": meta.get("capabilities") or {},
        "system": params.get("system") or "",
        "function_calling": params.get("function_calling") or "",
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def default_meta(tags: List[str], tool_ids: List[str]) -> Dict[str, Any]:
    return {
        "profile_image_url": "/static/favicon.png",
        "description": "OpenWebUI model profile managed by HybridRAG admin",
        "capabilities": {
            "citations": True,
            "status_updates": True,
            "builtin_tools": True,
        },
        "suggestion_prompts": None,
        "tags": clean_list(tags),
        "toolIds": clean_list(tool_ids),
    }


def default_params(system: str, function_calling: str) -> Dict[str, Any]:
    params = {}
    if system:
        params["system"] = system
    if function_calling:
        params["function_calling"] = function_calling
    return params


def tool_servers_from_config(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    row = conn.execute("SELECT data FROM config WHERE id = 1").fetchone()
    if not row:
        return []
    data = load_json(row["data"], {})
    connections = ((data.get("tool_server") or {}).get("connections") or [])
    servers = []
    for idx, item in enumerate(connections, 1):
        info = item.get("info") or {}
        server_id = str(info.get("id") or idx)
        servers.append({
            "tool_id": f"server:{server_id}",
            "name": info.get("name") or f"Tool server {server_id}",
            "description": info.get("description") or "",
            "url": item.get("url"),
            "path": item.get("path"),
            "enabled": bool((item.get("config") or {}).get("enable", True)),
        })
    return servers


def ollama_connections_from_config(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    row = conn.execute("SELECT data FROM config WHERE id = 1").fetchone()
    if not row:
        return []
    data = load_json(row["data"], {})
    ollama = data.get("ollama") or {}
    base_urls = ollama.get("base_urls") or []
    api_configs = ollama.get("api_configs") or {}
    names = ["cpu", "gpu", "laptop"]
    connections = []
    for idx, url in enumerate(base_urls):
        config = api_configs.get(str(idx)) or {}
        connections.append({
            "index": idx,
            "label": names[idx] if idx < len(names) else f"ollama-{idx}",
            "url": url,
            "enabled": bool(config.get("enable", True)),
            "tags": config.get("tags") or [],
            "prefix_id": config.get("prefix_id") or "",
            "connection_type": config.get("connection_type") or "",
        })
    return connections


def ollama_model_origins(connections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    origins: Dict[str, List[Dict[str, Any]]] = {}
    for connection in connections:
        if not connection.get("enabled"):
            continue
        url = str(connection.get("url") or "").rstrip("/")
        if not url:
            continue
        try:
            response = requests.get(url + "/api/tags", timeout=3)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            origins.setdefault("__errors__", []).append({
                "label": connection.get("label"),
                "url": url,
                "error": str(exc)[:240],
            })
            continue
        for model in data.get("models") or []:
            name = model.get("name")
            if not name:
                continue
            origins.setdefault(name, []).append({
                "label": connection.get("label"),
                "url": url,
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
            })
    return origins


@router.get("/admin/openwebui-models", response_class=HTMLResponse)
def openwebui_models_ui() -> HTMLResponse:
    return HTMLResponse(OPENWEBUI_MODELS_HTML)


@router.get("/admin/openwebui-models/api/models")
def list_openwebui_models() -> Dict[str, Any]:
    conn = openwebui_db()
    try:
        rows = conn.execute("SELECT * FROM model ORDER BY name").fetchall()
        models = [model_to_public(row) for row in rows]
        ollama_connections = ollama_connections_from_config(conn)
        origins = ollama_model_origins(ollama_connections)
        for model in models:
            lookup_id = model.get("base_model_id") or model.get("id")
            model["origin_servers"] = origins.get(lookup_id, [])
        tags = sorted({tag for model in models for tag in model["tags"]})
        return {
            "db_path": str(OPENWEBUI_DB_PATH),
            "models": models,
            "tags": tags,
            "tool_servers": tool_servers_from_config(conn),
            "ollama_connections": ollama_connections,
            "ollama_origin_errors": origins.get("__errors__", []),
            "default_rag_prompt": DEFAULT_RAG_PROMPT,
        }
    finally:
        conn.close()


@router.post("/admin/openwebui-models/api/models")
def create_openwebui_profile(payload: OpenWebUIProfileRequest) -> Dict[str, Any]:
    model_id = payload.id.strip()
    name = payload.name.strip()
    if not model_id or not name:
        raise HTTPException(status_code=422, detail="id and name are required")

    now = int(time.time())
    conn = openwebui_db()
    try:
        user_id = admin_user_id(conn)
        existing = conn.execute("SELECT * FROM model WHERE id = ?", (model_id,)).fetchone()
        meta = default_meta(payload.tags, payload.tool_ids)
        params = default_params(payload.system, payload.function_calling)
        if existing:
            conn.execute(
                """
                UPDATE model
                SET user_id = ?, base_model_id = ?, name = ?, meta = ?, params = ?,
                    updated_at = ?, is_active = ?
                WHERE id = ?
                """,
                (
                    user_id,
                    payload.base_model_id,
                    name,
                    json.dumps(meta, ensure_ascii=False),
                    json.dumps(params, ensure_ascii=False),
                    now,
                    1 if payload.is_active else 0,
                    model_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO model (id, user_id, base_model_id, name, meta, params, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    user_id,
                    payload.base_model_id,
                    name,
                    json.dumps(meta, ensure_ascii=False),
                    json.dumps(params, ensure_ascii=False),
                    now,
                    now,
                    1 if payload.is_active else 0,
                ),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM model WHERE id = ?", (model_id,)).fetchone()
        return model_to_public(row)
    finally:
        conn.close()


@router.patch("/admin/openwebui-models/api/models/{model_id:path}")
def patch_openwebui_profile(model_id: str, payload: OpenWebUIProfilePatch) -> Dict[str, Any]:
    conn = openwebui_db()
    try:
        row = conn.execute("SELECT * FROM model WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="model profile not found")
        meta = load_json(row["meta"], {})
        params = load_json(row["params"], {})

        updates = payload.dict(exclude_unset=True)
        name = updates.get("name", row["name"])
        base_model_id = updates.get("base_model_id", row["base_model_id"])
        is_active = updates.get("is_active", bool(row["is_active"]))
        if "tags" in updates:
            meta["tags"] = clean_list(updates["tags"])
        if "tool_ids" in updates:
            meta["toolIds"] = clean_list(updates["tool_ids"])
        if "system" in updates:
            params["system"] = updates["system"] or ""
        if "function_calling" in updates:
            params["function_calling"] = updates["function_calling"] or ""

        conn.execute(
            """
            UPDATE model
            SET name = ?, base_model_id = ?, meta = ?, params = ?, updated_at = ?, is_active = ?
            WHERE id = ?
            """,
            (
                name,
                base_model_id,
                json.dumps(meta, ensure_ascii=False),
                json.dumps(params, ensure_ascii=False),
                int(time.time()),
                1 if is_active else 0,
                model_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM model WHERE id = ?", (model_id,)).fetchone()
        return model_to_public(row)
    finally:
        conn.close()


@router.delete("/admin/openwebui-models/api/models/{model_id:path}")
def delete_openwebui_profile(model_id: str) -> Dict[str, Any]:
    conn = openwebui_db()
    try:
        cur = conn.execute("DELETE FROM model WHERE id = ?", (model_id,))
        conn.commit()
        if cur.rowcount < 1:
            raise HTTPException(status_code=404, detail="model profile not found")
        return {"status": "deleted"}
    finally:
        conn.close()


OPENWEBUI_MODELS_HTML = r"""
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenWebUI Model Profiles</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#181b1f; --line:#2a3038; --text:#eef2f6; --muted:#9aa6b2; --blue:#69a7ff; --ok:#35c46b; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif; }
    header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
    h1 { font-size:20px; margin:0; }
    main { padding:20px 24px 40px; display:grid; gap:18px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    h2 { font-size:16px; margin:0 0 12px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }
    th { color:var(--muted); }
    input,select,textarea { width:100%; background:#0c0f12; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:8px; }
    textarea { min-height:140px; }
    button { background:#263142; color:var(--text); border:1px solid #3b485a; border-radius:6px; padding:7px 10px; cursor:pointer; }
    button:hover { border-color:var(--blue); }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .wide { grid-column:span 2; }
    .full { grid-column:1 / -1; }
    .muted { color:var(--muted); }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; margin:1px; color:#cdd6df; }
    .actions { display:flex; gap:6px; flex-wrap:wrap; }
    pre { background:#0c0f12; border:1px solid var(--line); border-radius:6px; padding:10px; overflow:auto; }
    @media (max-width:900px){ .grid{grid-template-columns:1fr}.wide{grid-column:span 1} }
  </style>
</head>
<body>
<header>
  <h1>OpenWebUI Model Profiles</h1>
  <div class="muted">Zakładki, tagi, Ollama CPU/GPU/laptop, MCP tools</div>
</header>
<main>
  <section>
    <h2>Connections</h2>
    <div id="connections" class="muted"></div>
  </section>
  <section>
    <h2>Create / Update Profile</h2>
    <div class="grid">
      <input id="profile_id" placeholder="profile id, e.g. qwen-rag-gpu">
      <input id="profile_name" placeholder="display name, e.g. Qwen RAG GPU">
      <select id="base_model"></select>
      <input id="tags" placeholder="tags: raghybrid,gpu">
      <input id="tool_ids" placeholder="tool ids: server:1">
      <select id="function_calling"><option value="native">native</option><option value="">off/default</option></select>
      <select id="is_active"><option value="true">active</option><option value="false">inactive</option></select>
      <button onclick="saveProfile()">Save profile</button>
      <textarea id="system" class="full" placeholder="system prompt"></textarea>
    </div>
    <div class="actions" style="margin-top:10px">
      <button onclick="fillRag()">Use RAG prompt</button>
      <button onclick="setTags('raghybrid')">tag Raghybrid</button>
      <button onclick="setTags('gpu')">tag GPU</button>
      <button onclick="setTags('cpu')">tag CPU</button>
      <button onclick="setTags('laptop')">tag Laptop</button>
      <button onclick="setToolServer1()">attach HybridRAG MCP</button>
    </div>
  </section>
  <section>
    <h2>Profiles</h2>
    <table>
      <thead><tr><th>Name</th><th>Base</th><th>Server</th><th>Tags</th><th>Tools</th><th>Function calling</th><th>Active</th><th>Actions</th></tr></thead>
      <tbody id="models"></tbody>
    </table>
  </section>
</main>
<script>
let state = {models: [], tags: [], tool_servers: [], ollama_connections: [], default_rag_prompt: ''};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}
function pillList(items) { return (items || []).map(x => `<span class="pill">${esc(x)}</span>`).join(' '); }
function originList(items) {
  if (!items || !items.length) return '<span class="muted">unknown/custom</span>';
  return items.map(x => `<span class="pill">${esc(x.label)} ${esc(x.url)}</span>`).join(' ');
}
async function load() {
  state = await api('/admin/openwebui-models/api/models');
  $('connections').innerHTML = `
    <b>OpenWebUI DB:</b> ${esc(state.db_path)}<br>
    <b>Ollama:</b> ${state.ollama_connections.map(c => `${esc(c.label)}=${esc(c.url)}`).join(' | ')}<br>
    <b>Tool servers:</b> ${state.tool_servers.map(t => `${esc(t.tool_id)} ${esc(t.name)}`).join(' | ')}
  `;
  const baseOptions = [''].concat(state.models.map(m => m.id)).sort();
  $('base_model').innerHTML = baseOptions.map(id => `<option value="${esc(id)}">${esc(id || 'base_model_id empty')}</option>`).join('');
  $('models').innerHTML = state.models.map(m => `
    <tr>
      <td><b>${esc(m.name)}</b><br><span class="muted">${esc(m.id)}</span></td>
      <td>${esc(m.base_model_id || '')}</td>
      <td>${originList(m.origin_servers)}</td>
      <td>${pillList(m.tags)}</td>
      <td>${pillList(m.tool_ids)}</td>
      <td>${esc(m.function_calling)}</td>
      <td>${m.is_active ? 'yes' : 'no'}</td>
      <td class="actions">
        <button onclick="editProfile('${esc(m.id)}')">edit</button>
        <button onclick="quickPatch('${esc(m.id)}', {tags:[...new Set([...(state.models.find(x=>x.id==='${esc(m.id)}').tags||[]),'raghybrid'])], tool_ids:['server:1'], function_calling:'native', system: state.default_rag_prompt})">RAG</button>
        <button onclick="quickPatch('${esc(m.id)}', {is_active:false})">disable</button>
      </td>
    </tr>`).join('');
}
function editProfile(id) {
  const m = state.models.find(x => x.id === id);
  if (!m) return;
  $('profile_id').value = m.id;
  $('profile_name').value = m.name;
  $('base_model').value = m.base_model_id || '';
  $('tags').value = (m.tags || []).join(',');
  $('tool_ids').value = (m.tool_ids || []).join(',');
  $('function_calling').value = m.function_calling || '';
  $('is_active').value = m.is_active ? 'true' : 'false';
  $('system').value = m.system || '';
}
function fillRag() { $('system').value = state.default_rag_prompt; }
function setToolServer1() {
  const ids = new Set(($('tool_ids').value || '').split(',').map(x => x.trim()).filter(Boolean));
  ids.add('server:1');
  $('tool_ids').value = Array.from(ids).join(',');
}
function setTags(tag) {
  const tags = new Set(($('tags').value || '').split(',').map(x => x.trim()).filter(Boolean));
  tags.add(tag);
  $('tags').value = Array.from(tags).join(',');
}
async function saveProfile() {
  const payload = {
    id: $('profile_id').value.trim(),
    name: $('profile_name').value.trim(),
    base_model_id: $('base_model').value || null,
    tags: ($('tags').value || '').split(',').map(x => x.trim()).filter(Boolean),
    tool_ids: ($('tool_ids').value || '').split(',').map(x => x.trim()).filter(Boolean),
    system: $('system').value,
    function_calling: $('function_calling').value,
    is_active: $('is_active').value === 'true'
  };
  await api('/admin/openwebui-models/api/models', {method:'POST', body:JSON.stringify(payload)});
  await load();
}
async function quickPatch(id, patch) {
  await api('/admin/openwebui-models/api/models/' + encodeURIComponent(id), {method:'PATCH', body:JSON.stringify(patch)});
  await load();
}
load().catch(err => alert(err.message));
</script>
</body>
</html>
"""
