from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chromadb
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from git import Repo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(tempfile.gettempdir()) / "codebase_reader" / "server"
REPOS_DIR = DATA_DIR / "repos"
CHROMA_DIR = DATA_DIR / "chroma"

ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".scss",
    ".go",
    ".java",
    ".rb",
    ".php",
    ".rs",
    ".cs",
    ".c",
    ".h",
    ".cpp",
    ".sh",
    ".sql",
    ".dockerfile",
}
ALLOWED_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE"}
SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}
MAX_FILE_SIZE_BYTES = 250_000
EMBEDDING_DIMENSION = 384
DEFAULT_TOP_K = 5
MAX_HTML_FILES = 60

app = FastAPI(title="Codebase Q&A Tool", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class RepoRecord:
    repo_url: str
    repo_dir: Path
    collection_name: str
    repo_slug: str


class LocalHashEmbedding:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokens(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def _tokens(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9_./-]+", " ", text.lower())
        words = [word for word in cleaned.split() if len(word) > 1]
        ngrams: list[str] = []
        joined = "".join(words)
        for size in (3, 4):
            ngrams.extend(joined[i : i + size] for i in range(max(0, len(joined) - size + 1)))
        return words + ngrams


@lru_cache(maxsize=1)
def get_embedding_function() -> LocalHashEmbedding:
    return LocalHashEmbedding()


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)


def normalize_repo_url(repo_url: str) -> str:
    cleaned = repo_url.strip()
    if cleaned.startswith("file://"):
        cleaned = cleaned[7:]
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return cleaned.rstrip("/")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "repo"


def repo_hash(repo_url: str) -> str:
    return hashlib.sha1(normalize_repo_url(repo_url).encode("utf-8", errors="ignore")).hexdigest()[:12]


def get_repo_record(repo_url: str) -> RepoRecord:
    normalized = normalize_repo_url(repo_url)
    parsed = urlparse(normalized)
    candidate_name = Path(parsed.path if parsed.scheme else normalized).name
    repo_slug = f"{safe_slug(candidate_name)}-{repo_hash(normalized)}"
    repo_dir = REPOS_DIR / repo_slug
    collection_name = f"repo_{repo_slug.replace('-', '_')}"
    return RepoRecord(repo_url=normalized, repo_dir=repo_dir, collection_name=collection_name, repo_slug=repo_slug)


def clone_or_open_repo(repo_url: str) -> Path:
    record = get_repo_record(repo_url)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    source_path = Path(record.repo_url)
    if source_path.exists():
        return source_path.resolve()

    temp_dir = REPOS_DIR / f"{record.repo_slug}-{uuid.uuid4().hex}"
    clone_command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--single-branch",
        record.repo_url,
        str(temp_dir),
    ]
    clone_result = subprocess.run(clone_command, capture_output=True, text=True)
    if clone_result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        stderr = clone_result.stderr.strip() or clone_result.stdout.strip() or "git clone failed"
        raise HTTPException(status_code=400, detail=f"Unable to clone repository: {stderr}")

    # Attempt a safe, Windows-friendly replace of the temp clone into the cache
    def _atomic_replace_dir(src: Path, dst: Path, attempts: int = 5, delay: float = 0.5) -> None:
        try:
            if not dst.exists():
                src.replace(dst)
                return

            # Try removing the target directory and renaming
            for _ in range(attempts):
                try:
                    if dst.exists():
                        shutil.rmtree(dst, ignore_errors=False)
                    src.replace(dst)
                    return
                except PermissionError:
                    time.sleep(delay)

            # As a last resort, copy the files into place and remove the temp dir
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to move repo into cache: {exc}")

    # If an existing cache exists, try to replace it safely.
    _atomic_replace_dir(temp_dir, record.repo_dir)
    return record.repo_dir


def is_indexable_file(path: Path) -> bool:
    if path.name in ALLOWED_FILENAMES:
        return True
    suffix = path.suffix.lower()
    return suffix in ALLOWED_EXTENSIONS or path.name.lower() == "dockerfile"


def read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def collect_repo_files(repo_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if is_indexable_file(path):
            files.append(path)
    return sorted(files)


def chunk_file_content(content: str) -> list[str]:
    splitter = get_splitter()
    return splitter.split_text(content)


def build_collection(repo_url: str) -> dict[str, Any]:
    repo_path = clone_or_open_repo(repo_url)
    record = get_repo_record(repo_url)
    client = get_chroma_client()
    embedding = get_embedding_function()

    try:
        client.delete_collection(record.collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=record.collection_name, metadata={"repo_url": record.repo_url})

    files = collect_repo_files(repo_path)
    chunks: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for file_path in files:
        content = read_text_file(file_path)
        if not content:
            continue
        relative_path = str(file_path.relative_to(repo_path)).replace("\\", "/")
        for chunk_index, chunk in enumerate(chunk_file_content(content)):
            chunks.append(chunk)
            metadatas.append(
                {
                    "path": relative_path,
                    "chunk_index": chunk_index,
                    "source": relative_path,
                }
            )
            ids.append(f"{record.repo_slug}:{relative_path}:{chunk_index}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No indexable text files were found in this repository.")

    embeddings = embedding.embed_documents(chunks)
    collection.add(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)

    return {
        "repo_url": record.repo_url,
        "repo_slug": record.repo_slug,
        "repo_path": str(repo_path),
        "collection_name": record.collection_name,
        "file_count": len(files),
        "chunk_count": len(chunks),
    }


def search_collection(repo_url: str, question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    record = get_repo_record(repo_url)
    client = get_chroma_client()
    embedding = get_embedding_function()

    try:
        collection = client.get_collection(name=record.collection_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Repository not indexed yet. Call /api/ingest first.") from exc

    query_embedding = embedding.embed_query(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches: list[dict[str, Any]] = []
    for index, document in enumerate(result.get("documents", [[]])[0]):
        metadata = result.get("metadatas", [[]])[0][index] if result.get("metadatas") else {}
        distance = result.get("distances", [[]])[0][index] if result.get("distances") else None
        matches.append(
            {
                "path": metadata.get("path", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "distance": distance,
                "snippet": document[:900],
            }
        )

    answer = synthesize_answer(question, matches)
    return {
        "repo_url": record.repo_url,
        "question": question,
        "answer": answer,
        "matches": matches,
    }


def synthesize_answer(question: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "I could not find any relevant code chunks for that question."

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_GENAI_MODEL", "gemini-1.5-flash"),
                temperature=0.2,
                google_api_key=api_key,
            )
            context_lines = []
            for match in matches[:5]:
                context_lines.append(f"File: {match['path']}\n{match['snippet']}")
            prompt = (
                "You are a senior codebase assistant. Answer the user's question using only the provided code context. "
                "Be concise, mention file paths when useful, and if the evidence is incomplete say so.\n\n"
                f"Question: {question}\n\nContext:\n" + "\n\n---\n\n".join(context_lines)
            )
            response = llm.invoke(prompt)
            if getattr(response, "content", None):
                return str(response.content).strip()
        except Exception:
            pass

    first_lines = []
    for match in matches[:3]:
        snippet = match["snippet"].strip().splitlines()
        preview = snippet[0] if snippet else ""
        first_lines.append(f"{match['path']}: {preview[:180]}")

    lines = [
        f"I found the most relevant code in {len(matches)} chunk(s).",
        "Top signals:",
        *[f"- {line}" for line in first_lines],
        "",
        "If you want, I can also turn these matches into a step-by-step flow for the feature.",
    ]
    return "\n".join(lines)


def build_tree_preview(repo_path: Path) -> list[str]:
    paths = collect_repo_files(repo_path)
    preview: list[str] = []
    for path in paths[:MAX_HTML_FILES]:
        preview.append(str(path.relative_to(repo_path)).replace("\\", "/"))
    return preview


def render_workflow_html(summary: dict[str, Any], tree_preview: list[str], question: str | None = None) -> str:
    title = html.escape(summary.get("repo_slug", "repo"))
    workflow_json = html.escape(json.dumps(summary, indent=2))
    tree_html = "".join(f"<li>{html.escape(item)}</li>" for item in tree_preview) or "<li>No files indexed yet</li>"
    question_html = html.escape(question or "Ask a question to inspect the flow.")

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Codebase Flow - {title}</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f3ea;
      --panel: #fffdf8;
      --text: #1f1a17;
      --muted: #6e6258;
      --accent: #c46b3d;
      --accent-soft: #fde4d7;
      --line: #e4d7ca;
      --shadow: 0 24px 70px rgba(43, 25, 12, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(196, 107, 61, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(34, 98, 157, 0.12), transparent 24%),
        var(--bg);
      color: var(--text);
    }}
    .shell {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 20px;
      align-items: stretch;
      margin-bottom: 20px;
    }}
    .card {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 5vw, 3.8rem); line-height: 0.95; letter-spacing: -0.05em; }}
    p {{ line-height: 1.6; color: var(--muted); }}
    .pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 8px 14px;
      font-weight: 700;
      font-size: 0.9rem;
      margin-bottom: 16px;
    }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-bottom: 20px; }}
    .metric {{
      border-radius: 20px;
      background: #fff;
      border: 1px solid var(--line);
      padding: 18px;
    }}
    .metric strong {{ display: block; font-size: 1.8rem; margin-bottom: 6px; }}
    .flow {{ display: grid; gap: 20px; grid-template-columns: 1.15fr 0.85fr; }}
    .flow-block {{ min-height: 420px; }}
    .codebox {{
      background: #111;
      color: #f0efe8;
      border-radius: 18px;
      padding: 16px;
      overflow: auto;
      font-size: 0.92rem;
      line-height: 1.5;
    }}
    ol {{ margin: 0; padding-left: 20px; }}
    li {{ margin-bottom: 6px; }}
    .section-title {{ margin: 0 0 12px; font-size: 1.15rem; }}
    .subtitle {{ margin: 0 0 16px; color: var(--muted); }}
    .diagram {{ background: white; border-radius: 18px; border: 1px solid var(--line); padding: 18px; }}
    .stack {{ display: grid; gap: 14px; }}
    .tagline {{ color: var(--accent); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.78rem; }}
    @media (max-width: 960px) {{
      .hero, .flow, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="card">
        <div class="pill">AI Codebase Q&A Workflow</div>
        <h1>Understand the repo in plain English.</h1>
        <p>{question_html}</p>
        <div class="stack">
          <div>
            <div class="tagline">How it works</div>
            <p>Clone the repo, chunk every indexable file, embed the chunks in ChromaDB, retrieve the most relevant code, then explain the answer in a short developer-friendly summary.</p>
          </div>
          <div>
            <div class="tagline">Best use</div>
            <p>Use this page when you want a feature map, a mental model of the codebase, or a quick path to the functions that handle auth, payments, API calls, or UI state.</p>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="section-title">Current build summary</div>
        <div class="grid" style="grid-template-columns: 1fr; margin-bottom: 0;">
          <div class="metric"><strong>{html.escape(str(summary.get('file_count', 0)))}</strong><span>Files indexed</span></div>
          <div class="metric"><strong>{html.escape(str(summary.get('chunk_count', 0)))}</strong><span>Chunks stored in ChromaDB</span></div>
          <div class="metric"><strong>{html.escape(summary.get('collection_name', 'n/a'))}</strong><span>Vector collection</span></div>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="metric"><strong>1</strong><span>GitPython clones the repository to local storage.</span></div>
      <div class="metric"><strong>2</strong><span>RecursiveCharacterTextSplitter breaks files into searchable chunks.</span></div>
      <div class="metric"><strong>3</strong><span>ChromaDB stores embeddings and retrieves relevant code for each question.</span></div>
    </div>

    <div class="flow">
      <div class="card flow-block">
        <div class="section-title">Visual workflow</div>
        <div class="diagram mermaid">
flowchart LR
    A[GitHub repo URL] --> B[Clone repo]
    B --> C[Filter source files]
    C --> D[Chunk text]
    D --> E[Embed into ChromaDB]
    E --> F[Ask question]
    F --> G[Retrieve top chunks]
    G --> H[Explain answer]
        </div>
      </div>
      <div class="card flow-block">
        <div class="section-title">File tree preview</div>
        <div class="subtitle">Top files from the current repo snapshot.</div>
        <ol>{tree_html}</ol>
      </div>
    </div>

    <div class="card" style="margin-top: 20px;">
      <div class="section-title">API payload snapshot</div>
      <pre class="codebox">{workflow_json}</pre>
    </div>
  </div>
  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'base', themeVariables: {{ primaryColor: '#c46b3d', primaryTextColor: '#1f1a17', lineColor: '#b9a694' }} }});
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html_body = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Codebase Q&A Tool</title>
  <style>
    body { font-family: ui-sans-serif, system-ui; background: #f7f4ee; margin: 0; color: #1f1a17; }
    .wrap { max-width: 900px; margin: 0 auto; padding: 48px 20px; }
    .card { background: white; border-radius: 20px; padding: 24px; box-shadow: 0 24px 70px rgba(43, 25, 12, 0.12); border: 1px solid #e5d7c8; margin-bottom: 18px; }
    input, button, textarea { width: 100%; border-radius: 14px; border: 1px solid #d7c9ba; padding: 14px; font: inherit; }
    button { background: #c46b3d; color: white; border: none; font-weight: 700; cursor: pointer; }
    textarea { min-height: 120px; }
    .row { display: grid; gap: 12px; }
    .note { color: #6e6258; line-height: 1.6; }
    h1 { margin-top: 0; line-height: 1; letter-spacing: -0.04em; }
    code { background: #f2ebe2; padding: 2px 6px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>AI Codebase Q&A Tool</h1>
      <p class="note">Clone a GitHub repo, index it into ChromaDB, then ask questions like <code>where is auth handled?</code> or <code>explain the payment flow</code>.</p>
    </div>
    <div class="card">
      <form method="post" action="/api/ingest">
        <div class="row">
          <input name="repo_url" placeholder="https://github.com/owner/repo" />
          <button type="submit">Index repository</button>
        </div>
      </form>
    </div>
    <div class="card">
      <form method="get" action="/workflow">
        <div class="row">
          <input name="repo_url" placeholder="https://github.com/owner/repo" />
          <button type="submit">Open workflow view</button>
        </div>
      </form>
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(html_body)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def parse_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        return dict(body)

    form = await request.form()
    return dict(form)


@app.post("/api/ingest")
async def ingest(request: Request) -> JSONResponse:
    payload = await parse_payload(request)
    repo_url = str(payload.get("repo_url", "")).strip()
    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url is required")
    result = await run_in_threadpool(build_collection, repo_url)
    return JSONResponse(result)


@app.post("/api/ask")
async def ask(request: Request) -> JSONResponse:
    payload = await parse_payload(request)
    repo_url = str(payload.get("repo_url", "")).strip()
    question = str(payload.get("question", "")).strip()
    top_k_value = int(payload.get("top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K)
    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url is required")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    result = await run_in_threadpool(search_collection, repo_url, question, top_k_value)
    return JSONResponse(result)


@app.get("/workflow", response_class=HTMLResponse)
async def workflow(repo_url: str = Query(..., description="Repository URL or local path")) -> HTMLResponse:
    summary = await run_in_threadpool(build_collection, repo_url)
    repo_path = Path(summary["repo_path"])
    tree_preview = await run_in_threadpool(build_tree_preview, repo_path)
    rendered = render_workflow_html(summary, tree_preview)
    return HTMLResponse(rendered)


@app.get("/api/repo-summary")
async def repo_summary(repo_url: str = Query(..., description="Repository URL or local path")) -> JSONResponse:
    summary = await run_in_threadpool(build_collection, repo_url)
    repo_path = Path(summary["repo_path"])
    summary["tree_preview"] = await run_in_threadpool(build_tree_preview, repo_path)
    return JSONResponse(summary)
