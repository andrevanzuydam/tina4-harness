"""Clone a git repo and explore its code with the local model — Groundwire keeps the whole
repository off the GPU and feeds the model only the chunks that match your question.

    from harness.repo import open_repo, ask_repo, search_repo
    open_repo("https://github.com/tina4stack/groundwire")      # clone + index (once)
    print(ask_repo("where is the BM25 index built?"))          # grounded answer
    for path_title, score in search_repo("retrieve top-k"):    # raw retrieval
        print(path_title)

The repo is cloned into the project's ``repos/`` folder; only the matching code chunks ever reach
the model, so a large codebase works the same as a small one.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .download import PROJECT_ROOT
from .model import load_model

REPOS_DIR = PROJECT_ROOT / "repos"
CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh",
}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", "target"}

_index = None
_indexed_path = None
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clone(url: str, name: str | None = None, depth: int = 1) -> Path:
    """Shallow-clone ``url`` into repos/<name> (reused if already present)."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    name = name or url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = REPOS_DIR / name
    if dest.exists() and any(dest.iterdir()):
        return dest  # already cloned
    subprocess.run(
        ["git", "clone", "--depth", str(depth), url, str(dest)],
        check=True, capture_output=True, text=True, timeout=600,
    )
    return dest


def index(path: Path | str, k: int = 5, max_words: int = 80):
    """Index a repo's code with Groundwire (BM25, off-GPU). Returns the number of files indexed."""
    from groundwire import Groundwire

    global _index, _indexed_path
    path = Path(path)
    gw = Groundwire(memory="bm25", k=k, max_words=max_words)
    count = 0
    for f in path.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in CODE_EXT:
            continue
        if _SKIP_DIRS & set(f.parts):
            continue
        try:
            if f.stat().st_size > 250_000:
                continue
            gw.ingest_code(f.read_text(encoding="utf-8", errors="ignore"),
                           title=str(f.relative_to(path)))
            count += 1
        except Exception:
            continue
    _index, _indexed_path = gw, str(path)
    return count


def open_repo(url: str, name: str | None = None, depth: int = 1):
    """Clone (if needed) and index a repo. Returns (path, files_indexed)."""
    dest = clone(url, name=name, depth=depth)
    n = index(dest)
    return dest, n


def search_repo(query: str, k: int | None = None) -> list[tuple[str, float]]:
    """Return (text, score) code chunks matching a query (titles carry the file path)."""
    if _index is None:
        raise RuntimeError("Call open_repo(url) or index(path) first.")
    return [(text, score) for _cid, text, score in _index.retrieve(query, k=k)]


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return _THINK.sub("", text).strip()


def _repo_root() -> Path:
    if _indexed_path is None:
        raise RuntimeError("Open a repo first with open_repo(url).")
    return Path(_indexed_path)


def list_files(pattern: str = "*.py", limit: int = 40) -> list[str]:
    """List files in the open repo matching a glob (relative paths), e.g. '*.py' or '**/*.md'."""
    root = _repo_root()
    hits = []
    for f in root.rglob(pattern):
        if f.is_file() and not (_SKIP_DIRS & set(f.parts)):
            hits.append(str(f.relative_to(root)))
            if len(hits) >= limit:
                break
    return sorted(hits)


def read_file(rel_path: str) -> str:
    """Read one file from the open repo by its relative path."""
    f = _repo_root() / rel_path
    if not f.is_file():
        matches = [p for p in _repo_root().rglob("*") if p.is_file() and p.name == rel_path]
        if matches:
            f = matches[0]
    if not f.is_file():
        return f"ERROR: {rel_path} not found in the repo."
    return f.read_text(encoding="utf-8", errors="ignore")


def explain_file(rel_path: str, model=None, max_tokens: int = 700) -> str:
    """Read one file and explain its logic — what it does, the key parts, and how it flows."""
    code = read_file(rel_path)
    if code.startswith("ERROR:"):
        return code
    model = model or load_model()
    messages = [
        {"role": "system", "content":
            "Explain this Python file for a learner. Cover: (1) what it does in one line, (2) the "
            "key functions/classes and what each does, (3) how the logic flows step by step, and "
            "(4) anything notable or a possible bug. Be clear and concrete; short paragraphs or bullets."},
        {"role": "user", "content": f"File: {rel_path}\n```python\n{code[:6000]}\n```"},
    ]
    return _strip_think(model.generate(messages, max_tokens=max_tokens))


def ask_repo(question: str, model=None, k: int = 5, max_tokens: int = 400) -> str:
    """Answer a question about the cloned codebase, grounded in the retrieved code."""
    if _index is None:
        raise RuntimeError("Call open_repo(url) or index(path) first.")
    chunks = _index.retrieve(question, k=k)
    context = "\n\n".join(text for _cid, text, _s in chunks)
    model = model or load_model()
    messages = [
        {"role": "system", "content":
            "You answer questions about a codebase using ONLY the code excerpts provided. Be "
            "specific and name the file/function when you can. If it is not in the excerpts, say so."},
        {"role": "user", "content": f"Code from the repository:\n{context[:5000]}\n\nQuestion: {question}"},
    ]
    return _strip_think(model.generate(messages, max_tokens=max_tokens))
