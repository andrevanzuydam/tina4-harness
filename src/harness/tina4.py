"""Let the harness drive the Tina4 CLI and understand Tina4's paradigms.

Two jobs:
  1. RUN the tooling — ``tina4 init`` to provision a project, ``tina4 generate`` to scaffold a
     piece, ``tina4 serve`` to run it (non-blocking, so a notebook cell never hangs).
  2. GROUND the model — ``PARADIGMS`` is a concise, accurate summary of how a Tina4 Python app is
     built (routing, responses, templates, ORM, migrations), fed into the chat so the local model
     writes idiomatic Tina4 code instead of generic Flask/Django.

Everything shells out to the ``tina4`` client (the Rust CLI); nothing here reimplements it.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

# The non-negotiable conventions, kept short so they fit a small model's context. Verified against
# tina4_python.core.router (get/post/put/patch/delete/websocket/secured/noauth/role/can/template).
PARADIGMS = """Tina4 Python conventions (follow these exactly):
- Routes live in files under src/routes/. A file in that folder is auto-discovered — no manual registration.
- Define a route with a decorator from tina4_python.core.router and an async handler taking (request, response):
    from tina4_python.core.router import get, post
    @get("/hello")
    async def hello(request, response):
        return response({"message": "Hello"})        # dict/list -> JSON automatically
- Path params: @get("/user/{id}") -> async def h(request, response): request.params["id"].
- Other verbs: post, put, patch, delete, websocket. Protect writes with @secured() (don't use @noauth()).
- Render a template with response(template("page.twig", {"key": value})); templates live in src/templates (Twig/Frond).
- Read JSON body via request.body (already parsed); query via request.params.
- Database/ORM lives in src/orm; change schema ONLY through migrations (tina4 generate migration <name>), never ad-hoc SQL.
- Run the app with `tina4 serve` (never `python app.py`) — it compiles SCSS and hot-reloads.
Return ONE src/routes/*.py file's content unless asked otherwise. Do not invent Flask/FastAPI/Django APIs."""

_server: subprocess.Popen | None = None
_server_url: str | None = None


def _client() -> str:
    return os.environ.get("TINA4_BIN", "tina4")


def run(args: list[str], cwd: str | Path | None = None, timeout: int = 600) -> str:
    """Run a `tina4 ...` command and return its combined output (raises on non-zero exit)."""
    cmd = [_client(), *args]
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout + p.stderr).strip()
    if p.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {p.returncode}):\n{out}")
    return out


def available() -> bool:
    """Is the tina4 client installed?"""
    try:
        run(["--version"], timeout=20)
        return True
    except Exception:
        return False


def init(path: str | Path, language: str = "python") -> str:
    """Provision a new Tina4 project at ``path`` (scaffold only — does not start the server)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Answer the "start server now?" prompt with 'n' so this never blocks.
    cmd = [_client(), "init", language, str(path)]
    p = subprocess.run(cmd, input="n\n", capture_output=True, text=True, timeout=600)
    out = (p.stdout + p.stderr).strip()
    if p.returncode != 0:
        raise RuntimeError(f"tina4 init failed (exit {p.returncode}):\n{out}")
    return out


def generate(kind: str, name: str, cwd: str | Path) -> str:
    """Scaffold a piece inside a project: kind is model | route | migration | middleware."""
    return run(["generate", kind, name], cwd=cwd)


def is_project(path: str | Path) -> bool:
    """Does ``path`` look like a Tina4 project (has app.py and depends on tina4)?"""
    path = Path(path)
    if not (path / "app.py").is_file():
        return False
    pyproject = path / "pyproject.toml"
    try:
        return pyproject.is_file() and "tina4" in pyproject.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def find_project(start: str | Path | None = None) -> Path | None:
    """Walk up from ``start`` (default cwd) to find the enclosing Tina4 project, else None."""
    d = Path(start or Path.cwd()).resolve()
    for _ in range(8):
        if is_project(d):
            return d
        if d.parent == d:
            break
        d = d.parent
    # Also check one level of children of cwd (common in the notebook area).
    base = Path(start or Path.cwd())
    try:
        for child in base.iterdir():
            if child.is_dir() and is_project(child):
                return child.resolve()
    except Exception:
        pass
    return None


def serve(cwd: str | Path, port: int = 7146, wait: bool = True) -> str:
    """Start `tina4 serve` in the BACKGROUND (so a cell never blocks). Returns the app URL."""
    global _server, _server_url
    stop()
    log = Path(cwd) / "logs" / "harness-serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    _server = subprocess.Popen(
        [_client(), "serve", "--no-browser", "--no-reload", "--port", str(port)],
        cwd=str(cwd), stdout=log.open("w"), stderr=subprocess.STDOUT, env=env,
    )
    _server_url = f"http://localhost:{port}/"
    if wait:
        _wait_up(_server_url)
    return _server_url


def _wait_up(url: str, timeout: int = 30) -> bool:
    import urllib.request

    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return True
        except Exception:
            time.sleep(1)
    return False


def status() -> str:
    """Is the harness-started server running?"""
    if _server is not None and _server.poll() is None:
        return f"running at {_server_url} (pid {_server.pid})"
    return "not running"


def stop() -> str:
    """Stop a harness-started server."""
    global _server
    if _server is not None and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=5)
        except Exception:
            _server.kill()
    _server = None
    return "stopped"


# --- paradigm grounding over the downloaded framework docs (optional, best-effort) ---------------
_docs_index = None


def ensure_docs(cwd: str | Path) -> Path | None:
    """Download Tina4 docs into .tina4-docs/ (once) so answers can be grounded in them."""
    d = Path(cwd) / ".tina4-docs"
    if d.is_dir() and any(d.rglob("*.md")):
        return d
    try:
        run(["docs"], cwd=cwd, timeout=120)
    except Exception:
        return None
    return d if d.is_dir() else None


def ask_tina4(question: str, cwd: str | Path | None = None, k: int = 4) -> str:
    """Retrieve the most relevant Tina4 documentation chunks for a question (grounding text)."""
    global _docs_index
    base = Path(cwd or Path.cwd())
    docs = ensure_docs(base)
    if docs is None:
        return ""
    if _docs_index is None:
        try:
            from groundwire import Groundwire

            gw = Groundwire(memory="bm25", k=k, max_words=120)
            for f in docs.rglob("*.md"):
                try:
                    gw.ingest_code(f.read_text(encoding="utf-8", errors="ignore"), title=f.name)
                except Exception:
                    continue
            _docs_index = gw
        except Exception:
            return ""
    try:
        hits = _docs_index.retrieve(question, k=k)
        return "\n\n".join(text[:600] for _cid, text, _s in hits)
    except Exception:
        return ""
