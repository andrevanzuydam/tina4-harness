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
- Render a template with the @template decorator placed BELOW @get; the handler RETURNS a dict (the
  context). Templates live in src/templates (Twig/Frond). Correct pattern:
    @get("/dash")
    @template("dash.twig")
    async def dash(request, response):
        return {"title": "Dashboard"}
- Read JSON body via request.body (already parsed); query via request.params.
- Database/ORM lives in src/orm; change schema ONLY through migrations (tina4 generate migration <name>), never ad-hoc SQL.
- Run the app with `tina4 serve` (never `python app.py`) — it compiles SCSS and hot-reloads.
Return ONE src/routes/*.py file's content unless asked otherwise. Do not invent Flask/FastAPI/Django APIs."""

_server: subprocess.Popen | None = None
_server_url: str | None = None
_server_project: Path | None = None


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


def projects_in(area: str | Path) -> dict[str, Path]:
    """Map {name: path} of Tina4 projects directly under ``area`` (e.g. the notebook folder)."""
    found: dict[str, Path] = {}
    try:
        for child in Path(area).iterdir():
            if child.is_dir() and is_project(child):
                found[child.name] = child.resolve()
    except Exception:
        pass
    return found


def add_route(project: str | Path, name: str, code: str) -> str:
    """Write a route file to <project>/src/routes/<name>.py. Returns its project-relative path."""
    project = Path(project)
    routes = project / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    f = routes / f"{_slug(name)}.py"
    f.write_text(code.rstrip() + "\n", encoding="utf-8")
    return str(f.relative_to(project))


def add_template(project: str | Path, name: str, html: str) -> str:
    """Write a template to <project>/src/templates/<name>.twig. Returns its project-relative path."""
    project = Path(project)
    tpl = project / "src" / "templates"
    tpl.mkdir(parents=True, exist_ok=True)
    f = tpl / f"{_slug(name)}.twig"
    f.write_text(html.rstrip() + "\n", encoding="utf-8")
    return str(f.relative_to(project))


def write_file(project: str | Path, rel_path: str, content: str) -> str:
    """Write any file inside a project (relative path). Returns the project-relative path."""
    project = Path(project)
    f = (project / rel_path).resolve()
    if project.resolve() not in f.parents:
        raise ValueError(f"{rel_path} is outside the project")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content.rstrip() + "\n", encoding="utf-8")
    return str(f.relative_to(project))


def _slug(name: str) -> str:
    import re as _re

    s = _re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return s or "index"


def free_port(start: int = 7146, tries: int = 30) -> int:
    """First free TCP port at/after ``start`` (so a second app doesn't collide with the first)."""
    import socket

    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


def serve(cwd: str | Path, port: int = 7146, wait: bool = True, reload: bool = True) -> str:
    """Start `tina4 serve` in the BACKGROUND (so a cell never blocks). Returns the app URL.

    Hot-reload is ON by default so the preview auto-refreshes when the harness writes a file."""
    global _server, _server_url, _server_project
    stop()
    _server_project = Path(cwd).resolve()
    log = Path(cwd) / "logs" / "harness-serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    args = [_client(), "serve", "--no-browser", "--port", str(port)]
    if not reload:
        args.insert(2, "--no-reload")
    _server = subprocess.Popen(
        args, cwd=str(cwd), stdout=log.open("w"), stderr=subprocess.STDOUT, env=env,
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


def served_project() -> Path | None:
    """The project the harness currently serves (if any) — a strong 'what am I working on' signal."""
    if _server is not None and _server.poll() is None:
        return _server_project
    return None


def project_for_port(port: int) -> Path | None:
    """Best-effort: which Tina4 project is being served on ``port`` (by inspecting the serve process
    whose command names that port, then its working directory)."""
    if _server is not None and _server.poll() is None and _server_url and f":{port}" in _server_url:
        return _server_project
    # Fall back to scanning for a `tina4 serve --port <port>` process and reading its cwd.
    try:
        out = subprocess.run(["pgrep", "-f", f"tina4 serve.*{port}"], capture_output=True, text=True, timeout=5)
        for pid in out.stdout.split():
            cwd = subprocess.run(["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in cwd.splitlines():
                if line.startswith("n"):
                    p = Path(line[1:])
                    if is_project(p):
                        return p.resolve()
    except Exception:
        pass
    return None


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


# --- paradigm grounding over the framework docs -------------------------------------------------
# Docs are bundled with the harness (data/tina4-docs/<python|js>/) so grounding works FULLY OFFLINE —
# no network, no `tina4 docs`. A project-local .tina4-docs is used too when present, and only if
# nothing is on disk at all do we fall back to downloading.
from .download import PROJECT_ROOT  # noqa: E402

_BUNDLED_DOCS = PROJECT_ROOT / "data" / "tina4-docs"
_docs_index = None


def ensure_docs(cwd: str | Path) -> Path | None:
    """Download Tina4 docs into .tina4-docs/ (online, last resort) so answers can be grounded."""
    d = Path(cwd) / ".tina4-docs"
    if d.is_dir() and any(d.rglob("*.md")):
        return d
    try:
        run(["docs"], cwd=cwd, timeout=120)
    except Exception:
        return None
    return d if d.is_dir() else None


def _doc_dirs(cwd: str | Path) -> list[tuple[str, Path]]:
    """(label, dir) of *.md doc sets to ground on, offline-first: the bundled python+js docs shipped
    with the harness, plus a project-local .tina4-docs if present. Only when NOTHING is on disk do we
    try a network download."""
    dirs: list[tuple[str, Path]] = []
    if _BUNDLED_DOCS.is_dir():
        for sub in sorted(_BUNDLED_DOCS.iterdir()):
            if sub.is_dir() and any(sub.glob("*.md")):
                dirs.append((sub.name, sub))
    local = Path(cwd) / ".tina4-docs"
    if local.is_dir() and any(local.rglob("*.md")):
        dirs.append(("project", local))
    if not dirs:
        d = ensure_docs(cwd)  # last resort: download (needs network)
        if d is not None:
            dirs.append(("project", d))
    return dirs


def ask_tina4(question: str, cwd: str | Path | None = None, k: int = 4) -> str:
    """Retrieve the most relevant Tina4 documentation chunks for a question (grounding text). Indexes
    the bundled python + js docs (and any project-local docs), tagging each chunk with its source so
    the model can tell tina4-python from tina4-js. Works offline."""
    global _docs_index
    base = Path(cwd or Path.cwd())
    if _docs_index is None:
        dirs = _doc_dirs(base)
        if not dirs:
            return ""
        try:
            from groundwire import Groundwire

            gw = Groundwire(memory="bm25", k=k, max_words=120)
            for label, d in dirs:
                for f in sorted(d.rglob("*.md")):
                    try:
                        gw.ingest_code(f.read_text(encoding="utf-8", errors="ignore"),
                                       title=f"{label}/{f.name}")
                    except Exception:
                        continue
            _docs_index = gw
        except Exception:
            return ""
    try:
        hits = _docs_index.retrieve(question, k=k)
        # Each chunk already carries a "# file: <label>/<name>" header from ingest, so the model can
        # see whether it's python or js docs.
        return "\n\n".join(text[:600] for _cid, text, _s in hits)
    except Exception:
        return ""
