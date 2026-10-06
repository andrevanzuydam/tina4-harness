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

from .model import load_model


def _repos_dir() -> Path:
    """Clone into the NOTEBOOK's working directory (the 'notebook space') so it shows in the
    file browser. Override with the HARNESS_REPOS_DIR env var."""
    import os

    base = os.environ.get("HARNESS_REPOS_DIR")
    return (Path(base) if base else Path.cwd() / "repos")
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
    repos_dir = _repos_dir()
    repos_dir.mkdir(parents=True, exist_ok=True)
    name = name or url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = repos_dir / name
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


def ensure_indexed() -> bool:
    """If nothing is indexed in this kernel, index the most recent repo already cloned on disk."""
    global _index
    if _index is not None:
        return True
    cands: list[Path] = []
    for base in (_repos_dir(), Path.cwd() / "repos", Path.cwd().parent / "repos"):
        try:
            if base.exists():
                cands += [d for d in base.iterdir() if d.is_dir() and any(d.iterdir())]
        except Exception:
            pass
    if not cands:
        return False
    index(max(cands, key=lambda p: p.stat().st_mtime))
    return True


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


def search_roots() -> list[Path]:
    """Where to look for files to load: the notebook's own folder first, then the open repo.

    This is what lets the chat 'load' a file from ANY folder in the notebook space — not only from
    a cloned repo. Override/extend the notebook root with the HARNESS_NOTEBOOK_DIR env var."""
    import os

    roots: list[Path] = []
    nb = os.environ.get("HARNESS_NOTEBOOK_DIR")
    roots.append(Path(nb) if nb else Path.cwd())
    if _indexed_path:
        roots.append(Path(_indexed_path))
    seen, out = set(), []
    for r in roots:
        rp = r.resolve()
        if rp not in seen and rp.exists():
            seen.add(rp)
            out.append(r)
    return out


def _candidate_files(limit: int = 8000) -> list[Path]:
    """Every loadable code/text file under the notebook folder and the open repo."""
    files: list[Path] = []
    for root in search_roots():
        try:
            for f in root.rglob("*"):
                if len(files) >= limit:
                    break
                if not f.is_file() or f.suffix.lower() not in CODE_EXT:
                    continue
                if _SKIP_DIRS & set(f.parts) or ".ipynb_checkpoints" in f.parts:
                    continue
                files.append(f)
        except Exception:
            continue
    return files


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_file(hint: str, exact: bool = False) -> Path | None:
    """Resolve a user's file reference (path, name, or bare stem) to a real file in the
    notebook folder or the open repo. Case/separator-insensitive; prefers an exact match.

    With ``exact=True`` only a direct path or an exact name/stem match counts (no partial
    substring match) — use it when the reference is a bare word that might be ordinary prose."""
    hint = (hint or "").strip().strip("`'\"")
    if not hint:
        return None
    # 1) Direct path, absolute or relative to a search root.
    p = Path(hint)
    if p.is_absolute() and p.is_file():
        return p
    for root in search_roots():
        cand = root / hint
        if cand.is_file():
            return cand
    # 2) Match by file name or stem across all candidate files.
    want = _norm(hint)
    want_stem = _norm(Path(hint).stem)
    partial = None
    for f in _candidate_files():
        name, stem = _norm(f.name), _norm(f.stem)
        if want in (name, stem) or want_stem in (name, stem):
            return f  # exact name/stem match wins immediately
        if not exact and ((want and want in name) or (want_stem and want_stem in stem)):
            partial = partial or f
    return partial


# Line-bounded so a bare `import x` never swallows following lines (\s would include newlines).
_IMPORT_RE = re.compile(r"^[ \t]*(?:from[ \t]+([.\w]+)[ \t]+import\b|import[ \t]+([\w.,\t ]+))", re.M)


def _resolve_module(dotted: str) -> Path | None:
    """Resolve a (possibly dotted) module name to a local file: 'models.user' → models/user.py or
    models/user/__init__.py. Falls back to a unique file whose stem matches the last component."""
    rel = dotted.strip().lstrip(".").replace(".", "/")
    if not rel:
        return None
    for root in search_roots():
        for suffix in (rel + ".py", rel + "/__init__.py"):
            p = root / suffix
            if p.is_file():
                return p
    return find_file(rel.split("/")[-1] + ".py", exact=True)


def referenced_files(code: str, limit: int = 6, max_chars: int = 2500) -> dict[str, str]:
    """Follow the imports in some code and return {relative_path: source} for each LOCAL module it
    references (a real file in the notebook folder or the open repo). Resolves dotted paths and
    relative imports. Lets the chat see code that an open file depends on even when that file isn't
    open. Stdlib/third-party imports resolve to no local file and are simply skipped."""
    names: list[str] = []
    for m in _IMPORT_RE.finditer(code or ""):
        if m.group(1):  # from <X.Y> import ...
            names.append(m.group(1).strip())
        else:           # import a, b.c as d
            for part in m.group(2).split(","):
                nm = part.strip().split(" as ")[0].strip()
                if nm:
                    names.append(nm)
    found: dict[str, str] = {}
    for name in names:
        if len(found) >= limit:
            break
        f = _resolve_module(name)
        if f is None:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            key = str(f.resolve().relative_to(Path.cwd().resolve()))
        except Exception:
            key = f.name
        if key in found:
            continue
        found[key] = text[:max_chars] + ("\n# …(truncated)…" if len(text) > max_chars else "")
    return found


def find_definitions(names, max_chars: int = 1400, limit: int = 4) -> dict:
    """Locate where functions/classes are defined across the notebook space and open repo — for ANY
    name in ANY file. Returns {name: (relpath, snippet)}. Lets the model call a method it can't see
    because the file isn't open, without guessing its signature."""
    wanted = [n for n in dict.fromkeys(names) if n and n.isidentifier()]
    if not wanted:
        return {}
    pats = {
        n: re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+" + re.escape(n) + r"\b", re.M)
        for n in wanted
    }
    found: dict = {}
    for f in _candidate_files():
        if len(found) >= limit or len(found) == len(wanted):
            break
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for n, pat in pats.items():
            if n in found:
                continue
            m = pat.search(text)
            if m:
                try:
                    key = str(f.resolve().relative_to(Path.cwd().resolve()))
                except Exception:
                    key = f.name
                found[n] = (key, text[m.start():m.start() + max_chars])
    return found


def load_file(hint: str) -> tuple[str, str, str]:
    """Load a file to drop into a cell. Returns (display_name, language, source).
    Raises FileNotFoundError if nothing matches."""
    f = find_file(hint)
    if f is None:
        raise FileNotFoundError(hint)
    code = f.read_text(encoding="utf-8", errors="ignore")
    ext = f.suffix.lower().lstrip(".")
    lang = {"py": "python", "rb": "ruby", "yml": "yaml", "md": "markdown"}.get(ext, ext or "text")
    try:
        name = str(f.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        name = f.name
    return name, lang, code


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


def ask_repo(question: str, model=None, k: int = 5, max_tokens: int = 400, history=None,
             open_code: str = "") -> str:
    """Answer a question about the cloned codebase, grounded in the retrieved code.

    ``history`` (the chat so far) keeps follow-ups in context: a terse "explain its ties logic"
    still retrieves the right file, and the model sees what was already discussed.
    ``open_code`` is whatever is currently open in the notebook's cells — when the user has a file
    loaded, "this"/"the file"/"it" mean THAT, so it is put first in the context."""
    if _index is None:
        raise RuntimeError("Call open_repo(url) or index(path) first.")
    # Structure/overview questions: list EVERY top-level folder deterministically (nothing dropped),
    # described from its README first line, else its files. Standard one-bullet-per-folder format.
    if re.search(r"folder|director|structure|organi|overview|each|layout", question, re.I):
        return folder_overview()
    # "list / show / what programs|files|scripts" → the ACTUAL files in the repo, deterministically
    # (never a model-invented list). Grounded, so it can't hallucinate variants of one file.
    if re.search(r"\b(list|show|which|what|name)\b", question, re.I) and \
       re.search(r"\b(program|file|script|module|example|demo|command|app)s?\b", question, re.I):
        return program_overview()
    prior = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    # Retrieve using the recent user turns too, so a short follow-up still finds the right file.
    recent_user = " ".join(m["content"] for m in prior if m["role"] == "user")[-500:]
    query = (recent_user + " " + question).strip() if recent_user else question
    chunks = _index.retrieve(query, k=k)
    context = "\n\n".join(text for _cid, text, _s in chunks)
    # Only let the OPEN file dominate when the question is about it ("this"/"the file"/"it") — never
    # for a repo-wide question, where it would bias the answer toward whatever happens to be open.
    if open_code.strip() and re.search(r"\b(this|the file|that|it|open|current|selection|here|above)\b",
                                       question, re.I):
        context = ("Currently OPEN in the notebook — the user likely means THIS when they say "
                   "'this', 'the file', or 'it':\n" + open_code.strip()[:4000] + "\n\n"
                   "Other relevant code from the repository:\n" + context)
    model = model or load_model()
    system = {"role": "system", "content":
        "You answer questions about a codebase using ONLY the code excerpts provided and the "
        "conversation so far. Be specific and name the file/function. Answer in at most 5 sentences; "
        "if it is not in the excerpts, say so. No headings."}
    convo = [{"role": m["role"], "content": m["content"]} for m in prior[:-1][-6:]]  # drop the current turn
    user = {"role": "user", "content": f"Code from the repository:\n{context[:5000]}\n\nQuestion: {question}"}
    return _strip_think(model.generate([system] + convo + [user], max_tokens=max_tokens))


def _first_doc_line(d: Path) -> str:
    """One-line description from a folder's README (any case/extension), else ''."""
    for pat in ("README*", "readme*", "Readme*"):
        for r in sorted(d.glob(pat)):
            if not r.is_file():
                continue
            try:
                for ln in r.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = ln.strip().lstrip("#").strip().strip("=").strip()
                    if s and not s.startswith("!["):  # skip blank lines and badge images
                        return s[:120]
            except Exception:
                return ""
    return ""


def _py_summary(f: Path) -> str:
    """One-line description of a Python file: its module docstring's first line, else first comment."""
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    m = re.search(r'^\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text, re.S)
    if m:
        for ln in m.group(1).splitlines():
            s = ln.strip()
            if s:
                return s[:100]
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            s = s.lstrip("#").strip()
            if s:
                return s[:100]
        elif s:
            break
    return ""


def program_overview(limit: int = 50) -> str:
    """A deterministic, grounded list of the actual Python programs in the open repo (never
    model-invented). Each line: the real relative path + its module docstring's first line."""
    root = _repo_root()
    files = []
    for f in sorted(root.rglob("*.py")):
        if _SKIP_DIRS & set(f.parts) or ".ipynb_checkpoints" in f.parts:
            continue
        files.append(f)
    total = len(files)
    shown = files[:limit]
    head = (f"**{total} Python programs in {root.name}**"
            + (f" (showing the first {limit}):" if total > limit else ":"))
    lines = [head, ""]
    for f in shown:
        rel = f.relative_to(root)
        desc = _py_summary(f)
        lines.append(f"- `{rel}`" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def folder_overview() -> str:
    """Standardized, complete overview of the open repo — every top-level folder (nothing dropped),
    described from its README first line else its files. Works for any repo layout, including ones
    with code at the root or only a few folders."""
    root = _repo_root()
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name not in _SKIP_DIRS and not d.name.startswith(".")
    )
    root_files = sorted(
        f.name for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in CODE_EXT and not f.name.startswith(".")
    )
    lines: list[str] = []
    if dirs:
        lines.append(f"**{len(dirs)} top-level folder(s) in {root.name}:**")
        lines.append("")
        for d in dirs:
            desc = _first_doc_line(d)
            if not desc:
                names = sorted(f.name for f in d.iterdir() if f.is_file())[:5]
                desc = ", ".join(names) if names else "(subfolders only)"
            lines.append(f"- **{d.name}/** — {desc}")
    # Always surface root-level code files too — some repos keep everything at the root.
    if root_files:
        if dirs:
            lines += ["", f"**{len(root_files)} file(s) at the repo root:** " + ", ".join(root_files[:20])
                      + ("…" if len(root_files) > 20 else "")]
        else:
            lines.append(f"**{root.name}** has no sub-folders; {len(root_files)} file(s) at the root:")
            lines += [f"- {n}" for n in root_files[:40]]
    return "\n".join(lines) if lines else f"{root.name} appears to be empty."
