"""Grounded Python help — so a tiny local model can answer standard-library questions.

It does three things:
  1. **build the help file** — render the full standard library's documentation with ``pydoc``
     into ``data/python-help.txt`` (about 11 MB; done once, cached).
  2. **index it with Groundwire** — the whole corpus stays off the GPU; only the few chunks that
     match a question are ever put in front of the model.
  3. **answer** — ``ask_python(question)`` retrieves the relevant docs and has the local model
     answer from them; ``python_help(query)`` is the same retrieval exposed as a harness tool.

This is the RAG pattern from the notebook, pointed at Python's own docs: retrieval substitutes
for a context window the 2B model does not have.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .download import PROJECT_ROOT

HELP_FILE = PROJECT_ROOT / "data" / "python-help.txt"
_docs = None  # cached Groundwire index for this session
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Importing these for pydoc has side effects (antigravity opens a browser; this prints the Zen),
# so we describe them from a short note instead of importing them.
_SKIP = {
    "antigravity": "antigravity — the Python easter egg: `import antigravity` opens the xkcd comic "
                   "353 (https://xkcd.com/353/) in your web browser. It also defines a geohash() function.",
    "this": "this — the Python easter egg: `import this` prints 'The Zen of Python' by Tim Peters.",
}


def build_help(path: Path | str = HELP_FILE, max_modules: int | None = None, rebuild: bool = False) -> Path:
    """Render stdlib documentation to a text file (cached). Returns the path."""
    import pydoc

    path = Path(path)
    if path.exists() and not rebuild:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [n for n in sorted(getattr(sys, "stdlib_module_names", ())) if not n.startswith("_")]
    if max_modules:
        names = names[:max_modules]

    parts: list[str] = []
    for topic in ("builtins",):
        try:
            parts.append(f"===== {topic.upper()} =====\n" + pydoc.render_doc(topic, renderer=pydoc.plaintext))
        except Exception:
            pass
    for name in names:
        if name in _SKIP:
            parts.append(f"\n\n===== MODULE: {name} =====\n{_SKIP[name]}")  # avoid import side effects
            continue
        try:
            parts.append(f"\n\n===== MODULE: {name} =====\n" + pydoc.render_doc(name, renderer=pydoc.plaintext))
        except Exception:
            continue  # some modules refuse to import on this platform — skip them

    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def load(path: Path | str = HELP_FILE, k: int = 4, max_words: int = 120, rebuild: bool = False):
    """Build (if needed) and index the Python help with Groundwire. Cached per session."""
    global _docs
    if _docs is not None and not rebuild:
        return _docs
    from groundwire import Groundwire

    build_help(path, rebuild=rebuild)
    gw = Groundwire(memory="bm25", k=k, max_words=max_words)   # off-GPU, lexical, exact identifiers
    gw.ingest(Path(path).read_text(encoding="utf-8"), title="python-help")
    _docs = gw
    return gw


def search(query: str, k: int | None = None) -> list[tuple[str, float]]:
    """Return the top (text, score) help chunks for a query."""
    gw = load()
    return [(text, score) for _cid, text, score in gw.retrieve(query, k=k)]


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return _THINK.sub("", text).strip()


def ask_python(question: str, model=None, k: int = 4, max_tokens: int = 256) -> str:
    """Answer a Python question grounded in the local stdlib documentation."""
    gw = load(k=k)
    context = "\n\n".join(text for _cid, text, _s in gw.retrieve(question, k=k))
    if model is None:
        from .model import load_model

        model = load_model()
    messages = [
        {"role": "system", "content":
            "Answer the Python question using ONLY the documentation context. Be concise and show "
            "a short example if useful. If the answer is not in the context, say you could not find it."},
        {"role": "user", "content": f"Documentation:\n{context[:4000]}\n\nQuestion: {question}"},
    ]
    return _strip_think(model.generate(messages, max_tokens=max_tokens))
