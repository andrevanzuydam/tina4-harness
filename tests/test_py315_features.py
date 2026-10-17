"""Real, executable tests for EVERY new Python 3.15 language/builtin feature.

The harness's whole thesis is that small models hallucinate APIs, so every assertion here is
grounded in what the 3.15.0rc2 interpreter *actually does* — each exact value, error-message wording
and signature was captured from the real interpreter before it was written down, not copied from the
"what's new" page (several PEP write-ups there differ in detail from the shipped rc).

Every test is guarded with ``@pytest.mark.skipif(sys.version_info < (3, 15))`` so the suite SKIPS
cleanly on the 3.12 fallback venv (``.venv-notebook``) and runs for real on ``.venv-315``.

Most features are exercised by real execution. The few that are pure *syntax* (lazy import at module
scope, comprehension unpacking, the ``match`` pattern) are run either by ``exec`` of a real snippet
or, where module scope / a separate process is required, by writing a tiny real module to a tmp file
and running it on this same interpreter — never by mocking.

Run:  PYTHONPATH=src .venv-315/bin/python -m pytest tests/test_py315_features.py -q
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

import pytest

py315 = pytest.mark.skipif(sys.version_info < (3, 15), reason="Python 3.15 feature")


def _run_module(tmp_path: Path, files: dict[str, str], entry: str, *flags: str):
    """Write real ``.py`` files to a tmp dir and run ``entry`` on THIS interpreter. Returns the
    completed process (stdout/stderr/returncode) — real execution, no mock."""
    for name, src in files.items():
        (tmp_path / name).write_text(textwrap.dedent(src), encoding="utf-8")
    cmd = [sys.executable, *flags, str(tmp_path / entry)]
    return subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=60)


# --- PEP 810: lazy import ------------------------------------------------------------------------

@py315
def test_pep810_lazy_import_defers_loading(tmp_path):
    # A `lazy import` must NOT run the imported module until its first attribute use. We prove this
    # with a real module whose import prints a marker: the marker appears AFTER "BEFORE", only when
    # the value is actually touched. Module scope + a real import side effect need a real process.
    p = _run_module(
        tmp_path,
        {
            "sidefx.py": "import sys\nsys.stderr.write('IMPORTED\\n')\nVALUE = 99\n",
            "main.py": """
                lazy import sidefx
                import sys
                sys.stderr.write('BEFORE\\n')
                v = sidefx.VALUE
                sys.stderr.write('AFTER=%d\\n' % v)
            """,
        },
        "main.py",
    )
    assert p.returncode == 0, p.stderr
    # Deferred: the import marker lands between BEFORE and AFTER, not before BEFORE.
    assert p.stderr.splitlines() == ["BEFORE", "IMPORTED", "AFTER=99"]


@py315
def test_pep810_lazy_import_runtime_api_present():
    # The runtime switches that go with PEP 810 are real, callable functions on the live interpreter.
    assert callable(sys.get_lazy_imports)
    assert callable(sys.set_lazy_imports)
    import types
    assert hasattr(types, "LazyImportType")


@py315
def test_pep810_lazy_import_inside_function_is_syntax_error():
    # `lazy import` is restricted to module scope — inside a function it is a SyntaxError (verified
    # wording from the rc). Parse-only (compile) because the point is that it never compiles.
    with pytest.raises(SyntaxError) as ei:
        compile("def f():\n    lazy import os\n", "<test>", "exec")
    assert "lazy import not allowed inside functions" in str(ei.value)


# --- PEP 814: frozendict -------------------------------------------------------------------------

@py315
def test_pep814_frozendict_is_builtin_not_dict_subclass():
    import builtins
    assert hasattr(builtins, "frozendict")
    # It inherits directly from object, NOT from dict (per PEP 814, confirmed on the rc).
    assert not issubclass(frozendict, dict)  # noqa: F821  (3.15 builtin)


@py315
def test_pep814_frozendict_immutable():
    fd = frozendict(a=1)  # noqa: F821
    with pytest.raises(TypeError) as ei:
        fd["b"] = 2  # type: ignore[index]
    # Exact wording captured from the rc interpreter.
    assert str(ei.value) == "'frozendict' object does not support item assignment"


@py315
def test_pep814_frozendict_hashable_and_usable_as_key():
    fd = frozendict(a=1, b=2)  # noqa: F821
    assert isinstance(hash(fd), int)
    d = {fd: "ok"}
    assert d[frozendict(a=1, b=2)] == "ok"  # noqa: F821


@py315
def test_pep814_frozendict_equality_ignores_order():
    assert frozendict(a=1, b=2) == frozendict(b=2, a=1)  # noqa: F821


# --- PEP 798: * / ** unpacking in comprehensions -------------------------------------------------
# These forms are genuinely NEW syntax, so writing them literally here would make the whole module
# fail to PARSE on the 3.12 fallback venv (a SyntaxError at collection, not a skip). We therefore run
# each real form through eval() of a source string — still real execution of the new 3.15 syntax,
# just compiled from a string so the enclosing test module parses (and skips) everywhere.

@py315
def test_pep798_star_unpacking_in_list_comprehension():
    assert eval("[*L for L in [[1, 2], [3]]]") == [1, 2, 3]


@py315
def test_pep798_star_unpacking_in_set_comprehension():
    assert eval("{*s for s in [{1, 2}, {3}]}") == {1, 2, 3}


@py315
def test_pep798_doublestar_unpacking_in_dict_comprehension():
    assert eval("{**d for d in [{'a': 1}, {'b': 2}]}") == {"a": 1, "b": 2}


@py315
def test_pep798_star_unpacking_in_generator_expression():
    # The new unpack form in a generator expression, compiled from source and materialised.
    assert list(eval("(*L for L in [[1, 2], [3]])")) == [1, 2, 3]


# --- PEP 661: sentinel -----------------------------------------------------------------------------

@py315
def test_pep661_sentinel_constructor_signature():
    import inspect
    # Signature captured from the real builtin — NOT guessed. PEP drafts showed variants; the rc is:
    assert str(inspect.signature(sentinel)) == "(name, /, *, repr=None)"  # noqa: F821


@py315
def test_pep661_sentinel_identity_preserved_across_copy():
    import copy
    MISSING = sentinel("MISSING")  # noqa: F821
    assert copy.copy(MISSING) is MISSING
    assert copy.deepcopy(MISSING) is MISSING


@py315
def test_pep661_sentinel_custom_repr():
    S = sentinel("S", repr="<S>")  # noqa: F821
    assert repr(S) == "<S>"


# --- match: unary-plus literal pattern -----------------------------------------------------------

@py315
def test_match_unary_plus_literal_pattern():
    # `case +1:` mirrors `case -1:`. Exec a real match statement and check both arms.
    ns: dict = {}
    exec(
        "def classify(v):\n"
        "    match v:\n"
        "        case +1:\n"
        "            return 'one'\n"
        "        case _:\n"
        "            return 'other'\n",
        ns,
    )
    assert ns["classify"](1) == "one"
    assert ns["classify"](2) == "other"


# --- slice is generic (slice[int]) ---------------------------------------------------------------

@py315
def test_slice_is_subscriptable():
    # slice[int] used to raise TypeError; in 3.15 slice is a generic type.
    assert slice[int] is not None
    assert "slice" in repr(slice[int])


# --- __slots__ may declare __dict__ --------------------------------------------------------------

@py315
def test_slots_may_declare_dict():
    ns: dict = {}
    exec(
        "class C:\n"
        "    __slots__ = ('x', '__dict__')\n",
        ns,
    )
    c = ns["C"]()
    c.x = 1          # the declared slot
    c.anything = 2   # arbitrary attr, allowed because __dict__ is in __slots__
    assert (c.x, c.anything) == (1, 2)


# --- improved AttributeError suggestions ---------------------------------------------------------

@py315
def test_attribute_error_suggests_append_for_list_push():
    try:
        [].push(1)  # type: ignore[attr-defined]
    except AttributeError as e:
        base = str(e)
        rendered = "".join(traceback.format_exception_only(type(e), e))
    # The bare message is unchanged; the suggestion is added by the traceback renderer.
    assert base == "'list' object has no attribute 'push'"
    assert "Did you mean '.append'?" in rendered


@py315
def test_attribute_error_hint_for_dict_put():
    try:
        {}.put(1, 2)  # type: ignore[attr-defined]
    except AttributeError as e:
        rendered = "".join(traceback.format_exception_only(type(e), e))
    assert "Use d[k] = v." in rendered


# --- bytearray.take_bytes ------------------------------------------------------------------------

@py315
def test_bytearray_take_bytes_all():
    ba = bytearray(b"hello")
    assert ba.take_bytes() == b"hello"
    assert bytes(ba) == b""  # take_bytes removes what it returns


@py315
def test_bytearray_take_bytes_n():
    ba = bytearray(b"hello")
    assert ba.take_bytes(2) == b"he"
    assert bytes(ba) == b"llo"


# --- collections.Counter __xor__ / __ixor__ (symmetric difference) -------------------------------

@py315
def test_counter_xor_symmetric_difference():
    from collections import Counter
    assert dict(Counter(a=1, b=2) ^ Counter(b=1, c=3)) == {"a": 1, "b": 1, "c": 3}


@py315
def test_counter_ixor_in_place():
    from collections import Counter
    c = Counter(a=1, b=2)
    c ^= Counter(b=1, c=3)
    assert dict(c) == {"a": 1, "b": 1, "c": 3}


# --- math.integer and new float helpers ----------------------------------------------------------

@py315
def test_math_integer_module():
    import math
    assert hasattr(math, "integer")
    # The integer-only functions live under math.integer (PEP 791).
    assert math.integer.factorial(5) == 120
    assert math.integer.gcd(12, 18) == 6


@py315
def test_math_new_float_classifiers():
    import math
    assert math.isnormal(1.0) is True
    assert math.isnormal(0.0) is False
    assert math.signbit(-0.0) is True
    assert math.fmax(2.0, 3.0) == 3.0
    assert math.fmin(2.0, 3.0) == 2.0


# --- re.prefixmatch (match() soft-deprecated) ----------------------------------------------------

@py315
def test_re_prefixmatch_module_and_pattern():
    import re
    m = re.prefixmatch(r"ab", "abcdef")
    assert m is not None and m.group(0) == "ab"
    assert re.prefixmatch(r"zz", "abcdef") is None
    pat = re.compile(r"ab")
    assert pat.prefixmatch("abcdef") is not None


# --- asyncio.TaskGroup.cancel --------------------------------------------------------------------

@py315
def test_asyncio_taskgroup_cancel():
    import asyncio

    async def main():
        cancelled = {"n": 0}

        async def slow():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled["n"] += 1
                raise

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(slow())
                tg.create_task(slow())
                await asyncio.sleep(0.02)
                tg.cancel()  # new in 3.15: cancel the whole group
        except* asyncio.CancelledError:
            pass
        return cancelled["n"]

    assert asyncio.run(main()) == 2


# --- tomllib: TOML 1.1.0 -------------------------------------------------------------------------

@py315
def test_tomllib_toml_1_1_trailing_comma_in_inline_table():
    import tomllib
    # Trailing comma in an inline table is TOML 1.1.0 (rejected under 1.0).
    assert tomllib.loads("t = {a = 1, b = 2,}") == {"t": {"a": 1, "b": 2}}


# --- PEP 686: UTF-8 is the default I/O encoding --------------------------------------------------

@py315
def test_pep686_utf8_is_default_text_encoding(tmp_path):
    p = tmp_path / "t.txt"
    f = open(p, "w")  # NO encoding= argument
    try:
        assert f.encoding.lower() == "utf-8"
        f.write("héllo")
    finally:
        f.close()
    # Round-trips as UTF-8 bytes, proving the default encoding really was UTF-8.
    assert p.read_bytes() == "héllo".encode("utf-8")


# --- __cached__ no longer set (use __spec__.cached) ----------------------------------------------

@py315
def test_cached_attribute_removed_from_modules():
    import json  # a normal imported module
    assert not hasattr(json, "__cached__")
    # The replacement is on the module spec and still points at the .pyc.
    assert json.__spec__.cached is not None
    assert json.__spec__.cached.endswith(".pyc")


# --- GROUNDWIRE: 3.15 paradigms retrieved from the offline corpus (no model) ---------------------
# These are NOT version-guarded: retrieval is pure BM25 over bundled .md text, so it works the same
# on both the 3.15 and 3.12 venvs. The point is that a query about a 3.15 paradigm pulls back the
# right distilled chunk — the grounding the chat/help path leans on so the small model doesn't guess.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harness import pyhelp  # noqa: E402


@pytest.mark.parametrize(
    ("query", "expected_file"),
    [
        ("lazy import", "pep-810-lazy-import.md"),
        ("frozendict", "pep-814-frozendict.md"),
        ("unpacking in comprehensions", "pep-798-unpacking-in-comprehensions.md"),
        ("sentinel", "pep-661-sentinel.md"),
        ("utf-8 default encoding", "pep-686-utf8-default.md"),
        ("bytearray take_bytes", "builtins-and-collections.md"),
        ("re.prefixmatch", "stdlib-math-re-asyncio-tomllib.md"),
    ],
)
def test_py315_query_grounds_to_the_right_chunk(query, expected_file):
    pyhelp.reset_py315()  # fresh index -> deterministic
    hits = pyhelp.search_py315(query, k=1)
    assert hits, f"no 3.15 grounding retrieved for {query!r}"
    top_chunk = hits[0][0]
    # Each chunk carries a `# file: <name>` header, so the source feature/PEP is identifiable.
    assert top_chunk.splitlines()[0] == f"# file: {expected_file}"


def test_py315_grounding_tags_source_and_is_offline():
    pyhelp.reset_py315()
    hits = pyhelp.search_py315("how does lazy import work", k=2)
    assert hits
    joined = "\n".join(t for t, _s in hits)
    assert "# file:" in joined            # chunks are tagged with their source file
    assert "SyntaxError" in joined or "defer" in joined.lower()  # real doc text, not a canned string
