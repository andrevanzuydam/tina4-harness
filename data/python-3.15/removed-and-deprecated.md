# Removed and deprecated in Python 3.15

## __cached__ no longer set
The import system no longer sets or uses the module `__cached__` attribute. Use `__spec__.cached`
instead to find a module's compiled `.pyc` path.

```python
import json
hasattr(json, '__cached__')   # False in 3.15
json.__spec__.cached          # '.../json/__pycache__/__init__.cpython-315.pyc'
```

## profiling package (PEP 799); profile deprecated
The new `profiling` package contains `profiling.tracing` (deterministic, moved from `cProfile`) and
`profiling.sampling` (the Tachyon statistical sampler that attaches to a running process). `cProfile`
stays as an alias. The old `profile` module is deprecated and will be removed in Python 3.17.

## Selected removals
- `http.server.CGIHTTPRequestHandler` and the `--cgi` flag removed.
- `pathlib.PurePath.is_reserved()` removed — use `os.path.isreserved()`.
- `importlib.Loader.load_module()` and `zipimport.zipimporter.load_module()` removed.
- `glob.glob0()` / `glob.glob1()` removed.
- `platform.java_ver()` removed.
- `sre_compile`, `sre_constants`, `sre_parse` modules removed.
- `typing`: keyword-argument `NamedTuple("P", x=int)` form removed; zero-field `TypedDict` removed;
  `typing.no_type_check_decorator` removed; `ByteString` removed from `typing.__all__`.

## Selected deprecations
- `re.match()` / `Pattern.match()` soft-deprecated (use `prefixmatch()`); will not be removed.
- `collections.abc.ByteString` and `typing.ByteString` emit `DeprecationWarning`.
- `__version__` / `version` / `VERSION` deprecated in many stdlib modules (use `sys.version_info`).
- `struct` `'F'` / `'D'` type codes soft-deprecated in favour of `'Zf'` / `'Zd'`.
