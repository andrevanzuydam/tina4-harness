# PEP 810 — lazy import (Python 3.15)

Python 3.15 adds a `lazy` soft keyword that defers importing a module until the first time one of
its attributes is actually used.

```python
lazy import json
lazy from pathlib import Path
```

The module is NOT loaded when the `lazy import` statement runs; it is loaded the first time the name
is accessed. This speeds up start-up when an expensive import is only needed on some code paths.

Restrictions (each is a `SyntaxError`):
- `lazy import` is only allowed at module scope — never inside a function, a class body, or a
  `try` / `except` / `finally` block. Inside a function it raises
  `SyntaxError: lazy import not allowed inside functions`.
- `lazy from m import *` is not allowed.
- `lazy from __future__ import ...` is not allowed.

Control it globally with the `-X lazy_imports=all|normal` interpreter option or the
`PYTHON_LAZY_IMPORTS` environment variable (`normal` is the default). The explicit `lazy` keyword
defers regardless of the flag.

Runtime API: `sys.set_lazy_imports()`, `sys.get_lazy_imports()`, and
`sys.set_lazy_imports_filter(callable)` (the filter receives `(importing, imported, fromlist)` and
returns `True` to allow lazy loading). The proxy object has type `types.LazyImportType`. A module can
list fully-qualified names in `__lazy_modules__` to make their plain `import` statements lazy.
