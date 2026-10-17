# PEP 661 — sentinel builtin (Python 3.15)

`sentinel` is a new built-in type for creating unique sentinel values — the standard way to express
"no argument was given", distinct from `None`.

Constructor signature (verified on the interpreter):

```python
sentinel(name, /, *, repr=None)
```

- `name` is positional-only.
- `repr` is an optional keyword-only custom string representation.

```python
MISSING = sentinel('MISSING')

def get(d, key, default=MISSING):
    if default is MISSING:
        ...
```

Properties:
- Identity is preserved across `copy.copy` and `copy.deepcopy` — `copy.copy(MISSING) is MISSING`.
- It is picklable when importable by module and name.
- It supports `|` in type expressions.
- A custom repr is shown as given: `sentinel('S', repr='<S>')` has `repr(...) == '<S>'`.
