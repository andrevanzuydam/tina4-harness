# Improved error messages (Python 3.15)

`AttributeError` now suggests the right method when you call a method from a different type on a
builtin container. The suggestion is added by the traceback renderer; the exception's own message
(`str(e)`) is unchanged.

```python
[].push(1)
# AttributeError: 'list' object has no attribute 'push'. Did you mean '.append'?

{}.put(1, 2)
# AttributeError: 'dict' object has no attribute 'put'. Use d[k] = v.

(1, 2, 3).append(4)
# suggests using a list
```

- `str(e)` is still just `'list' object has no attribute 'push'`; the `Did you mean '.append'?`
  text appears in the formatted traceback (e.g. `traceback.format_exception_only`).
- `AttributeError` can also suggest a nested attribute like `.inner.area` when an inner member has
  the attribute.
- `delattr()` suggests similar names.
- `ImportError` / `ModuleNotFoundError` repr shows `name=` and `path=` when given as keywords.
