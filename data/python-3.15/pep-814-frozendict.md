# PEP 814 — frozendict builtin (Python 3.15)

`frozendict` is a new built-in: an immutable, hashable mapping. It inherits directly from `object`,
NOT from `dict`, so `issubclass(frozendict, dict)` is `False`.

```python
fd = frozendict(x=1, y=2)
fd['z'] = 3   # TypeError: 'frozendict' object does not support item assignment
```

Properties:
- Immutable — item assignment raises
  `TypeError: 'frozendict' object does not support item assignment`.
- Hashable when all keys and values are hashable, so a `frozendict` can be a dict key or set member.
- Preserves insertion order.
- Equality ignores order: `frozendict(a=1, b=2) == frozendict(b=2, a=1)` is `True`.

It is accepted by `copy`, `decimal`, `json`, `marshal`, `plistlib` (serialization only), `pickle`,
`pprint`, and `xml.etree.ElementTree`, and by `eval()` / `exec()` for the globals argument and by
`type()` and `str.maketrans()` for their dict argument.
