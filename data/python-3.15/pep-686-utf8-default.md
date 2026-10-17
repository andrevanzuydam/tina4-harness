# PEP 686 — UTF-8 is the default I/O encoding (Python 3.15)

From Python 3.15, text I/O without an explicit `encoding` argument uses UTF-8 instead of the
locale's preferred encoding.

```python
with open('t.txt', 'w') as f:   # no encoding= -> UTF-8
    f.write('héllo')
```

- `open(path, 'w').encoding` is `'utf-8'`.
- This makes code behave the same across platforms and locales by default, removing a common source
  of `UnicodeDecodeError` on Windows.
- Revert to the old locale behaviour with `PYTHONUTF8=0` or `-X utf8=0`.
- Pass `encoding='locale'` to explicitly request the locale encoding for a single call.
