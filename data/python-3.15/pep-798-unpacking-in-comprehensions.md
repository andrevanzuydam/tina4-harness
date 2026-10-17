# PEP 798 — unpacking in comprehensions (Python 3.15)

Python 3.15 allows `*` and `**` unpacking inside list, set and dict comprehensions and inside
generator expressions (including async generator expressions).

```python
[*L for L in lists]        # flatten a list of lists into one list
{*s for s in sets}         # union of sets into one set
{**d for d in dicts}       # merge a sequence of dicts into one dict
(*L for L in lists)        # the same, as a generator expression
```

Examples:
- `[*L for L in [[1, 2], [3]]]` produces `[1, 2, 3]`.
- `{*s for s in [{1, 2}, {3}]}` produces `{1, 2, 3}`.
- `{**d for d in [{'a': 1}, {'b': 2}]}` produces `{'a': 1, 'b': 2}`.

Before 3.15 this was a `SyntaxError: iterable unpacking cannot be used in comprehension`. The new
syntax is a concise replacement for `itertools.chain` style flattening and for `dict` merging loops.
