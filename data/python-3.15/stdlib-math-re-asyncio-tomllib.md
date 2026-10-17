# Notable stdlib additions (Python 3.15): math, re, asyncio, tomllib

## math.integer (PEP 791) and new float helpers
Integer-only math functions are grouped under the new `math.integer` submodule, e.g.
`math.integer.factorial`, `math.integer.gcd`, `math.integer.lcm`, `math.integer.isqrt`,
`math.integer.comb`, `math.integer.perm`.

`math` also gains `isnormal()`, `issubnormal()`, `fmax()`, `fmin()`, and `signbit()`:

```python
math.isnormal(1.0)   # True
math.isnormal(0.0)   # False
math.signbit(-0.0)   # True
math.fmax(2.0, 3.0)  # 3.0
```

## re.prefixmatch (re.match soft-deprecated)
`re.prefixmatch()` and `Pattern.prefixmatch()` are the new names for `re.match()` / `Pattern.match()`,
which are now soft-deprecated (not removed) because "match" was confusing next to `fullmatch`/`search`.

```python
re.prefixmatch(r'ab', 'abcdef')   # matches at the start
```

## asyncio.TaskGroup.cancel
`asyncio.TaskGroup` gains a `cancel()` method that cancels every task in the group at once.

## tomllib — TOML 1.1.0
`tomllib` now supports TOML 1.1.0: newlines and trailing commas inside inline tables, `\xHH` and `\e`
string escapes, and optional seconds in times.

```python
tomllib.loads('t = {a = 1, b = 2,}')   # trailing comma OK in 1.1.0
```
