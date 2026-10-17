# New builtin methods and collections changes (Python 3.15)

## bytearray.take_bytes
`bytearray.take_bytes(n=None, /)` removes and returns bytes from the front of the bytearray without
copying. With no argument it takes everything.

```python
ba = bytearray(b'hello')
ba.take_bytes(2)   # b'he'   -> ba is now bytearray(b'llo')
ba.take_bytes()    # b'llo'  -> ba is now empty
```

## collections.Counter symmetric difference
`Counter` gains `__xor__` and `__ixor__` (`^` and `^=`) for multiset symmetric difference:

```python
from collections import Counter
Counter(a=1, b=2) ^ Counter(b=1, c=3)   # Counter({'c': 3, 'a': 1, 'b': 1})
```

For each key the result count is the absolute difference of the two counts; keys present in only one
side keep their count.

## bytes.replace keyword count
`bytes.replace()` now accepts its `count` argument by keyword.
