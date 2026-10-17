# Small language changes (Python 3.15): match +literal, slice generic, __slots__

## Unary plus in match literal patterns
`match` patterns now allow a unary plus before a number literal, mirroring the existing unary minus:

```python
match v:
    case +1:
        ...
    case -1:
        ...
```

## slice is subscriptable (generic)
`slice` is now a generic type, so `slice[int]` is valid. Before 3.15 this raised `TypeError`.

```python
s = slice[int]       # valid in 3.15
```

## __slots__ may declare __dict__ and __weakref__
Any class may now list `__dict__` (and `__weakref__`) in `__slots__`, so an instance gets both the
named slots and a normal attribute dict:

```python
class C:
    __slots__ = ('x', '__dict__')

c = C()
c.x = 1          # the declared slot
c.anything = 2   # allowed, because __dict__ is in __slots__
```

Additionally, any `__slots__` is now allowed for `tuple` subclasses, including `namedtuple` classes.
