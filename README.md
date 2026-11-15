# Gelfand–Tsetlin patterns for `gl_N`

The first example follows [Molev, §2, equations (2.1) and (2.4)](https://arxiv.org/abs/math/0211289). Read the rows from the highest weight at the top down to the length-one row.

```python
from fractions import Fraction

from gelfand_tsetlin import RowArray, is_gt_pattern

rows = RowArray([[2, 1, 0], [2, 0], [1]])
assert is_gt_pattern(rows)
assert not is_gt_pattern(RowArray([[2, 1, 0], [3, 0], [1]]))

half = Fraction(1, 2)
assert is_gt_pattern([[2 + half, half], [1 + half]])
```

`RowArray` stores the supplied rows as a tuple of tuples. The predicate checks for positive rank, row lengths `N, N-1, ..., 1`, and the two nonnegative integer differences for each entry below the top row:

```text
upper[j] - lower[j]
lower[j] - upper[j + 1]
```

Every entry may share a common complex shift, since the shift cancels from both differences.
The validator accepts Python integers, `Fraction`, and finite `Decimal`, `float`, and `complex` values. It uses the exact stored values of floats and complex components. Use `Fraction` or `Decimal` when the shift is intended to be exact in decimal.

## Run locally

Build with CPython and a C compiler for that Python installation.

```sh
python3 -m pip install .
python3 -m unittest discover -s tests
```

Python provides the row-array interface and the C extension implements the validity check.
