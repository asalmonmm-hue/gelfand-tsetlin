# Gelfand–Tsetlin patterns for `gl_N`

The examples follow [Molev, §2](https://arxiv.org/abs/math/0211289). Read the rows from the highest weight at the top down to the length-one row.

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

## Matrix coefficients

`coefficient(i, j, source, target)` returns the coordinate of `target` in
`E[i,j]` applied to `source`, in Molev's Theorem 2.3 basis. The indices are
one-based and may be equal or adjacent. Both patterns must have the same
highest weight. A valid target outside the support has coefficient zero.

```python
from fractions import Fraction

from gelfand_tsetlin import coefficient

source = [[4, 2, 0], [3, 1], [2]]
assert coefficient(2, 2, source, source) == 2
assert coefficient(2, 3, source, [[4, 2, 0], [4, 1], [2]]) == Fraction(10, 3)
assert coefficient(2, 3, source, source) == 0
```

The standalone C kernel borrows packed integer rows and uses constant stack
space with no heap allocation. Python validates and packs the input and boxes
the output. The kernel uses fixed-width arithmetic and raises `OverflowError`
through the Python interface if a normalized entry or a coefficient calculation
exceeds its range. The real results are exact. For a nonreal common shift, a
diagonal result is returned as a Python `complex`, with its usual precision.

## Run locally

Build with CPython and a C compiler for that Python installation.

```sh
python3 -m pip install .
python3 -m unittest discover -s tests
```

Python provides the row-array interface and the C extension implements the validity check.
