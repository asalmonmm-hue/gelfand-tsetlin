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

## Contravariant form

`inner_product(left, right)` evaluates Molev's Proposition 2.4 on two GT
basis vectors with the same highest weight. Distinct patterns pair to zero.
For a matching pair, each quotient of factorials reduces to a product of
consecutive integers. The C kernel multiplies those factors without allocating
or computing factorials. It returns an exact integer and reports overflow above
`INT64_MAX`.

```python
from gelfand_tsetlin import inner_product

assert inner_product([[2, 0], [1]], [[2, 0], [1]]) == 2
assert inner_product([[2, 0], [1]], [[2, 0], [2]]) == 0
```

## Universal enveloping algebra

`EnvelopingElement` represents elements of the rational form of `U(gl_N)`
as sparse sums of PBW monomials in row-major matrix-unit order. Use
`matrix_unit(rank, i, j)` for a generator and `scalar(rank, value)` for
a constant. The constructor also accepts a mapping from words of one-based
pairs to integers or fractions. The empty word is the identity.

```python
from gelfand_tsetlin import EnvelopingElement

e12 = EnvelopingElement.matrix_unit(2, 1, 2)
e21 = EnvelopingElement.matrix_unit(2, 2, 1)
e11 = EnvelopingElement.matrix_unit(2, 1, 1)
e22 = EnvelopingElement.matrix_unit(2, 2, 2)
assert e21 * e12 == e12 * e21 + e22 - e11
assert (e12 ** 2).terms == {((1, 2), (1, 2)): 1}
```

The C multiplication kernel normal-orders products using the matrix-unit
commutators and combines equal terms. Symbolic products allocate storage for
their terms and can grow as commutators expand.

## Molev’s raising and lowering operators

`raising_operator(rank, i, level=None)` and `lowering_operator(rank, i, level=None)`
construct Molev’s `z[i,level]` and `z[level,i]` from equations (2.9) and
(2.10). All indices are one-based. The level defaults to the ambient rank.
A lower level selects the corresponding upper-left subalgebra.

`apply_enveloping(element, source)` applies the element to one GT vector,
acting with the rightmost factor first. It returns a dictionary of target
patterns and nonzero coefficients. It uses the existing C coefficients for
adjacent generators and composes sparse commutators for other matrix units.
The action accepts real common shifts and uses exact rational coefficients
within the range of the fixed-width coefficient kernel.

```python
from gelfand_tsetlin import RowArray, apply_enveloping, lowering_operator

highest = RowArray([[1, 0, 0], [1, 0], [1]])
third = RowArray([[1, 0, 0], [0, 0], [0]])
assert apply_enveloping(lowering_operator(3, 1), highest) == {third: 1}
```

## Run locally

Build with CPython and a C compiler for that Python installation.

```sh
python3 -m pip install .
python3 -m unittest discover -s tests
```

Python provides the row-array interface and the C extension implements the validity check.
