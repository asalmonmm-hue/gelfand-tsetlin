# Capabilities

See the [README](README.md) for setup and benchmarks.

## Comparison

The comparison uses passageMath’s Sage-compatible API and Macaulay2’s
`LieAlgebraRepresentations` package. **Unverified** means an equivalent
API was not established in this comparison.

| Capability | gelfand-tsetlin | SageMath / passageMath | Macaulay2 |
| --- | --- | --- | --- |
| Raw pattern storage | `RowArray` preserves nested rows before validation | Native nested sequences and `GelfandTsetlinPattern` | Flat entry list and type A `GTPattern` |
| Pattern validation | Checks shape, integrality of gaps, and common shifts | Membership checks interlacing on supplied rows | Type A constructor checks triangular entry count and interlacing |
| Full diagonal weights | `weight(pattern)` | `weight()` | Pattern content and simple-coroot coordinates |
| Fixed-top-row enumeration | Planned | `GelfandTsetlinPatterns(top_row=...)` | `gtPatterns` for partition top rows |
| Semistandard tableau conversion | Planned | `to_tableau()` and construction from tableaux | Unverified |
| GT generator action | Single coefficients and sparse columns for `gl_N` | GT-specific action unverified | `GTrepresentationMatrices` for irreducible `sl_N` modules |
| Exact dense generator matrices | Assemble from sparse columns | Unverified | Direct via `GTrepresentationMatrices` |
| Contravariant pairing in Molev’s basis | `inner_product` | Unverified | Unverified |
| Rational PBW arithmetic | `EnvelopingElement` in row-major matrix-unit order | `pbw_basis()` | `universalEnvelopingAlgebra` on the documented semisimple domain |
| Molev’s raising and lowering operators | Constructors and sparse action | Unverified | Unverified |

Exact checks against passageMath agreed on 59,808 validation decisions,
12,045 weight coordinates, and 375 rational PBW products. Generator actions
agreed with Macaulay2 on all 39,400 matrix entries across four `sl_3` modules
after matching the GT basis order.

[SageMath pattern API](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/gelfand_tsetlin_patterns.html) ·
[SageMath PBW API](https://doc.sagemath.org/html/en/reference/algebras/sage/algebras/lie_algebras/poincare_birkhoff_witt.html) ·
[Macaulay2 LieAlgebraRepresentations](https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/LieAlgebraRepresentations/html/index.html)

## Patterns and validation

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
Built-in integer lists and tuples use a direct C check on the supplied rows.

## Weights

`weight(pattern)` returns all the diagonal weights in order from `E[1,1]`
through `E[N,N]`. It validates the pattern once and sums each row once.
Real weights are exact. A nonreal common shift uses Python's `complex`
precision, as for the single-entry diagonal API.

```python
from gelfand_tsetlin import weight

assert weight([[2, 1, 0], [1, 0], [1]]) == (1, 0, 2)
```

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

The C multiplication kernel joins already ordered word pairs directly and
normal-orders other products using the matrix-unit commutators. It combines
equal terms. Symbolic products allocate storage for
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
