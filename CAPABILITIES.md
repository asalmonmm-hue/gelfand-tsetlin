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
| PBW coproduct | Exact sparse pairs of ordered words via `coproduct()` | Unverified | Unverified |
| Quadratic Casimir | Trace-normalized `gl_N` element and exact highest-weight eigenvalue | `casimir_element()` for supported Lie algebras | Casimir operator and scalar for semisimple representations, with its own normalization |
| Standard and dual tensor decomposition | Multiplicity-one highest weights via `tensor_with_standard()` | Character products via `WeylCharacterRing` | Tensor products of semisimple Lie algebra modules |
| Dual-standard summand projectors | Exact sparse action in the uncoupled tensor GT basis | Unverified | Unverified |
| Quantum minors and the $\tau,A,B,C$ polynomials | PBW polynomial coefficients and right evaluation, including the Capelli determinant | Unverified | Unverified |
| Quantum-minor construction of GT basis vectors | Ordered $\kappa$ factors and exact scale in Molev’s GT normalization | Unverified | Unverified |
| Gelfand–Tsetlin subalgebra | Joint eigenvalues of the standard Capelli generators | Unverified | Unverified |
| Molev’s raising and lowering operators | Constructors, general sparse action, and direct highest-line action | Unverified | Unverified |
| Type B, C, and D matrix generators | Signed-index $F_{ij}$ in the ambient type A PBW basis | Matrix Lie algebras, with permutation between index conventions | Unverified |
| Type B, C, and D normalized branching generators | PBW representatives of the $z_{ia}$ and $z_{ai}$ for the same-type reductions | Unverified | Unverified |
| Classical interpolation polynomials | $Z_{n,-n}(u)$ at exact Cartan weights, with direct node operators | Unverified | Unverified |
| Type B patterns and basis construction | Validation and ordered factors of Theorem 3.3 | Unverified | Unverified |
| Yangian evaluation-module irreducibility | Exact endpoint tests for the general position of strings | Unverified | Unverified |

Exact checks against passageMath agreed on 59,808 validation decisions,
12,045 weight coordinates, and 375 rational PBW products. Generator actions
agreed with Macaulay2 on all 39,400 matrix entries across four `sl_3` modules
after matching the GT basis order.
The Casimir matched live Macaulay2 on 105 GT vectors and matched passageMath
symbolically on `sl_3` after converting to each tool's normalization.
All 20 tested standard or dual tensor products matched Macaulay2 after
conversion to Dynkin coordinates.
The classical matrix-generator comparison matched 226 generator matrices,
9,970 ordered Lie brackets, and six basis spans against passageMath 10.8.13
over $\mathbf{Q}$ in types $B_2,B_3,C_2,C_3,D_3,D_4$ after permuting indices.

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

### Coproduct and quadratic Casimir

`element.coproduct()` returns a read-only sparse mapping from pairs of
PBW words to exact coefficients. The empty word is the unit in either
factor. The coproduct sends `E[i,j]` to `E[i,j] ⊗ 1 + 1 ⊗ E[i,j]` and
extends multiplicatively. The C kernel splits the multiplicity of each
generator and evaluates the corresponding binomial coefficients. Both words
retain their PBW order.

`EnvelopingElement.quadratic_casimir(rank)` returns Molev’s trace-normalized
Casimir $C = \sum_{i,j} E_{ij}E_{ji}$ in PBW form. Its coproduct has cross
term $\Delta(C)-C\otimes 1-1\otimes C=2\sum_{i,j}E_{ij}\otimes E_{ji}$.
[Molev, §2.4](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import EnvelopingElement

x = EnvelopingElement.matrix_unit(2, 1, 2)
word = ((1, 2),)
assert (x**2).coproduct() == {(word * 2, ()): 1, (word, word): 2, ((), word * 2): 1}

casimir = EnvelopingElement.quadratic_casimir(2)
assert casimir * x == x * casimir
assert casimir.coproduct()[word, ((2, 1),)] == 2
```

### Casimir eigenvalues and tensoring with the standard module

`quadratic_casimir_eigenvalue(highest_weight)` evaluates
$c(\lambda)=\sum_i\lambda_i(\lambda_i+N-2i+1)$ in exact arithmetic.
`tensor_with_standard(highest_weight, dual=False)` returns the dominant
summand labels $\lambda+\varepsilon_i$, each with multiplicity one.
With `dual=True` it returns the dominant $\lambda-\varepsilon_i$ labels.
The inputs are real highest weights with nonnegative integral gaps.

The Casimir differences give the scalar of the cross term on each summand.
For the standard module, the half cross term acts by $\lambda_i-i+1$.
For the dual standard module, Molev’s matrix $E$ is the negative of the
half cross term and acts by $\lambda_i+N-i$.
[Molev, equations (2.30) and (2.31)](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import quadratic_casimir_eigenvalue, tensor_with_standard

assert quadratic_casimir_eigenvalue((2, 1, 0)) == 9
assert tensor_with_standard((2, 2, 0)) == ((3, 2, 0), (2, 2, 1))
assert tensor_with_standard((2, 2, 0), dual=True) == ((2, 1, 0), (2, 2, -1))
```

### Dual-standard tensor-summand projectors

`apply_dual_standard_projector(source, j, r)` applies Molev's projector
$P_r$ to the uncoupled vector $\varepsilon_j\otimes\xi_\Lambda$.
The result maps pairs `(i, RowArray)` to exact coefficients. The index
`r` identifies the summand with highest weight $\lambda-\varepsilon_r$.
An index in range with an inadmissible summand gives an empty result.

The implementation applies the Lagrange polynomial in the characteristic
operator along one sparse vector, using the admissible eigenvalues
$\lambda_s+N-s$. It uses the unnormalized GT basis and the same real-shift
and coefficient range rules as the general sparse action.
[Molev, equations (2.32)–(2.34)](https://arxiv.org/abs/math/0211289).

```python
from fractions import Fraction
from gelfand_tsetlin import RowArray, apply_dual_standard_projector

first = RowArray([[1, 0], [1]])
second = RowArray([[1, 0], [0]])
assert apply_dual_standard_projector(first, 1, 1) == {
    (1, first): Fraction(1, 2),
    (2, second): Fraction(1, 2),
}
```

## Quantum minors

`quantum_minor(rank, rows, columns, u=None)` uses Molev's column-ordered
formula with the factors at $u,u-1,\ldots$. Row and column indices are
one-based and their supplied order determines the sign. Repeated indices
give zero and the empty minor is the unit. The full principal minor is
the Capelli determinant.

Omit `u` to get a tuple of `EnvelopingElement` coefficients in ascending
degree. Pass an integer, fraction, or same-rank `EnvelopingElement` to
evaluate. In operator-valued evaluation, each power of `u` multiplies its
coefficient on the right. [Molev, equations (2.35) and (2.38)](https://arxiv.org/abs/math/0211289).

For scalar or formal $u$, the commutator with $E_{ab}$ in (2.37) is a
sum of at most two minors. One replaces row index $b$ by $a$, and the
other replaces column index $a$ by $b$ and is subtracted. The public PBW
arithmetic evaluates commutators using the general multiplication kernel.

```python
from gelfand_tsetlin import EnvelopingElement, quantum_minor

e11 = EnvelopingElement.matrix_unit(2, 1, 1)
e12 = EnvelopingElement.matrix_unit(2, 1, 2)
e21 = EnvelopingElement.matrix_unit(2, 2, 1)
e22 = EnvelopingElement.matrix_unit(2, 2, 2)
constant, linear, leading = quantum_minor(2, (1, 2), (1, 2))
assert constant == e11 * (e22 - 1) - e21 * e12
assert linear == e11 + e22 - 1
assert leading == 1
```

### Tau polynomials

`tau_lowering(rank, i, u=None, *, level=None)` and
`tau_raising(rank, i, u=None, *, level=None)` construct Molev’s
$\tau_{ki}(u)$ and $\tau_{ik}(u)$, where $k$ is `level` and defaults to
`rank`. They accept $1\le i<k\le N$ and return PBW coefficients or an
evaluated element using the same convention as `quantum_minor`.

On a $\mathfrak{gl}_{k-1}$-highest vector of weight $\mu$, evaluate
the lowering polynomial at $u=-\mu_i$ and the raising polynomial at
$u=-\mu_i+i-1$. The equivalent substitutions within the enveloping
algebra are $-E_{ii}$ and $-E_{ii}+i-1$, respectively. These
specializations agree with the chain operators on highest vectors.
Their PBW expressions can differ by terms in the left ideal generated
by the $\mathfrak{gl}_{k-1}$ raising generators.
For the action at these specializations, use `apply_highest_lowering`
or `apply_highest_raising` with `level=k` to skip constructing the PBW
polynomial.
[Molev, Propositions 2.18 and 2.19](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import EnvelopingElement, tau_raising

e11 = EnvelopingElement.matrix_unit(3, 1, 1)
e13 = EnvelopingElement.matrix_unit(3, 1, 3)
e21 = EnvelopingElement.matrix_unit(3, 2, 1)
e23 = EnvelopingElement.matrix_unit(3, 2, 3)
assert tau_raising(3, 2) == (e21 * e13 - e23 * e11, -e23)
```

### A, B, and C polynomials

Write $I=(1,\ldots,m)$ and $J=(1,\ldots,m-1,m+1)$. The functions
`a_polynomial(rank, m, u=None)`, `b_polynomial(rank, m, u=None)`, and
`c_polynomial(rank, m, u=None)` construct the minors with row and column
lists $(I,I)$, $(I,J)$, and $(J,I)$, respectively. The principal minor
$A_m$ is the Capelli determinant of the upper-left $\mathfrak{gl}_m$.
$B_m$ and $C_m$ require $m<\mathrm{rank}$. All three use the same
coefficient and evaluation conventions as `quantum_minor`.
[Molev, §2.5 and Theorem 2.20](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import a_polynomial, b_polynomial, c_polynomial
from gelfand_tsetlin import quantum_minor, tau_raising

assert a_polynomial(3, 2) == quantum_minor(3, (1, 2), (1, 2))
assert b_polynomial(3, 2, 0) == -tau_raising(3, 2, 0)
assert c_polynomial(3, 2) == quantum_minor(3, (1, 3), (1, 2))
```

### Quantum-minor construction of the GT basis

`kappa_factors(pattern)` yields pairs `(m, u)` for the product of
$C_m(u)$ factors in Molev’s Theorem 2.21, in the order in which the
product is written. The rightmost factor acts first.
`kappa_scale(pattern)` computes the exact integer $s$ in
$\kappa_\Lambda=s\,\xi_\Lambda$ for the same choice of highest vector.
It groups the single-step coefficients of (2.40) into products of
consecutive integers and needs no PBW expansion. Both functions accept
real common shifts. [Molev, Theorems 2.20 and 2.21](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import kappa_factors, kappa_scale

pattern = ((4, 2, 0), (2, 1), (1,))
assert list(kappa_factors(pattern)) == [(2, -3), (2, -4), (1, -2), (1, -3), (1, -4), (2, -1)]
assert kappa_scale(pattern) == 18
```

### Gelfand–Tsetlin subalgebra

`gt_eigenvalues(pattern)` returns the joint eigenvalues of the standard
generators $a_{mi}$ of the GT subalgebra. The outer tuple lists levels
from $m=1$ to $N$. The $m$-th inner tuple lists $\alpha_{m1},\ldots,\alpha_{mm}$
in descending degree, omitting the monic leading coefficient of
$\prod_{i=1}^m(u+\lambda_{mi}-i+1)$. The eigenvalues are exact for real
common shifts. To evaluate a polynomial in the generators on a GT vector,
substitute these eigenvalues for its variables.
[Molev, equation (2.41) and Proposition 2.22](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import gt_eigenvalues

assert gt_eigenvalues(((2, 1, 0), (2, 1), (1,))) == ((1,), (2, 0), (0, -4, 0))
```

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

### Direct highest-line action

`apply_highest_raising(source, i, level=None)` and
`apply_highest_lowering(source, i, level=None)` act on a
`gl_(level-1)`-highest GT vector. The level defaults to the ambient rank.
A source is highest at that step when every row below the length-`level-1`
row is a prefix of it. An admissible move changes entry `i` in that row
and in every shorter prefix that contains it.

Each call returns a sparse dictionary with at most one target. An
inadmissible shift yields an empty dictionary. An invalid pattern or
a pattern off the highest line raises `ValueError`. As with
`apply_enveloping`, the source can have a real common shift.

In Molev’s normalization, lowering has coefficient one and raising uses
the product in Lemma 2.13. The C evaluator takes `O(level)` arithmetic
operations for that product and uses exact Python integers of arbitrary
size. Input validation and target construction are separate costs.
The direct calls avoid constructing or applying a PBW expression.
[Molev, equations (2.18) and (2.28)](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import RowArray, apply_highest_lowering, apply_highest_raising

highest = RowArray([[1, 0, 0], [1, 0], [1]])
third = RowArray([[1, 0, 0], [0, 0], [0]])
assert apply_highest_lowering(highest, 1) == {third: 1}
assert apply_highest_raising(third, 1) == {highest: 2}
```

## Orthogonal and symplectic matrix generators

`classical_matrix_unit(kind, rank, i, j)` constructs Molev’s
$F_{ij}=E_{ij}-\theta_{ij}E_{-j,-i}$ in the ambient
`EnvelopingElement` representation. Use `"B"` for $\mathfrak o_{2n+1}$,
`"C"` for $\mathfrak{sp}_{2n}$, and `"D"` for $\mathfrak o_{2n}$, with
`rank=n`. The signed indices run from $-n$ to $n$ with zero included only
in type B. PBW terms use the corresponding one-based ambient matrix
positions. $\theta_{ij}$ is $1$ in types B and D and
$\operatorname{sgn}(i)\operatorname{sgn}(j)$ in type C.
[Molev, equations (3.3) and (3.4)](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import EnvelopingElement, classical_matrix_unit

assert classical_matrix_unit("C", 2, 2, -2) == 2 * EnvelopingElement.matrix_unit(4, 4, 1)
assert classical_matrix_unit("D", 2, 2, -2) == 0
```

### Normalized branching generators

`classical_branching_generator(kind, rank, i, j)` constructs a PBW
representative of Molev’s normalized $z_{ij}$ from §3.1. Exactly one
index is an outer label $a\in\{-n,n\}$, and the other is an inner index
of $\mathfrak g_{n-1}$. Positive inner indices give raising operators
$z_{ia}$ and lowering operators $z_{ai}$. The type B zero-index operators
preserve the $\mathfrak g_{n-1}$ highest weight.

The implementation groups the descending chains of (3.8) and keeps
Cartan factors on the right. The returned representatives act as the
normalized generators on $\mathfrak g_{n-1}$-highest vectors. Reconstructing
matrix-unit actions with Lemma 3.1 uses products that omit $j=i$ in
types B and C, and $j=\pm i$ in type D. Type D chains through $-i$ can
still contribute to the normalized generator.
[Molev, equation (3.8) and Lemma 3.1](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import classical_branching_generator, classical_matrix_unit

def f(i, j):
    return classical_matrix_unit("C", 2, i, j)

assert classical_branching_generator("C", 2, 1, 2) == (
    f(1, 2) * (2 * f(1, 1) - 2) + f(1, -1) * f(-1, 2)
)
```

### Interpolation polynomials

`classical_interpolation_polynomial(kind, rank, weight, u=None)`
specializes the right Cartan functions of $Z_{n,-n}(u)$ at `weight`,
the eigenvalues of $F_{11},\ldots,F_{nn}$ on the input vector. Omit `u`
to get coefficients in ascending powers of $u^2$. Supply an integer or
fraction to evaluate. The PBW representatives give the specialized
action on the highest-vector space. For successive actions, the input
weight is the weight at that step. At a collision with the outer node,
types B and C use the leading coefficient and the inner nodes. Type D
uses the signed-node formula (3.31) at coincident squared nodes.

`classical_interpolation_node(kind, rank, i)` gives the symbolic value at
$u=g_i$ without specializing a weight. Use $1\le i\le n$ in B and C,
and $1\le i<n$ in D. In B and C the last value is the standalone
$z_{n,-n}$. [Molev, equations (3.10)–(3.12)](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import classical_interpolation_node, classical_interpolation_polynomial

assert classical_interpolation_polynomial("C", 1, (-1,), 0) == (
    classical_interpolation_node("C", 1, 1)
)
```

### Type B basis patterns

A type B pattern is a sequence of `(sigma, highest_row, intermediate_row)`
triples in descending level order. At level $k$, both rows have length
$k$. All entries are nonpositive integers or all are nonpositive
half-integers. `is_type_b_gt_pattern` checks both interlacing rules and
the sign bit. `type_b_basis_factors` yields the product in written order,
with factors `("z", k, i, j)` or `("Z", k, u)`. Apply the rightmost
factor first, starting at a highest vector. Each factor belongs to the
rank-$k$ subalgebra. [Molev, Lemma 3.2 and Theorem 3.3](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import is_type_b_gt_pattern, type_b_basis_factors

pattern = ((1, (-2,), (-1,)),)
assert is_type_b_gt_pattern(pattern)
assert list(type_b_basis_factors(pattern)) == [("z", 1, 1, 0), ("Z", 1, -2)]
```

## Yangian evaluation-module strings

`yangian_strings_in_general_position(first, second)` accepts two
highest-weight pairs `(alpha, beta)` for $L(\alpha,\beta)$. The corresponding
string is $\{\beta,\beta+1,\ldots,\alpha-1\}$. The test compares endpoints
and unit-step lattices. `yangian_tensor_is_irreducible(highest_weights)`
checks the condition for every pair, as in Molev’s Theorem 3.8.
[Molev, §3.3](https://arxiv.org/abs/math/0211289).

```python
from gelfand_tsetlin import yangian_strings_in_general_position, yangian_tensor_is_irreducible

assert yangian_strings_in_general_position((3, 0), (2, 1))
assert not yangian_strings_in_general_position((2, 0), (4, 2))
assert yangian_tensor_is_irreducible([(2, 0), (5, 3)])
```
