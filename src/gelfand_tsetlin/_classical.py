"""Matrix and branching generators in Molev's classical conventions."""

from fractions import Fraction
from operator import index

from ._enveloping import EnvelopingElement, _checked_rank


def _classical_dimension(kind, rank):
    if kind not in ("B", "C", "D"):
        raise ValueError("kind must be 'B', 'C', or 'D'")
    rank = index(rank)
    if rank < 1:
        raise ValueError("rank must be positive")
    return _checked_rank(2 * rank + (kind == "B"))


def _classical_indices(kind, rank, i, j):
    dimension = _classical_dimension(kind, rank)
    rank = dimension // 2
    i, j = index(i), index(j)
    if not (-rank <= i <= rank and -rank <= j <= rank):
        raise ValueError("signed matrix-unit indices must lie within the rank")
    if kind != "B" and (i == 0 or j == 0):
        raise ValueError("index zero is only valid in type B")
    return dimension, i, j


def _matrix_unit(kind, dimension, i, j):
    rank = dimension // 2
    has_zero = kind == "B"
    row = i + rank + (has_zero or i < 0)
    column = j + rank + (has_zero or j < 0)
    direct = EnvelopingElement.matrix_unit(dimension, row, column)
    reflected = EnvelopingElement.matrix_unit(dimension, dimension + 1 - column, dimension + 1 - row)
    if kind == "C" and (i < 0) != (j < 0):
        return direct + reflected
    return direct - reflected


def classical_matrix_unit(kind, rank, i, j):
    """Return Molev's F[i,j] in the defining matrix embedding.

    The rank is positive and kind is B, C, or D. The signed indices run
    from -rank to rank with zero included only in type B. The resulting
    enveloping element uses consecutive one-based indices in that order.
    """
    dimension, i, j = _classical_indices(kind, rank, i, j)
    return _matrix_unit(kind, dimension, i, j)


def _shifted_cartan(kind, dimension, i):
    if i == 0:
        return EnvelopingElement.scalar(dimension, Fraction(-1, 2))
    shift = -i
    if kind == "B":
        shift += Fraction(1 if i > 0 else -1, 2)
    elif kind == "D":
        shift += 1 if i > 0 else -1
    return _matrix_unit(kind, dimension, i, i) + shift


def _ordered_paths(kind, dimension, i, a, labels, factors):
    starts = (*labels, i)
    paths = {start: _matrix_unit(kind, dimension, start, a) for start in starts}
    # Scan upward so paths[j] already accounts for labels below j.
    # A chain from a higher start either skips j or takes j as its first step.
    for pos, j in enumerate(labels):
        for start in starts[pos + 1:]:
            edge = _matrix_unit(kind, dimension, start, j)
            paths[start] = paths[start] * factors[j] + edge * paths[j]
    return paths[i]


def _opposite_prefix(dimension, i, labels, cartan):
    # Store P(0) and the right evaluation of (P(t) - P(0)) / t at f_i + 1.
    # When a label j is skipped, P(t) is multiplied on the right by t - f_j.
    starts = (*labels, i)
    zero = EnvelopingElement.scalar(dimension, 0)
    prefix = {start: (_matrix_unit("D", dimension, start, -i), zero) for start in starts}
    for pos, j in enumerate(labels):
        f_j = _shifted_cartan("D", dimension, j)
        shifted = cartan + 1 - f_j
        step_constant, step_quotient = prefix[j]
        for start in starts[pos + 1:]:
            constant, quotient = prefix[start]
            edge = _matrix_unit("D", dimension, start, j)
            prefix[start] = (
                edge * step_constant - constant * f_j,
                edge * step_quotient + constant + quotient * shifted,
            )
    # The full opposite prefix has P(0) = 0 in type D.
    return prefix[i][1]


def _branching_generator(kind, dimension, i, a):
    rank = dimension // 2
    labels = tuple(j for j in range(1 - rank, i) if j or kind == "B")
    cartan = _shifted_cartan(kind, dimension, i)
    factors = {j: cartan - _shifted_cartan(kind, dimension, j) for j in labels}
    if kind != "D" or i < 0:
        return _ordered_paths(kind, dimension, i, a, labels, factors)

    without_opposite = tuple(j for j in labels if j != -i)
    result = _ordered_paths(kind, dimension, i, a, without_opposite, factors)
    between = tuple(j for j in labels if j > -i)
    if between:
        prefix = _opposite_prefix(dimension, i, between, cartan)
        below = tuple(j for j in labels if j < -i)
        suffix = _ordered_paths(kind, dimension, -i, a, below, factors)
        # The suffix has f_i-weight -1, so right evaluation at f_i + 1
        # before the suffix is equivalent to right evaluation at f_i after it.
        return result + (prefix * suffix) * Fraction(1, 2)
    return result


def classical_branching_generator(kind, rank, i, j):
    """Return a normalized classical branching generator from Molev §3.1.

    Exactly one signed index must be -rank or rank. The result is a PBW
    representative in the ambient general linear enveloping algebra.
    """
    dimension, i, j = _classical_indices(kind, rank, i, j)
    rank = dimension // 2
    if (abs(i) == rank) == (abs(j) == rank):
        raise ValueError("exactly one branching index must be -rank or rank")
    if abs(j) == rank:
        return _branching_generator(kind, dimension, i, j)

    element = _branching_generator(kind, dimension, -j, -i)
    exponent = rank - j + (kind == "C" and i < 0)
    return -element if exponent % 2 else element


def _interpolation_node(kind, dimension, i):
    """Return the numerator at g_i in Molev's interpolation formula."""
    rank = dimension // 2
    lowering = _branching_generator(kind, dimension, i, -rank)
    if i == rank:
        # The last node in types B and C uses z[n, n] = 1 and the full chain.
        return lowering
    return classical_branching_generator(kind, rank, rank, i) * lowering
