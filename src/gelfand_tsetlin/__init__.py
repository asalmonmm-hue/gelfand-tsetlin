"""Row arrays and interlacing for the finite-dimensional gl_N patterns."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from numbers import Rational
from operator import index
from typing import Iterable

from ._gt import check_integer_rows, check_rows, coefficient_i64, inner_product_i64
from ._enveloping import EnvelopingElement
from ._minors import (
    a_polynomial,
    b_polynomial,
    c_polynomial,
    quantum_minor,
    tau_lowering,
    tau_raising,
)
from ._molev import lowering_operator, raising_operator


__all__ = [
    "EnvelopingElement",
    "RowArray",
    "a_polynomial",
    "apply_enveloping",
    "apply_dual_standard_projector",
    "apply_highest_lowering",
    "apply_highest_raising",
    "b_polynomial",
    "c_polynomial",
    "coefficient",
    "gt_eigenvalues",
    "inner_product",
    "is_gt_pattern",
    "kappa_factors",
    "kappa_scale",
    "lowering_operator",
    "quadratic_casimir_eigenvalue",
    "quantum_minor",
    "raising_operator",
    "tau_lowering",
    "tau_raising",
    "tensor_with_standard",
    "weight",
]


@dataclass(frozen=True, slots=True, init=False)
class RowArray:
    """A top-first sequence of rows, preserving the supplied entries."""

    rows: tuple[tuple[object, ...], ...]

    def __init__(self, rows: Iterable[Iterable[object]]) -> None:
        object.__setattr__(self, "rows", tuple(tuple(row) for row in rows))


def _exact_coordinates(entry: object) -> tuple[int | Fraction, int | Fraction] | None:
    if isinstance(entry, bool):
        return None
    if isinstance(entry, int):
        return int(entry), 0
    try:
        if isinstance(entry, complex):
            return Fraction(entry.real), Fraction(entry.imag)
        if isinstance(entry, (Rational, float, Decimal)):
            return Fraction(entry), 0
    except (ValueError, OverflowError):
        return None  # Nonfinite real or imaginary component.
    return None


def _pattern_coordinates(array):
    if not isinstance(array, RowArray):
        try:
            array = RowArray(array)
        except TypeError:
            return None
    return tuple(
        tuple(_exact_coordinates(entry) for entry in row) for row in array.rows
    )


def is_gt_pattern(array: RowArray | Iterable[Iterable[object]]) -> bool:
    """Test for row lengths N, ..., 1 and nonnegative integral interlacing gaps.

    N must be positive. The entries can share a nonintegral real or complex
    shift. Float and complex components are interpreted as exact binary values.
    """
    rows = array.rows if type(array) is RowArray else array
    result = check_integer_rows(rows)
    return check_rows(_pattern_coordinates(array)) if result is None else result


def _normalized_pair(source, target):
    source_rows = _pattern_coordinates(source)
    target_rows = source_rows if source is target else _pattern_coordinates(target)
    if not check_rows(source_rows) or (source_rows is not target_rows and not check_rows(target_rows)):
        raise ValueError("both arguments must be valid Gelfand–Tsetlin patterns")
    if source_rows[0] != target_rows[0]:
        raise ValueError("patterns must have the same highest weight")

    origin_real, origin_imag = source_rows[0][-1]
    source_offsets = tuple(int(real - origin_real) for row in source_rows for real, _ in row)
    target_offsets = source_offsets if source_rows is target_rows else tuple(
        int(real - origin_real) for row in target_rows for real, _ in row
    )
    return len(source_rows), origin_real, origin_imag, source_offsets, target_offsets


def coefficient(i, j, source, target):
    """Return the coefficient of target in E[i,j] applied to source.

    Indices are one-based. The diagonal and adjacent formulas are in
    Molev's Theorem 2.3. A missing transition between valid patterns is zero.
    Real results are exact integers or fractions. A diagonal result with a
    nonreal shift is returned as a Python complex value.
    """
    i, j = index(i), index(j)
    rank, origin_real, origin_imag, source, target = _normalized_pair(source, target)
    if not (1 <= i <= rank and 1 <= j <= rank and abs(i - j) <= 1):
        raise ValueError("indices must be diagonal or adjacent and within the pattern rank")

    numerator, denominator = coefficient_i64(rank, i, j, source, target)
    if i == j and source == target:
        value = numerator + origin_real
        return complex(value, origin_imag) if origin_imag else value
    return numerator if denominator == 1 else Fraction(numerator, denominator)


def inner_product(left, right):
    """Return Molev's contravariant pairing of two GT basis vectors.

    Proposition 2.4 gives an exact integer for matching patterns and zero
    for distinct patterns. Both inputs must be valid and have the same
    highest weight. The pairing is independent of the common shift.
    """
    rank, _, _, left, right = _normalized_pair(left, right)
    numerator, _ = inner_product_i64(rank, left, right)
    return numerator


# These modules use the pattern and coefficient helpers defined above.
from ._action import apply_enveloping
from ._character import gt_eigenvalues
from ._highest import apply_highest_lowering, apply_highest_raising
from ._kappa import kappa_factors, kappa_scale
from ._projectors import apply_dual_standard_projector
from ._tensor import quadratic_casimir_eigenvalue, tensor_with_standard
from ._weight import weight
