"""Row arrays and interlacing for the finite-dimensional gl_N patterns."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from numbers import Rational
from typing import Iterable

from ._gt import check_rows


__all__ = ["RowArray", "is_gt_pattern"]


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


def is_gt_pattern(array: RowArray | Iterable[Iterable[object]]) -> bool:
    """Test for row lengths N, ..., 1 and nonnegative integral interlacing gaps.

    N must be positive. The entries can share a nonintegral real or complex
    shift. Float and complex components are interpreted as exact binary values.
    """
    if not isinstance(array, RowArray):
        try:
            array = RowArray(array)
        except TypeError:
            return False
    coordinates = tuple(
        tuple(_exact_coordinates(entry) for entry in row) for row in array.rows
    )
    return check_rows(coordinates)
