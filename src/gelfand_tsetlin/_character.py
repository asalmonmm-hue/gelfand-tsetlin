"""Joint eigenvalues for the Gelfand–Tsetlin subalgebra."""

from collections.abc import Iterable
from fractions import Fraction

from . import RowArray, _pattern_coordinates
from ._gt import check_integer_rows, check_rows, gt_row_eigenvalues


def gt_eigenvalues(
    pattern: RowArray | Iterable[Iterable[object]],
) -> tuple[tuple[int | Fraction, ...], ...]:
    """Return the joint eigenvalues of Molev's generators a[m, i].

    Row m contains the m non-leading coefficients of the polynomial
    product of (u + lambda[m, j] - j + 1) in descending degree order.
    Rows are ordered by increasing m. The pattern may have a common real
    shift, and the coefficients are computed over exact rationals.
    """
    try:
        array = pattern if isinstance(pattern, RowArray) else RowArray(pattern)
    except TypeError as exc:
        raise ValueError("argument must be a valid Gelfand–Tsetlin pattern") from exc

    valid = check_integer_rows(array.rows)
    if valid is True:
        return tuple(map(gt_row_eigenvalues, reversed(array.rows)))

    coordinates = _pattern_coordinates(array) if valid is None else None
    if coordinates is None or not check_rows(coordinates):
        raise ValueError("argument must be a valid Gelfand–Tsetlin pattern")
    if coordinates[0][-1][1]:
        raise ValueError("Gelfand–Tsetlin eigenvalues require a common real shift")

    return tuple(
        gt_row_eigenvalues(tuple(real for real, _ in row))
        for row in reversed(coordinates)
    )
