"""Diagonal weights of validated Gelfand–Tsetlin patterns."""

from collections.abc import Iterable
from fractions import Fraction

from . import RowArray, _pattern_coordinates
from ._gt import check_integer_rows, check_rows


def _row_differences(bottom_first_sums):
    previous = 0
    for total in bottom_first_sums:
        yield total - previous
        previous = total


def weight(
    pattern: RowArray | Iterable[Iterable[object]],
) -> tuple[int | Fraction | complex, ...]:
    """Return all diagonal weights in order from E[1,1] to E[N,N].

    Validate the pattern once and compute each row sum once. Real weights
    use arbitrary-precision integers or exact fractions. For a nonreal
    common shift, the weights use Python's complex type, as in coefficient().
    """
    try:
        array = pattern if isinstance(pattern, RowArray) else RowArray(pattern)
    except TypeError as exc:
        raise ValueError("argument must be a valid Gelfand–Tsetlin pattern") from exc

    valid = check_integer_rows(array.rows)
    if valid is True:
        return tuple(_row_differences(map(sum, reversed(array.rows))))

    coordinates = _pattern_coordinates(array) if valid is None else None
    if coordinates is None or not check_rows(coordinates):
        raise ValueError("argument must be a valid Gelfand–Tsetlin pattern")

    real_sums = (sum(real for real, _ in row) for row in reversed(coordinates))
    origin_real, origin_imag = coordinates[0][-1]
    if origin_imag:
        # Taking differences of the row lengths leaves one imaginary shift.
        return tuple(
            complex(value, origin_imag) for value in _row_differences(real_sums)
        )
    # Preserve the same integer/fraction convention as diagonal coefficients.
    return tuple(
        int(value - origin_real) + origin_real
        for value in _row_differences(real_sums)
    )
