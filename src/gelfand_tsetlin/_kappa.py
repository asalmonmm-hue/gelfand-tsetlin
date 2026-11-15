"""Molev's quantum-minor construction of the GT basis."""

from math import prod

from . import RowArray, _pattern_coordinates
from ._gt import check_integer_rows, check_rows


def _kappa_rows(pattern):
    try:
        array = pattern if isinstance(pattern, RowArray) else RowArray(pattern)
    except TypeError as exc:
        raise ValueError("pattern must be a valid Gelfand–Tsetlin pattern") from exc

    is_integer = check_integer_rows(array.rows)
    if is_integer is True:
        return array.rows

    coordinates = _pattern_coordinates(array) if is_integer is None else None
    if coordinates is None or not check_rows(coordinates):
        raise ValueError("pattern must be a valid Gelfand–Tsetlin pattern")
    if coordinates[0][-1][1]:
        raise ValueError("kappa construction requires a real common shift")
    return tuple(tuple(real for real, _ in row) for row in coordinates)


def _kappa_factors(rows):
    rank = len(rows)
    top = rows[0]
    for column, highest in enumerate(top[:-1]):
        for row_index in range(1, rank - column):
            target = rows[row_index][column]
            first = column - target - 1
            for offset in range(int(highest - target)):
                yield rank - row_index, first - offset


def kappa_factors(pattern):
    """Yield (m, u) for the ordered product of C_m(u) in Theorem 2.21.

    Pairs are in written product order, so the rightmost factor acts
    first on the highest vector. The pattern may have a real common shift.
    """
    return _kappa_factors(_kappa_rows(pattern))


def kappa_scale(pattern):
    """Return the exact scalar N_Lambda in kappa_Lambda = N_Lambda xi_Lambda.

    Both constructions start from the same highest vector. The scalar is
    independent of the real common shift of the pattern.
    """
    rows = _kappa_rows(pattern)
    top = rows[0]
    scale = 1
    for row, lower in zip(rows[1:], rows[2:]):
        for column, target in enumerate(row):
            highest = top[column]
            steps = int(highest - target)
            if not steps:
                continue
            for i, lower_target in enumerate(lower):
                # In action order, columns to the left still have their
                # highest entries. The other lower entries are at target.
                lower_entry = top[i] if i < column else lower_target
                first = int(lower_entry - highest) + column - i
                scale *= prod(range(first, first + steps))
    return scale
