"""Molev's raising and lowering actions on highest lines."""

from operator import index

from . import RowArray, _pattern_coordinates
from ._gt import check_integer_rows, check_rows, highest_line_coefficient


def _validated_rows(source):
    try:
        array = source if isinstance(source, RowArray) else RowArray(source)
    except TypeError as exc:
        raise ValueError("source must be a valid Gelfand–Tsetlin pattern") from exc

    is_integer = check_integer_rows(array.rows)
    if is_integer is True:
        return array.rows, True

    coordinates = _pattern_coordinates(array) if is_integer is None else None
    if coordinates is None or not check_rows(coordinates):
        raise ValueError("source must be a valid Gelfand–Tsetlin pattern")
    if coordinates[0][-1][1]:
        raise ValueError("highest-line action requires a real common shift")
    return tuple(tuple(real for real, _ in row) for row in coordinates), False


def _apply_highest(source, i, level, step):
    i = index(i)
    level = None if level is None else index(level)
    rows, is_integer = _validated_rows(source)
    rank = len(rows)
    level = rank if level is None else level
    if not 1 <= i < level <= rank:
        raise ValueError("operator indices must satisfy 1 <= i < level <= rank")

    middle_index = rank - level + 1
    top = rows[middle_index - 1]
    middle = rows[middle_index]
    if any(
        entry != highest
        for row in rows[middle_index + 1 :]
        for entry, highest in zip(row, middle)
    ):
        raise ValueError("source must be highest for gl_(level-1)")

    if is_integer:
        coefficient = highest_line_coefficient(top, middle, i, step)
    else:
        origin = top[-1]
        top_offsets = tuple(int(entry - origin) for entry in top)
        middle_offsets = tuple(int(entry - origin) for entry in middle)
        coefficient = highest_line_coefficient(top_offsets, middle_offsets, i, step)
    if coefficient == 0:
        return {}

    column = i - 1
    shifted = middle[:column] + (middle[column] + step,) + middle[column + 1 :]
    # Each shorter highest row is a prefix of the new middle row.
    changed = tuple(shifted[:width] for width in range(level - 1, i - 1, -1))
    target = RowArray(rows[:middle_index] + changed + rows[rank - i + 1 :])
    return {target: coefficient}


def apply_highest_raising(source, i, level=None):
    """Apply Molev's z[i, level] to a gl_(level-1)-highest GT basis vector.

    Indices are one-based, and the level defaults to the ambient rank.
    Return a sparse mapping with one target or the empty mapping at the
    branching boundary. A common real shift is allowed.
    """
    return _apply_highest(source, i, level, 1)


def apply_highest_lowering(source, i, level=None):
    """Apply Molev's z[level, i] to a gl_(level-1)-highest GT basis vector.

    Indices are one-based, and the level defaults to the ambient rank.
    Return a sparse mapping with coefficient one or the empty mapping at the
    branching boundary. A common real shift is allowed.
    """
    return _apply_highest(source, i, level, -1)
