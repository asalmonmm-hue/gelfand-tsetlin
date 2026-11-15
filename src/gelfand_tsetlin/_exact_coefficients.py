"""Arbitrary-precision intermediates for the fixed-width coefficient API."""

from fractions import Fraction
from math import prod


def _row_offset(rank, length):
    return (rank * (rank + 1) - length * (length + 1)) // 2


def _exact_coefficient(row, neighbor, column, step):
    pivot = row[column] - column
    numerator = -step * prod(pivot - entry + index for index, entry in enumerate(neighbor))
    if not numerator:
        return 0
    denominator = prod(
        pivot - entry + index for index, entry in enumerate(row) if index != column
    )
    value = Fraction(numerator, denominator)
    limit = 1 << 63
    if not -limit <= value.numerator < limit or value.denominator >= limit:
        raise OverflowError("GT arithmetic exceeded its fixed-width range")
    return value.numerator if value.denominator == 1 else value


def adjacent_coefficient_exact(rank, i, j, source, target):
    """Return an adjacent coefficient for validated, normalized flat patterns."""
    length = min(i, j)
    start = _row_offset(rank, length)
    stop = start + length
    step = 1 if i < j else -1
    changed = None
    for index, (before, after) in enumerate(zip(source, target)):
        if before == after:
            continue
        if changed is not None or not start <= index < stop or after - before != step:
            return 0
        changed = index
    if changed is None:
        return 0

    neighbor = source[start - length - 1 : start] if step == 1 else source[stop : stop + length - 1]
    return _exact_coefficient(source[start:stop], neighbor, changed - start, step)


def adjacent_action_exact(rank, i, j, source):
    """Yield (column, coefficient) pairs for a validated, normalized source."""
    length = min(i, j)
    start = _row_offset(rank, length)
    stop = start + length
    step = 1 if i < j else -1
    row = source[start:stop]
    upper = source[start - length - 1 : start]
    lower = source[stop : stop + length - 1]
    neighbor = upper if step == 1 else lower
    for column, entry in enumerate(row):
        shifted = entry + step
        if not upper[column + 1] <= shifted <= upper[column]:
            continue
        if column > 0 and shifted > lower[column - 1]:
            continue
        if column + 1 < length and shifted < lower[column]:
            continue
        value = _exact_coefficient(row, neighbor, column, step)
        if value:
            yield column, value
