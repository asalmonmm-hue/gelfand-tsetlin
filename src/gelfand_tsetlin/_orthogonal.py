"""Patterns for the alternating orthogonal chain in Molev §4."""

from fractions import Fraction
from itertools import pairwise
from operator import index


def _entry_denominator(entry):
    if isinstance(entry, (int, Fraction)) and not isinstance(entry, bool):
        return entry.denominator
    return None


def _dominant(dimension, row):
    coordinates = row if dimension % 2 else (*row[:-1], abs(row[-1]))
    return coordinates[-1] >= 0 and all(upper >= lower for upper, lower in pairwise(coordinates))


def _branching_bounds(dimension, row):
    if dimension % 2:
        return row, (*row[1:], -row[-1])
    return row[:-1], (*row[1:-1], abs(row[-1]))


def is_orthogonal_gt_pattern(pattern):
    """Test top-first rows for o_N down to o_2 in Molev §4.

    Row lengths are floor(d/2) for d=N,...,2. Entries are all integers or
    all proper half-integers. Only the last coordinate of an even row
    may be negative.
    """
    from . import RowArray

    try:
        rows = pattern.rows if isinstance(pattern, RowArray) else tuple(map(tuple, pattern))
    except TypeError:
        return False
    dimension = len(rows) + 1
    if not rows or any(len(row) != d // 2 for d, row in enumerate(reversed(rows), 2)):
        return False
    denominator = _entry_denominator(rows[0][0])
    if denominator not in (1, 2) or any(
        _entry_denominator(entry) != denominator for row in rows for entry in row
    ):
        return False
    if not _dominant(dimension, rows[0]):
        return False

    for level, (top, bottom) in zip(range(dimension, 2, -1), pairwise(rows)):
        upper, lower = _branching_bounds(level, top)
        if any(not lo <= entry <= hi for lo, entry, hi in zip(lower, bottom, upper)):
            return False
    return True


def _descending_weights(upper, lower):
    current = list(upper)
    while True:
        yield tuple(current)
        for i in range(len(current) - 1, -1, -1):
            if current[i] > lower[i]:
                current[i] -= 1
                break
            current[i] = upper[i]
        else:
            return


def orthogonal_branching(dimension, highest_weight):
    """Yield the o_(N-1) summand labels of an o_N highest weight.

    Here N is dimension, with N >= 3. The multiplicity-free weights are
    yielded lazily, in decreasing lexicographic order.
    """
    dimension = index(dimension)
    if dimension < 3:
        raise ValueError("orthogonal branching requires dimension at least three")
    try:
        row = tuple(highest_weight)
    except TypeError as exc:
        raise ValueError("highest_weight must be a sequence of orthogonal weights") from exc
    if len(row) != dimension // 2:
        raise ValueError("highest_weight must have floor(dimension / 2) entries")
    denominator = _entry_denominator(row[0])
    if denominator not in (1, 2) or any(_entry_denominator(entry) != denominator for entry in row):
        raise ValueError("highest_weight entries must all be integers or all be proper half-integers")
    if not _dominant(dimension, row):
        raise ValueError("highest_weight must be dominant for its orthogonal dimension")

    return _descending_weights(*_branching_bounds(dimension, row))
