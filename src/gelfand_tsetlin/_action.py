"""Sparse action of enveloping-algebra elements in Molev's GT basis."""

from fractions import Fraction

from . import RowArray, _pattern_coordinates, check_rows
from ._enveloping import EnvelopingElement
from ._exact_coefficients import adjacent_action_exact
from ._gt import adjacent_coefficients_i64


def _add_scaled(result, terms, scale):
    for pattern, value in terms.items():
        total = result.get(pattern, 0) + scale * value
        if total:
            result[pattern] = total
        else:
            result.pop(pattern, None)


def _shift_entry(rows, row_index, column, step):
    row = rows[row_index]
    shifted = row[:column] + (row[column] + step,) + row[column + 1 :]
    return RowArray(rows[:row_index] + (shifted,) + rows[row_index + 1 :])


def _adjacent_action(i, j, source, offsets_cache):
    length = min(i, j)
    step = 1 if i < j else -1
    rows = source.rows
    rank = len(rows)
    row_index = rank - length

    offsets = offsets_cache.get(source)
    if offsets is None:
        origin = rows[0][-1]
        offsets = tuple(int(entry - origin) for row in rows for entry in row)
        offsets_cache[source] = offsets

    try:
        terms = adjacent_coefficients_i64(rank, i, j, offsets)
    except OverflowError:
        coefficients = adjacent_action_exact(rank, i, j, offsets)
    else:
        coefficients = (
            (pos, numerator if denominator == 1 else Fraction(numerator, denominator))
            for pos, numerator, denominator in terms
        )
    return {_shift_entry(rows, row_index, pos, step): value for pos, value in coefficients}


def _row_sum(row, row_sums):
    total = row_sums.get(row)
    if total is None:
        total = row_sums[row] = sum(row)
    return total


def _unit_action(i, j, source, cache, offsets_cache, row_sums):
    key = (i, j, source)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if i == j:
        rows = source.rows
        pos = len(rows) - i
        value = _row_sum(rows[pos], row_sums)
        if i > 1:
            value -= _row_sum(rows[pos + 1], row_sums)
        result = {source: value} if value else {}
    elif abs(i - j) == 1:
        result = _adjacent_action(i, j, source, offsets_cache)
    else:
        # E[i,j] = [E[i,k], E[k,j]], with E[i,k] adjacent.
        k = i + (1 if i < j else -1)
        forward = _unit_action(k, j, source, cache, offsets_cache, row_sums)
        result = _apply_unit(i, k, forward, cache, offsets_cache, row_sums)
        backward = _unit_action(i, k, source, cache, offsets_cache, row_sums)
        reverse = _apply_unit(k, j, backward, cache, offsets_cache, row_sums)
        _add_scaled(result, reverse, -1)
    cache[key] = result
    return result


def _apply_unit(i, j, terms, cache, offsets_cache, row_sums):
    result = {}
    for pattern, scale in terms.items():
        _add_scaled(result, _unit_action(i, j, pattern, cache, offsets_cache, row_sums), scale)
    return result


def apply_enveloping(element, source):
    """Return the sparse action of an element on one valid GT basis pattern.

    The pattern must have the element's rank and a real common shift.
    Products act rightmost factor first. The returned dictionary keys are
    RowArray instances and its coefficients are exact integers or fractions.
    """
    if not isinstance(element, EnvelopingElement):
        raise TypeError("element must be an EnvelopingElement")
    rows = _pattern_coordinates(source)
    if not check_rows(rows) or len(rows) != element.rank:
        raise ValueError("source must be a valid GT pattern of the element's rank")
    if rows[0][-1][1]:
        raise ValueError("sparse action requires a real common shift")

    source = RowArray(tuple(real for real, _ in row) for row in rows)
    result, cache, offsets_cache, row_sums = {}, {}, {}, {}
    for word, scale in element._terms.items():
        terms = {source: scale}
        for unit in reversed(word):
            i, j = divmod(unit, element.rank)
            terms = _apply_unit(i + 1, j + 1, terms, cache, offsets_cache, row_sums)
            if not terms:
                break
        _add_scaled(result, terms, 1)
    return result
