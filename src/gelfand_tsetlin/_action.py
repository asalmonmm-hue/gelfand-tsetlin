"""Sparse action of enveloping-algebra elements in Molev's GT basis."""

from . import RowArray, _pattern_coordinates, check_rows, coefficient
from ._enveloping import EnvelopingElement


def _add_scaled(result, terms, scale):
    for pattern, value in terms.items():
        total = result.get(pattern, 0) + scale * value
        if total:
            result[pattern] = total
        else:
            result.pop(pattern, None)


def _adjacent_action(i, j, source):
    length = min(i, j)
    step = 1 if i < j else -1
    rows = source.rows
    row_index = len(rows) - length
    upper, row = rows[row_index - 1 : row_index + 1]
    lower = rows[row_index + 1] if length > 1 else ()

    result = {}
    for pos, old in enumerate(row):
        entry = old + step
        if (
            not upper[pos] >= entry >= upper[pos + 1]
            or (pos > 0 and lower[pos - 1] < entry)
            or (pos + 1 < length and entry < lower[pos])
        ):
            continue
        new_row = row[:pos] + (entry,) + row[pos + 1 :]
        target = RowArray(rows[:row_index] + (new_row,) + rows[row_index + 1 :])
        value = coefficient(i, j, source, target)
        if value:
            result[target] = value
    return result


def _unit_action(i, j, source, cache):
    key = (i, j, source)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if i == j:
        value = coefficient(i, i, source, source)
        result = {source: value} if value else {}
    elif abs(i - j) == 1:
        result = _adjacent_action(i, j, source)
    else:
        # E[i,j] = [E[i,k], E[k,j]], with E[i,k] adjacent.
        k = i + (1 if i < j else -1)
        result = _apply_unit(i, k, _unit_action(k, j, source, cache), cache)
        reverse = _apply_unit(k, j, _unit_action(i, k, source, cache), cache)
        _add_scaled(result, reverse, -1)
    cache[key] = result
    return result


def _apply_unit(i, j, terms, cache):
    result = {}
    for pattern, scale in terms.items():
        _add_scaled(result, _unit_action(i, j, pattern, cache), scale)
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
    result, cache = {}, {}
    for word, scale in element.terms.items():
        terms = {source: scale}
        for i, j in reversed(word):
            terms = _apply_unit(i, j, terms, cache)
            if not terms:
                break
        _add_scaled(result, terms, 1)
    return result
