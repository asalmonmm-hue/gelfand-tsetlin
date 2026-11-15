"""Sparse summand projections on the dual-standard tensor product."""

from fractions import Fraction
from operator import index

from . import RowArray, _pattern_coordinates, check_rows
from ._action import _unit_action


def _apply_shifted_characteristic(terms, rank, root, cache, offsets_cache):
    result = {key: -root * value for key, value in terms.items()} if root else {}
    for (j, pattern), scale in terms.items():
        for i in range(1, rank + 1):
            for target, value in _unit_action(i, j, pattern, cache, offsets_cache).items():
                key = (i, target)
                total = result.get(key, 0) + scale * value
                if total:
                    result[key] = total
                else:
                    result.pop(key, None)
    return result


def apply_dual_standard_projector(source, j, r):
    """Project epsilon_j tensor source onto the summand L(lambda - epsilon_r).

    The dual-standard index j and summand index r are one-based. The result
    maps (dual-standard index, RowArray) to exact coefficients in Molev's
    unnormalized GT basis. An in-range r with no summand gives an empty
    result. The source may have a real common shift.
    """
    j, r = index(j), index(r)
    coordinates = _pattern_coordinates(source)
    if not check_rows(coordinates):
        raise ValueError("source must be a valid Gelfand–Tsetlin pattern")
    if coordinates[0][-1][1]:
        raise ValueError("dual-standard projector requires a real common shift")
    rank = len(coordinates)
    if not (1 <= j <= rank and 1 <= r <= rank):
        raise ValueError("tensor and summand indices must be within the pattern rank")

    source = RowArray(tuple(real for real, _ in row) for row in coordinates)
    highest = source.rows[0]
    if r < rank and highest[r - 1] == highest[r]:
        return {}

    alpha = highest[r - 1] + rank - r
    terms, denominator = {(j, source): 1}, 1
    cache, offsets_cache = {}, {}
    for s, entry in enumerate(highest, 1):
        if s == r or (s < rank and entry == highest[s]):
            continue
        root = entry + rank - s
        terms = _apply_shifted_characteristic(terms, rank, root, cache, offsets_cache)
        if not terms:
            return {}
        denominator *= alpha - root
    if denominator == 1:
        return terms
    return {key: Fraction(value, denominator) for key, value in terms.items()}
