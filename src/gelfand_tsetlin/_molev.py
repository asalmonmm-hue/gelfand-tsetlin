"""Molev's raising and lowering operators for the chain of gl algebras."""

from itertools import combinations
from operator import index

from ._enveloping import EnvelopingElement


def _chain_terms(rank, i, level, raising):
    candidates = range(i - 1, 0, -1) if raising else range(i + 1, level)
    unit = EnvelopingElement.matrix_unit
    diagonal = {j: unit(rank, i, i) - unit(rank, j, j) + (j - i) for j in candidates}
    for length in range(len(candidates) + 1):
        for chain in combinations(candidates, length):
            path = (i, *chain, level)
            selected = set(chain)
            edges = zip(path, path[1:])
            # Keep the path factors in traversal order in both formulas.
            word = tuple(edges) if raising else tuple((v, u) for u, v in edges)
            term = EnvelopingElement(rank, {word: 1})
            # The omitted h_i - h_j factors commute and belong on the right.
            for j in candidates:
                if j not in selected:
                    term = term * diagonal[j]
            yield term


def _operator(rank, i, level, *, raising):
    rank, i = index(rank), index(i)
    level = rank if level is None else index(level)
    if not 1 <= i < level <= rank:
        raise ValueError("operator indices must satisfy 1 <= i < level <= rank")
    return EnvelopingElement._sum_ordered(rank, _chain_terms(rank, i, level, raising))


def raising_operator(rank, i, level=None):
    """Return Molev's z[i, level] in the upper-left subalgebra.

    Indices are one-based. The level defaults to the ambient rank.
    """
    return _operator(rank, i, level, raising=True)


def lowering_operator(rank, i, level=None):
    """Return Molev's z[level, i] in the upper-left subalgebra.

    Indices are one-based. The level defaults to the ambient rank.
    """
    return _operator(rank, i, level, raising=False)
