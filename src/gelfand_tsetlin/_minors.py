"""Molev's column-ordered quantum minors in the PBW basis."""

from operator import index

from ._enveloping import EnvelopingElement, _add_terms, _checked_rank, _rational


def _minor_coefficients(rank, rows, columns):
    unit = EnvelopingElement.matrix_unit
    states = {0: ({(): 1},)}
    for shift, column in enumerate(columns):
        factors = []
        for row in rows:
            diagonal = row == column
            constant = unit(rank, row, column)
            factors.append((constant - shift if diagonal and shift else constant, diagonal))
        next_states = {}
        for mask, prefix in states.items():
            for position, (constant, diagonal) in enumerate(factors):
                bit = 1 << position
                if mask & bit:
                    continue
                selected = mask | bit
                if selected not in next_states:
                    next_states[selected] = tuple({} for _ in range(shift + 2))
                target = next_states[selected]
                # Appending this row adds one inversion per earlier row after it.
                sign = -1 if (mask >> (position + 1)).bit_count() % 2 else 1
                for degree, packed in enumerate(prefix):
                    if not packed:
                        continue
                    coefficient = EnvelopingElement._from_ordered(rank, packed)
                    if sign < 0:
                        coefficient = -coefficient
                    _add_terms(target[degree], (coefficient * constant)._terms)
                    if diagonal:
                        _add_terms(target[degree + 1], coefficient._terms)
        states = next_states

    packed = states[(1 << len(rows)) - 1]
    degree = len(packed) - 1
    while degree and not packed[degree]:
        degree -= 1
    return tuple(
        EnvelopingElement._from_ordered(rank, coefficient)
        for coefficient in packed[:degree + 1]
    )


def quantum_minor(rank, rows, columns, u=None):
    """Return a quantum minor of uI + E using Molev's column convention.

    Rows and columns are ordered, one-based index lists of equal length.
    Omit u for a tuple of PBW coefficients in ascending powers of u. An
    exact rational u evaluates the polynomial. A rank-matched enveloping
    element may also be substituted, with powers placed to the right.
    """
    rank = _checked_rank(rank)
    rows = tuple(map(index, rows))
    columns = tuple(map(index, columns))
    if len(rows) != len(columns):
        raise ValueError("quantum minor rows and columns must have equal length")
    if any(not 1 <= item <= rank for item in (*rows, *columns)):
        raise ValueError("quantum minor indices must lie in 1..rank")
    if isinstance(u, EnvelopingElement):
        if u.rank != rank:
            raise ValueError("substitution must have the quantum minor's rank")
    elif u is not None:
        u = _rational(u)

    if len(set(rows)) != len(rows) or len(set(columns)) != len(columns):
        coefficients = (EnvelopingElement.scalar(rank, 0),)
    else:
        coefficients = _minor_coefficients(rank, rows, columns)
    if u is None:
        return coefficients

    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * u + coefficient
    return value
