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


def _tau_indices(rank, i, level):
    rank, i = _checked_rank(rank), index(i)
    level = rank if level is None else index(level)
    if not 1 <= i < level <= rank:
        raise ValueError("operator indices must satisfy 1 <= i < level <= rank")
    return rank, i, level


def tau_lowering(rank, i, u=None, *, level=None):
    """Return Molev's tau[level, i](u) in the upper-left subalgebra.

    Indices are one-based and level defaults to rank. Omit u for ascending
    PBW coefficients, or pass an exact scalar or a rank-matched element as
    in quantum_minor.
    """
    rank, i, level = _tau_indices(rank, i, level)
    return quantum_minor(rank, range(i + 1, level + 1), range(i, level), u)


def tau_raising(rank, i, u=None, *, level=None):
    """Return Molev's tau[i, level](u) in the upper-left subalgebra.

    Indices are one-based and level defaults to rank. Omit u for ascending
    PBW coefficients, or pass an exact scalar or a rank-matched element as
    in quantum_minor.
    """
    rank, i, level = _tau_indices(rank, i, level)
    minor = quantum_minor(rank, range(1, i + 1), (*range(1, i), level), u)
    if i % 2:
        return minor
    if u is None:
        return tuple(-coefficient for coefficient in minor)
    return -minor


def a_polynomial(rank, m, u=None):
    """Return Molev's A_m(u), the upper-left Capelli determinant.

    The level m satisfies 1 <= m <= rank. The return and evaluation
    conventions are those of quantum_minor.
    """
    rank, m = _checked_rank(rank), index(m)
    if not 1 <= m <= rank:
        raise ValueError("A polynomial level must satisfy 1 <= m <= rank")
    indices = range(1, m + 1)
    return quantum_minor(rank, indices, indices, u)


def _adjacent_minor_indices(rank, m):
    rank, m = _checked_rank(rank), index(m)
    if not 1 <= m < rank:
        raise ValueError("B and C polynomial levels must satisfy 1 <= m < rank")
    return rank, range(1, m + 1), (*range(1, m), m + 1)


def b_polynomial(rank, m, u=None):
    """Return Molev's B_m(u), with column m replaced by m+1.

    The level m satisfies 1 <= m < rank. The return and evaluation
    conventions are those of quantum_minor.
    """
    rank, principal, shifted = _adjacent_minor_indices(rank, m)
    return quantum_minor(rank, principal, shifted, u)


def c_polynomial(rank, m, u=None):
    """Return Molev's C_m(u), with row m replaced by m+1.

    The level m satisfies 1 <= m < rank. The return and evaluation
    conventions are those of quantum_minor.
    """
    rank, principal, shifted = _adjacent_minor_indices(rank, m)
    return quantum_minor(rank, shifted, principal, u)
