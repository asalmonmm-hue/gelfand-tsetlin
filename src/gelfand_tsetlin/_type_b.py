"""Type B Gelfand–Tsetlin patterns in Molev's conventions."""

from fractions import Fraction
from operator import index


def _interlaces(upper, intermediate):
    return all(v >= w for v, w in zip(intermediate, upper)) and all(
        v >= w for v, w in zip(upper, intermediate[1:])
    )


def _pattern_rows(pattern):
    try:
        entries = tuple(pattern)
        rows = []
        for level, entry in zip(range(len(entries), 0, -1), entries):
            sigma, highest, intermediate = entry
            sigma = index(sigma)
            highest, intermediate = tuple(highest), tuple(intermediate)
            if sigma not in (0, 1) or len(highest) != level or len(intermediate) != level:
                return None
            rows.append((sigma, highest, intermediate))
    except (TypeError, ValueError):
        return None
    if not rows:
        return None

    top_entry = rows[0][1][0]
    if isinstance(top_entry, bool) or not isinstance(top_entry, (int, Fraction)):
        return None
    denominator = top_entry.denominator
    if denominator not in (1, 2):
        return None
    for sigma, highest, intermediate in rows:
        for entry in (*highest, *intermediate):
            if (
                isinstance(entry, bool)
                or not isinstance(entry, (int, Fraction))
                or entry.denominator != denominator
                or entry > 0
            ):
                return None
        if (sigma and intermediate[0] == 0) or not _interlaces(highest, intermediate):
            return None
    for (_, _, intermediate), (_, lower, _) in zip(rows, rows[1:]):
        if not _interlaces(lower, intermediate):
            return None
    return rows


def is_type_b_gt_pattern(pattern):
    """Test a top-first sequence of (sigma, highest, intermediate) triples.

    At level k both rows have k entries. All entries are nonpositive exact
    integers or all are nonpositive exact proper half-integers. The binary
    sigma is zero when the first intermediate entry is zero.
    """
    return _pattern_rows(pattern) is not None


def _basis_factors(rows):
    lower = ()
    for level, (sigma, highest, intermediate) in enumerate(reversed(rows), 1):
        if sigma:
            yield ("z", level, level, 0)
        for i, (mu, lam, nu) in enumerate(zip(lower, highest, intermediate), 1):
            for _ in range(int(nu - mu)):
                yield ("z", level, level, i)
            for _ in range(int(nu - lam)):
                yield ("z", level, i, -level)

        # The rightmost Z first raises the last coordinate from l_kk.
        # In written order the parameters therefore descend from l'_kk - 1.
        last = intermediate[-1] - level
        for offset in range(int(intermediate[-1] - highest[-1])):
            yield ("Z", level, last - offset)
        lower = highest


def type_b_basis_factors(pattern):
    """Yield each factor of Molev's type B basis product in written order.

    Entries are ("z", k, i, j) for z[i,j] or ("Z", k, u) for Z[k,-k](u).
    The rightmost factor acts first. The pattern is validated before return.
    """
    rows = _pattern_rows(pattern)
    if rows is None:
        raise ValueError("argument must be a valid type B Gelfand–Tsetlin pattern")
    return _basis_factors(rows)
