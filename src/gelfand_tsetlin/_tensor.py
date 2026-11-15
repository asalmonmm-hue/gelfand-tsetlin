"""Closed-form operations on dominant highest weights."""

from itertools import pairwise

from . import _exact_coordinates


def _dominant_real_weight(highest_weight):
    try:
        coordinates = tuple(map(_exact_coordinates, highest_weight))
    except TypeError as exc:
        raise ValueError("highest_weight must be a nonempty sequence of real weights") from exc
    if not coordinates or any(entry is None or entry[1] for entry in coordinates):
        raise ValueError("highest_weight must be a nonempty sequence of finite real weights")

    weight = tuple(real for real, _ in coordinates)
    for upper, lower in pairwise(weight):
        gap = upper - lower
        if gap < 0 or gap % 1:
            raise ValueError("consecutive highest-weight gaps must be nonnegative integers")
    return weight


def quadratic_casimir_eigenvalue(highest_weight):
    """Return the exact eigenvalue of the trace-normalized quadratic Casimir.

    The coordinates may share a real shift. Consecutive gaps must be
    nonnegative integers, as for a finite-dimensional gl_N highest weight.
    """
    weight = _dominant_real_weight(highest_weight)
    rank = len(weight)
    return sum(value * (value + rank + 1 - 2 * i)
               for i, value in enumerate(weight, 1))


def tensor_with_standard(highest_weight, *, dual=False):
    """Return the highest weights in tensoring with the standard module.

    Set dual=True for the dual standard module. Each summand occurs once.
    The output is ordered by the index of the coordinate that changes.
    """
    weight = _dominant_real_weight(highest_weight)
    if dual:
        step = -1
        indices = (i for i, value in enumerate(weight)
                   if i == len(weight) - 1 or value > weight[i + 1])
    else:
        step = 1
        indices = (i for i, value in enumerate(weight)
                   if i == 0 or weight[i - 1] > value)
    return tuple(
        weight[:i] + (weight[i] + step,) + weight[i + 1:]
        for i in indices
    )
