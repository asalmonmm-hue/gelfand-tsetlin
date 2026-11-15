"""Irreducibility criteria for tensor products of Yangian modules."""

from itertools import combinations

from . import _exact_coordinates


def _string_bounds(highest_weight):
    try:
        alpha, beta = highest_weight
    except (TypeError, ValueError) as exc:
        raise ValueError("highest weight must be a pair (alpha, beta)") from exc
    alpha, beta = _exact_coordinates(alpha), _exact_coordinates(beta)
    if alpha is None or beta is None:
        raise ValueError("string endpoints must be finite real or complex numbers")
    upper, alpha_imag = alpha
    lower, beta_imag = beta
    length = upper - lower
    if alpha_imag != beta_imag or length < 0 or length.denominator != 1:
        raise ValueError("alpha - beta must be a nonnegative integer")
    return lower, upper, beta_imag


def _in_general_position(first, second):
    lower, upper, imag = first
    other_lower, other_upper, other_imag = second
    if lower == upper or other_lower == other_upper:
        return True
    if imag != other_imag or (lower - other_lower).denominator != 1:
        return True

    # The upper endpoints are excluded. Equality with the other lower
    # endpoint means adjacent strings with no missing lattice point.
    if upper < other_lower or other_upper < lower:
        return True
    return (lower <= other_lower and other_upper <= upper) or (
        other_lower <= lower and upper <= other_upper
    )


def yangian_strings_in_general_position(first, second):
    """Test the two strings S(alpha, beta) for Molev's general position.

    Each input is a highest-weight pair (alpha, beta) with nonnegative
    integral difference. Empty strings are in general position with any
    string. Endpoints are checked exactly, including finite binary floats.
    """
    return _in_general_position(_string_bounds(first), _string_bounds(second))


def yangian_tensor_is_irreducible(highest_weights):
    """Test irreducibility of a tensor product of Y(2) evaluation modules.

    Each factor is supplied as (alpha, beta). Molev's Theorem 3.8 reduces
    irreducibility to pairwise general position of the associated strings.
    """
    strings = tuple(map(_string_bounds, highest_weights))
    return all(_in_general_position(first, second) for first, second in combinations(strings, 2))
