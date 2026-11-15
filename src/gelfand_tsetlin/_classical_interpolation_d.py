"""Type D interpolation at a collision of squared Cartan nodes."""

from fractions import Fraction
from math import prod

from ._classical import _interpolation_node
from ._classical_interpolation import _node_product
from ._enveloping import EnvelopingElement


def _specialize_type_d_collision(dimension, weight):
    """Return PBW representatives of the even coefficients in (3.31).

    The representative acts as stated on highest vectors of the given
    weight. The division remainder and odd coefficients belong to the
    annihilator of these vectors after specialization.
    """
    rank = dimension // 2
    labels = range(1, rank)
    half = Fraction(1, 2)
    nodes = {}
    for i in labels:
        shifted = weight[i - 1] - i + 1
        nodes[i] = half + shifted
        nodes[-i] = half - shifted

    signed_labels = tuple(nodes)
    zero = EnvelopingElement.scalar(dimension, 0)
    numerator = [zero] * (2 * rank - 2)
    for i in labels:
        others = tuple(j for j in signed_labels if abs(j) != i)
        denominator = prod(nodes[i] - nodes[j] for j in others)
        if not denominator:
            raise ValueError("type D signed interpolation nodes with distinct absolute indices must be distinct")

        # Opposite indices have the same denominator and common polynomial.
        scale = Fraction(1, denominator)
        positive = _interpolation_node("D", dimension, i) * scale
        negative = _interpolation_node("D", dimension, -i) * scale
        linear = positive + negative
        constant = positive * nodes[-i] + negative * nodes[i]
        for degree, scalar in enumerate(_node_product(-nodes[j] for j in others)):
            numerator[degree] += constant * scalar
            numerator[degree + 1] += linear * scalar

    # Divide by 2u + 1 before retaining the even powers of u.
    coefficients = [zero] * (rank - 1)
    quotient = zero
    for degree in range(len(numerator) - 2, -1, -1):
        quotient = (numerator[degree + 1] - quotient) * half
        if degree % 2 == 0:
            coefficients[degree // 2] = quotient
    return tuple(coefficients)
