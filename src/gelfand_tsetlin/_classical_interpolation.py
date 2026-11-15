"""Molev's even interpolation polynomials for the same-type reductions."""

from fractions import Fraction
from math import prod
from operator import index

from ._classical import _classical_dimension, _interpolation_node, _matrix_unit
from ._enveloping import EnvelopingElement, _rational


def _node_count(kind, dimension):
    count = dimension // 2 - (kind == "D")
    if count == 0:
        raise ValueError("type D interpolation requires rank at least two")
    return count


def classical_interpolation_node(kind, rank, i):
    """Return the node operator Z[n,-n](g_i) in (3.12)."""
    dimension, i = _classical_dimension(kind, rank), index(i)
    if not 1 <= i <= _node_count(kind, dimension):
        raise ValueError("interpolation node is outside the range for this type and rank")
    return _interpolation_node(kind, dimension, i)


def _node_product(nodes):
    coefficients = [1]
    for node in nodes:
        coefficients.append(0)
        for degree in range(len(coefficients) - 1, 0, -1):
            coefficients[degree] = coefficients[degree - 1] - node * coefficients[degree]
        coefficients[0] *= -node
    return coefficients


def _leading_coefficient(kind, dimension, rank):
    def unit(i, j):
        return _matrix_unit(kind, dimension, i, j)

    if kind == "C":
        return unit(rank, -rank)
    terms = EnvelopingElement._sum_ordered(
        dimension, (unit(rank, j) * unit(j, -rank) for j in range(1, rank))
    )
    return 2 * terms - unit(rank, 0) ** 2


def classical_interpolation_polynomial(kind, rank, weight, u=None):
    """Specialize the right Cartan factors of Molev's Z[n,-n](u).

    The weight lists the exact F[i,i] eigenvalues for i=1,...,rank.
    Omit u for PBW coefficients in ascending powers of u squared. An
    exact scalar u returns the evaluated PBW representative. B and C
    handle an outer-node collision using the leading coefficient. Type D
    uses the signed-node formula (3.31) at collisions.
    """
    dimension = _classical_dimension(kind, rank)
    rank = dimension // 2
    count = _node_count(kind, dimension)
    weight = tuple(map(_rational, weight))
    if len(weight) != rank:
        raise ValueError("weight must contain one Cartan eigenvalue for each rank index")
    parameter = None if u is None else _rational(u) ** 2
    shift = {"B": 1, "C": Fraction(1, 2), "D": Fraction(3, 2)}[kind]
    nodes = tuple((weight[i - 1] - i + shift) ** 2 for i in range(1, count + 1))

    leading = None
    if len(set(nodes)) != count and kind != "D" and nodes[-1] in nodes[:-1]:
        nodes = nodes[:-1]
        leading = _leading_coefficient(kind, dimension, rank)
    if len(set(nodes)) != len(nodes):
        if kind == "D":
            from ._classical_interpolation_d import _specialize_type_d_collision

            coefficients = _specialize_type_d_collision(dimension, weight)
            if parameter is None:
                return coefficients
            value = coefficients[-1]
            for coefficient in reversed(coefficients[:-1]):
                value = value * parameter + coefficient
            return value
        raise ValueError("squared interpolation nodes must be distinct after specialization")

    if parameter in nodes:
        return _interpolation_node(kind, dimension, nodes.index(parameter) + 1)

    operator_weights = []
    for i, node in enumerate(nodes, 1):
        denominator = prod(node - other for other in nodes if other != node)
        operator_weights.append(_interpolation_node(kind, dimension, i) * Fraction(1, denominator))

    if parameter is not None:
        product = prod(parameter - node for node in nodes)
        result = EnvelopingElement._sum_ordered(
            dimension,
            (operator * Fraction(product, parameter - node)
             for operator, node in zip(operator_weights, nodes)),
        )
        return result if leading is None else result + leading * product

    product_coefficients = _node_product(nodes)
    # Synthetic division gives each cardinal polynomial without its denominator.
    quotients = []
    for node in nodes:
        descending = [1]
        for coefficient in reversed(product_coefficients[1:-1]):
            descending.append(coefficient + node * descending[-1])
        quotients.append(descending[::-1])

    coefficients = [
        EnvelopingElement._sum_ordered(
            dimension, (op * quotient[degree] for op, quotient in zip(operator_weights, quotients))
        )
        for degree in range(len(nodes))
    ]
    if leading is not None:
        coefficients = [
            coefficient + leading * value
            for coefficient, value in zip(coefficients, product_coefficients)
        ]
        coefficients.append(leading)
    return tuple(coefficients)
