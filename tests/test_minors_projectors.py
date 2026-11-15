from fractions import Fraction
from itertools import product
from unittest import TestCase

from gelfand_tsetlin import (
    EnvelopingElement,
    RowArray,
    apply_dual_standard_projector,
    apply_enveloping,
    inner_product,
    lowering_operator,
    quantum_minor,
    raising_operator,
)


def matrix_units(rank):
    return {
        (i, j): EnvelopingElement.matrix_unit(rank, i, j)
        for i, j in product(range(1, rank + 1), repeat=2)
    }


class QuantumMinorTests(TestCase):
    def test_column_shifts_and_right_substitution(self):
        e = matrix_units(3)
        constant = e[1, 1] * (e[2, 2] - 1) - e[2, 1] * e[1, 2]
        linear = e[1, 1] + e[2, 2] - 1
        one = EnvelopingElement.scalar(3, 1)
        self.assertEqual(quantum_minor(3, (1, 2), (1, 2)), (constant, linear, one))
        value = Fraction(2, 3)
        self.assertEqual(
            quantum_minor(3, (1, 2), (1, 2), value),
            constant + linear * value + value**2,
        )
        operator = e[1, 3]
        self.assertNotEqual(linear * operator, operator * linear)
        self.assertEqual(
            quantum_minor(3, (1, 2), (1, 2), operator),
            constant + linear * operator + operator**2,
        )

    def test_rank_three_operators_from_218_and_219(self):
        e = matrix_units(3)
        self.assertEqual(
            quantum_minor(3, (2, 3), (1, 2), -e[1, 1]),
            lowering_operator(3, 1),
        )
        self.assertEqual(
            -quantum_minor(3, (1, 2), (1, 3), 1 - e[2, 2]),
            raising_operator(3, 2),
        )
        self.assertEqual(quantum_minor(3, (3,), (2,)), (lowering_operator(3, 2),))
        self.assertEqual(quantum_minor(3, (1,), (3,)), (raising_operator(3, 1),))

    def test_rank_four_operators_modulo_the_highest_vector_ideal(self):
        e = matrix_units(4)
        tau41 = quantum_minor(4, (2, 3, 4), (1, 2, 3), -e[1, 1])
        tau34 = quantum_minor(4, (1, 2, 3), (1, 2, 4), 2 - e[3, 3])
        z41 = lowering_operator(4, 1)
        z34 = raising_operator(4, 3)
        self.assertEqual(
            tau41 - z41,
            (e[3, 1] * e[4, 2] - e[3, 2] * e[4, 1]) * e[2, 3],
        )
        self.assertEqual(
            tau34 - z34,
            (e[2, 4] * e[3, 1] - e[2, 1] * e[3, 4]) * e[1, 2],
        )
        source = RowArray(((3, 2, 1, 0), (3, 2, 0), (3, 2), (3,)))
        for minor, operator in ((tau41, z41), (tau34, z34)):
            expected = apply_enveloping(operator, source)
            self.assertTrue(expected)
            self.assertEqual(apply_enveloping(minor, source), expected)

    def test_alternating_and_empty_minors(self):
        value = Fraction(1, 5)
        minor = quantum_minor(3, (1, 2), (1, 3), value)
        self.assertTrue(minor)
        self.assertEqual(quantum_minor(3, (2, 1), (1, 3), value), -minor)
        self.assertEqual(quantum_minor(3, (1, 2), (3, 1), value), -minor)
        self.assertEqual(quantum_minor(3, (1, 1), (1, 3)), (EnvelopingElement.scalar(3, 0),))
        self.assertEqual(quantum_minor(3, (1, 2), (3, 3), value), 0)
        self.assertEqual(quantum_minor(3, (), (), value), 1)

    def test_capelli_coefficients_are_central(self):
        indices = (1, 2, 3)
        coefficients = quantum_minor(3, indices, indices)
        self.assertEqual(len(coefficients), 4)
        self.assertEqual(coefficients[-1], 1)
        for coefficient, generator in product(coefficients, matrix_units(3).values()):
            self.assertEqual(coefficient * generator, generator * coefficient)

    def test_validates_indices_and_parameter(self):
        with self.assertRaises(ValueError):
            quantum_minor(3, (1, 2), (1,))
        with self.assertRaises(ValueError):
            quantum_minor(3, (0,), (1,))
        with self.assertRaises(ValueError):
            quantum_minor(3, (1,), (4,))
        with self.assertRaises(ValueError):
            quantum_minor(3, (1,), (1,), EnvelopingElement.scalar(2, 1))
        with self.assertRaises(TypeError):
            quantum_minor(3, (1,), (1,), 0.5)


def add_scaled(result, terms, scale):
    for key, value in terms.items():
        total = result.get(key, 0) + scale * value
        if total:
            result[key] = total
        else:
            result.pop(key, None)


def rank_three_patterns(top):
    for left, right in product(range(top[1], top[0] + 1), range(top[2], top[1] + 1)):
        for bottom in range(right, left + 1):
            yield RowArray((top, (left, right), (bottom,)))


class DualStandardProjectorTests(TestCase):
    def test_trace_projection_on_the_rank_two_standard_module(self):
        for shift in (0, Fraction(1, 3)):
            top = (1 + shift, shift)
            e1, e2 = (RowArray((top, (bottom,))) for bottom in top)
            trace_piece = {(1, e1): Fraction(1, 2), (2, e2): Fraction(1, 2)}
            self.assertEqual(apply_dual_standard_projector(e1, 1, 1), trace_piece)
            self.assertEqual(apply_dual_standard_projector(e2, 2, 1), trace_piece)
            self.assertEqual(apply_dual_standard_projector(e1, 2, 1), {})
            self.assertEqual(apply_dual_standard_projector(e1, 1, 2), {
                (1, e1): Fraction(1, 2), (2, e2): Fraction(-1, 2),
            })

    def test_rank_three_projectors_resolve_the_identity_and_characteristic_operator(self):
        generators = matrix_units(3)
        for top in ((2, 1, 0), (2, 2, 0)):
            patterns = tuple(rank_three_patterns(top))
            basis = tuple(product(range(1, 4), patterns))
            norm = {pattern: inner_product(pattern, pattern) for pattern in patterns}
            roots = {
                r: top[r - 1] + 3 - r
                for r in range(1, 4)
                if r == 3 or top[r - 1] > top[r]
            }
            projections = {
                r: {key: apply_dual_standard_projector(key[1], key[0], r) for key in basis}
                for r in range(1, 4)
            }
            for j, source in basis:
                key = j, source
                resolution, spectral_action = {}, {}
                for r, columns in projections.items():
                    column = columns[key]
                    if r not in roots:
                        self.assertEqual(column, {})
                        continue
                    add_scaled(resolution, column, 1)
                    add_scaled(spectral_action, column, roots[r])
                    for target, value in column.items():
                        self.assertEqual(
                            norm[target[1]] * value,
                            norm[source] * columns[target].get(key, 0),
                        )
                    for s in roots:
                        composition = {}
                        for target, value in column.items():
                            add_scaled(composition, projections[s][target], value)
                        self.assertEqual(composition, column if r == s else {})
                self.assertEqual(resolution, {key: 1})
                expected = {
                    (i, target): value
                    for i in range(1, 4)
                    for target, value in apply_enveloping(generators[i, j], source).items()
                }
                self.assertEqual(spectral_action, expected)

    def test_rank_one_and_invalid_arguments(self):
        scalar = RowArray(((5,),))
        self.assertEqual(apply_dual_standard_projector(scalar, 1, 1), {(1, scalar): 1})
        with self.assertRaises(ValueError):
            apply_dual_standard_projector(scalar, 0, 1)
        with self.assertRaises(ValueError):
            apply_dual_standard_projector(scalar, 1, 2)
        with self.assertRaises(ValueError):
            apply_dual_standard_projector([[2, 0], [3]], 1, 1)
