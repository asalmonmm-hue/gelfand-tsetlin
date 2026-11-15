from fractions import Fraction
from itertools import product
from unittest import TestCase

from gelfand_tsetlin import (
    EnvelopingElement,
    RowArray,
    a_polynomial,
    apply_dual_standard_projector,
    apply_enveloping,
    b_polynomial,
    c_polynomial,
    inner_product,
    lowering_operator,
    quantum_minor,
    raising_operator,
    tau_lowering,
    tau_raising,
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
        tau41 = tau_lowering(4, 1, -e[1, 1])
        tau34 = tau_raising(4, 3, 2 - e[3, 3])
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

    def test_matrix_unit_commutator_replaces_rows_and_columns(self):
        e32 = EnvelopingElement.matrix_unit(3, 3, 2)
        value = Fraction(2, 3)
        minor = quantum_minor(3, (1, 2), (1, 3), value)
        replaced_row = quantum_minor(3, (1, 3), (1, 3), value)
        replaced_column = quantum_minor(3, (1, 2), (1, 2), value)
        self.assertEqual(e32 * minor - minor * e32, replaced_row - replaced_column)

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


class ABCPolynomialTests(TestCase):
    def test_low_degree_coefficients_and_evaluation(self):
        e = matrix_units(3)
        one = EnvelopingElement.scalar(3, 1)
        self.assertEqual(a_polynomial(3, 1), (e[1, 1], one))
        self.assertEqual(b_polynomial(3, 1), (e[1, 2],))
        self.assertEqual(c_polynomial(3, 1), (e[2, 1],))

        b_constant = e[1, 1] * e[2, 3] - e[2, 1] * e[1, 3]
        c_constant = e[1, 1] * e[3, 2] - e[3, 1] * e[1, 2]
        self.assertEqual(b_polynomial(3, 2), (b_constant, e[2, 3]))
        self.assertEqual(c_polynomial(3, 2), (c_constant, e[3, 2]))
        self.assertEqual(b_polynomial(3, 2, e[3, 3]), b_constant + e[2, 3] * e[3, 3])
        self.assertEqual(a_polynomial(3, 3), quantum_minor(3, (1, 2, 3), (1, 2, 3)))

    def test_theorem_220_at_a_rank_three_pattern(self):
        source = RowArray(((3, 1, 0), (2, 1), (1,)))
        raised = RowArray(((3, 1, 0), (3, 1), (1,)))
        lowered = RowArray(((3, 1, 0), (1, 1), (1,)))
        value = Fraction(2, 3)
        self.assertEqual(apply_enveloping(a_polynomial(3, 2, value), source), {
            source: (value + 2) * value,
        })
        self.assertEqual(apply_enveloping(b_polynomial(3, 2, -2), source), {raised: -8})
        self.assertEqual(apply_enveloping(c_polynomial(3, 2, -2), source), {lowered: -1})

    def test_level_bounds(self):
        for constructor, rank, level in (
            (a_polynomial, 3, 0), (a_polynomial, 3, 4),
            (b_polynomial, 3, 0), (b_polynomial, 3, 3),
            (c_polynomial, 3, 0), (c_polynomial, 3, 3),
        ):
            with self.subTest(constructor=constructor.__name__, rank=rank, level=level):
                with self.assertRaises(ValueError):
                    constructor(rank, level)


class TauPolynomialTests(TestCase):
    def test_rank_three_coefficients_and_exact_evaluation(self):
        e = matrix_units(3)
        tau31 = e[2, 1] * e[3, 2] - e[3, 1] * (e[2, 2] - 1), -e[3, 1]
        tau23 = -e[2, 3] * e[1, 1] + e[2, 1] * e[1, 3], -e[2, 3]
        self.assertEqual(tau_lowering(3, 1), tau31)
        self.assertEqual(tau_raising(3, 2), tau23)
        self.assertEqual(tau_lowering(3, 2), (e[3, 2],))
        self.assertEqual(tau_raising(3, 1), (e[1, 3],))

        value = Fraction(2, 3)
        self.assertEqual(tau_lowering(3, 1, value), tau31[0] + tau31[1] * value)
        self.assertEqual(tau_raising(3, 2, value), tau23[0] + tau23[1] * value)
        parameter = e[3, 3]
        self.assertNotEqual(tau23[1] * parameter, parameter * tau23[1])
        self.assertEqual(tau_raising(3, 2, parameter), tau23[0] + tau23[1] * parameter)

    def test_embedding_and_index_validation(self):
        e = matrix_units(5)
        self.assertEqual(tau_lowering(5, 1, -e[1, 1], level=3), lowering_operator(5, 1, level=3))
        self.assertEqual(tau_raising(5, 2, 1 - e[2, 2], level=3), raising_operator(5, 2, level=3))
        for constructor in (tau_lowering, tau_raising):
            for rank, i, level in ((1, 1, None), (3, 0, None), (5, 3, 3), (5, 1, 6)):
                with self.subTest(constructor=constructor.__name__, rank=rank, i=i, level=level):
                    with self.assertRaises(ValueError):
                        constructor(rank, i, level=level)
            with self.assertRaises(ValueError):
                constructor(5, 1, EnvelopingElement.scalar(3, 1), level=3)


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
