from fractions import Fraction
from functools import partial
from itertools import product
from unittest import TestCase

from gelfand_tsetlin import (
    RowArray,
    apply_enveloping,
    classical_branching_generator,
    classical_interpolation_node,
    classical_interpolation_polynomial,
    classical_matrix_unit,
)


class ClassicalMatrixUnitTests(TestCase):
    def test_molev_signed_indices_in_the_ambient_pbw_basis(self):
        # In B2 the signed order is (-2, -1, 0, 1, 2).
        b = classical_matrix_unit("B", 2, -1, 0)
        self.assertEqual(b.rank, 5)
        self.assertEqual(b.terms, {((2, 3),): 1, ((3, 4),): -1})
        self.assertEqual(classical_matrix_unit("B", 2, 0, 0), 0)

        # In C2 and D2 the signed order is (-2, -1, 1, 2).
        for kind, sign in (("C", 1), ("D", -1)):
            with self.subTest(kind=kind):
                f = classical_matrix_unit(kind, 2, 2, -1)
                self.assertEqual(f.rank, 4)
                self.assertEqual(f.terms, {((4, 2),): 1, ((3, 1),): sign})

    def test_generators_on_the_antidiagonal(self):
        for kind in ("B", "D"):
            with self.subTest(kind=kind):
                self.assertEqual(classical_matrix_unit(kind, 2, 2, -2), 0)
        self.assertEqual(classical_matrix_unit("C", 2, 2, -2).terms, {((4, 1),): 2})
        self.assertEqual(classical_matrix_unit("C", 2, -2, 2).terms, {((1, 4),): 2})

    def test_lie_brackets_in_pbw_form(self):
        for kind, left, right, result in (
            ("B", (1, 0), (0, 2), (1, 2)),
            ("C", (2, 1), (1, -2), (2, -2)),
        ):
            with self.subTest(kind=kind):
                x = classical_matrix_unit(kind, 2, *left)
                y = classical_matrix_unit(kind, 2, *right)
                self.assertEqual(x * y - y * x, classical_matrix_unit(kind, 2, *result))

        x = classical_matrix_unit("D", 2, 2, -1)
        y = classical_matrix_unit("D", 2, -1, 2)
        expected = classical_matrix_unit("D", 2, 2, 2) + classical_matrix_unit("D", 2, 1, 1)
        self.assertEqual(x * y - y * x, expected)

    def test_generators_span_the_expected_dimension(self):
        for kind, rank in product(("B", "C", "D"), (2, 3)):
            indices = tuple(i for i in range(-rank, rank + 1) if i or kind == "B")
            supports = {
                frozenset(classical_matrix_unit(kind, rank, i, j).terms)
                for i, j in product(indices, repeat=2)
            } - {frozenset()}
            with self.subTest(kind=kind, rank=rank):
                # Each nonzero reflection orbit spans one line with disjoint support.
                self.assertEqual(len(set().union(*supports)), sum(map(len, supports)))
                dimension = rank * (2 * rank - 1 if kind == "D" else 2 * rank + 1)
                self.assertEqual(len(supports), dimension)

    def test_invalid_types_ranks_and_signed_indices(self):
        for args in (
            ("A", 2, 1, 2),
            ("B", 0, 0, 0),
            ("B", 2, 3, 1),
            ("B", 2, 1, -3),
            ("C", 2, 0, 1),
            ("D", 2, 1, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                classical_matrix_unit(*args)
        with self.assertRaises(TypeError):
            classical_matrix_unit("C", 2, 1.0, 2)


class ClassicalBranchingGeneratorTests(TestCase):
    def test_type_c_normalized_inner_first_and_reflected_outer_first(self):
        f = partial(classical_matrix_unit, "C", 2)
        z = partial(classical_branching_generator, "C", 2)
        for a in (-2, 2):
            with self.subTest(a=a):
                self.assertEqual(z(-1, a), f(-1, a))
                expected = 2 * f(1, a) * (f(1, 1) - 1) + f(1, -1) * f(-1, a)
                self.assertEqual(z(1, a), expected)
        self.assertEqual(z(2, 1), -f(-1, -2))
        self.assertEqual(z(-2, 1), f(-1, 2))

    def test_type_c_operators_preserve_highest_vectors_and_shift_weight(self):
        # In the ambient standard representation the signed labels -2,
        # -1 and 2 correspond to positions 1, 2 and 4 of the gl4 basis.
        top = (1, 0, 0, 0)
        outer_negative = RowArray((top, (1, 0, 0), (1, 0), (1,)))
        outer_positive = RowArray((top, (0, 0, 0), (0, 0), (0,)))
        inner_highest = RowArray((top, (1, 0, 0), (1, 0), (0,)))
        inner_positive_root = classical_matrix_unit("C", 2, -1, 1)
        cartan = classical_matrix_unit("C", 2, 1, 1)
        raise_weight = classical_branching_generator("C", 2, 1, 2)
        lower_weight = classical_branching_generator("C", 2, 2, 1)

        self.assertEqual(apply_enveloping(cartan, inner_highest), {inner_highest: -1})
        self.assertEqual(apply_enveloping(cartan, outer_negative), {})
        for source in (inner_highest, outer_negative, outer_positive):
            self.assertEqual(apply_enveloping(inner_positive_root, source), {})
        self.assertEqual(apply_enveloping(raise_weight, inner_highest), {outer_negative: 4})
        self.assertEqual(apply_enveloping(lower_weight, outer_negative), {inner_highest: -1})
        self.assertEqual(apply_enveloping(raise_weight, outer_positive), {})

    def test_type_b_zero_inner_label_preserves_inner_weight(self):
        f = partial(classical_matrix_unit, "B", 2)
        z = partial(classical_branching_generator, "B", 2)
        expected = f(0, 2) * (f(1, 1) - 1) + f(0, -1) * f(-1, 2)
        actual = z(0, 2)
        self.assertTrue(actual)
        self.assertEqual(actual, expected)
        self.assertEqual(z(-2, 0), actual)
        self.assertEqual(f(1, 1) * actual - actual * f(1, 1), 0)

    def test_type_d_normalization_retains_the_mirror_label_in_the_chains(self):
        f = partial(classical_matrix_unit, "D", 3)
        f2 = f(2, 2) - 1
        f1 = f(1, 1)
        expected = (
            f(2, 3) * (f2 - f1) * (f2 + f1)
            + f(2, 1) * f(1, 3) * (f2 + f1)
            + f(2, -1) * f(-1, 3) * (f2 - f1)
            - f(2, 1) * f(2, -1) * f(-2, 3)
        )
        self.assertEqual(classical_branching_generator("D", 3, 2, 3), expected)

    def test_type_d_rank_two_and_outer_first_reflection(self):
        f = partial(classical_matrix_unit, "D", 2)
        z = partial(classical_branching_generator, "D", 2)
        for a in (-2, 2):
            with self.subTest(a=a):
                self.assertEqual(z(1, a), f(1, a))
                self.assertEqual(z(a, 1), -f(-1, -a))

    def test_requires_one_inner_and_one_outer_signed_label(self):
        for args in (
            ("A", 2, 1, 2),
            ("B", 0, 0, 0),
            ("B", 2, 2, -2),
            ("C", 2, 1, -1),
            ("C", 2, 0, 2),
            ("C", 2, -2, 3),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                classical_branching_generator(*args)


class ClassicalInterpolationTests(TestCase):
    def test_rank_one_outer_node_uses_the_type_specific_chain(self):
        f_b = partial(classical_matrix_unit, "B", 1)
        f_c = partial(classical_matrix_unit, "C", 1)
        expected = {"B": f_b(1, 0) * f_b(0, -1), "C": f_c(1, -1)}
        for kind, node in expected.items():
            with self.subTest(kind=kind):
                self.assertEqual(classical_interpolation_node(kind, 1, 1), node)
                self.assertEqual(classical_interpolation_polynomial(kind, 1, (0,)), (node,))
                self.assertEqual(
                    classical_interpolation_polynomial(kind, 1, (0,), Fraction(7, 3)), node
                )

    def test_type_c_outer_node_and_exact_polynomial_coefficients(self):
        f = partial(classical_matrix_unit, "C", 2)
        z = partial(classical_branching_generator, "C", 2)
        inner = z(2, 1) * z(1, -2)
        f1, f2 = f(1, 1) - 1, f(2, 2) - 2
        outer = (
            f(2, -2) * (f2 + f1) * (f2 - f1)
            + f(2, 1) * f(1, -2) * (f2 + f1)
            + f(2, -1) * f(-1, -2) * (f2 - f1)
            + f(2, 1) * f(1, -1) * f(-1, -2)
        )
        self.assertEqual(classical_interpolation_node("C", 2, 1), inner)
        self.assertEqual(classical_interpolation_node("C", 2, 2), outer)
        # The shifted nodes of weight (-1, -2) are -3/2 and -7/2.
        polynomial = partial(classical_interpolation_polynomial, "C", 2, (-1, -2))
        constant = Fraction(49, 40) * inner - Fraction(9, 40) * outer
        quadratic = Fraction(1, 10) * (outer - inner)
        self.assertEqual(polynomial(), (constant, quadratic))
        self.assertEqual(polynomial(Fraction(3, 2)), inner)
        self.assertEqual(polynomial(Fraction(-7, 2)), outer)
        self.assertEqual(polynomial(Fraction(2, 3)), constant + Fraction(4, 9) * quadratic)

    def test_type_b_shifted_nodes_at_a_half_integral_weight(self):
        weight = (Fraction(-1, 2), Fraction(-3, 2))
        for i, u in enumerate((Fraction(-1, 2), Fraction(-5, 2)), 1):
            with self.subTest(i=i):
                self.assertEqual(
                    classical_interpolation_polynomial("B", 2, weight, u),
                    classical_interpolation_node("B", 2, i),
                )

    def test_type_c_interpolation_moves_inside_a_multiplicity_space(self):
        top = (1, 0, 0, 0)
        source = RowArray((top, (1, 0, 0), (1, 0), (1,)))
        target = RowArray((top, (0, 0, 0), (0, 0), (0,)))
        # Both vectors are sp2-highest of inner weight zero. The source
        # has outer weight -1, so the squared interpolation nodes are 1/4
        # and 25/4. The inner node kills it and the outer node yields 2.
        polynomial = partial(classical_interpolation_polynomial, "C", 2, (0, -1))
        self.assertEqual(apply_enveloping(polynomial(Fraction(1, 2)), source), {})
        self.assertEqual(apply_enveloping(polynomial(Fraction(5, 2)), source), {target: 2})
        self.assertEqual(apply_enveloping(polynomial(0), source), {target: Fraction(-1, 12)})

    def test_type_d_uses_only_inner_nodes(self):
        z = partial(classical_branching_generator, "D", 3)
        polynomial = partial(classical_interpolation_polynomial, "D", 3)
        weight = (0, -1, -2)
        for i, u in enumerate((Fraction(1, 2), Fraction(-3, 2)), 1):
            with self.subTest(i=i):
                node = z(3, i) * z(i, -3)
                self.assertEqual(classical_interpolation_node("D", 3, i), node)
                self.assertEqual(polynomial(weight, u), node)
        self.assertEqual(polynomial(weight), polynomial((0, -1, -4)))

    def test_type_c_outer_node_collision_in_a_symmetric_power(self):
        top = (3, 0, 0, 0)
        source = RowArray((top, (2, 0, 0), (2, 0), (1,)))
        target = RowArray((top, (1, 0, 0), (1, 0), (0,)))
        # In Sym^3 of the defining module this source is proportional to
        # e_{-2} e_{-1} e_2. Both shifted nodes equal -3/2 on it.
        polynomial = partial(classical_interpolation_polynomial, "C", 2, (-1, 0))
        self.assertEqual(apply_enveloping(polynomial(0), source), {target: Fraction(-1, 24)})
        self.assertEqual(apply_enveloping(polynomial(Fraction(1, 2)), source), {})
        self.assertEqual(apply_enveloping(polynomial(1), source), {target: Fraction(1, 8)})

    def test_type_b_outer_node_collision_in_a_symmetric_power(self):
        top = (3, 0, 0, 0, 0)
        source = RowArray((top, (2, 0, 0, 0), (2, 0, 0), (2, 0), (1,)))
        target = RowArray((top, (1, 0, 0, 0), (1, 0, 0), (1, 0), (0,)))
        # The middle zero label changes the rho shift. Both nodes are -1.
        polynomial = partial(classical_interpolation_polynomial, "B", 2, (-1, 0))
        self.assertEqual(apply_enveloping(polynomial(0), source), {})
        self.assertEqual(apply_enveloping(polynomial(1), source), {target: Fraction(1, 12)})
        self.assertEqual(apply_enveloping(polynomial(2), source), {target: Fraction(1, 3)})

    def test_type_d_opposite_node_collision_in_the_defining_module(self):
        top = (1, 0, 0, 0, 0, 0)
        source = RowArray(tuple(top[:size] for size in range(6, 0, -1)))
        target = RowArray((top, (0,) * 5, (0,) * 4, (0,) * 3, (0,) * 2, (0,)))
        # e_{-3} and e_3 are both inner-highest, with inner weight (0,0).
        # The GT normalization contributes 1/5! to the e_{-3} -> e_3 move.
        polynomial = partial(classical_interpolation_polynomial, "D", 3, (0, 0, -1))
        self.assertEqual(apply_enveloping(polynomial(0), source), {target: Fraction(-1, 240)})
        self.assertEqual(apply_enveloping(polynomial(Fraction(-1, 2)), source), {})
        self.assertEqual(apply_enveloping(polynomial(1), source), {target: Fraction(1, 80)})

    def test_invalid_nodes_weights_and_parameters(self):
        with self.assertRaises(ValueError):
            classical_interpolation_node("D", 3, 3)
        with self.assertRaises(ValueError):
            classical_interpolation_node("B", 2, 0)
        with self.assertRaises(ValueError):
            classical_interpolation_polynomial("D", 3, (0, -1))
        with self.assertRaises(ValueError):
            classical_interpolation_polynomial("D", 3, (0, 1, 0))
        with self.assertRaises(ValueError):
            classical_interpolation_polynomial("D", 1, (0,))
        with self.assertRaises(TypeError):
            classical_interpolation_polynomial("C", 1, (0,), 1.5)
