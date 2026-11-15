from fractions import Fraction
from math import comb
from unittest import TestCase

from gelfand_tsetlin import EnvelopingElement


def matrix_unit(i, j):
    return EnvelopingElement.matrix_unit(3, i, j)


class EnvelopingTests(TestCase):
    def test_reversed_pair_adds_the_commutator(self):
        e12, e21 = matrix_unit(1, 2), matrix_unit(2, 1)
        e11, e22 = matrix_unit(1, 1), matrix_unit(2, 2)

        self.assertEqual(e12 * e21 - e21 * e12, e11 - e22)
        self.assertEqual(
            dict((e21 * e12).terms),
            {((1, 2), (2, 1)): 1, ((1, 1),): -1, ((2, 2),): 1},
        )

    def test_commutators_with_only_one_delta_and_with_neither_delta(self):
        e12, e23, e31 = matrix_unit(1, 2), matrix_unit(2, 3), matrix_unit(3, 1)
        self.assertEqual(e12 * e23 - e23 * e12, matrix_unit(1, 3))
        self.assertEqual(e12 * e31 - e31 * e12, -matrix_unit(3, 2))
        self.assertEqual(e12 * matrix_unit(3, 3), matrix_unit(3, 3) * e12)

    def test_repeated_cartan_run_has_binomial_coefficients(self):
        exponent = 96
        for cartan, root, on_left, sign in (
            ((2, 2), (2, 1), True, 1),
            ((2, 2), (1, 2), True, -1),
            ((1, 1), (2, 1), False, 1),
            ((2, 2), (2, 3), False, -1),
        ):
            with self.subTest(cartan=cartan, root=root, on_left=on_left):
                run = matrix_unit(*cartan) ** exponent
                unit = matrix_unit(*root)
                actual = run * unit if on_left else unit * run
                expected = {
                    ((root,) + (cartan,) * degree if on_left else (cartan,) * degree + (root,)):
                    comb(exponent, degree) * sign ** (exponent - degree)
                    for degree in range(exponent + 1)
                }
                self.assertEqual(dict(actual.terms), expected)

    def test_constructor_collects_cartan_run_corrections_with_other_terms(self):
        prefix, cartan, root, suffix = (1, 1), (2, 2), (1, 2), (3, 3)
        coefficients = Fraction(2, 3)
        element = EnvelopingElement(3, {
            (prefix,) + (cartan,) * 3 + (root, suffix): coefficients,
            (prefix, root, suffix): coefficients,
        })
        self.assertEqual(dict(element.terms), {
            (prefix, root) + (cartan,) * degree + (suffix,):
            coefficients * comb(3, degree) * (-1) ** (3 - degree)
            for degree in range(1, 4)
        })

    def test_constructor_normal_orders_words_and_collects_equal_terms(self):
        raw = {
            ((2, 1), (1, 2)): Fraction(2, 3),
            ((1, 2), (2, 1)): Fraction(-2, 3),
            (): Fraction(1, 5),
        }
        element = EnvelopingElement(3, raw)
        self.assertEqual(element.rank, 3)
        self.assertEqual(
            dict(element.terms),
            {
                ((1, 1),): Fraction(-2, 3),
                ((2, 2),): Fraction(2, 3),
                (): Fraction(1, 5),
            },
        )

    def test_scalar_identity_powers_and_exact_cancellation(self):
        e12 = matrix_unit(1, 2)
        one = EnvelopingElement.scalar(3, 1)
        zero = EnvelopingElement.scalar(3, 0)
        self.assertEqual(e12**0, one)
        self.assertEqual((one + e12) ** 2, one + 2 * e12 + e12 * e12)
        self.assertEqual((Fraction(1, 3) * e12) * 3, e12)
        self.assertEqual(e12 + (-e12), zero)
        self.assertFalse(zero)
        self.assertEqual(dict(zero.terms), {})

    def test_product_is_associative(self):
        e12, e21, e32 = matrix_unit(1, 2), matrix_unit(2, 1), matrix_unit(3, 2)
        a, b, c = e32 + e21, e12 + Fraction(1, 7), e21 - e32
        self.assertEqual((a * b) * c, a * (b * c))

    def test_ordered_term_products_and_identity_collect_and_cancel(self):
        a, b = matrix_unit(1, 2), matrix_unit(1, 3)
        left = 1 - Fraction(1, 3) * a
        right = Fraction(1, 3) * a + Fraction(2, 5) * b + 1
        self.assertEqual(dict((left * right).terms), {
            (): 1,
            ((1, 3),): Fraction(2, 5),
            ((1, 2), (1, 2)): Fraction(-1, 9),
            ((1, 2), (1, 3)): Fraction(-2, 15),
        })

    def test_molev_lowering_operator(self):
        e11, e22 = matrix_unit(1, 1), matrix_unit(2, 2)
        e31, e32, e21 = matrix_unit(3, 1), matrix_unit(3, 2), matrix_unit(2, 1)

        # z_31 = E_31 (h_1 - h_2) + E_21 E_32, with h_i = E_ii - i + 1.
        z31 = e31 * (e11 - e22 + 1) + e21 * e32
        self.assertEqual(
            dict(z31.terms),
            {
                ((1, 1), (3, 1)): 1,
                ((2, 2), (3, 1)): -1,
                ((2, 1), (3, 2)): 1,
                ((3, 1),): 2,
            },
        )

    def test_term_view_is_read_only(self):
        element = matrix_unit(1, 2)
        with self.assertRaises(TypeError):
            element.terms[((1, 2),)] = 0
        self.assertEqual(dict(element.terms), {((1, 2),): 1})

    def test_invalid_rank_unit_and_scalar(self):
        with self.assertRaises(ValueError):
            EnvelopingElement.scalar(0, 1)
        with self.assertRaises(ValueError):
            EnvelopingElement.matrix_unit(3, 0, 1)
        with self.assertRaises(ValueError):
            EnvelopingElement.matrix_unit(3, 1, 4)
        with self.assertRaises(TypeError):
            EnvelopingElement.scalar(3, 0.5)
