from fractions import Fraction
from math import comb
from unittest import TestCase

from gelfand_tsetlin import EnvelopingElement


def multiply_tensor_terms(rank, left, right):
    terms = {}
    for (a, b), x in left.items():
        for (c, d), y in right.items():
            ac = EnvelopingElement(rank, {a: 1}) * EnvelopingElement(rank, {c: 1})
            bd = EnvelopingElement(rank, {b: 1}) * EnvelopingElement(rank, {d: 1})
            for u, p in ac.terms.items():
                for v, q in bd.terms.items():
                    key = u, v
                    terms[key] = terms.get(key, 0) + x * y * p * q
    return {key: value for key, value in terms.items() if value}


class CoproductTests(TestCase):
    def test_zero_scalars_and_primitive_generators(self):
        scalar = EnvelopingElement.scalar
        self.assertEqual(scalar(3, 0).coproduct(), {})
        self.assertEqual(scalar(3, 1).coproduct(), {((), ()): 1})
        self.assertEqual(scalar(3, Fraction(-3, 7)).coproduct(),
                         {((), ()): Fraction(-3, 7)})

        word = ((3, 1),)
        terms = EnvelopingElement.matrix_unit(3, 3, 1).coproduct()
        self.assertEqual(terms, {(word, ()): 1, ((), word): 1})
        with self.assertRaises(TypeError):
            terms[word, ()] = 2

    def test_repeated_word_uses_exact_binomial_coefficients(self):
        word = ((1, 2),)
        element = EnvelopingElement(2, {word * 100: Fraction(5, 7)})
        expected = {
            (word * i, word * (100 - i)): Fraction(5 * comb(100, i), 7)
            for i in range(101)
        }
        self.assertEqual(element.coproduct(), expected)

    def test_multiple_runs_stay_in_pbw_order_in_both_factors(self):
        a, b = ((1, 2),), ((2, 1),)
        element = EnvelopingElement(2, {a * 3 + b * 2: 1})
        expected = {
            (a * i + b * j, a * (3 - i) + b * (2 - j)): comb(3, i) * comb(2, j)
            for i in range(4)
            for j in range(3)
        }
        self.assertEqual(element.coproduct(), expected)

    def test_coproduct_preserves_products_with_noncommuting_factors(self):
        e12 = EnvelopingElement.matrix_unit(2, 1, 2)
        e21 = EnvelopingElement.matrix_unit(2, 2, 1)
        left = e21**2 + Fraction(2, 5) * e12
        right = e12 * e21 - 1
        self.assertEqual(
            (left * right).coproduct(),
            multiply_tensor_terms(2, left.coproduct(), right.coproduct()),
        )

    def test_coproduct_is_coassociative_on_a_sparse_element(self):
        e12 = EnvelopingElement.matrix_unit(2, 1, 2)
        e21 = EnvelopingElement.matrix_unit(2, 2, 1)
        terms = (e21 * e12**2 + Fraction(1, 3) * e12 - 2).coproduct()
        left, right = {}, {}
        for (a, b), coefficient in terms.items():
            for (u, v), value in EnvelopingElement(2, {a: 1}).coproduct().items():
                key = u, v, b
                left[key] = left.get(key, 0) + coefficient * value
            for (u, v), value in EnvelopingElement(2, {b: 1}).coproduct().items():
                key = a, u, v
                right[key] = right.get(key, 0) + coefficient * value
        self.assertEqual(left, right)


class CasimirTests(TestCase):
    def test_quadratic_casimir_has_trace_normalization_and_is_central(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                units = {
                    (i, j): EnvelopingElement.matrix_unit(rank, i, j)
                    for i in range(1, rank + 1)
                    for j in range(1, rank + 1)
                }
                casimir = EnvelopingElement.quadratic_casimir(rank)
                self.assertEqual(
                    casimir, sum(unit * units[j, i] for (i, j), unit in units.items())
                )
                for unit in units.values():
                    self.assertEqual(casimir * unit, unit * casimir)

    def test_quadratic_casimir_coproduct_has_the_trace_cross_term(self):
        rank = 3
        casimir = EnvelopingElement.quadratic_casimir(rank)
        expected = {
            key: coefficient
            for word, coefficient in casimir.terms.items()
            for key in ((word, ()), ((), word))
        }
        expected.update({
            (((i, j),), ((j, i),)): 2
            for i in range(1, rank + 1)
            for j in range(1, rank + 1)
        })
        self.assertEqual(casimir.coproduct(), expected)
