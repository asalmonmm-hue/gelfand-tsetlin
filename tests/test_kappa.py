from fractions import Fraction
from itertools import islice
from unittest import TestCase

from gelfand_tsetlin import (
    RowArray,
    apply_enveloping,
    c_polynomial,
    kappa_factors,
    kappa_scale,
)


class KappaBasisTests(TestCase):
    def test_written_order_and_pbw_action(self):
        highest_rows = ((4, 2, 0), (4, 2), (4,))
        pattern_rows = ((4, 2, 0), (2, 1), (1,))
        written_order = ((2, -3), (2, -4), (1, -2), (1, -3), (1, -4), (2, -1))
        for shift in (0, Fraction(2, 3)):
            with self.subTest(shift=shift):
                highest = RowArray(tuple(entry + shift for entry in row) for row in highest_rows)
                pattern = RowArray(tuple(entry + shift for entry in row) for row in pattern_rows)
                factors = tuple(kappa_factors(pattern))
                self.assertEqual(factors, tuple((m, u - shift) for m, u in written_order))
                coefficient = 1
                source = highest
                for m, u in reversed(factors):
                    (source, value), = apply_enveloping(c_polynomial(3, m, u), source).items()
                    coefficient *= value
                self.assertEqual(source, pattern)
                self.assertEqual(coefficient, 18)
                self.assertEqual(kappa_scale(pattern), coefficient)

    def test_scale_sign_and_identity(self):
        self.assertEqual(kappa_scale(((3, 1, 0), (2, 0), (2,))), -3)
        for pattern in (((Fraction(1, 2),),), ((4, 2, 0), (4, 2), (4,))):
            with self.subTest(pattern=pattern):
                self.assertEqual(tuple(kappa_factors(pattern)), ())
                self.assertEqual(kappa_scale(pattern), 1)

    def test_lazy_factors_and_invalid_patterns(self):
        factors = kappa_factors(((10**30, 0), (0,)))
        self.assertIs(iter(factors), factors)
        self.assertEqual(tuple(islice(factors, 3)), ((1, -1), (1, -2), (1, -3)))
        for pattern in ((), ((3, 1), (4,)), ((2, 0), (Fraction(1, 2),)), ((2 + 1j, 1j), (1 + 1j,))):
            for construct in (kappa_factors, kappa_scale):
                with self.subTest(pattern=pattern, construct=construct.__name__):
                    with self.assertRaises(ValueError):
                        construct(pattern)
