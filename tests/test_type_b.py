from fractions import Fraction
from itertools import islice
from unittest import TestCase

from gelfand_tsetlin import is_type_b_gt_pattern, type_b_basis_factors


class TypeBPatternTests(TestCase):
    def test_factor_order_and_multiplicities(self):
        pattern = (
            (1, (-2, -5, -8), (-1, -3, -7)),
            (0, (-2, -6), (-1, -5)),
            (1, (-3,), (-1,)),
        )
        self.assertTrue(is_type_b_gt_pattern(pattern))
        self.assertEqual(tuple(type_b_basis_factors(pattern)), (
            ("z", 1, 1, 0), ("Z", 1, -2), ("Z", 1, -3),
            ("z", 2, 2, 1), ("z", 2, 2, 1), ("z", 2, 1, -2), ("Z", 2, -7),
            ("z", 3, 3, 0), ("z", 3, 3, 1), ("z", 3, 1, -3),
            ("z", 3, 3, 2), ("z", 3, 3, 2), ("z", 3, 3, 2),
            ("z", 3, 2, -3), ("z", 3, 2, -3), ("Z", 3, -10),
        ))

    def test_sign_and_exact_half_integral_entries(self):
        integral = ((0, (-1,), (0,)),)
        self.assertTrue(is_type_b_gt_pattern(integral))
        self.assertEqual(tuple(type_b_basis_factors(integral)), (("Z", 1, -1),))

        half = Fraction(1, 2)
        half_integral = ((1, (-3 * half,), (-half,)),)
        self.assertTrue(is_type_b_gt_pattern(half_integral))
        self.assertEqual(tuple(type_b_basis_factors(half_integral)), (
            ("z", 1, 1, 0), ("Z", 1, -3 * half),
        ))

    def test_invalid_patterns(self):
        invalid_patterns = (
            (),
            ((0, (0,)),),
            ((2, (-1,), (-1,)),),
            ((1, (-1,), (0,)),),
            ((0, (-1,), (Fraction(-1, 2),)),),
            ((0, (-1,), (-2,)),),
            ((0, (0, -2), (0, -2)), (0, (-3,), (-2,))),
            ((0, (1,), (1,)),),
        )
        for pattern in invalid_patterns:
            with self.subTest(pattern=pattern):
                self.assertFalse(is_type_b_gt_pattern(pattern))
                with self.assertRaises(ValueError):
                    tuple(type_b_basis_factors(pattern))

    def test_factors_are_lazy_for_large_weight_gaps(self):
        factors = type_b_basis_factors(((0, (-10**30,), (0,)),))
        self.assertIs(iter(factors), factors)
        self.assertEqual(tuple(islice(factors, 3)), (
            ("Z", 1, -1), ("Z", 1, -2), ("Z", 1, -3),
        ))
