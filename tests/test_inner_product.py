from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import RowArray, coefficient, inner_product


class InnerProductTests(TestCase):
    def test_gl2_norms_in_molevs_basis(self):
        # The pattern with bottom entry a has norm (5-a)! 3! / (a-2)!.
        for a, expected in ((5, 1), (4, 3), (3, 12), (2, 36)):
            with self.subTest(a=a):
                pattern = RowArray([[5, 2], [a]])
                result = inner_product(pattern, pattern)
                self.assertIs(type(result), int)
                self.assertEqual(result, expected)

    def test_distinct_patterns_of_the_same_weight_are_orthogonal(self):
        left = [[2, 1, 0], [2, 0], [1]]
        right = [[2, 1, 0], [1, 1], [1]]
        self.assertEqual(inner_product(left, right), 0)
        self.assertEqual(inner_product(right, left), 0)
        self.assertEqual(inner_product(left, left), 2)

    def test_orthogonality_beyond_native_offset_range(self):
        left = ((2**63, 0), (0,))
        right = ((2**63, 0), (1,))
        self.assertEqual(inner_product(left, right), 0)
        self.assertEqual(inner_product(right, left), 0)

    def test_rank_three_raising_and_lowering_are_adjoint(self):
        lower = [[4, 2, 0], [3, 1], [2]]
        for middle in ([4, 1], [3, 2]):
            with self.subTest(middle=middle):
                upper = [[4, 2, 0], middle, [2]]
                self.assertEqual(
                    coefficient(2, 3, lower, upper) * inner_product(upper, upper),
                    coefficient(3, 2, upper, lower) * inner_product(lower, lower),
                )

    def test_norm_is_invariant_under_a_common_shift(self):
        for shift in (-8, Fraction(1, 3), 2j):
            with self.subTest(shift=shift):
                pattern = [[5 + shift, 2 + shift], [3 + shift]]
                self.assertEqual(inner_product(pattern, pattern), 12)
        self.assertEqual(inner_product([[2j]], [[2j]]), 1)

    def test_reports_invalid_patterns_and_incompatible_highest_weights(self):
        valid = [[4, 2, 0], [3, 1], [2]]
        for invalid in ([[4, 2, 0], [3, 1]], [[4, 2, 0], [5, 1], [2]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                inner_product(valid, invalid)
        with self.assertRaises(ValueError):
            inner_product([[5, 2, 0], [3, 1], [2]], valid)

    def test_reports_unrepresentable_exact_norm(self):
        pattern = [[21, 0], [0]]
        with self.assertRaises(OverflowError):
            inner_product(pattern, pattern)

    def test_highest_weight_norm_needs_no_large_factorials(self):
        largest = 2**63 - 1
        pattern = [[largest, 0], [largest]]
        self.assertEqual(inner_product(pattern, pattern), 1)
