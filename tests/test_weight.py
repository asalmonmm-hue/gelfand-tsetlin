from decimal import Decimal
from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import RowArray, coefficient, weight


class WeightTests(TestCase):
    def test_matches_diagonal_coefficients_in_order(self):
        source = RowArray([[4, 2, 0], [3, 0], [1]])
        expected = tuple(coefficient(k, k, source, source) for k in (1, 2, 3))
        self.assertEqual(weight(source), expected)
        self.assertEqual(expected, (1, 2, 3))

    def test_accepts_one_shot_row_iterables(self):
        rows = (iter(row) for row in [[4, 2, 0], [3, 0], [1]])
        self.assertEqual(weight(rows), (1, 2, 3))

    def test_large_integer_weights_are_exact(self):
        wide = 2**100
        source = [[wide, -3], [wide - 1]]
        self.assertEqual(weight(source), (wide - 1, -2))
        self.assertEqual(coefficient(1, 1, source, source), wide - 1)
        self.assertEqual(coefficient(2, 2, source, source), -2)
        self.assertEqual(coefficient(1, 1, source, [[wide, -3], [wide]]), 0)

    def test_exact_common_real_shifts(self):
        third = Fraction(1, 3)
        self.assertEqual(weight([[5 + third, 2 + third], [3 + third]]), (3 + third, 4 + third))
        self.assertEqual(weight([[Decimal("2.1"), Decimal("0.1")], [Decimal("1.1")]]),
                         (Fraction(11, 10), Fraction(11, 10)))

    def test_nonreal_common_shift_matches_diagonal_coefficients(self):
        pattern = [[4 + 2j, 1 + 2j], [3 + 2j]]
        self.assertEqual(weight(pattern), (3 + 2j, 2 + 2j))

    def test_invalid_patterns_raise_value_error(self):
        invalid = (None, [], [[1, 0]], [[False]], [[1, 0], [2]], [[1, 0], [Fraction(1, 2)]])
        for pattern in invalid:
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                weight(pattern)
