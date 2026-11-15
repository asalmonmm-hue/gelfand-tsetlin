from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import (
    yangian_strings_in_general_position,
    yangian_tensor_is_irreducible,
)


class YangianStringTests(TestCase):
    def test_unit_step_overlap_adjacency_and_containment(self):
        string = (3, 0)
        cases = (
            ((5, 3), False),  # {3,4} is adjacent to {0,1,2}.
            ((4, 2), False),  # {2,3} overlaps without containment.
            ((2, 1), True),
            ((4, -1), True),
            ((3, 0), True),
            ((5, 4), True),
            ((2, 2), True),
        )
        for other, expected in cases:
            with self.subTest(other=other):
                self.assertEqual(yangian_strings_in_general_position(string, other), expected)
                self.assertEqual(yangian_strings_in_general_position(other, string), expected)

    def test_fractional_and_complex_unit_step_residue_classes(self):
        half = Fraction(1, 2)
        quarter = Fraction(1, 4)
        self.assertFalse(yangian_strings_in_general_position((2 + half, half), (3 + half, 2 + half)))
        self.assertTrue(yangian_strings_in_general_position((2 + half, half), (2 + quarter, quarter)))
        self.assertFalse(yangian_strings_in_general_position((2 + 1j, 1j), (4 + 1j, 2 + 1j)))
        self.assertTrue(yangian_strings_in_general_position((2 + 1j, 1j), (4 + 2j, 2 + 2j)))

    def test_very_long_strings_are_checked_by_their_endpoints(self):
        large = 10**100
        wide = (large, -large)
        self.assertFalse(yangian_strings_in_general_position(wide, (large + 1, large)))
        self.assertTrue(yangian_strings_in_general_position(wide, (large - 1, 1 - large)))
        self.assertTrue(yangian_strings_in_general_position(wide, (large + 2, large + 1)))

    def test_tensor_criterion_uses_all_pairs_and_accepts_empty_factors(self):
        self.assertTrue(yangian_tensor_is_irreducible(()))
        self.assertTrue(yangian_tensor_is_irreducible(((0, 0),)))
        self.assertTrue(yangian_tensor_is_irreducible(((8, 0), (4, 1), (3, 2), (2, 2), (12, 10), (4, 1))))
        self.assertFalse(yangian_tensor_is_irreducible(((2, 0), (12, 10), (4, 2))))

    def test_invalid_highest_weights(self):
        for invalid in ((1,), (1, 0, 0), (-1, 0), (Fraction(1, 2), 0), (1 + 1j, 0), (float("inf"), 0)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    yangian_strings_in_general_position((1, 0), invalid)
                with self.assertRaises(ValueError):
                    yangian_tensor_is_irreducible(((1, 0), invalid))
