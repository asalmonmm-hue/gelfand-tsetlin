from fractions import Fraction
from itertools import islice
from unittest import TestCase

from gelfand_tsetlin import RowArray, is_orthogonal_gt_pattern, orthogonal_branching


class OrthogonalBranchingTests(TestCase):
    def test_odd_to_even_includes_both_signs(self):
        self.assertEqual(
            tuple(orthogonal_branching(5, (2, 1))),
            ((2, 1), (2, 0), (2, -1), (1, 1), (1, 0), (1, -1)),
        )

    def test_even_to_odd_uses_absolute_value_of_last_coordinate(self):
        for last in (-1, 1):
            with self.subTest(last=last):
                self.assertEqual(
                    tuple(orthogonal_branching(6, (3, 2, last))),
                    ((3, 2), (3, 1), (2, 2), (2, 1)),
                )

    def test_half_integral_branches(self):
        half = Fraction(1, 2)
        self.assertEqual(
            tuple(orthogonal_branching(5, (3 * half, half))),
            ((3 * half, half), (3 * half, -half), (half, half), (half, -half)),
        )
        self.assertEqual(
            tuple(orthogonal_branching(4, (3 * half, -half))),
            ((3 * half,), (half,)),
        )
        self.assertEqual(tuple(orthogonal_branching(3, (half,))), ((half,), (-half,)))

    def test_known_dimensions_from_iterated_restriction(self):
        def basis_size(dimension, highest_weight):
            if dimension == 2:
                return 1
            return sum(
                basis_size(dimension - 1, lower)
                for lower in orthogonal_branching(dimension, highest_weight)
            )

        half = Fraction(1, 2)
        fixtures = (
            (5, (1, 0), 5),
            (5, (1, 1), 10),
            (5, (half, half), 4),
            (6, (1, 0, 0), 6),
            (6, (1, 1, 0), 15),
            (6, (half, half, half), 4),
            (6, (half, half, -half), 4),
        )
        for dimension, highest_weight, expected in fixtures:
            with self.subTest(dimension=dimension, highest_weight=highest_weight):
                self.assertEqual(basis_size(dimension, highest_weight), expected)

    def test_iterator_is_lazy(self):
        upper = 10**100
        weights = orthogonal_branching(3, (upper,))
        self.assertIs(iter(weights), weights)
        self.assertEqual(tuple(islice(weights, 3)), ((upper,), (upper - 1,), (upper - 2,)))

    def test_invalid_highest_weights(self):
        for dimension, highest_weight in ((2, (0,)), (5, (2,)), (5, (2, -1)), (4, (1, -2))):
            with self.subTest(dimension=dimension, highest_weight=highest_weight):
                with self.assertRaises(ValueError):
                    tuple(orthogonal_branching(dimension, highest_weight))


class OrthogonalPatternTests(TestCase):
    def test_full_pattern_and_terminal_row(self):
        half = Fraction(1, 2)
        spinor_pattern = ((3 * half, half), (half, -half), (half,), (-half,))
        integer_pattern = ((2, 1, -1), (2, 1), (1, -1), (1,), (-1,))
        self.assertTrue(is_orthogonal_gt_pattern(spinor_pattern))
        self.assertTrue(is_orthogonal_gt_pattern(RowArray(integer_pattern)))
        self.assertTrue(is_orthogonal_gt_pattern(((-3 * half,),)))

    def test_invalid_layout_parity_and_interlacing(self):
        half = Fraction(1, 2)
        invalid = (
            (),
            ((1, 0), (0,)),
            ((half,), (0,)),
            ((1,), (-2,)),
            ((2, -1), (0,), (0,)),
            ((2, 1), (1, -2), (1,), (1,)),
            ((Fraction(1, 3),),),
        )
        for pattern in invalid:
            with self.subTest(pattern=pattern):
                self.assertFalse(is_orthogonal_gt_pattern(pattern))
