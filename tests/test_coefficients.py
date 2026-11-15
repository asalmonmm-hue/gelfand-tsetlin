from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import RowArray, coefficient


class CoefficientTests(TestCase):
    def test_diagonal_is_the_difference_of_consecutive_row_sums(self):
        source = RowArray([[4, 2, 0], [3, 0], [1]])
        self.assertEqual(
            [coefficient(k, k, source, source) for k in (1, 2, 3)],
            [1, 2, 3],
        )
        self.assertEqual(coefficient(1, 1, [[-3]], [[-3]]), -3)

    def test_length_one_row_raising_and_lowering(self):
        source = [[5, 2], [3]]
        self.assertEqual(coefficient(1, 2, source, [[5, 2], [4]]), 4)
        self.assertEqual(coefficient(2, 1, source, [[5, 2], [2]]), 1)

    def test_length_two_row_has_exact_rational_coefficients(self):
        source = [[4, 2, 0], [3, 1], [2]]
        transitions = (
            (2, 3, [4, 1], Fraction(10, 3)),
            (2, 3, [3, 2], Fraction(8, 3)),
            (3, 2, [2, 1], Fraction(1, 3)),
            (3, 2, [3, 0], Fraction(2, 3)),
        )
        for i, j, middle, expected in transitions:
            with self.subTest(i=i, j=j, middle=middle):
                result = coefficient(i, j, source, [[4, 2, 0], middle, [2]])
                self.assertIsInstance(result, Fraction)
                self.assertEqual(result, expected)

    def test_missing_transitions_are_zero(self):
        source = [[4, 2, 0], [3, 1], [2]]
        tests = (
            (1, 1, [3, 1], 3),  # Diagonal action changes no entries.
            (2, 3, [3, 1], 2),  # Adjacent action has no diagonal term.
            (2, 3, [2, 1], 2),  # This is a lowering move.
            (2, 3, [3, 1], 3),  # This changes the wrong row.
            (2, 3, [4, 2], 2),  # This changes two entries.
            (2, 3, [4, 1], 3),  # This changes the right row and another row.
        )
        for i, j, middle, bottom in tests:
            with self.subTest(i=i, j=j, middle=middle, bottom=bottom):
                self.assertEqual(
                    coefficient(i, j, source, [[4, 2, 0], middle, [bottom]]),
                    0,
                )

    def test_common_shift_only_changes_diagonal_coefficients(self):
        shift = Fraction(1, 3)
        source = [[5 + shift, 2 + shift], [3 + shift]]
        target = [[5 + shift, 2 + shift], [4 + shift]]
        self.assertEqual(coefficient(1, 1, source, source), 3 + shift)
        self.assertEqual(coefficient(2, 2, source, source), 4 + shift)
        self.assertEqual(coefficient(1, 2, source, target), 4)

        complex_source = [[5 + 2j, 2 + 2j], [3 + 2j]]
        complex_target = [[5 + 2j, 2 + 2j], [4 + 2j]]
        self.assertEqual(coefficient(1, 2, complex_source, complex_target), 4)
        self.assertEqual(coefficient(1, 1, complex_source, complex_source), 3 + 2j)
        self.assertEqual(coefficient(1, 1, complex_source, complex_target), 0)

    def test_reports_invalid_indices_and_patterns(self):
        source = [[4, 2, 0], [3, 1], [2]]
        for i, j in ((0, 1), (1, 4), (1, 3)):
            with self.subTest(i=i, j=j), self.assertRaises(ValueError):
                coefficient(i, j, source, source)
        with self.assertRaises(TypeError):
            coefficient(1.0, 1, source, source)
        for invalid in ([[4, 2, 0], [3, 1]], [[4, 2, 0], [5, 1], [2]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                coefficient(1, 1, source, invalid)
        with self.assertRaises(ValueError):
            coefficient(1, 1, [[5, 2, 0], [3, 1], [2]], source)

    def test_reports_unrepresentable_exact_results(self):
        top = [2**63 - 1, 0]
        source = [top, [2**62]]
        target = [top, [2**62 + 1]]
        with self.assertRaises(OverflowError):
            coefficient(1, 2, source, target)

    def test_cancels_factors_whose_magnitude_exceeds_signed_int64(self):
        largest = 2**63 - 1
        top = [largest, 1, 0]
        source = [top, [largest, 0], [1]]
        target = [top, [largest, 1], [1]]
        self.assertEqual(coefficient(2, 3, source, target), 1)

    def test_rank_three_raising_lowering_commutator(self):
        patterns = [
            ((2, 1, 0), (a, b), (c,))
            for a in (1, 2)
            for b in (0, 1)
            for c in range(b, a + 1)
        ]
        raising = {
            (target, source): coefficient(2, 3, source, target)
            for source in patterns
            for target in patterns
        }
        lowering = {
            (target, source): coefficient(3, 2, source, target)
            for source in patterns
            for target in patterns
        }
        for source in patterns:
            for target in patterns:
                with self.subTest(source=source, target=target):
                    bracket = sum(
                        raising[target, intermediate] * lowering[intermediate, source]
                        - lowering[target, intermediate] * raising[intermediate, source]
                        for intermediate in patterns
                    )
                    expected = (
                        coefficient(2, 2, source, target)
                        - coefficient(3, 3, source, target)
                    )
                    self.assertEqual(bracket, expected)
