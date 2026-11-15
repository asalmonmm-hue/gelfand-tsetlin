from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import RowArray, a_polynomial, apply_enveloping, gt_eigenvalues, weight


class GTCharacterTests(TestCase):
    def test_exact_shifted_capelli_coefficients(self):
        shift = Fraction(1, 3)
        rows = ((3, 1, 0), (2, 1), (1,))
        pattern = tuple(tuple(entry + shift for entry in row) for row in rows)
        self.assertEqual(gt_eigenvalues(pattern), (
            (Fraction(4, 3),),
            (Fraction(8, 3), Fraction(7, 9)),
            (2, -5, Fraction(-50, 27)),
        ))

    def test_large_integer_coefficients(self):
        shift = 2**100
        self.assertEqual(gt_eigenvalues(((shift + 1, shift), (shift,))), (
            (shift,), (2 * shift, shift**2 - 1),
        ))

    def test_pbw_capelli_coefficients_act_by_the_joint_eigenvalues(self):
        shift = Fraction(1, 3)
        rows = ((2, 1, 0), (2, 0), (1,))
        for offset in (0, shift):
            pattern = RowArray(tuple(tuple(entry + offset for entry in row) for row in rows))
            rank = len(pattern.rows)
            for level, eigenvalues in enumerate(gt_eigenvalues(pattern), 1):
                operators = reversed(a_polynomial(rank, level)[:-1])
                for i, (eigenvalue, operator) in enumerate(zip(eigenvalues, operators, strict=True), 1):
                    with self.subTest(offset=offset, level=level, i=i):
                        expected = {pattern: eigenvalue} if eigenvalue else {}
                        self.assertEqual(apply_enveloping(operator, pattern), expected)

    def test_joint_eigenvalues_distinguish_patterns_of_equal_weight(self):
        first = RowArray(((2, 1, 0), (2, 0), (1,)))
        second = RowArray(((2, 1, 0), (1, 1), (1,)))
        self.assertEqual(weight(first), weight(second))
        self.assertEqual(gt_eigenvalues(first)[1], (1, -2))
        self.assertEqual(gt_eigenvalues(second)[1], (1, 0))

    def test_rejects_invalid_and_nonreal_patterns(self):
        invalid = ((), ((2, 0), (3,)), ((2, 0), (1, 1)), ((2 + 1j, 1j), (1 + 1j,)))
        for pattern in invalid:
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                gt_eigenvalues(pattern)
