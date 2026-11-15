from decimal import Decimal
from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import RowArray, is_gt_pattern


class RowArrayTests(TestCase):
    def test_preserves_even_invalid_rows_as_a_snapshot(self):
        rows = [[0], ["x", 3]]
        array = RowArray(rows)
        rows[1][0] = "y"
        self.assertEqual(array.rows, ((0,), ("x", 3)))
        self.assertFalse(is_gt_pattern(array))


class PatternTests(TestCase):
    def test_accepts_weak_interlacing_and_signed_entries(self):
        self.assertTrue(is_gt_pattern(RowArray([[2, 1, -1], [2, -1], [-1]])))
        self.assertTrue(is_gt_pattern([[-4]]))

    def test_rejects_malformed_rows(self):
        for rows in ([], [[2, 1]], [[2, 1], []], [[2, 1], 1], [[2, 1], [1, 1]]):
            with self.subTest(rows=rows):
                self.assertFalse(is_gt_pattern(rows))

    def test_rejects_violations_of_either_inequality(self):
        self.assertFalse(is_gt_pattern([[2, 0], [3]]))
        self.assertFalse(is_gt_pattern([[2, 0], [-1]]))

    def test_accepts_common_nonintegral_shifts(self):
        for shift in (Fraction(1, 3), Decimal("0.1"), 0.5, 0.5 + 2j):
            with self.subTest(shift=shift):
                self.assertTrue(is_gt_pattern([[shift + 2, shift], [shift + 1]]))

    def test_rejects_ordered_but_nonintegral_gaps(self):
        self.assertFalse(is_gt_pattern([[2, 0], [Fraction(1, 2)]]))
        self.assertFalse(is_gt_pattern([[2 + 1j, 1j], [1 + 2j]]))
        self.assertFalse(is_gt_pattern([[1.1, 0.1], [0.1]]))
        # Subtraction in binary floating point alone rounds this gap to 2**60.
        self.assertFalse(is_gt_pattern([[float(2**60), -0.5], [-0.5]]))

    def test_rejects_unsupported_or_nonfinite_scalars_at_rank_one(self):
        for entry in (True, "1", float("inf"), complex(1, float("nan")), Decimal("NaN")):
            with self.subTest(entry=entry):
                self.assertFalse(is_gt_pattern([[entry]]))

    def test_accepts_unbounded_integer_entries(self):
        value = 10**100
        self.assertTrue(is_gt_pattern([[value + 1, value], [value]]))
        self.assertFalse(is_gt_pattern([[value, 0], [Fraction(1, 2)]]))

    def test_integer_fast_path_preserves_iterable_and_scalar_fallback(self):
        class Integer(int):
            def __int__(self):
                return 1

        self.assertTrue(is_gt_pattern(iter((iter((2, 0)), iter((1,))))))
        self.assertTrue(is_gt_pattern(((2, 0), (Integer(3),))))

        def broken_row():
            raise RuntimeError("input iteration failed")
            yield 0

        with self.assertRaisesRegex(RuntimeError, "input iteration failed"):
            is_gt_pattern([[2, 1, 0], [5, 0], broken_row()])
