from decimal import Decimal
from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import (
    EnvelopingElement,
    RowArray,
    apply_enveloping,
    lowering_operator,
    raising_operator,
)


def matrix_unit(i, j):
    return EnvelopingElement.matrix_unit(3, i, j)


class MolevOperatorTests(TestCase):
    def test_gl3_operators_match_molevs_examples(self):
        h1, h2 = matrix_unit(1, 1), matrix_unit(2, 2) - 1
        self.assertEqual(raising_operator(3, 1), matrix_unit(1, 3))
        self.assertEqual(raising_operator(3, 2), matrix_unit(2, 3) * (h2 - h1)
                         + matrix_unit(2, 1) * matrix_unit(1, 3))
        self.assertEqual(lowering_operator(3, 2), matrix_unit(3, 2))
        self.assertEqual(lowering_operator(3, 1), matrix_unit(3, 1) * (h1 - h2)
                         + matrix_unit(2, 1) * matrix_unit(3, 2))

    def test_operators_from_lower_levels_embed_in_ambient_rank(self):
        self.assertEqual(raising_operator(3, 1, level=2), matrix_unit(1, 2))
        self.assertEqual(lowering_operator(3, 1, level=2), matrix_unit(2, 1))
        for constructor in (raising_operator, lowering_operator):
            for index, level in ((0, 3), (2, 2), (1, 4), (1, 1)):
                with self.subTest(constructor=constructor, index=index, level=level):
                    with self.assertRaises(ValueError):
                        constructor(3, index, level=level)

    def test_raising_on_a_gl2_highest_line(self):
        e3 = RowArray([[1, 0, 0], [0, 0], [0]])
        e1 = RowArray([[1, 0, 0], [1, 0], [1]])
        self.assertEqual(apply_enveloping(raising_operator(3, 1), e3), {e1: 2})
        # The two summands in z_23 cancel, since (0,1) is not a gl_2 highest weight.
        self.assertEqual(apply_enveloping(raising_operator(3, 2), e3), {})

    def test_lowering_matches_molevs_normalization_of_the_gl3_standard_module(self):
        highest = RowArray([[1, 0, 0], [1, 0], [1]])
        lowered = RowArray([[1, 0, 0], [0, 0], [0]])
        self.assertEqual(apply_enveloping(lowering_operator(3, 1), highest), {lowered: 1})


class EnvelopingActionTests(TestCase):
    def test_adjacent_column_emits_all_admissible_moves(self):
        top = (4, 2, 0)
        source = RowArray([top, (3, 1), (2,)])
        self.assertEqual(
            apply_enveloping(matrix_unit(2, 3), source),
            {RowArray([top, (4, 1), (2,)]): Fraction(10, 3),
             RowArray([top, (3, 2), (2,)]): Fraction(8, 3)},
        )
        self.assertEqual(
            apply_enveloping(matrix_unit(3, 2), source),
            {RowArray([top, (2, 1), (2,)]): Fraction(1, 3),
             RowArray([top, (3, 0), (2,)]): Fraction(2, 3)},
        )

    def test_adjacent_column_preserves_absent_support_beyond_native_range(self):
        top = (1 << 64, 0)
        source = RowArray([top, (top[0],)])
        self.assertEqual(apply_enveloping(EnvelopingElement.matrix_unit(2, 1, 2), source), {})
        with self.assertRaises(OverflowError):
            apply_enveloping(EnvelopingElement.matrix_unit(2, 2, 1), source)

    def test_products_act_in_operator_order(self):
        source = RowArray([[1, 0, 0], [1, 0], [1]])
        e12, e21 = matrix_unit(1, 2), matrix_unit(2, 1)
        self.assertEqual(apply_enveloping(e12 * e21, source), {source: 1})
        self.assertEqual(apply_enveloping(e21 * e12, source), {})

    def test_scalar_action_is_exact_and_zero_has_empty_support(self):
        source = RowArray([[1, 0, 0], [1, 0], [1]])
        value = Fraction(-3, 7)
        self.assertEqual(apply_enveloping(EnvelopingElement.scalar(3, value), source),
                         {source: value})
        self.assertEqual(apply_enveloping(EnvelopingElement.scalar(3, 0), source), {})

    def test_real_shift_is_exact_and_complex_shift_is_rejected(self):
        low, high = Decimal("0.1"), Decimal("1.1")
        source = RowArray([[high, low, low], [high, low], [high]])
        target = RowArray([[high, low, low], [high, low], [low]])
        self.assertEqual(apply_enveloping(matrix_unit(1, 1), source),
                         {source: Fraction(11, 10)})
        self.assertEqual(apply_enveloping(matrix_unit(2, 1), source), {target: 1})
        complex_source = RowArray([[1 + 2j, 2j, 2j], [1 + 2j, 2j], [1 + 2j]])
        with self.assertRaises(ValueError):
            apply_enveloping(matrix_unit(1, 1), complex_source)
