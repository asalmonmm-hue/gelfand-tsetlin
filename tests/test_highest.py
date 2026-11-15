from fractions import Fraction
from itertools import product
from unittest import TestCase

from gelfand_tsetlin import (
    RowArray,
    apply_enveloping,
    apply_highest_lowering,
    apply_highest_raising,
    lowering_operator,
    raising_operator,
)


class HighestLineTests(TestCase):
    def assert_matches_enveloping_action(self, source, i, level=None):
        rank = len(source.rows)
        for direct, symbolic in (
            (apply_highest_raising, raising_operator),
            (apply_highest_lowering, lowering_operator),
        ):
            with self.subTest(source=source, i=i, level=level, operator=direct.__name__):
                self.assertEqual(
                    direct(source, i, level=level),
                    apply_enveloping(symbolic(rank, i, level=level), source),
                )

    def test_agrees_with_enveloping_action_on_complete_highest_lines(self):
        for top in ((2, -1), (4, 2, 0), (3, 2, 1, 0)):
            intervals = (range(lo, hi + 1) for hi, lo in zip(top, top[1:]))
            for middle in product(*intervals):
                source = RowArray([top, *(middle[:k] for k in range(len(middle), 0, -1))])
                for i in range(1, len(top)):
                    self.assert_matches_enveloping_action(source, i)

    def test_embedded_level_changes_exactly_the_relevant_prefixes(self):
        top, level_row = (7, 5, 2, 0), (5, 3, 1)
        source = RowArray([top, level_row, (4, 2), (4,)])
        self.assert_matches_enveloping_action(source, 1, level=3)
        self.assert_matches_enveloping_action(source, 2, level=3)
        self.assertEqual(
            apply_highest_lowering(source, 1, level=3),
            {RowArray([top, level_row, (3, 2), (3,)]): 1},
        )
        self.assertEqual(
            apply_highest_lowering(source, 2, level=3),
            {RowArray([top, level_row, (4, 1), (4,)]): 1},
        )
        self.assert_matches_enveloping_action(source, 1, level=2)

    def test_real_rational_shift_and_iterable_source(self):
        shift = Fraction(1, 3)
        rows = tuple(tuple(x + shift for x in row) for row in ((4, 2, 0), (3, 1), (3,)))
        source = RowArray(rows)
        self.assert_matches_enveloping_action(source, 1)
        self.assertEqual(apply_highest_raising(iter(rows), 1), apply_highest_raising(source, 1))

    def test_exact_coefficient_without_fixed_width_limit(self):
        large = 1 << 100
        top = (large, 0, -large)
        source = RowArray([top, (0, 0), (0,)])
        target = RowArray([top, (1, 0), (1,)])
        self.assertEqual(apply_highest_raising(source, 1), {target: large * (large + 2)})
        self.assertEqual(apply_highest_lowering(source, 1), {})
        self.assertEqual(apply_highest_lowering(source, 2), {RowArray([top, (0, -1), (0,)]): 1})

    def test_rejects_invalid_patterns_levels_and_nonhighest_vectors(self):
        source = RowArray([(4, 2, 0), (3, 1), (3,)])
        for direct in (apply_highest_raising, apply_highest_lowering):
            for i, level in ((0, None), (3, None), (2, 2), (1, 4), (1, 1)):
                with self.subTest(operator=direct.__name__, i=i, level=level):
                    with self.assertRaises(ValueError):
                        direct(source, i, level=level)
            for bad_rows in (
                [(4, 2, 0), (3, 1), (2,)],
                [(4, 2, 0), (3, 1), (4,)],
                [(4, 2, 0), (3, 1)],
                [(4 + 1j, 2 + 1j, 1j), (3 + 1j, 1 + 1j), (3 + 1j,)],
            ):
                with self.subTest(operator=direct.__name__, bad_rows=bad_rows):
                    with self.assertRaises(ValueError):
                        direct(bad_rows, 1)

    def test_commuting_and_same_index_relations_on_highest_lines(self):
        top = (4, 2, 0)
        source = RowArray([top, (3, 1), (3,)])

        def compose(left, i, right, j):
            ((intermediate, scale),) = right(source, j).items()
            return {target: scale * value for target, value in left(intermediate, i).items()}

        lower, raise_ = apply_highest_lowering, apply_highest_raising
        self.assertEqual(compose(lower, 1, lower, 2), compose(lower, 2, lower, 1))
        self.assertEqual(compose(raise_, 1, lower, 2), compose(lower, 2, raise_, 1))
        # The two same-index compositions evaluate the shifted Capelli product.
        self.assertEqual(compose(raise_, 1, lower, 1), {source: 8})
        self.assertEqual(compose(lower, 1, raise_, 1), {source: 10})
