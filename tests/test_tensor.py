from decimal import Decimal
from fractions import Fraction
from unittest import TestCase

from gelfand_tsetlin import quadratic_casimir_eigenvalue, tensor_with_standard


class HighestWeightTests(TestCase):
    def test_quadratic_casimir_eigenvalue(self):
        self.assertEqual(quadratic_casimir_eigenvalue((4, 2, 0)), 28)
        self.assertEqual(quadratic_casimir_eigenvalue((1, 0, 0)), 3)
        self.assertEqual(quadratic_casimir_eigenvalue((0, 0, -1)), 3)
        self.assertEqual(
            quadratic_casimir_eigenvalue(iter(map(Decimal, ("1.5", "0.5", "-0.5")))),
            Fraction(27, 4),
        )
        self.assertEqual(quadratic_casimir_eigenvalue((10**100,)), 10**200)

    def test_pieri_chooses_the_ends_of_equal_weight_blocks(self):
        weight = (2, 2, 0, 0)
        self.assertEqual(tensor_with_standard(weight), ((3, 2, 0, 0), (2, 2, 1, 0)))
        self.assertEqual(tensor_with_standard(weight, dual=True), ((2, 1, 0, 0), (2, 2, 0, -1)))
        half = Fraction(1, 2)
        self.assertEqual(tensor_with_standard((half,), dual=True), ((-half,),))
        self.assertEqual(
            tensor_with_standard((half, half - 1)),
            ((half + 1, half - 1), (half, half)),
        )

    def test_casimir_differences_on_tensor_summands(self):
        weight = (4, 2, 0)
        rank = len(weight)
        value = quadratic_casimir_eigenvalue(weight)
        standard_summands = tensor_with_standard(weight)
        dual_summands = tensor_with_standard(weight, dual=True)
        for i, (standard, dual) in enumerate(zip(standard_summands, dual_summands), 1):
            self.assertEqual(
                quadratic_casimir_eigenvalue(standard) - value - rank,
                2 * (weight[i - 1] - i + 1),
            )
            self.assertEqual(
                value + rank - quadratic_casimir_eigenvalue(dual),
                2 * (weight[i - 1] + rank - i),
            )

    def test_invalid_highest_weights(self):
        invalid_weights = ((), (0, 1), (1, Fraction(1, 2)), (1j,), (float("inf"),), (True,))
        for weight in invalid_weights:
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    quadratic_casimir_eigenvalue(weight)
                with self.assertRaises(ValueError):
                    tensor_with_standard(weight, dual=True)
