from __future__ import annotations

import math
import unittest

from pt101.math_utils import log_sigmoid, softmax


class MathTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self) -> None:
        probabilities = softmax([1000.0, 1000.0])
        self.assertEqual(probabilities, [0.5, 0.5])
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_log_sigmoid_at_zero(self) -> None:
        self.assertAlmostEqual(log_sigmoid(0.0), -math.log(2.0))

    def test_softmax_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            softmax([])


if __name__ == "__main__":
    unittest.main()
