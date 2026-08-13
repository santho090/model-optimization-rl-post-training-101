from __future__ import annotations

import unittest

from pt101.pipeline import run_toy_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_is_deterministic_and_promotes(self) -> None:
        first = run_toy_pipeline()
        self.assertEqual(first, run_toy_pipeline())
        self.assertEqual(first["evidence"], "simulated")
        self.assertGreater(
            first["evaluation"]["preferred_probability_after"],
            first["evaluation"]["preferred_probability_before"],
        )
        self.assertTrue(first["promotion"]["promote"])


if __name__ == "__main__":
    unittest.main()
