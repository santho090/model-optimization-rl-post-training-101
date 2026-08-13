from __future__ import annotations

import unittest

from pt101.math_utils import mean, softmax
from pt101.stages import dpo_step, grpo_advantages, ppo_surrogate, reward_model_step, sft_step


class StageTests(unittest.TestCase):
    def test_sft_increases_target_probability(self) -> None:
        result = sft_step([0.0, 0.0], target=1)
        self.assertGreater(softmax(result.after)[1], softmax(result.before)[1])
        self.assertGreater(result.loss, 0.0)

    def test_reward_model_increases_chosen_gap(self) -> None:
        weight, loss, probability = reward_model_step(0.0, 1.0, 0.0)
        self.assertGreater(weight, 0.0)
        self.assertGreater(loss, 0.0)
        self.assertEqual(probability, 0.5)

    def test_dpo_increases_preference_gap(self) -> None:
        updated, loss = dpo_step(0.0, 0.0)
        self.assertGreater(updated, 0.0)
        self.assertGreater(loss, 0.0)

    def test_ppo_clips_large_positive_ratio(self) -> None:
        result = ppo_surrogate(1.35, 0.8)
        self.assertTrue(result["was_clipped"])
        self.assertAlmostEqual(float(result["clipped_ratio"]), 1.2)
        self.assertLess(float(result["objective"]), float(result["unclipped_objective"]))

    def test_grpo_advantages_are_centered(self) -> None:
        advantages = grpo_advantages([0.0, 1.0, 1.0, 0.5])
        self.assertAlmostEqual(mean(advantages), 0.0)

    def test_grpo_all_equal_has_no_signal(self) -> None:
        self.assertEqual(grpo_advantages([1.0, 1.0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
