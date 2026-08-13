from __future__ import annotations

from typing import Any

from .math_utils import softmax
from .stages import (
    dpo_step,
    grpo_advantages,
    kl_categorical,
    ppo_surrogate,
    reward_model_step,
    sft_step,
)


def run_toy_pipeline() -> dict[str, Any]:
    """Run a deterministic miniature post-training loop with explicit evidence boundaries."""
    base_logits = [0.0, 0.0]
    sft = sft_step(base_logits, target=1)
    rm_weight, rm_loss, pair_probability = reward_model_step(0.0, 1.0, 0.0)
    sft_gap = sft.after[1] - sft.after[0]
    dpo_gap, dpo_loss = dpo_step(sft_gap, reference_gap=0.0)
    ppo = ppo_surrogate(ratio=1.35, advantage=0.8)
    group_rewards = [0.0, 1.0, 1.0, 0.5]
    grpo = grpo_advantages(group_rewards)
    promoted_logits = [-dpo_gap / 2.0, dpo_gap / 2.0]
    quality_before = softmax(base_logits)[1]
    quality_after = softmax(promoted_logits)[1]
    kl = kl_categorical(promoted_logits, base_logits)
    promotion = quality_after > quality_before and kl < 0.2
    return {
        "evidence": "simulated",
        "boundary": "A two-action teaching model; not an LLM training result or benchmark.",
        "base": {"logits": base_logits, "preferred_probability": quality_before},
        "sft": {
            "loss": sft.loss,
            "logits": sft.after,
            "preferred_probability": softmax(sft.after)[1],
        },
        "reward_model": {
            "loss": rm_loss,
            "weight": rm_weight,
            "chosen_probability": pair_probability,
        },
        "dpo": {"loss": dpo_loss, "policy_gap": dpo_gap},
        "ppo": ppo,
        "grpo": {"rewards": group_rewards, "advantages": grpo},
        "evaluation": {
            "preferred_probability_before": quality_before,
            "preferred_probability_after": quality_after,
            "kl_from_base": kl,
        },
        "promotion": {
            "quality_improved": quality_after > quality_before,
            "kl_below_limit": kl < 0.2,
            "promote": promotion,
        },
    }
