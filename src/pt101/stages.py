from __future__ import annotations

import math
from dataclasses import dataclass

from .math_utils import clip, log_sigmoid, mean, population_std, sigmoid, softmax


@dataclass(frozen=True)
class StepResult:
    algorithm: str
    before: list[float]
    after: list[float]
    loss: float
    note: str


def sft_step(logits: list[float], target: int, learning_rate: float = 0.4) -> StepResult:
    """One cross-entropy gradient step on a categorical toy policy."""
    probabilities = softmax(logits)
    loss = -math.log(probabilities[target])
    gradients = probabilities[:]
    gradients[target] -= 1.0
    updated = [
        value - learning_rate * gradient for value, gradient in zip(logits, gradients, strict=True)
    ]
    return StepResult(
        "sft", logits, updated, loss, "increase the demonstrated action's log-probability"
    )


def reward_model_step(
    weight: float, chosen_feature: float, rejected_feature: float, learning_rate: float = 0.2
) -> tuple[float, float, float]:
    """Fit a one-weight Bradley-Terry preference model for one pair."""
    feature_gap = chosen_feature - rejected_feature
    reward_gap = weight * feature_gap
    loss = -log_sigmoid(reward_gap)
    gradient = (sigmoid(reward_gap) - 1.0) * feature_gap
    updated = weight - learning_rate * gradient
    return updated, loss, sigmoid(reward_gap)


def dpo_step(
    policy_gap: float,
    reference_gap: float,
    beta: float = 0.1,
    learning_rate: float = 1.0,
) -> tuple[float, float]:
    """One DPO step where gaps are chosen-minus-rejected log-probabilities."""
    margin = beta * (policy_gap - reference_gap)
    loss = -log_sigmoid(margin)
    gradient = beta * (sigmoid(margin) - 1.0)
    return policy_gap - learning_rate * gradient, loss


def reinforce_step(
    logits: list[float],
    action: int,
    reward: float,
    baseline: float = 0.0,
    learning_rate: float = 0.2,
) -> StepResult:
    """One REINFORCE step using a scalar baseline."""
    probabilities = softmax(logits)
    advantage = reward - baseline
    gradients = probabilities[:]
    gradients[action] -= 1.0
    updated = [
        value - learning_rate * advantage * gradient
        for value, gradient in zip(logits, gradients, strict=True)
    ]
    loss = -advantage * math.log(probabilities[action])
    return StepResult("reinforce", logits, updated, loss, "reward-weighted log-probability update")


def ppo_surrogate(ratio: float, advantage: float, epsilon: float = 0.2) -> dict[str, float | bool]:
    """Return the unclipped and clipped PPO surrogate for one sampled action."""
    unclipped = ratio * advantage
    clipped_ratio = clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped_value = clipped_ratio * advantage
    objective = min(unclipped, clipped_value)
    return {
        "ratio": ratio,
        "advantage": advantage,
        "clipped_ratio": clipped_ratio,
        "unclipped_objective": unclipped,
        "clipped_objective": clipped_value,
        "objective": objective,
        "was_clipped": not math.isclose(ratio, clipped_ratio),
    }


def grpo_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    """Normalize rewards within one prompt's sampled completion group."""
    standard_deviation = population_std(rewards)
    if standard_deviation < epsilon:
        return [0.0 for _ in rewards]
    average = mean(rewards)
    return [(reward - average) / (standard_deviation + epsilon) for reward in rewards]


def kl_categorical(policy_logits: list[float], reference_logits: list[float]) -> float:
    policy = softmax(policy_logits)
    reference = softmax(reference_logits)
    return sum(p * math.log(p / q) for p, q in zip(policy, reference, strict=True))
