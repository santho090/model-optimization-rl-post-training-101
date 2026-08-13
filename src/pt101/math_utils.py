from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def log_sigmoid(value: float) -> float:
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise ValueError("softmax requires at least one logit")
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    average = mean(values)
    return math.sqrt(mean([(value - average) ** 2 for value in values]))


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
