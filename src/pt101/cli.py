from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .pipeline import run_toy_pipeline
from .stages import dpo_step, grpo_advantages, ppo_surrogate, reward_model_step, sft_step


def _write(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pt101", description="CPU-only post-training teaching labs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("pipeline", "sft", "reward-model", "dpo", "ppo", "grpo"):
        child = subparsers.add_parser(name)
        child.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pipeline":
        payload = run_toy_pipeline()
    elif args.command == "sft":
        result = sft_step([0.0, 0.0], target=1)
        payload = {
            "evidence": "simulated",
            "loss": result.loss,
            "before": result.before,
            "after": result.after,
        }
    elif args.command == "reward-model":
        weight, loss, probability = reward_model_step(0.0, 1.0, 0.0)
        payload = {
            "evidence": "simulated",
            "loss": loss,
            "weight": weight,
            "chosen_probability": probability,
        }
    elif args.command == "dpo":
        gap, loss = dpo_step(0.4, 0.0)
        payload = {"evidence": "simulated", "loss": loss, "updated_policy_gap": gap}
    elif args.command == "ppo":
        payload = {"evidence": "simulated", **ppo_surrogate(1.35, 0.8)}
    else:
        rewards = [0.0, 1.0, 1.0, 0.5]
        payload = {
            "evidence": "simulated",
            "rewards": rewards,
            "advantages": grpo_advantages(rewards),
        }
    _write(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
