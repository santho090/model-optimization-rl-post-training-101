# Bridge from the toy model to real training

The canonical labs use no external dependencies. This bridge maps the concepts to a current real stack without claiming that a run occurred.

| Teaching object | Real stack counterpart |
| --- | --- |
| list of logits | transformer vocabulary logits |
| cross-entropy step | PyTorch autograd plus `SFTTrainer` or a custom loop |
| scalar Bradley-Terry scorer | sequence-classification reward model plus `RewardTrainer` |
| chosen/rejected log-probability gap | `DPOTrainer` |
| stored ratio and advantage | `PPOTrainer`-style online loop |
| group-normalized rewards | `GRPOTrainer` |
| JSON trace | tracker plus immutable checkpoints and per-example artifacts |

## Capability-gated first real run

Status: `specified-not-executed`.

1. Choose a small openly licensed causal model that fits the available accelerator. Record the exact immutable revision and license.
2. Create 32 training and 32 held-out records for one verifiable task. Deduplicate by problem family.
3. Run the held-out baseline with a fixed template, decoding configuration, and exact checker.
4. Run a tiny SFT job first. Save resolved configuration, logs, checkpoints, and environment.
5. Re-run the frozen evaluation and inspect every changed example.
6. Create preference pairs from candidate samples. Keep source policy and decoding IDs.
7. Compare continued SFT with DPO. Use online RL only if a trusted environment reward exists and the system can preserve rollout-policy identity.
8. Reject the candidate on any critical regression. Otherwise use a bounded, reversible canary.

Current API shapes can change. Before a real run, consult the exact installed-version documentation for [TRL](https://huggingface.co/docs/trl/main/index), [PyTorch](https://docs.pytorch.org/docs/stable/), and the chosen model.

## Stop conditions

- dataset license or consent is unresolved;
- train/test leakage is detected;
- baseline cannot be reproduced;
- loss becomes non-finite;
- checkpoint or rollout policy identity is ambiguous;
- reward rises while independent task success or a critical guardrail regresses;
- the hardware memory ledger has no headroom or cleanup plan.
