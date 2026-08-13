# Evidence contract

This repository separates what a result **is** from what we wish it implied.

| Label | Meaning | Example | It does not prove |
| --- | --- | --- | --- |
| `derived` | Arithmetic from stated inputs | LoRA parameter count | runtime or quality |
| `simulated` | Deterministic teaching-model output | PPO clipping trace | real model behavior |
| `measured` | Observation from a named run | held-out pass rate with artifacts | universal generalization |
| `reported` | External source's result | a paper's benchmark table | local reproduction |
| `specified-not-executed` | Procedure exists but was not run | GPU SFT manifest | any training outcome |

A `measured` record names:

1. base and candidate model identities;
2. tokenizer and chat template;
3. dataset snapshots and split policy;
4. training code and full configuration;
5. hardware and software environment;
6. evaluator, decoding, seed, and per-example artifacts;
7. failures, exclusions, and stop conditions.

The CPU labs prove equations and invariants only. Upgrade their evidence label only after replacing the teaching object with an observed system and preserving the required record.
