# End-to-end capstone packet

Goal: improve one narrow, exactly scorable behavior while preserving three guardrails. Produce evidence another engineer can audit.

## Phase 1: contract and baseline

- State the behavior in input/output terms.
- Freeze at least 32 held-out items by problem family.
- Pin prompt template, decoding, evaluator, and baseline model.
- Run the baseline and inspect every failure.

Stop if the evaluator accepts incorrect answers or the baseline cannot be reproduced.

## Phase 2: data

- Create demonstrations without copying held-out problem families.
- Preserve raw source, transform, and provenance.
- Validate schemas and duplicates.
- Generate preference pairs only after recording source policy and decoding.

Stop on unresolved rights, consent, contamination, or leakage.

## Phase 3: candidates

1. Train SFT as the required baseline.
2. Choose continued SFT, DPO, or online RL using the [algorithm chooser](../reference/algorithm-chooser.md).
3. Save every resolved configuration, log, and checkpoint.
4. For online RL, prove rollout weights and stored old log-probabilities share one identity.

Stop on non-finite loss, runaway KL, corrupt artifacts, or critical online-evaluation regression.

## Phase 4: independent evaluation

- Run baseline and candidate with identical frozen settings.
- Report per-example transitions: pass→pass, fail→pass, pass→fail, fail→fail.
- Audit safety, formatting, general capability, latency, and cost.
- Red-team the reward or judge separately.

## Phase 5: decision

The promotion record must bind all artifacts and say `promote` or `reject`. A higher optimized reward is insufficient. Any hard guardrail failure rejects the candidate.

## Required deliverables

- experiment record;
- demonstration and preference dataset cards;
- training manifest and environment lock;
- raw per-example baseline and candidate outputs;
- evaluator calibration and adversarial cases;
- failure register;
- model card and rollback plan;
- final promotion decision.
