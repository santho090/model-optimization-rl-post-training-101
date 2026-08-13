# Choose the smallest method that fits the feedback

| Available feedback | Start with | Why | Main limitation |
| --- | --- | --- | --- |
| high-quality target responses | SFT | direct, stable imitation baseline | bounded by demonstrations |
| fixed chosen/rejected pairs | DPO after SFT | no online rollout or separate critic loop | cannot discover uncovered failures |
| reusable preference scorer plus online samples | reward model + PPO | separates judging from policy updates | complex, proxy exploitation and policy lag |
| exact outcome reward with multiple samples per prompt | GRPO-style online RL | group baseline avoids a learned critic in the stated design | rollout cost and zero-signal groups |
| multi-step tool outcome | environment-based online RL | learns from consequences | credit assignment, sandbox, nondeterminism |

Decision flow:

```text
Do demonstrations solve the task?
  yes -> establish SFT baseline
  no  -> fix task/data/evaluation before choosing RL

Do you only have a frozen preference dataset?
  yes -> compare DPO with continued SFT

Can the policy interact with a trusted reward source?
  yes -> consider online RL
        exact/verifiable groups -> GRPO-style route
        learned reward + value baseline -> PPO-style route
```

Run every method against the same frozen task contract. Algorithm choice is a hypothesis, not a status symbol.
