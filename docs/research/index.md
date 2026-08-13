# Important and current research track

**Snapshot date: 2026-08-12.** Read the stable [beginner spine](../spine/index.md) first. This track is chronological and argumentative: foundations establish the machinery, 2024–2025 systems show the reasoning-RL wave, and 2026 lessons test its limits and newest directions.

```text
policy-gradient foundations
  -> human preferences and process feedback
  -> open end-to-end post-training recipes
  -> reasoning RL and system stabilization
  -> capability-versus-sharpening debate
  -> generalization, curricula, re-solving, agents, and scientific rewards
```

Use the [interactive research map](../playgrounds/research-map.html) to compare papers by feedback source, online/offline learning, and claim strength.

| ID | Lesson | First publication | Status checked 2026-08-12 |
| --- | --- | --- | --- |
| R00 | [Read post-training research without being fooled](r00-read-research-without-being-fooled.md) | 2026-08-12 | course synthesis; not a research paper |
| R01 | [LoRA: learn a low-rank weight update instead of every weight](r01-lora.md) | 2021 | ICLR 2022 conference paper |
| R02 | [QLoRA: quantize the frozen base while training adapters](r02-qlora.md) | 2023 | NeurIPS 2023 paper |
| R03 | [From REINFORCE to RLOO: the simple policy-gradient line](r03-reinforce-and-rloo.md) | 1992; RLOO paper 2024 | REINFORCE: journal paper; RLOO: ACL 2024 paper |
| R04 | [PPO: the general-purpose optimizer that entered RLHF](r04-ppo.md) | 2017 | arXiv technical paper; widely used algorithm |
| R05 | [InstructGPT: demonstrations, preferences, reward model, and PPO](r05-instructgpt.md) | 2022 | NeurIPS 2022 paper |
| R06 | [Constitutional AI: principles, self-revision, and AI feedback](r06-constitutional-ai.md) | 2022 | arXiv research paper |
| R07 | [Let's Verify Step by Step: outcome versus process supervision](r07-process-supervision.md) | 2023 | arXiv research paper; PRM800K data released |
| R08 | [DPO: turn preference optimization into a classification loss](r08-dpo.md) | 2023 | NeurIPS 2023 paper |
| R09 | [KTO: learn from desirable and undesirable examples without pairs](r09-kto.md) | 2024 | ICML 2024 conference paper |
| R10 | [SimPO: reference-free preference optimization with a length-normalized reward](r10-simpo.md) | 2024 | NeurIPS 2024 conference paper |
| R11 | [DeepSeekMath: GRPO and verifiable mathematical rewards](r11-deepseekmath-and-grpo.md) | 2024 | arXiv technical report |
| R12 | [Tulu 3: an open post-training pipeline, not one magic loss](r12-tulu-3.md) | 2024 | arXiv technical report; open artifacts |
| R13 | [DeepSeek-R1: pure RL experiment versus the production recipe](r13-deepseek-r1.md) | 2025 | arXiv technical report; open model artifacts |
| R14 | [Kimi k1.5: long-context RL and long-to-short transfer](r14-kimi-k1-5.md) | 2025 | arXiv technical report |
| R15 | [DAPO: make large-scale reasoning RL trainable and inspectable](r15-dapo.md) | 2025 | arXiv technical report; code and data released |
| R16 | [Dr. GRPO: find length bias before celebrating longer reasoning](r16-dr-grpo.md) | 2025 | arXiv critical study; code released |
| R17 | [ProRL: does prolonged RL expand the reasoning boundary?](r17-prorl.md) | 2025 | arXiv research paper; weights released |
| R18 | [Breaking Barriers: RL gains often fail to cross domains](r18-cross-domain-generalization.md) | 2026 conference publication; first posted 2025 | ICLR 2026 conference paper |
| R19 | [Easy-to-hard curriculum RL: keep the policy in a learnable zone](r19-easy-to-hard-curriculum.md) | 2026 conference publication; first posted 2025 | ICLR 2026 conference paper |
| R20 | [Scalable Power Sampling: test distribution sharpening before training](r20-power-sampling.md) | 2026 | ICML 2026 conference paper |
| R21 | [Re²: teach the policy to abandon a bad reasoning path](r21-re2-resolving.md) | 2026-03-07 | arXiv preprint; frontier snapshot |
| R22 | [Agent² RL-Bench: can agents engineer their own post-training loop?](r22-agent2-rl-bench.md) | 2026-04-12 | arXiv preprint and Microsoft Research publication page; frontier snapshot |
| R23 | [RL with Verifiable Physics: replace binary reward with graded reality](r23-rl-with-verifiable-physics.md) | 2026-07-11 | arXiv preprint; latest dated frontier lesson in this snapshot |

## How selection works

A paper belongs here when it changed the conceptual or operational post-training stack, supplies an important negative result, or represents a current frontier direction with a clear evidence boundary. Citation count and benchmark rank alone are insufficient. Preprints remain labeled preprints.

The track is curated rather than exhaustive. Its machine-readable source record is [research-snapshot.json](../reference/research-snapshot.json).
