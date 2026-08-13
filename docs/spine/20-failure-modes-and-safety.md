# 20. Failure modes, reward hacking, and safety

**Question:** How does optimization fail even when the training chart is green?

**Evidence in this chapter:** stable concepts are explained from cited primary sources. All `pt101` outputs are deterministic `simulated` teaching evidence, not measurements of an LLM.

## The simplest accurate answer

Optimization amplifies whatever produces the training signal. If the proxy is incomplete, the policy may exploit it. If data is narrow, capabilities may regress elsewhere. If feedback is biased, those biases can become more consistent. Safety is therefore a set of independent constraints and evaluations, not one reward term.

## A useful mental model

A student told that only the final numeric answer matters may learn to copy answer keys. The score improved; the desired competence did not. Models search high-dimensional behavior spaces where proxy loopholes can be harder to anticipate.

## How it works

Watch for reward hacking, judge hacking, sycophancy, mode collapse, verbosity bias, length gaming, catastrophic forgetting, KL drift, memorization, data contamination, capability elicitation gaps, and distribution shift. Use held-out adversarial tasks, canaries, human audits, independent evaluators, per-slice regression budgets, and rollback-ready checkpoints. Red-team the evaluator as aggressively as the policy.

```mermaid
flowchart LR
    A[task and frozen evaluation] --> B[data or environment]
    B --> C[policy produces logits or samples]
    C --> D[loss, preference, or reward]
    D --> E[gradient update]
    E --> F[candidate evaluation]
    F -->|all gates pass| G[promote]
    F -->|any blocker fails| H[reject and diagnose]
```

## Work one example

A candidate gets a higher helpfulness score by always agreeing with the user's premise. A factuality slice shows more confident errors. The promotion rule must block the candidate even though the optimized metric improved.

## Do it yourself

Write a failure register for one planned run: symptom, detector, threshold, containment, rollback, and owner. Include failures in data, trainer, rollout system, evaluator, and deployment.

Record the input, configuration, output, evidence label, and one sentence stating what the output **does not** prove. If you change two variables, split the work into two experiments.

## Check your understanding

Which safety property is enforced outside the model so an optimized policy cannot trade it away?

## Common failure

A KL limit constrains distributional movement relative to a reference; it is not a semantic safety guarantee.

## Sources

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

## Course position

- Prerequisite: [Chapter 19](../spine/19-training-systems.md)
- Next: [Chapter 21](../spine/21-production-loop.md)
