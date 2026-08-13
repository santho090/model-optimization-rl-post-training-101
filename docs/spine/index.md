# The canonical beginner path

Read these chapters in order. Each chapter introduces only the machinery needed by the next one. The course uses one continuous teaching loop so that SFT, reward modeling, DPO, PPO, and GRPO are not isolated vocabulary.

```text
task contract
  -> frozen baseline evaluation
  -> demonstrations and SFT
  -> preferences and reward models
  -> choose offline preference optimization or online RL
  -> distributed execution with artifact identities
  -> independent gates
  -> staged promotion or rejection
```

| # | Chapter | Guiding question |
| ---: | --- | --- |
| 00 | [Map the post-training stack](00-map-the-stack.md) | What are we changing, and why does a pretrained model need another training stack? |
| 01 | [Numbers, probability, and sampling](01-numbers-probability-and-sampling.md) | How does a model turn scores into a choice? |
| 02 | [Vectors, matrices, and neural networks](02-vectors-matrices-and-neural-networks.md) | What are the objects inside a model before they become probabilities? |
| 03 | [Parameters, forward passes, losses, and gradients](03-parameters-forward-loss-gradient.md) | What physically changes when a model learns? |
| 04 | [A language model from tokens to loss](04-language-model-from-tokens-to-loss.md) | Where do post-training losses attach to a transformer? |
| 05 | [Objectives, baselines, and experiments](05-objectives-and-experiments.md) | How do we know an update caused a useful improvement? |
| 06 | [Build the data pipeline](06-data-pipeline.md) | How do raw interactions become trustworthy training records? |
| 07 | [Build the evaluation harness first](07-evaluation-harness.md) | What must be measured before any training run? |
| 08 | [Supervised fine-tuning](08-supervised-fine-tuning.md) | How does imitation turn examples into a usable instruction model? |
| 09 | [LoRA, adapters, and training memory](09-lora-and-memory.md) | How can we update a large model without training every weight? |
| 10 | [Preference data and feedback](10-preference-data.md) | How do we turn human or AI judgments into usable comparisons? |
| 11 | [Reward models](11-reward-models.md) | How can comparisons train a scalar scorer? |
| 12 | [RL from bandits to Markov decision processes](12-rl-from-bandits-to-mdps.md) | What makes reinforcement learning different from supervised learning? |
| 13 | [Policy gradients and REINFORCE](13-policy-gradients.md) | How can a non-differentiable reward change differentiable model weights? |
| 14 | [PPO and KL control](14-ppo-and-kl-control.md) | Why constrain how far the policy moves? |
| 15 | [Direct Preference Optimization](15-dpo.md) | Can we optimize preferences without an explicit reward-model-and-PPO loop? |
| 16 | [Verifiable rewards, human feedback, and AI feedback](16-verifiable-rewards-and-rlaif.md) | Where should rewards come from? |
| 17 | [GRPO and group-relative advantages](17-grpo.md) | How can a group of completions provide a baseline without a separate critic? |
| 18 | [Agents, tools, and environments](18-agents-tools-and-environments.md) | What changes when the model acts over multiple steps? |
| 19 | [The training system: memory, parallelism, and rollouts](19-training-systems.md) | How does an algorithm become a reliable distributed job? |
| 20 | [Failure modes, reward hacking, and safety](20-failure-modes-and-safety.md) | How does optimization fail even when the training chart is green? |
| 21 | [Promotion, deployment, monitoring, and iteration](21-production-loop.md) | When is a trained checkpoint ready to serve? |
| 22 | [Capstone: optimize one small model end to end](22-capstone.md) | Can you operate the entire stack without confusing a proxy, a simulation, and a measured result? |

The first run needs Python 3.12 and no GPU, external model, or network after installation. Read [the evidence contract](../reference/evidence.md) before interpreting output.
