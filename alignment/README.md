# DPO toy demo

A small, from-scratch implementation of Direct Preference Optimization
(Rafailov et al., 2023) on a tiny word-level LSTM language model, trained
on ~200 synthetic preference pairs.

## What DPO actually does

Instead of the RLHF pipeline of training a separate reward model and then
running PPO against it, DPO reframes preference learning as a direct loss
on the policy model itself:

```
L_DPO = -log(sigmoid(beta * ((logp_policy(chosen) - logp_ref(chosen))
                            - (logp_policy(rejected) - logp_ref(rejected)))))
```

- `logp_policy(y)` — the sequence log-likelihood the policy model assigns
  the response `y` (teacher-forced sum of log-softmax token probabilities).
- `logp_ref(y)` — the same quantity from a **frozen** copy of the model
  taken *before* any DPO updates. This is what keeps the policy from
  drifting arbitrarily far from its starting behavior — without it, the
  loss can be driven to zero by making the policy maximally confident
  about anything, not specifically about preferring chosen over rejected.
- `beta` — a temperature controlling how strongly the loss penalizes
  violating the preference; higher beta = sharper preference, more
  aggressive updates.

The whole point of DPO relative to classic RLHF is that this needs no
reward model and no RL rollout loop — it's a single supervised-style loss
computed directly from two log-probabilities.

## What's simplified here, and why

- **Toy scale**: a word-level LSTM LM with a small vocabulary, trained on
  ~200 synthetic preference pairs of financial-headline summaries (chosen
  = concise/neutral, rejected = verbose/sensational). This is enough to
  demonstrate the mechanism (the loss goes down, the policy's preference
  margin over the reference grows) without needing a pretrained LLM or
  GPU-hours of compute.
- **No SFT stage**: real DPO pipelines start from a supervised
  fine-tuned model. Here the "policy" starts from random initialization
  and the reference is just a frozen copy of that same random init — the
  point is to show the DPO loss and reference-model mechanics work
  correctly, not to reproduce a full RLHF pipeline.
- **Sequence-level log-likelihood, not sampling**: this script scores
  fixed (prompt, chosen, rejected) triples; it doesn't include the
  generation/sampling loop you'd use to actually produce candidate
  responses from the policy during training.

## Running it

```bash
python alignment/dpo_toy_demo.py
```

Prints the DPO loss decreasing over training steps and the average
preference margin (`logp(chosen) - logp(rejected)`, relative to the
reference model) before and after training — the metric that shows the
policy actually learned to prefer the chosen responses more than its
frozen starting point did.
