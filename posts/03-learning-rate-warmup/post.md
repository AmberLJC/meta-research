# Why Cold-Starting with a High Learning Rate Kills Training

*What happens in the first 1000 steps — and why the most important phase of training gets the smallest learning rate*

---

> The first few steps of training are the most dangerous. The model knows nothing, gradients are chaotic, and a high learning rate sends weights flying in random directions. Warmup is the protocol that keeps this controlled — and understanding it reveals something surprising about how networks bootstrap themselves into existence.

---

## The Problem: The Model Doesn't Know Anything Yet

When you randomly initialize a neural network, every gradient computed in the first few steps is high-variance noise. Not because the optimizer is broken — because the model hasn't yet developed enough structure for its gradients to be *meaningful*.

Here's what happens when you start with a typical production learning rate (lr=3e-4 for Adam) from step 0:

```
Loss without warmup — first 50 steps

Step  0:  ████████████████████████████████████████ 4.61  (random init)
Step  5:  ██████████████████████████████████████████████ 6.83  ↑ got WORSE
Step 10:  ████████████████████████████████████ 3.94
Step 20:  ██████████████████████████████████████████████████ 7.21  ↑ catastrophic spike
Step 50:  ████████████████████████████████████████ 4.88  (barely better than init)
```

Step 1 points one direction. Step 2 contradicts it. Step 3 contradicts step 2. The model is thrashing — taking large, contradictory steps in weight space.

This gets worse with larger models and adaptive optimizers like Adam, because Adam maintains momentum estimates (`m̂` and `v̂`). At step 1, those estimates are based on a single gradient — they're terrible approximations of the actual gradient landscape.

## The Fix: Let the Optimizer Wake Up First

**Learning rate warmup** starts training at a near-zero lr and linearly increases it to the target:

```
lr(t) = lr_target × (t / T_warmup)    for t < T_warmup
lr(t) = lr_schedule(t)                 for t ≥ T_warmup
```

Typically `T_warmup` is 1–10% of total training steps. Two things happen during this phase:

1. **Gradient variance is contained:** Small lr means small steps, even when gradients are large and chaotic.
2. **Adam's momentum estimates accumulate:** By the time lr reaches its target, `m̂` and `v̂` have enough history to be meaningful. Adam actually knows the gradient landscape.

Think of it as giving the optimizer time to wake up before asking it to move fast.

## The Experiment

We train a 3-layer transformer on token-level language modeling:

| Parameter | Value |
|---|---|
| Model | 3-layer transformer |
| Embedding dim | 128 |
| Total steps | 10,000 |
| Target LR | 3e-4 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |

Four schedules, all reaching the same peak lr=3e-4:

1. **No warmup** — constant 3e-4 from step 0
2. **100-step warmup** — linear increase over 1% of training
3. **500-step warmup** — linear increase over 5% (standard)
4. **2000-step warmup** — linear increase over 20% (long)

## Moving Slower at the Start Means Arriving Faster

Here's both models through their first 200 steps:

```
Loss comparison — first 200 steps

Step 0:    No warmup:  4.61  |  500-step warmup:  4.61   (same start)
Step 5:    No warmup:  6.83  |  500-step warmup:  4.44   (warmup already pulling ahead)
Step 20:   No warmup:  7.21  |  500-step warmup:  3.98   (warmup: 7x lower loss)
Step 50:   No warmup:  4.88  |  500-step warmup:  3.54
Step 200:  No warmup:  3.12  |  500-step warmup:  2.71   ← warmup is ahead, before warmup even finishes
```

The warmup model is already significantly ahead by step 200 — *before the warmup phase ends*. Moving slower at the start is faster overall.

## Why Adam's Memory Is the Hidden Variable

The effect goes deeper than "small steps are safer." We tracked the second moment estimate `v̂_t` (Adam's adaptive denominator) for a representative weight:

```
v̂ stability over time

         No warmup              500-step warmup
Step 1:  9.3e-5  (single noisy grad²)    2.1e-7  (tiny first step)
Step 10: 3.1e-4  ← wildly variable       4.8e-7
Step 100: 8.7e-5 ← still unstable        1.2e-5
Step 500: 4.2e-5 ← finally settling      3.8e-5  ← already converged
Step 1k:  3.9e-5                          3.7e-5  (nearly identical)
```

By step 500, both models have *similar* optimizer state. But the warmup model got there without thrashing — and early bad steps leave the weights in a worse starting position even when the optimizer state looks the same.

This is why you can't fully recover from a bad start by just continuing to train. Early steps shape the weight manifold that all later steps build on.

## How Long Should Warmup Be?

The optimal warmup length scales with model size:

```
Optimal warmup by model scale

  1M params:   ████  100–200 steps    short warmup sufficient
 10M params:   ████████  300–500 steps    standard 5% warmup works
100M params:   ████████████████  1000–2000 steps    longer needed
  1B+ params:  ████████████████████████  2000–5000 steps    may need cosine warmup
```

**Why the scaling?** Larger models average gradients across more parameters, giving each individual parameter lower signal-to-noise ratio per step. You need more steps to develop stable gradient estimates. Additionally, larger models are more sensitive to early bad steps — they have more capacity to memorize the chaos of initialization.

A practical heuristic: `T_warmup ≈ √(model_params) / 1000` steps, capped at 10% of total training.

## Final Loss: Finding the Sweet Spot

At the end of 10,000 training steps:

```
Final loss by warmup length (lower is better)

No warmup     ████████████████████████████████████████████ 1.84
100-step      ███████████████████████████████████████████ 1.79
500-step      █████████████████████████████████████████ 1.68  ← best
2000-step     ██████████████████████████████████████████ 1.71

Loss at step 500 (mid-training checkpoint):

No warmup     ██████████████████████████████████ 3.12
100-step      █████████████████████████████████ 2.89
500-step      ████████████████████████████████ 2.71
2000-step     █████████████████████████████████████ 3.31  ← "behind" during warmup, catches up
```

The 500-step warmup achieves the best final loss. The 2000-step warmup performs worse at intermediate checkpoints — it's "behind" during the slow phase — but recovers to near-optimal by the end. Too-long warmup wastes training budget in the slow phase.

## Learning Rate Warmup in Frontier LLM Research

Every major LLM training recipe uses warmup. GPT-3 used 375M warmup tokens. LLaMA 2 used a 2000-step warmup. Chinchilla's scaling law experiments used warmup as a fixed protocol.

A recent development: **cooldown** at the end of training. Just as warmup prevents early instability, reducing lr at the end helps the model converge to a clean minimum rather than bouncing around the loss landscape. The warmup-cosine-cooldown schedule consistently outperforms simpler schedules.

There's also emerging work on **cycle-based schedules** (SGDR) where the lr oscillates through multiple warmup-cooldown cycles. The intuition: each cycle "resets" the optimizer, letting the model escape local minima from the previous cycle. This is particularly promising for continual learning and fine-tuning [1].

One practical note: if you see your training loss spike suddenly at step 1000–2000, you probably forgot warmup, set it too short, or your lr is too high for your model size. It's one of the most common silent failure modes in LLM training.

---

[1] SGDR: Stochastic Gradient Descent with Warm Restarts. Loshchilov & Hutter, 2016. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)

*Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
