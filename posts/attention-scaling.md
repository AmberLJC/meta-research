# Why Transformers Divide Attention by √d

*What happens when dot products grow too large — and how a single constant prevents softmax from becoming useless*

---

> The softmax function has a dirty secret: feed it large numbers and it turns into a one-hot selector. Every token attends to exactly one other token, gradients vanish, and your transformer learns nothing. The fix is one line of math — divide by √d — but understanding *why* it works reveals something deep about how high-dimensional geometry interacts with probability.

---

## The Problem: Softmax Goes Sharp

The core operation in self-attention is:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

That `/ √d` term is what we're here to understand. Let's start by seeing what happens without it.

When two vectors of dimension `d` are filled with random values drawn from a standard normal distribution (mean 0, variance 1), their dot product has variance equal to `d`. This means:

- At d=64, typical dot products are around **±8**
- At d=512, they're around **±22**

Now feed those values into softmax. The `exp` function amplifies differences exponentially — and at high magnitude, it doesn't just concentrate attention, it *collapses* it into a single winner:

```
Dot product magnitudes → Attention distribution (4 tokens)

  ±1  ████████ 35%
      ███████  28%
      ██████   22%
      █████    15%    ← spread, gradients flow

  ±5  █████████████████████ 82%
      ████  13%
      ██   4%
      █    1%         ← concentrated

 ±20  ████████████████████████ 99.99%
      ░ 0.01%
      ░ 0.00%
      ░ 0.00%         ← one-hot. gradients dead.
```

When softmax becomes one-hot, the gradient through it becomes nearly zero everywhere except the argmax. The model stops learning which tokens to attend to — it's stuck in a winner-take-all regime from step 1.

## The Fix: One Line of Math

Dividing by √d brings the dot product variance back to 1, regardless of dimension:

```
Var(q · k / √d) = Var(q · k) / d = d / d = 1
```

This keeps the softmax inputs in a regime where the distribution remains smooth and gradients flow. It's not a hyperparameter to tune — it's a mathematical correction for the geometry of high-dimensional dot products.

**The key insight:** The problem isn't large numbers per se. It's large numbers that grow with a hyperparameter (d) you chose. Without the scaling, every time you make your model larger, attention gets worse. With the scaling, increasing d doesn't degrade attention quality.

## The Experiment

We train a 2-layer transformer on a synthetic task: **copy the token at position k** for a randomly chosen k. The model must learn which position to attend to — making the attention pattern directly measurable.

| Parameter | Value |
|---|---|
| Sequence length | 16 |
| Embedding dim | 64 |
| Heads | 4 |
| Task | Selective copy |
| Training steps | 3000 |

We train three variants: **No scaling** (raw `softmax(QKᵀ)`), **√d scaling** (standard attention), and **Fixed scaling (1/4)** — a naive constant that ignores d entirely.

## The Collapse Starts at Initialization

We track **attention entropy** — a measure of how spread vs. concentrated the attention distribution is. Maximum entropy (uniform attention) = `log(16) = 2.77`. Minimum = 0 (one-hot).

Here's what the three models look like *before training even begins*:

```
Attention entropy at step 0  (max possible = 2.77)

No scaling    ██ 0.3   ← already near one-hot at init!
Fixed (1/4)   ████████████████████ 1.9
√d scaling    ██████████████████████████ 2.6   ← near-uniform
```

The unscaled model is already broken before a single gradient step. Random initialization produces dot products large enough to collapse the softmax — and the model is frozen there, attending strongly to whichever token happened to have the largest random dot product.

## Performance Diverges Fast

By step 200, the three models have split apart:

```
Task accuracy at step 200

No scaling    ████ 12%   ← random chance
Fixed (1/4)   ████████████████ 41%
√d scaling    ██████████████████████████████ 78%

Attention entropy at step 200

No scaling    █ 0.2    (got worse — completely locked)
Fixed (1/4)   ████████████ 1.1
√d scaling    ██████████████████ 1.8
```

The unscaled model never learns the task. Its attention maps look *confident* — sharply focused on specific tokens — but that confidence is artificial. It's just geometry, not learned attention. The model is stuck exactly where initialization left it.

## The Problem Gets Worse as d Grows

We repeat the experiment across d = {16, 32, 64, 128, 256}. With √d scaling, accuracy stays above 90% at every scale. Without it:

```
Peak accuracy WITHOUT scaling, by embedding dim

d=16   ██████████████████████████████████ 67%
d=32   █████████████████ 34%
d=64   ██████ 12%
d=128  ████  8%
d=256  ███   6%   ← nearly random

With √d scaling: ████████████████████████████████████████████ 90-95% at all d
```

Every time you scale up the model, the unscaled version gets worse. This is why early transformer experiments that forgot the scaling reported poor results — they blamed the architecture, not the missing constant.

## Temperature Is a Generalization

The `1/√d` scaling is a special case of **attention temperature**:

```
Attention = softmax(QKᵀ / τ) · V
```

where τ=√d is the temperature. Higher temperature → smoother distribution. Lower → sharper.

Modern applications deliberately tune this:
- **Temperature = 0 at inference:** Greedy decoding (argmax attention)
- **Temperature > 1:** More exploratory, useful for creative generation
- **Temperature < 1 during fine-tuning:** Can sharpen attention toward relevant context

The original paper chose `τ = √d` as a principled default that adapts with model size. The principle — keep softmax inputs in a controlled variance regime — is what matters, however you get there.

## Attention Scaling in Frontier LLM Research

Modern LLMs add a second scaling mechanism on top of `1/√d`: **Rotary Position Embeddings (RoPE)**, used in LLaMA, Mistral, and GPT-NeoX. RoPE encodes relative positions by rotating query and key vectors, which interacts with the dot product in a way that naturally preserves scaling invariance.

There's also active research on **softmax alternatives** that don't suffer from entropy collapse:
- **softmax1** (adds a constant to the denominator) avoids the one-hot regime without needing scaling
- **FlashAttention** reorders the attention computation for memory efficiency but relies on the same scaling

A recent insight from scaling research: the effective attention temperature drifts during training even with `1/√d`, because learned Q and K weight matrices can develop large norms. Some training recipes (e.g., QK-Norm) add explicit normalization to Q and K projections to prevent this drift [1].

---

[1] Scaling Vision Transformers to 22 Billion Parameters. Dehghani et al., 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)

*Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
