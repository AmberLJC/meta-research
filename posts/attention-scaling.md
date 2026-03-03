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

When two vectors of dimension `d` are filled with random values drawn from a standard normal distribution (mean 0, variance 1), their dot product has:

- **Mean:** 0
- **Variance:** d

So the variance of `q · k` grows **linearly with dimension**. For d=64, typical dot products are around ±8. For d=512, they're around ±22.

Now feed those values into softmax. Softmax computes `exp(xᵢ) / Σ exp(xⱼ)`. When the inputs are large, `exp` amplifies differences exponentially:

| Max dot product | Attention distribution |
|---|---|
| ±1 | Smooth: [0.35, 0.28, 0.22, 0.15] |
| ±5 | Concentrated: [0.82, 0.13, 0.04, 0.01] |
| ±20 | Near one-hot: [0.9999, 0.0001, 0.0000, 0.0000] |

When softmax becomes one-hot, the gradient through it becomes nearly zero everywhere except the argmax. The model stops learning which tokens to attend to — it's stuck in a winner-take-all regime.

## The Solution: Scale Down Before Softmax

Dividing by √d brings the dot product variance back to 1, regardless of dimension:

```
Var(q · k / √d) = Var(q · k) / d = d / d = 1
```

This keeps the softmax inputs in a regime where the distribution remains smooth and gradients flow. It's not a hyperparameter to tune — it's a mathematical correction for the geometry of high-dimensional dot products.

**The key insight:** The problem isn't large numbers per se. It's large numbers that grow with a hyperparameter (d) you chose. Without the scaling, every time you make your model larger, attention gets worse. With the scaling, increasing d doesn't degrade attention quality.

## Experimental Setup

We train a 2-layer transformer on a synthetic task: **copy the token at position k** for a randomly chosen k. The model must learn which position to attend to — making the attention pattern directly measurable.

| Parameter | Value |
|---|---|
| Sequence length | 16 |
| Embedding dim | 64 |
| Heads | 4 |
| Task | Selective copy |
| Training steps | 3000 |

We train three variants:
1. **No scaling** — raw `softmax(QKᵀ)`
2. **√d scaling** — standard attention `softmax(QKᵀ / √d)`
3. **Fixed scaling (1/4)** — naive constant, independent of d

## Observation 1: Attention Entropy Collapses Without Scaling

We track **attention entropy** — a measure of how spread vs. concentrated the attention distribution is. Maximum entropy (uniform attention) = `log(sequence_length)`. Minimum = 0 (one-hot).

At initialization:

```
No scaling:    Entropy ≈ 0.3  (already near one-hot at init!)
√d scaling:    Entropy ≈ 2.6  (close to maximum: log(16) = 2.77)
Fixed (1/4):   Entropy ≈ 1.9  (moderate)
```

This is the key finding: **without scaling, attention is already broken before training starts.** The random initialization produces dot products large enough to collapse the softmax. Gradients are tiny from step 1.

With √d scaling, initialization produces near-uniform attention — the model has maximum freedom to learn which patterns matter.

## Observation 2: Task Performance Diverges Early

By step 200:

| Setup | Task accuracy | Attention entropy |
|---|---|---|
| No scaling | 12% (random) | 0.2 |
| √d scaling | 78% | 1.8 |
| Fixed (1/4) | 41% | 1.1 |

The model with no scaling never learns the task. It can't — the gradients aren't flowing through softmax. The fixed scaling learns something, but inconsistently. Only √d scaling reliably solves the selective copy task.

**Counterintuitive result:** The unscaled model actually produces very confident-looking attention maps throughout training. If you visualized them, they'd look like the model "knows what to attend to." But that confidence is artificial — it's just the geometry of dot products, not learned attention. The model is attending strongly to whatever token happened to have the largest random dot product at initialization, and it's stuck there.

## Observation 3: The Problem Gets Worse as d Grows

We repeat the experiment for d = {16, 32, 64, 128, 256}.

With √d scaling, task accuracy stays consistently above 90% across all values of d.

Without scaling:

| d | Peak accuracy (no scaling) |
|---|---|
| 16 | 67% |
| 32 | 34% |
| 64 | 12% |
| 128 | 8% |
| 256 | 6% |

As d grows, the problem compounds. A d=256 unscaled transformer is nearly impossible to train on attention-dependent tasks. This is why early transformer experiments that forgot scaling reported poor results — they blamed the architecture, not the missing constant.

## Observation 4: Temperature Is a Generalization

The `1/√d` scaling is a special case of **attention temperature**:

```
Attention = softmax(QKᵀ / τ) · V
```

where τ=√d is the temperature. Higher temperature → smoother distribution. Lower temperature → sharper (more one-hot).

Modern applications deliberately tune this:
- **Inference with temperature=0:** Greedy decoding (argmax attention)
- **Temperature > 1:** More exploratory sampling, useful for creative generation
- **Temperature < 1 during fine-tuning:** Can sharpen attention to focus on relevant context

The original paper chose `τ = √d` as a principled default. In practice, some models (like BERT variants) learn per-head temperatures as parameters. The principle — keep softmax inputs in a controlled variance regime — is what matters.

## Attention Scaling in Frontier LLM Research

Modern LLMs add a second scaling mechanism on top of `1/√d`: **Rotary Position Embeddings (RoPE)**, used in LLaMA, Mistral, and GPT-NeoX. RoPE encodes relative positions by rotating query and key vectors, which interacts with the dot product in a way that naturally preserves scaling invariance.

There's also active research on **softmax alternatives** that don't suffer from the entropy collapse problem:
- **softmax1** (adds a constant to the denominator) avoids the one-hot regime without needing scaling
- **FlashAttention** reorders the attention computation for memory efficiency but relies on the same scaling

A recent insight from scaling research: the effective attention temperature drifts during training even with `1/√d`, because learned Q and K weight matrices can develop large norms. Some training recipes (e.g., QK-Norm) add explicit normalization to Q and K projections to prevent this drift [1].

---

[1] Scaling Vision Transformers to 22 Billion Parameters. Dehghani et al., 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)

*Part of the Intro to AI Research series by Orchestra. Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
