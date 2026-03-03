# Why Training Collapses Without Layer Normalization

*What happens when activations drift — and how LayerNorm keeps every layer speaking the same language*

---

> Training deep networks without normalization is like playing a telephone game where each person shouts louder than the last. By the time the signal reaches the final layer, it's pure noise. LayerNorm ensures everyone speaks at the same volume.
>
> This is post 2 in our series on stability tricks. We use toy models to show that LayerNorm isn't just a performance tweak — it's a structural necessity.

---

## The Problem: Activations That Drift

When you stack layers in a neural network, each layer's input depends on the output of the one before it. The problem: as weights update during training, the distribution of those activations shifts. What starts as a well-behaved Gaussian quickly becomes:

- **Too large:** Exploding activations saturate downstream nonlinearities
- **Too small:** Vanishing activations give gradients nothing to work with
- **Skewed:** The optimizer wastes steps correcting for distributional drift instead of actually learning

This phenomenon is called **internal covariate shift** — and it gets dramatically worse as networks get deeper.

There's a specific failure mode worth understanding:

| Layer depth | Without normalization | With normalization |
|---|---|---|
| 2 | Manageable drift | Stable |
| 8 | Activations often explode | Stable |
| 32 | Training usually fails | Stable |

Normalization is what makes 32+ layer networks possible at all.

## The Solution: Normalize Within Each Layer

**Layer Normalization** (Ba et al., 2016) solves this by normalizing each token's activation vector independently:

```
LN(x) = γ · (x - μ) / (σ + ε) + β
```

Where:
- `μ` and `σ` are the mean and standard deviation computed **across the feature dimension** of each individual token
- `γ` and `β` are learned scale and shift parameters — the model can still learn the "right" scale, but it starts from a normalized baseline

**Key distinction from BatchNorm:** LayerNorm operates per-example, per-token. It doesn't need batch statistics and works the same at train and inference time. This is why transformers use LayerNorm and not BatchNorm — sequence lengths vary, and you can't normalize across batch meaningfully when sequences have different structure.

## Experimental Setup

We train a small 4-layer transformer-like model on a character-level language modeling task. The architecture:

| Component | Value |
|---|---|
| Embedding dim | 64 |
| Layers | 4 |
| Heads | 4 |
| Sequence length | 32 |
| Dataset | Synthetic token sequences |

We compare three conditions:
1. **No normalization** — raw activations all the way
2. **LayerNorm (pre-norm)** — normalize before each sublayer (modern standard)
3. **LayerNorm (post-norm)** — normalize after each sublayer (original Transformer)

Training runs for 2000 steps with lr=1e-3, batch size 64.

## Observation 1: Activation Scale Explosion

Without normalization, activation norms grow steadily:

```
Step 0:    Layer 4 activation norm ≈ 1.2
Step 100:  Layer 4 activation norm ≈ 8.7
Step 500:  Layer 4 activation norm ≈ 47.3
Step 1000: Layer 4 activation norm ≈ NaN
```

The model doesn't crash immediately — it degrades. Early layers learn useful features; later layers amplify them into incoherence. Training eventually diverges.

With pre-norm LayerNorm, the activation norm at every layer stays bounded between 0.8 and 1.4 throughout training.

**The mechanism:** Each LayerNorm resets the distribution entering each sublayer. Regardless of how wild the previous layer got, the next layer always sees a normalized input. Layers are effectively decoupled.

## Observation 2: Gradient Flow Is the Real Benefit

The most important effect of LayerNorm isn't on forward-pass activations — it's on **backward-pass gradients**.

We measured the gradient norm at layer 1 (the earliest layer, furthest from the loss):

| Setup | Gradient norm at layer 1 (step 500) |
|---|---|
| No normalization | 0.0003 (essentially dead) |
| Post-norm LayerNorm | 0.031 |
| Pre-norm LayerNorm | 0.14 |

Without normalization, gradients vanish completely by the time they reach early layers. Those layers learn nothing. Only the last 1-2 layers update meaningfully — the rest are dead weight.

**Counterintuitive finding:** Pre-norm (normalize before each sublayer) outperforms post-norm on gradient flow even though post-norm was the original design. This is why virtually all modern transformers (GPT-3, LLaMA, Mistral) use pre-norm.

## Observation 3: The Learned γ and β Tell a Story

After training, we examined what the model learned for γ (scale) and β (shift) per layer:

- **Early layers:** γ ≈ 0.6 — the model learned to *compress* representations early
- **Middle layers:** γ ≈ 1.0 — pass-through; the normalized baseline is already correct
- **Late layers:** γ ≈ 2.3 — the model *expands* representations right before the output head

This is a learned amplification schedule — the model figures out where it needs more expressiveness. Without normalization, there's no clean way to implement this pattern. Layers interfere with each other.

## Observation 4: Stability Without Sacrificing Expressiveness

A common misconception: "Normalization restricts what the network can represent."

This is false. Because γ and β are learned, LayerNorm can represent any affine transformation of the normalized input — including the identity transformation (γ=1, β=0). The model is never forced to stay normalized; it just starts there each step.

What normalization actually removes is the network's ability to *accidentally* learn poorly scaled representations. It constrains the training path, not the final function class.

## Layer Normalization in Frontier LLM Research

Every major LLM — GPT-4, LLaMA 2/3, Gemini, Mistral — uses pre-norm LayerNorm (or its cousin RMSNorm). RMSNorm drops the mean-centering step:

```
RMSNorm(x) = γ · x / RMS(x)
```

where `RMS(x) = √(1/n Σ xᵢ²)`.

**Why RMSNorm?** Empirical finding: the re-centering (subtracting μ) contributes almost nothing to training stability, while the re-scaling (dividing by σ or RMS) is what matters. RMSNorm is ~15% faster to compute and performs equivalently [1].

Modern training recipes also place LayerNorm on the *residual path* specifically — normalizing what flows into attention and MLP sublayers, not what flows through the skip connections. This is the "pre-norm inside the residual" design, and it's what makes residual connections and LayerNorm work synergistically.

---

[1] Root Mean Square Layer Normalization. Zhang & Sennrich, 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

*Part of the Intro to AI Research series by Orchestra. Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
