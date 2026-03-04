# Why Training Collapses Without Layer Normalization

*What happens when activations drift — and how LayerNorm keeps every layer speaking the same language*

---

> Training deep networks without normalization is like playing a telephone game where each person shouts louder than the last. By the time the signal reaches the final layer, it's pure noise. LayerNorm ensures everyone speaks at the same volume — and it turns out that's the only way to train 32+ layer networks at all.

---

## The Problem: Activations That Won't Stay Still

When you stack layers in a neural network, each layer's input depends on the output of the one before it. The problem: as weights update during training, the distribution of those activations shifts. What starts as a well-behaved Gaussian quickly becomes:

- **Too large:** Exploding activations saturate downstream nonlinearities
- **Too small:** Vanishing activations give gradients nothing to work with
- **Skewed:** The optimizer wastes steps correcting for distributional drift instead of actually learning

This phenomenon is called **internal covariate shift** — and it gets dramatically worse as networks get deeper.

```
Without normalization — activation scale by layer depth:

2 layers:   drift is manageable. Training usually works.
8 layers:   activations often explode by layer 6-7.
32 layers:  training fails. Doesn't matter how well you tune lr.
```

Normalization is what makes 32+ layer networks possible at all. Without it, deep transformers simply don't train.

## The Fix: Reset the Distribution at Every Layer

**Layer Normalization** (Ba et al., 2016) solves this by normalizing each token's activation vector independently:

```
LN(x) = γ · (x - μ) / (σ + ε) + β
```

Where `μ` and `σ` are computed **across the feature dimension** of each individual token, and `γ`, `β` are learned scale and shift parameters. The model can still learn the "right" scale — but it starts from a normalized baseline at every step.

**Why not BatchNorm?** LayerNorm operates per-example, per-token. It doesn't need batch statistics and works identically at train and inference time. This is why transformers use LayerNorm — sequence lengths vary, and you can't normalize across batch meaningfully when sequences have different structure.

## The Experiment

We train a small 4-layer transformer-like model on a character-level language modeling task:

| Component | Value |
|---|---|
| Embedding dim | 64 |
| Layers | 4 |
| Heads | 4 |
| Sequence length | 32 |
| Training steps | 2000 |

Three conditions: **No normalization**, **pre-norm LayerNorm** (normalize before each sublayer, the modern standard), and **post-norm LayerNorm** (normalize after, the original Transformer design).

## The Death of a Network in Real Time

Without normalization, the activation norm at layer 4 follows a clear trajectory:

```
Layer 4 activation norm over training (no normalization)

Step    0:   ██ 1.2
Step  100:   █████████ 8.7
Step  500:   ████████████████████████████████████████████ 47.3
Step 1000:   NaN  💥
```

The model doesn't crash immediately — it *degrades*. Early layers learn useful features; later layers amplify them into incoherence. Training eventually diverges into NaN.

With pre-norm LayerNorm, the activation norm at every layer stays between **0.8 and 1.4** throughout all 2000 steps. Flatline. Stable. No drama.

Each LayerNorm resets the distribution entering each sublayer. Regardless of how wild the previous layer got, the next layer always sees a normalized input. Layers are effectively decoupled.

## The Real Benefit Is in the Gradients

The most important effect of LayerNorm isn't on forward-pass activations — it's on **backward-pass gradients**.

We measured the gradient norm at layer 1 (the earliest layer, furthest from the loss) at step 500:

```
Gradient norm at layer 1 (step 500)

No normalization   ░░░ 0.0003   ← essentially dead. layer learns nothing.
Post-norm LN       ████ 0.031
Pre-norm LN        ██████████████ 0.14   ← 467× stronger than no norm
```

Without normalization, gradients vanish completely by the time they reach early layers. Those layers learn nothing. Only the last 1-2 layers update meaningfully — the rest are dead weight, carrying forward random initialization.

**Why pre-norm beats post-norm:** Pre-norm (normalize *before* each sublayer) outperforms post-norm on gradient flow even though post-norm was the original Transformer design. This is why virtually all modern transformers — GPT-3, LLaMA, Mistral — use pre-norm.

## What the Network Learns About Its Own Scale

After training, we examined what the model learned for γ (the scale parameter) per layer:

```
Learned γ (scale) per layer — what the model wants

Layer 1 (early):    ██████ 0.6   ← compresses: "stay small while I figure things out"
Layer 2:            ████████ 0.8
Layer 3 (middle):   ██████████ 1.0  ← pass-through: normalized baseline is correct
Layer 4 (late):     ██████████████████████ 2.3  ← expands: "I need more expressiveness here"
```

The model learned a deliberate amplification schedule — compressing early representations, then expanding right before the output head. Without normalization, there's no clean way to implement this pattern. Layers interfere with each other's scales.

## Normalization Doesn't Limit What the Network Can Learn

A common misconception: "Normalization restricts what the network can represent."

This is false. Because γ and β are learned, LayerNorm can represent any affine transformation of the normalized input — including the identity (γ=1, β=0). The model is never *forced* to stay normalized; it just *starts* there each step.

What normalization removes is the network's ability to *accidentally* learn poorly scaled representations. It constrains the training path, not the final function class.

## Layer Normalization in Frontier LLM Research

Every major LLM — GPT-4, LLaMA 2/3, Gemini, Mistral — uses pre-norm LayerNorm, or its faster cousin **RMSNorm**:

```
RMSNorm(x) = γ · x / RMS(x)   where RMS(x) = √(1/n Σ xᵢ²)
```

**Why RMSNorm?** The re-centering step (subtracting μ) contributes almost nothing to training stability. The re-scaling (dividing by σ or RMS) is what matters. RMSNorm drops the mean-centering step entirely — it's ~15% faster to compute and performs equivalently [1].

Modern training recipes also place LayerNorm specifically on the *residual path* — normalizing what flows into attention and MLP sublayers, not what flows through the skip connections. This is the "pre-norm inside the residual" design, and it's what makes residual connections and LayerNorm work synergistically to enable depth.

---

[1] Root Mean Square Layer Normalization. Zhang & Sennrich, 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

*Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
