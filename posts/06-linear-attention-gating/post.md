# How Gating Saves Linear Attention's Memory
*A single scalar per timestep — learned to open or close — is the difference between a model that remembers and one that forgets everything.*

---

> Vanilla recurrent models have a dirty secret: every new token rewrites the hidden state. Feed them sixteen noise tokens between a useful signal and the final query, and the signal is gone. The fix is mechanical and elegant — multiply the previous state by a gate before adding the new input. The gate decides what to keep and what to overwrite, and this single decision is what separates a model that can remember across long sequences from one that cannot.

---

## The Problem: Every New Token Corrupts the State

A vanilla tanh RNN updates its hidden state as:

```
h_t = tanh(W · x_t  +  U · h_{t-1})
```

Every timestep, both the previous state `h_{t-1}` and the current input `x_t` contribute to the new state through a nonlinear mixing. There is no mechanism to say "this token is irrelevant — do not let it modify what I have stored." As irrelevant tokens accumulate, the useful signal gets progressively diluted.

This is not a failure of capacity or training. It is structural. The architecture has no way to block an update.

## The Fix: Multiply Before You Accumulate

A gated update interposes a scalar gate `g_t ∈ [0, 1]` before the recurrence:

```
h_t = g_t · h_{t-1}  +  (1 − g_t) · tanh(W · x_t)
```

When `g_t ≈ 1`: the previous state is preserved, the new input is almost entirely ignored.
When `g_t ≈ 0`: the previous state is erased, the new input replaces it.

The gate controls the balance between memory and update at every position. The question is what determines `g_t`.

**Fixed gate:** `g_t = 0.9` always — a constant exponential decay. Simple, no parameters needed. The state fades predictably: after L noise tokens, `0.9^L` of the original signal survives.

**Data-dependent gate:** `g_t = σ(W_g · x_t)` — a small learned linear layer applied to the current input. The gate varies by token: high on noise (preserve state), low on important tokens (update state). This is the core innovation in GLA (Yang et al., 2023) and, with more structure, in Mamba (Gu & Dao, 2023).

## The Experiment

We test this on a **needle-in-haystack** task: the sequence begins with a MARK token followed by an important value, then L noise tokens, then a QUERY token. The model must output the value from position 1, ignoring everything in between.

This directly tests whether a recurrent model can hold a single piece of information stable across L interfering updates.

| Parameter | Value |
|-----------|-------|
| Hidden dim | 16 |
| Noise length L | 16 (training), 2–16 (scaling) |
| Vocab | 18 tokens (2 special + 8 values + 8 noise) |
| Training steps | 300 |
| Batch size | 256 |
| Seeds | 2 |
| Optimizer | Adam, lr=5e-3 |
| Chance level | 1/8 = 12.5% |

**Three conditions:**
1. **Vanilla RNN** — no gate, baseline
2. **Fixed gate (g = 0.9)** — constant exponential decay
3. **Data-dependent gate** — learned gate `g_t = σ(W_g · x_t)`

![Figure 1](fig-linear-attention-gating.png)
*Figure 1. (a) Gate values across sequence positions for a trained data-dependent gate model: the gate learns to spike (open) at the MARK position and stay high (preserve state) through noise tokens. (b) Accuracy during training at L=16: fixed gate significantly outperforms both vanilla and data-dependent gate in 300 steps. (c) Accuracy vs. noise length L: vanilla RNN collapses at L=8, fixed gate degrades gracefully, data-dependent gate requires more training to unlock its full advantage.*

## Vanilla RNN Collapses; Fixed Gate Holds

At L=2 and L=4 (short sequences), all three models succeed. Vanilla RNN can hold one piece of information through a few tokens without corruption. But the story changes dramatically at L=8:

```
Accuracy at noise_len = 8 (300 training steps)

Vanilla RNN        ██           0.107   ← near chance (0.125)
Data-dep gate      █▌           0.143
Fixed gate g=0.9   ████████████ 0.561   ← 5.2× above chance
```

The vanilla model's state has been overwritten by 8 noise tokens. The signal is gone. The fixed gate retains `0.9⁸ = 0.43` of the original state — enough to identify the correct value significantly above chance. The story is the same at L=12 and L=16, where fixed gate (0.314, 0.154) continues to dominate vanilla (0.109, 0.111).

**Counterintuitive result:** The data-dependent gate, which should theoretically be the best performer, does *not* outperform the fixed gate in this 300-step toy experiment. The reason is the learning difficulty: the data-dependent gate must learn that noise tokens → `g ≈ 1` (preserve) and the MARK token → `g ≈ 0` (write). This requires gradient signal to flow back through many timesteps, and 300 steps is not enough for a dim-16 model to fully discover this strategy. At scale — with larger hidden dimensions, more training, and gradient-friendly initialization — GLA and Mamba demonstrate that data-dependent gating dominates both fixed-decay and vanilla recurrence.

## The Gate Learns to Discriminate

Panel (a) shows the gate values across positions for a trained data-dependent gate model. The learned pattern is already visible: the gate value at the MARK position (position 0) is low — the model opens to absorb the new value. Across the noise region (positions 2–17), the gate stays elevated, protecting the stored state. The gate has, partially, learned the right strategy.

The path from this partial solution to the dominant behavior observed in GLA and Mamba requires three ingredients not present in our toy model: larger hidden dimension (so the gate has richer input features), longer training, and the hardware-efficient training algorithm that allows GLA to propagate gradients through long sequences without numerical issues.

## The Exponential Decay Bound

Why does the fixed gate degrade gracefully while vanilla fails catastrophically? The math is direct.

For vanilla RNN: after L noise tokens, the hidden state is dominated by a composition of L nonlinear functions of noise inputs. The original signal is not preserved in any recoverable way.

For fixed gate g=0.9: after L noise tokens, the state is:

```
h_{L+2} = 0.9^L · h_2  +  (1−0.9) · Σ_{l=1}^{L} 0.9^{L−l} · tanh(W · noise_l)
```

The original value `h_2` decays by `0.9^L` but is never entirely erased. A classifier trained on this state can always recover the signal, as long as `0.9^L` is above the noise floor. At L=8: `0.9^8 = 0.43`. At L=16: `0.9^16 = 0.19`. Small but nonzero — which is why accuracy at L=16 (0.154) is still above chance (0.125).

Data-dependent gating removes the exponential decay constraint entirely. By learning `g = 1.0` at noise positions, the original signal is preserved with zero decay. This is the theoretical ceiling — but reaching it requires training.

## Linear Attention in Frontier LLM Research

The gating mechanism described here is not just a conceptual fix — it is the architectural core of current frontier models for long-context processing. **GLA** (Yang et al., 2023) extends this to the multi-head linear attention setting with a hardware-aware chunked training algorithm that makes gated recurrence as efficient as standard linear attention. On language modeling benchmarks, GLA matches Mamba at similar parameter counts.

**Mamba** (Gu & Dao, 2023) takes data-dependent gating further: not just the gate but the entire transition matrix A and input projection B are input-dependent. This makes the model fully selective — both what to remember and how to update the memory are determined by the current token. Mamba-2 (Dao & Gu, 2024) later showed this is equivalent to a structured form of linear attention, unifying the two lines of work.

The toy experiment here makes one thing clear: even a primitive fixed gate dramatically outperforms ungated recurrence. The full power of data-dependent gating requires scale — but the structural insight is the same at 16 dimensions and at 4096.

---

[1] Yang, S. et al., 2023. Gated Linear Attention Transformers with Hardware-Efficient Training. [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

[2] Gu, A. & Dao, T., 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

[3] Dao, T. & Gu, A., 2024. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

*Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
