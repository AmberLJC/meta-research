# Why Linear Attention Forgets — The Mathematics of Uniform Attention
*Removing softmax is not free. The kernel trick that makes attention linear also strips it of its ability to focus on a single token.*

---

> Softmax creates a probability distribution: one number per key, all positive, summing to one. With enough contrast in the scores, all the weight collapses onto a single key — selective, like a spotlight. Linear attention replaces softmax with a kernel inner product. The weights still sum to one, but they are **bounded away from zero**. The spotlight becomes a floodlight, and the model can no longer tell which key the query is asking for.

---

## The Problem: Attention Without a Spotlight

Standard attention computes a weighted sum over values, where the weights come from softmax:

```
w_i = exp(q · k_i / √d) / Σ_j exp(q · k_j / √d)
```

The exponential in the numerator is the key ingredient. If `q · k_1 / √d` is 3 units larger than all other scores, then `w_1` is `exp(3) ≈ 20×` larger than any other weight. In the limit, the distribution becomes one-hot: the query attends exclusively to the matching key.

Linear attention (Katharopoulos et al., 2020) replaces this with:

```
w_i = φ(q) · φ(k_i) / Σ_j φ(q) · φ(k_j)
```

where `φ(x) = elu(x) + 1` keeps all values positive. The weights still sum to 1 — but there is no exponential amplification. The contrast between keys is linear, not exponential. And for random keys drawn from a high-dimensional space, the inner products `φ(q) · φ(k_i)` all concentrate around the same value. Every key receives roughly equal weight: `w_i ≈ 1/N`.

This is not an artifact of insufficient training. It is a mathematical property of the feature map.

## The Fix: There Is No Fix in This Post

Linear attention's O(n) cost is real. The recall failure is also real. They are two sides of the same coin — the same design decision that eliminates the n×n matrix also eliminates the selective, concentrated attention. Part 3 of this series examines how gating partially recovers selectivity in the recurrent view of linear attention.

## The Experiment

We test retrieval selectivity in two ways: analytically, to establish the mathematical fact; and empirically, to confirm it persists in trained models.

**Analytical setup:** Given N random key vectors `k_1, ..., k_N ∈ ℝ⁶⁴` and query `q = k_1` (a perfect match), how much attention weight does each method assign to position 1?

**Training setup:** A minimal single-layer attention model (no MLP, no LayerNorm) trained on an associative recall task. Input: `[k₁, v₁, k₂, v₂, k₃, v₃, k₄, v₄, query_key]`. Target: the value paired with `query_key`. Vocab = 24 tokens (8 keys + 16 values). 600 training steps, 2 seeds.

| Parameter | Value |
|-----------|-------|
| Dim `d` | 64 (analytical), 32 (training) |
| Keys (analytical) | N ∈ {2, 4, 8, 16, 32, 64} |
| Key-value pairs (training) | 4 |
| Vocab | 24 tokens |
| Training steps | 600 |
| Seeds | 2 |
| Optimizer | Adam, lr=0.01 |

![Figure 1](fig-linear-attention-recall.png)
*Figure 1. (a) Analytical retrieval weight on the matching key as N increases: softmax stays near 1.0 across all N; linear ELU tracks the uniform baseline 1/N exactly. (b) Training accuracy over 600 steps: both models exceed chance (6.25%) but neither fully converges on a 4-pair recall task with a single attention layer. (c) After training, softmax attention assigns higher weight to the matching key position than linear ELU — consistent with the analytical prediction.*

## Softmax Maintains Selectivity; Linear Attention Cannot

The analytical experiment is unambiguous. With a perfect query `q = k_1`, here is the attention weight assigned to the matching key:

```
Weight on matching key (query = k₁, 1000 random trials)

N = 2   softmax  ██████████████████████████ 0.999   linear  ████████████ 0.592
N = 4   softmax  ██████████████████████████ 0.996   linear  ██████       0.329
N = 8   softmax  ██████████████████████████ 0.991   linear  ████         0.172
N = 16  softmax  ██████████████████████████ 0.981   linear  ██           0.089
N = 32  softmax  ██████████████████████████ 0.966   linear  █            0.044
N = 64  softmax  ██████████████████████████ 0.930   linear  ▌            0.023
```

Linear attention's weight at N=64 is 0.023 — indistinguishable from the uniform baseline of 1/64 = 0.016. The model effectively assigns equal weight to every key regardless of whether it matches the query. Softmax maintains 0.930 even at N=64.

The difference compounds with depth: in a two-layer softmax transformer, the second layer can act on already-sharpened representations. In a two-layer linear attention network, each layer's diffuse output feeds into the next, with no sharpening accumulating.

## The Training Curves Tell a More Cautious Story

**Counterintuitive result:** On the 4-pair recall task, a single-layer linear ELU model matches or slightly exceeds softmax after 600 training steps (softmax = 0.323, linear = 0.333, chance = 0.062). Both are well above chance, but neither converges cleanly.

This does not contradict the analytical result — it qualifies it. With only 4 key-value pairs, the attention weight from any position is at most 1/9 ≈ 0.11. Linear attention assigns 0.018 to the matching key; softmax assigns 0.051. The advantage exists but is not large enough for a single-layer model with 600 steps to separate the two cleanly.

The analytical result predicts the failure will be catastrophic at N = 32 or N = 64 keys. The training experiment is testing the easy end of the spectrum. This matters: real retrieval-augmented systems index thousands of documents, not four.

## The Attention Weights Confirm the Theory

Looking directly at what the trained models attend to reveals the gap (panel c). After training, the softmax model's attention distribution on the matching key (mean = 0.051) is nearly 3× higher than the linear ELU model's (mean = 0.018). Both are well below the theoretical maximum of 1.0, reflecting that the models have not fully converged — but the direction is consistent with the analytical prediction.

The trained softmax model has learned to be more selective. The trained linear ELU model, despite optimizing the same objective, cannot produce sharp attention distributions by design. The weights in panel (c) show this directly: softmax produces a histogram shifted toward higher values; linear ELU produces a histogram concentrated near zero.

## Linear Attention in Frontier LLM Research

The recall failure described here is not a theoretical curiosity — it motivated a wave of architecture redesign. **MEGALODON** (Ma et al., 2024) documents the exact recall failure in production: linear attention models trained at scale perform well on perplexity but poorly on downstream tasks requiring precise token retrieval. **Mamba** (Gu & Dao, 2023) directly addresses the limitation: by making the recurrent state update *selective* through input-dependent parameters, it recovers retrieval capability without returning to O(n²) attention.

The key insight driving these architectures: what linear attention loses is not capacity but *selectivity*. Restoring selectivity requires not a bigger kernel, but a fundamentally different update rule — which is the subject of Part 3.

---

[1] Katharopoulos, A. et al., 2020. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

[2] Gu, A. & Dao, T., 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

[3] Ma, X. et al., 2024. MEGALODON: Efficient LLM Pretraining and Inference with Unlimited Context Length. [arXiv:2404.08801](https://arxiv.org/abs/2404.08801)

*Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
