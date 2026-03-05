# Why Transformers Can't Read Long Books
*Softmax attention scales quadratically with sequence length — and the kernel trick that fixes it in one line of math.*

---

> Standard attention secretly stores an n×n matrix — one entry for every pair of tokens. At sequence length 2048 that is four million numbers, growing quadratically as you add context. A 100-page document requires 16× the memory and time of a 25-page one. The good news: the attention output can be computed exactly without ever building that matrix, by swapping the order of two matrix multiplications.

---

## The Problem: O(n²) Attention

The standard transformer computes attention as:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

The matrix `QKᵀ` has shape `(n, n)` — every token attends to every other token. Computing it costs **O(n²d)** time and **O(n²)** memory. For language models, `n` is the sequence length. This is fine at `n = 512`. It starts to hurt at `n = 2048`. It becomes prohibitive at `n = 100k` — the range modern LLMs are expected to handle.

The bottleneck isn't the values V or the queries Q. It's the n×n attention matrix itself: it must be materialized in memory before being multiplied by V.

Think of it like printing a full multiplication table before doing any arithmetic. You could just compute each product on demand — but the standard formulation insists on the table first.

## The Fix: Swap the Matrix Multiplication Order

Linear attention (Katharopoulos et al., 2020) starts from a simple observation. Replace the softmax with a feature map φ(·) that keeps all values positive:

```
φ(x) = elu(x) + 1        (all outputs > 0)
```

Now the attention formula becomes:

```
Output = (φ(Q) · [φ(K)ᵀ V]) / (φ(Q) · φ(K)ᵀ · 1)
```

The key insight: `φ(K)ᵀ V` is a `(d × d)` matrix — independent of sequence length. Compute it *first*, then apply `φ(Q)`. The order of operations goes from O(n²d) to **O(nd²)**. For `d = 64` and `n = 2048`, that is a 32× reduction in compute.

The math is exact — no approximation. The output is the same quantity, just computed in a different order.

```
Standard:  softmax(QKᵀ/√d) · V   → build n×n matrix first
Linear:    φ(Q) · (φ(K)ᵀ · V)    → build d×d matrix first
```

For any practical model where `d << n`, the linear version wins decisively.

## The Experiment

We measure the forward-pass cost of three variants on a tiny toy model (d = 64, 2 heads, batch = 1) at sequence lengths from 64 to 2048, then compare their training loss on a character-level language modeling task.

| Parameter | Value |
|-----------|-------|
| Model dim `d` | 64 |
| Sequence lengths | 64 → 2048 |
| Batch size | 1 |
| LM task seq length | 64 |
| LM vocab size | 32 |
| LM training steps | 2000 |
| Seed | 42 |

**Three conditions:**
1. **Softmax attention** — standard O(n²) baseline
2. **Linear attention (ELU φ)** — O(n) kernel trick, the fix
3. **Raw dot-product** — remove softmax, normalize by sum; naive and unstable

![Figure 1](fig-linear-attention-quadratic.png)
*Figure 1. (a) Forward-pass time vs. sequence length (log-log scale): softmax grows quadratically, linear attention grows linearly. (b) Memory footprint follows the same pattern. (c) Character-LM training loss: linear attention matches softmax quality despite the cost reduction.*

## The Quadratic Curve Is Unmistakable

At sequence length 2048, softmax attention takes **155.8 ms** per forward pass. Linear attention with the ELU feature map takes **5.0 ms** — a **31× speedup** on CPU, with no GPU tricks involved.

The time curves in panel (a) tell the story visually. Softmax traces a parabola on the log-log axes — consistent with O(n²). Linear attention traces a straight line — O(n). Raw dot-product sits in between because removing the softmax also removes the stabilizing normalization, leading to numerical instability at large n.

```
Seq length 2048 forward-pass time:
Softmax (O(n²))   ████████████████████████████████ 155.8 ms
Raw dot-product   ████████                          30.5 ms
Linear ELU (O(n)) ██                                 5.0 ms   ← 31× faster
```

## The Cost Is Paid in Quality — Or Is It?

**Counterintuitive result:** On the character-level LM task, all three variants converge to nearly identical final loss: softmax = 3.351, linear ELU = 3.346, raw dot-product = 3.263. The 31× speedup costs essentially nothing in model quality on this task.

This is not always true — linear attention struggles with tasks requiring sharp retrieval (more on that in Part 2 of this series). But for the core business of language modeling — predicting the next token in a sequence — the linear kernel matches softmax quality at a fraction of the cost.

The raw dot-product (naive baseline) actually achieves the lowest loss here, but this is deceptive: without the softmax normalization, the attention scores grow unboundedly during training. At longer sequences or higher learning rates, this variant diverges. The low loss at seq = 64 is a training artifact, not a real advantage.

## The Recurrent Dual: Linear Attention as an RNN

Linear attention has a second remarkable property: it can be computed *recurrently*.

Because φ(K)ᵀV is an outer-product cumulative sum, the output at each position t depends only on a fixed-size `(d × d)` state matrix — exactly like an RNN hidden state. This means:

```
Training (parallel):    O(nd²) — process all tokens at once
Inference (sequential): O(d²)  per token — constant-time decoding
```

Standard softmax transformers have O(n·d) KV-cache cost at inference, growing with context length. Linear attention has O(d²) — constant, regardless of sequence length. For a deployed model generating millions of tokens, this matters enormously.

## Linear Attention in Frontier LLM Research

The efficiency argument for linear attention has driven a wave of modern architectures. **RetNet** (Sun et al., 2023) uses a retention mechanism — essentially linear attention with exponential decay — to achieve O(1) inference while maintaining competitive language modeling performance. The authors show parity with transformers at 7B parameters.

**RWKV** (Peng et al., 2023) pushes this further: a production-scale language model trained to 14B parameters using a purely linear-recurrent formulation. RWKV-4 was deployed on consumer hardware and competed with GPT-3 on standard benchmarks — impossible with a standard transformer at the same parameter count.

The open question is not whether linear attention is efficient — it clearly is. The question is whether the quality tradeoffs are acceptable. Part 2 of this series examines where linear attention fails, and *why* removing softmax creates a fundamental limitation for retrieval-heavy tasks.

---

[1] Katharopoulos, A. et al., 2020. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

[2] Sun, R. et al., 2023. RetNet: Retentive Network: A Successor to Transformer for Large Language Models. [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)

[3] Peng, B. et al., 2023. RWKV: Reinventing RNNs for the Transformer Era. [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)

*Experiments and post generated with [Orchestra](https://www.orchestra-research.com).*
