"""
exp-linear-attention-quadratic.py
Experiment for: posts/linear-attention-quadratic.md

Compares:
  1. Softmax attention (standard, O(n^2))
  2. Linear attention with ELU feature map (O(n))
  3. Raw dot-product attention (no normalization, naive)

Measures: wall-clock time and memory vs sequence length, plus training loss.
"""

import time
import tracemalloc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── colour palette ──────────────────────────────────────────────────
C1 = '#1a1a2e'   # dark navy  — broken baseline (softmax)
C2 = '#e63946'   # red        — naive (raw dot-product)
C3 = '#2a9d8f'   # teal       — hero (linear attention)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.8, 'figure.dpi': 150,
})

# ── attention implementations ────────────────────────────────────────

def softmax_attention(Q, K, V):
    """Standard O(n^2) attention."""
    d = Q.shape[-1]
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, V)


def linear_attention(Q, K, V):
    """Linear attention via ELU feature map, O(n*d^2)."""
    phi = lambda x: F.elu(x) + 1.0
    Q2, K2 = phi(Q), phi(K)
    # compute KV context matrix first  (d x d), then apply Q
    KV = torch.bmm(K2.transpose(1, 2), V)       # (B, d, d)
    Z  = 1.0 / (torch.bmm(Q2, K2.sum(dim=1, keepdim=True).transpose(1, 2)).squeeze(-1) + 1e-6)
    out = torch.bmm(Q2, KV) * Z.unsqueeze(-1)
    return out


def raw_attention(Q, K, V):
    """Raw dot-product attention (no softmax, no normalization) — unstable."""
    d = Q.shape[-1]
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d ** 0.5)
    return torch.bmm(scores, V)


# ── timing / memory sweep ────────────────────────────────────────────

SEQ_LENS = [64, 128, 256, 512, 1024, 2048]
DIM = 64

def measure(attn_fn, seq_len, n_runs=3):
    """Return (mean_ms, peak_mb) over n_runs."""
    times, mems = [], []
    for _ in range(n_runs):
        Q = torch.randn(1, seq_len, DIM)
        K = torch.randn(1, seq_len, DIM)
        V = torch.randn(1, seq_len, DIM)
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = attn_fn(Q, K, V)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append((t1 - t0) * 1000)
        mems.append(peak / 1024 / 1024)
    return np.mean(times), np.mean(mems)


print("Running timing/memory sweep...")
results = {name: {'times': [], 'mems': []} for name in ['softmax', 'linear', 'raw']}
fns = {'softmax': softmax_attention, 'linear': linear_attention, 'raw': raw_attention}

for n in SEQ_LENS:
    for name, fn in fns.items():
        t, m = measure(fn, n)
        results[name]['times'].append(t)
        results[name]['mems'].append(m)
    print(f"  seq_len={n:4d}: softmax={results['softmax']['times'][-1]:.1f}ms  linear={results['linear']['times'][-1]:.1f}ms  raw={results['raw']['times'][-1]:.1f}ms")

# ── character LM training ────────────────────────────────────────────

class TinyTransformerLayer(nn.Module):
    def __init__(self, dim, attn_fn):
        super().__init__()
        self.attn_fn = attn_fn
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.ff  = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        attn_out = self.attn_fn(Q, K, V)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, dim, attn_fn):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layer = TinyTransformerLayer(dim, attn_fn)
        self.head  = nn.Linear(dim, vocab)

    def forward(self, x):
        h = self.embed(x)
        h = self.layer(h)
        return self.head(h)


VOCAB, SEQ, DIM_LM, STEPS, LR = 32, 64, 64, 2000, 3e-4

def make_batch(bs=16):
    data = torch.randint(0, VOCAB, (bs, SEQ + 1))
    return data[:, :-1], data[:, 1:]

def train_lm(attn_fn, label, clip=1.0):
    torch.manual_seed(42)
    model = TinyLM(VOCAB, DIM_LM, attn_fn)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        x, y = make_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % 50 == 0:
            losses.append(loss.item())
    print(f"  [{label}] final loss: {losses[-1]:.3f}")
    return losses

print("\nTraining character LMs...")
steps_axis = list(range(0, STEPS, 50))
loss_softmax = train_lm(softmax_attention, 'softmax')
loss_linear  = train_lm(linear_attention,  'linear')
loss_raw     = train_lm(raw_attention,     'raw-dot')

# ── plot ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
fig.suptitle('Linear Attention — Efficiency vs Softmax', fontsize=11, fontweight='bold', y=1.03)

# Panel (a): Time vs seq_len
ax = axes[0]
ax.plot(SEQ_LENS, results['softmax']['times'], color=C1, ls='--',       lw=1.8, label='Softmax O(n²)')
ax.plot(SEQ_LENS, results['raw']['times'],     color=C2, ls='-.',       lw=1.8, label='Raw dot-product')
ax.plot(SEQ_LENS, results['linear']['times'],  color=C3, ls='-',        lw=2.2, label='Linear (ELU)')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel('Sequence length (log scale)'); ax.set_ylabel('Time (ms, log scale)')
ax.set_title('(a) Forward-pass time vs sequence length')
ax.legend(fontsize=8)

# Panel (b): Memory vs seq_len
ax = axes[1]
ax.plot(SEQ_LENS, results['softmax']['mems'], color=C1, ls='--',  lw=1.8, label='Softmax O(n²)')
ax.plot(SEQ_LENS, results['raw']['mems'],     color=C2, ls='-.',  lw=1.8, label='Raw dot-product')
ax.plot(SEQ_LENS, results['linear']['mems'],  color=C3, ls='-',   lw=2.2, label='Linear (ELU)')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel('Sequence length (log scale)'); ax.set_ylabel('Peak memory (MB, log scale)')
ax.set_title('(b) Memory footprint vs sequence length')
ax.legend(fontsize=8)

# Panel (c): Training loss curves
ax = axes[2]
ax.plot(steps_axis, loss_softmax, color=C1, ls='--',  lw=1.8, label='Softmax')
ax.plot(steps_axis, loss_raw,     color=C2, ls='-.',  lw=1.8, label='Raw dot-product')
ax.plot(steps_axis, loss_linear,  color=C3, ls='-',   lw=2.2, label='Linear (ELU)')
ax.set_xlabel('Training step'); ax.set_ylabel('Cross-entropy loss')
ax.set_title('(c) Character LM training loss (seq=64)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('posts/fig-linear-attention-quadratic.png', dpi=200, bbox_inches='tight', pad_inches=0.12)
plt.close()
print("\nFigure saved: posts/fig-linear-attention-quadratic.png")

# print key numbers for the post
print(f"\nKey numbers:")
print(f"  Time at seq=2048:  softmax={results['softmax']['times'][-1]:.1f}ms  linear={results['linear']['times'][-1]:.1f}ms  raw={results['raw']['times'][-1]:.1f}ms")
print(f"  Mem  at seq=2048:  softmax={results['softmax']['mems'][-1]:.1f}MB   linear={results['linear']['mems'][-1]:.1f}MB   raw={results['raw']['mems'][-1]:.1f}MB")
print(f"  Final loss:        softmax={loss_softmax[-1]:.3f}  linear={loss_linear[-1]:.3f}  raw={loss_raw[-1]:.3f}")
