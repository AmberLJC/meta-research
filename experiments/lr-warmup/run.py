"""
Experiment: Learning Rate Warmup
Goal: Show that warmup stabilizes early training by controlling gradient step size
while Adam's moment estimates are unreliable.

Honest design:
- Character-level language model on synthetic data (bigram statistics)
- 2-layer transformer with causal attention
- Compare: no warmup, short (1%), standard (5%), long (20%) warmup
- Track: loss curves, gradient norm variance (measure of step chaos), final perplexity
- Also track Adam's v_hat estimate for a representative weight to show bootstrap effect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import math

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Synthetic language model data ─────────────────────────────────────────────
# Generate text with bigram statistics — simple but real structure to learn
def make_dataset(vocab_size=16, seq_len=16, n_seqs=4000, seed=42):
    rng = np.random.RandomState(seed)
    # Random bigram transition matrix
    trans = rng.dirichlet([0.5] * vocab_size, size=vocab_size)  # (vocab, vocab)
    seqs = []
    for _ in range(n_seqs):
        s = [rng.randint(0, vocab_size)]
        for _ in range(seq_len):
            s.append(rng.choice(vocab_size, p=trans[s[-1]]))
        seqs.append(s)
    data = torch.tensor(seqs, dtype=torch.long)  # (n_seqs, seq_len+1)
    split = int(0.85 * n_seqs)
    return data[:split], data[split:], trans

# ── Causal self-attention block ───────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, seq_len):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1,2) for t in qkv]
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:T,:T], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, seq_len)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab_size, d, n_layers, n_heads, seq_len):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        self.blocks  = nn.ModuleList([TransformerBlock(d, n_heads, seq_len) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

# ── LR schedule helpers ───────────────────────────────────────────────────────
def get_lr(step, warmup_steps, total_steps, lr_max, lr_min=1e-5):
    if step < warmup_steps:
        return lr_max * (step + 1) / max(warmup_steps, 1)
    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# ── Training ─────────────────────────────────────────────────────────────────
def train_lm(warmup_steps, total_steps=2000, batch=64, lr_max=3e-4,
             vocab_size=16, d=64, n_layers=2, n_heads=4, seq_len=16):
    torch.manual_seed(SEED)
    train_data, val_data, _ = make_dataset(vocab_size, seq_len)
    model = TinyLM(vocab_size, d, n_layers, n_heads, seq_len)

    # Track a single representative weight for optimizer state analysis
    tracked_param = list(model.blocks[0].attn.qkv.parameters())[0]  # shape (3d, d)
    tracked_idx = (0, 0)  # one scalar element

    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.999), weight_decay=0.01)

    history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'grad_norm': [],         # gradient norm each step (raw, not smoothed)
        'grad_norm_var': [],     # variance over 50-step windows
        'v_hat': [],             # Adam's 2nd moment estimate for tracked param
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
    }

    n_tr = len(train_data)
    grad_norms_window = []

    for step in range(total_steps):
        # Adjust LR
        lr = get_lr(step, warmup_steps, total_steps, lr_max)
        for pg in opt.param_groups:
            pg['lr'] = lr

        model.train()
        idx = torch.randint(0, n_tr, (batch,))
        batch_data = train_data[idx]       # (batch, seq_len+1)
        x = batch_data[:, :-1]             # (batch, seq_len)
        targets = batch_data[:, 1:]        # (batch, seq_len)

        logits = model(x)                  # (batch, seq_len, vocab_size)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.reshape(-1))

        opt.zero_grad()
        loss.backward()

        # Total gradient norm
        total_grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        grad_norms_window.append(total_grad_norm)

        opt.step()

        # Track Adam v_hat for the representative parameter
        state = opt.state.get(tracked_param, {})
        v_hat = state.get('exp_avg_sq', torch.zeros(1))
        v_hat_val = v_hat[tracked_idx].item() if hasattr(v_hat, '__getitem__') else v_hat.item()

        if step % 10 == 0:
            # Validation loss
            model.eval()
            with torch.no_grad():
                n_val = len(val_data)
                val_idx = torch.randint(0, n_val, (256,))
                vb = val_data[val_idx]
                vlogits = model(vb[:, :-1])
                val_loss = F.cross_entropy(vlogits.view(-1, vocab_size), vb[:, 1:].reshape(-1)).item()

            # Gradient norm variance over last 50 steps
            gn_var = float(np.var(grad_norms_window[-50:])) if len(grad_norms_window) >= 2 else 0.0

            history['step'].append(step)
            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss)
            history['lr'].append(lr)
            history['grad_norm'].append(total_grad_norm)
            history['grad_norm_var'].append(gn_var)
            history['v_hat'].append(v_hat_val)

    return history

# ── Run all warmup conditions ─────────────────────────────────────────────────
TOTAL_STEPS = 2000
configs = {
    'no_warmup':    0,
    'short_warmup': int(0.01 * TOTAL_STEPS),   # 20 steps
    'std_warmup':   int(0.05 * TOTAL_STEPS),   # 100 steps
    'long_warmup':  int(0.20 * TOTAL_STEPS),   # 400 steps
}

histories = {}
for name, warmup in configs.items():
    print(f"  Training: {name} (warmup={warmup} steps)")
    histories[name] = train_lm(warmup_steps=warmup, total_steps=TOTAL_STEPS)

# ── Save raw results ──────────────────────────────────────────────────────────
with open('experiments/lr-warmup/results.json', 'w') as f:
    json.dump(histories, f, indent=2)

# ── Figure 1: Validation loss curves ─────────────────────────────────────────
colors = {
    'no_warmup':    '#e74c3c',
    'short_warmup': '#e67e22',
    'std_warmup':   '#2ecc71',
    'long_warmup':  '#3498db',
}
labels = {
    'no_warmup':    f'No warmup (0 steps)',
    'short_warmup': f'Short warmup ({configs["short_warmup"]} steps, 1%)',
    'std_warmup':   f'Standard warmup ({configs["std_warmup"]} steps, 5%)',
    'long_warmup':  f'Long warmup ({configs["long_warmup"]} steps, 20%)',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, hist in histories.items():
    axes[0].plot(hist['step'], hist['val_loss'], color=colors[name], label=labels[name], linewidth=2, alpha=0.9)
axes[0].set_title('Validation Loss Over Training', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Training step'); axes[0].set_ylabel('Cross-entropy loss')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# Zoom in on first 10% of steps
early_cutoff = TOTAL_STEPS // 10
for name, hist in histories.items():
    early_steps = [s for s in hist['step'] if s <= early_cutoff]
    early_loss  = [hist['val_loss'][i] for i, s in enumerate(hist['step']) if s <= early_cutoff]
    axes[1].plot(early_steps, early_loss, color=colors[name], label=labels[name], linewidth=2, alpha=0.9)
axes[1].set_title(f'Early Training (First {early_cutoff} Steps)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Training step'); axes[1].set_ylabel('Cross-entropy loss')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.suptitle('Learning Rate Warmup Experiment: Validation Loss', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/lr-warmup/figures/val_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Gradient norm variance (chaos measure) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for name, hist in histories.items():
    ax.plot(hist['step'], hist['grad_norm_var'], color=colors[name], label=labels[name], linewidth=2, alpha=0.85)
# Mark warmup end for each
for name, warmup in configs.items():
    if warmup > 0:
        ax.axvline(warmup, color=colors[name], linestyle=':', alpha=0.5)
ax.set_title('Gradient Norm Variance (50-step rolling window)\nHigher = more chaotic step directions', fontweight='bold', fontsize=12)
ax.set_xlabel('Training step'); ax.set_ylabel('Gradient norm variance')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/lr-warmup/figures/grad_norm_variance.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: LR schedules ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for name, warmup in configs.items():
    lrs = [get_lr(s, warmup, TOTAL_STEPS, 3e-4) for s in range(TOTAL_STEPS)]
    ax.plot(range(TOTAL_STEPS), lrs, color=colors[name], label=labels[name], linewidth=2)
ax.set_title('Learning Rate Schedules', fontweight='bold', fontsize=12)
ax.set_xlabel('Training step'); ax.set_ylabel('Learning rate')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/lr-warmup/figures/lr_schedules.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 4: Adam v_hat over time ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for name, hist in histories.items():
    ax.plot(hist['step'], hist['v_hat'], color=colors[name], label=labels[name], linewidth=2, alpha=0.85)
ax.set_title("Adam's 2nd Moment Estimate (v̂) for a Representative Weight\nHigher early = optimizer state based on noisy gradients", fontweight='bold', fontsize=12)
ax.set_xlabel('Training step'); ax.set_ylabel('v̂ value')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/lr-warmup/figures/adam_vhat.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== RESULTS SUMMARY ===")
print(f"{'Config':<20} {'Warmup':>8} {'Loss@10%':>10} {'Loss@50%':>10} {'Final loss':>10} {'Early GN var':>14}")
step10 = TOTAL_STEPS // 10
step50 = TOTAL_STEPS // 2
for name, hist in histories.items():
    steps = hist['step']
    losses = hist['val_loss']
    gn_vars = hist['grad_norm_var']
    idx10 = min(range(len(steps)), key=lambda i: abs(steps[i] - step10))
    idx50 = min(range(len(steps)), key=lambda i: abs(steps[i] - step50))
    # Early gradient norm variance: first 100 steps
    early_gn_var = np.mean([gn_vars[i] for i, s in enumerate(steps) if s <= 100]) if any(s <= 100 for s in steps) else 0
    print(f"{name:<20} {configs[name]:>8} {losses[idx10]:>10.4f} {losses[idx50]:>10.4f} {losses[-1]:>10.4f} {early_gn_var:>14.4f}")

print("\nDone. Figures saved to experiments/lr-warmup/figures/")
