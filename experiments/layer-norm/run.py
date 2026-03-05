"""
Experiment: Layer Normalization
Goal: Show that without LayerNorm, activation norms grow and gradient flow dies in early layers.
Honest design: We use a real 4-layer MLP (not "transformer-like") to keep it reproducible on CPU.
We measure:
  1. Activation norm at each layer over training steps
  2. Gradient norm at each layer over training steps
  3. Validation loss curves
  4. Pre-norm vs Post-norm vs No-norm comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import random

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
# Task: 10-class classification on synthetic Gaussian clusters
# Simple enough to isolate normalization effects
def make_data(n=2000, d=64, n_classes=10, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, d) * 3
    X, y = [], []
    for i in range(n):
        c = i % n_classes
        x = centers[c] + rng.randn(d) * 0.5
        X.append(x)
        y.append(c)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    return (X[:split], y[:split]), (X[split:], y[split:])

# ── Model ─────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d, norm_type):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.norm_type = norm_type
        if norm_type in ('pre', 'post'):
            self.norm = nn.LayerNorm(d)
        self.activation = None  # filled during forward for measurement

    def forward(self, x):
        if self.norm_type == 'pre':
            x = self.fc(self.norm(x))
        elif self.norm_type == 'post':
            x = self.norm(self.fc(x))
        else:
            x = self.fc(x)
        x = F.relu(x)
        self.activation = x.detach()
        return x

class Net(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, n_layers, norm_type):
        super().__init__()
        self.embed = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList([Block(d_hidden, norm_type) for _ in range(n_layers)])
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ── Training loop ─────────────────────────────────────────────────────────────
def train(norm_type, n_layers=6, d_hidden=128, lr=1e-3, steps=800, batch=64):
    (X_tr, y_tr), (X_val, y_val) = make_data(n=3000, d=64, n_classes=10)
    model = Net(64, d_hidden, 10, n_layers, norm_type)
    # Init: use larger init to stress-test normalization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.3)
            nn.init.zeros_(m.bias)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'act_norms': [[] for _ in range(n_layers)],   # per-layer activation norm
        'grad_norms': [[] for _ in range(n_layers)],   # per-layer gradient norm
    }

    n_tr = len(X_tr)
    for step in range(steps):
        model.train()
        idx = torch.randint(0, n_tr, (batch,))
        xb, yb = X_tr[idx], y_tr[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()

        # Measure gradient norms per block
        for i, block in enumerate(model.blocks):
            gn = block.fc.weight.grad.norm().item() if block.fc.weight.grad is not None else 0.0
            history['grad_norms'][i].append(gn)

        opt.step()

        # Measure activation norms per block
        with torch.no_grad():
            _ = model(xb)
            for i, block in enumerate(model.blocks):
                if block.activation is not None:
                    history['act_norms'][i].append(block.activation.norm(dim=-1).mean().item())

        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = F.cross_entropy(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
                history['step'].append(step)
                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

    return history

# ── Run experiments ────────────────────────────────────────────────────────────
print("Running Layer Normalization experiment...")
print("  - No normalization")
hist_none = train('none')
print("  - Pre-norm LayerNorm")
hist_pre = train('pre')
print("  - Post-norm LayerNorm")
hist_post = train('post')

results = {
    'none': hist_none,
    'pre': hist_pre,
    'post': hist_post,
}

# Save raw numbers
with open('experiments/layer-norm/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Figure 1: Activation norms per layer (final step) ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
layer_labels = [f'L{i+1}' for i in range(6)]

for ax, (name, hist, color, label) in zip(axes, [
    ('none', hist_none, '#e74c3c', 'No normalization'),
    ('pre',  hist_pre,  '#2ecc71', 'Pre-norm LayerNorm'),
    ('post', hist_post, '#3498db', 'Post-norm LayerNorm'),
]):
    # Mean activation norm at last 50 steps per layer
    norms_raw = [np.mean(hist['act_norms'][i][-50:]) for i in range(6)]
    norms = [v if np.isfinite(v) else 0.0 for v in norms_raw]
    labels_bar = [f'{v:.2f}' if np.isfinite(v) else 'NaN/∞' for v in norms_raw]
    ax.bar(layer_labels, norms, color=color, alpha=0.85)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean activation norm')
    finite_norms = [v for v in norms if v > 0]
    ylim_top = max(finite_norms)*1.3 + 0.5 if finite_norms else 1.0
    ax.set_ylim(0, ylim_top)
    for j, (v, lbl) in enumerate(zip(norms, labels_bar)):
        ax.text(j, v + 0.02*ylim_top, lbl, ha='center', va='bottom', fontsize=9)

plt.suptitle('Activation Norms per Layer (averaged over last 50 steps)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/layer-norm/figures/activation_norms.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Gradient norms per layer ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, (name, hist, color, label) in zip(axes, [
    ('none', hist_none, '#e74c3c', 'No normalization'),
    ('pre',  hist_pre,  '#2ecc71', 'Pre-norm LayerNorm'),
    ('post', hist_post, '#3498db', 'Post-norm LayerNorm'),
]):
    norms_raw = [np.mean(hist['grad_norms'][i][-50:]) for i in range(6)]
    norms = [v if np.isfinite(v) else 0.0 for v in norms_raw]
    labels_bar = [f'{v:.4f}' if np.isfinite(v) else 'NaN' for v in norms_raw]
    ax.bar(layer_labels, norms, color=color, alpha=0.85)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean gradient norm (layer weight)')
    finite = [v for v in norms if v > 0]
    ylim_top = max(finite)*1.3 + 1e-5 if finite else 1e-4
    ax.set_ylim(0, ylim_top)
    for j, (v, lbl) in enumerate(zip(norms, labels_bar)):
        ax.text(j, v + 0.02*ylim_top, lbl, ha='center', va='bottom', fontsize=8)

plt.suptitle('Gradient Norms per Layer — Layer 1 is earliest (hardest to reach)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/layer-norm/figures/gradient_norms.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: Activation norm over time (layer 1 vs layer 6) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, layer_i, title in zip(axes, [0, 5], ['Layer 1 (earliest)', 'Layer 6 (latest)']):
    for hist, color, label in [
        (hist_none, '#e74c3c', 'No norm'),
        (hist_pre,  '#2ecc71', 'Pre-norm'),
        (hist_post, '#3498db', 'Post-norm'),
    ]:
        steps = list(range(len(hist['act_norms'][layer_i])))
        ax.plot(steps, hist['act_norms'][layer_i], color=color, label=label, alpha=0.8)
    ax.set_title(f'Activation Norm Over Time — {title}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Activation norm')
    ax.legend()

plt.tight_layout()
plt.savefig('experiments/layer-norm/figures/activation_over_time.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 4: Validation loss comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for hist, color, label in [
    (hist_none, '#e74c3c', 'No normalization'),
    (hist_pre,  '#2ecc71', 'Pre-norm LayerNorm'),
    (hist_post, '#3498db', 'Post-norm LayerNorm'),
]:
    axes[0].plot(hist['step'], hist['val_loss'], color=color, label=label, linewidth=2)
    axes[1].plot(hist['step'], [a*100 for a in hist['val_acc']], color=color, label=label, linewidth=2)

axes[0].set_title('Validation Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Step'); axes[0].set_ylabel('Cross-entropy loss')
axes[0].legend()
axes[1].set_title('Validation Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Step'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()

plt.suptitle('Training Dynamics with and without Layer Normalization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/layer-norm/figures/val_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Print summary stats ───────────────────────────────────────────────────────
print("\n=== RESULTS SUMMARY ===")
for name, hist, label in [
    ('none', hist_none, 'No normalization'),
    ('pre',  hist_pre,  'Pre-norm'),
    ('post', hist_post, 'Post-norm'),
]:
    final_loss = hist['val_loss'][-1]
    final_acc  = hist['val_acc'][-1] * 100
    l1_grad    = np.mean(hist['grad_norms'][0][-50:])
    l6_grad    = np.mean(hist['grad_norms'][5][-50:])
    l1_act     = np.mean(hist['act_norms'][0][-50:])
    l6_act     = np.mean(hist['act_norms'][5][-50:])
    ratio      = l6_grad / (l1_grad + 1e-12)
    print(f"\n{label}:")
    print(f"  Final val loss: {final_loss:.4f}  |  Final val acc: {final_acc:.1f}%")
    print(f"  Layer 1 grad norm (last 50 steps): {l1_grad:.6f}")
    print(f"  Layer 6 grad norm (last 50 steps): {l6_grad:.6f}")
    print(f"  Gradient ratio (L6/L1): {ratio:.2f}x")
    print(f"  Layer 1 act norm: {l1_act:.3f}  |  Layer 6 act norm: {l6_act:.3f}")

print("\nDone. Figures saved to experiments/layer-norm/figures/")
