"""
Experiment: Attention Scaling (1/√d)
Goal: Show that without √d scaling, softmax entropy collapses at initialization
and the model fails to learn attention-dependent tasks.

Honest design:
- Selective copy task: given a sequence, copy the token at a designated position
- Single self-attention layer so attention patterns are directly measurable
- Vary dimension d to show the problem worsens with scale
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

# ── Selective copy task ────────────────────────────────────────────────────────
# Sequence: [signal_pos (0-indexed), tok0, tok1, ..., tok_{L-1}]
# Target: tok_{signal_pos}
# The model must learn to attend to position signal_pos+1 in the sequence.

def make_copy_batch(batch_size, seq_len=8, vocab_size=8, device='cpu'):
    """
    Returns (x, target) where:
    - x shape: (batch, seq_len+1) token ids
    - x[:,0] = the position to copy (0..seq_len-1)
    - x[:,1:] = random tokens
    - target = x[range(batch), x[:,0]+1]
    """
    pos = torch.randint(0, seq_len, (batch_size,), device=device)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    x = torch.cat([pos.unsqueeze(1), tokens], dim=1)  # (batch, seq_len+1)
    target = tokens[torch.arange(batch_size), pos]     # (batch,)
    return x, target

# ── Single-head attention model ───────────────────────────────────────────────
class SingleHeadAttnModel(nn.Module):
    def __init__(self, vocab_size, seq_len, d, n_classes, scale=True):
        super().__init__()
        self.d = d
        self.scale = scale
        self.embed = nn.Embedding(vocab_size + seq_len, d)  # combined vocab for tokens and pos ids
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        # x: (batch, T) where T = seq_len + 1
        B, T = x.shape
        emb = self.embed(x)            # (B, T, d)
        Q = self.Wq(emb)               # (B, T, d)
        K = self.Wk(emb)               # (B, T, d)
        V = self.Wv(emb)               # (B, T, d)

        scores = torch.bmm(Q, K.transpose(1, 2))  # (B, T, T)
        if self.scale:
            scores = scores / math.sqrt(self.d)

        attn = torch.softmax(scores, dim=-1)       # (B, T, T)
        out = torch.bmm(attn, V)                   # (B, T, d)

        # Classify using the output at position 0 (the query position token)
        logits = self.head(out[:, 0, :])           # (B, n_classes)
        return logits, attn

    def attention_entropy(self, x):
        """Returns mean entropy of attention distribution at position 0."""
        with torch.no_grad():
            _, attn = self.forward(x)
        attn_row = attn[:, 0, :]   # (B, T) — attention from the query position
        # entropy: -sum(p * log(p + eps))
        entropy = -(attn_row * (attn_row + 1e-9).log()).sum(dim=-1)
        return entropy.mean().item()

# ── Train one model, tracking entropy and accuracy ────────────────────────────
def train_attn(d, scale, steps=1000, batch=128, lr=3e-4, seq_len=8, vocab_size=8):
    model = SingleHeadAttnModel(vocab_size, seq_len, d, vocab_size, scale=scale)
    # Standard init
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'step': [],
        'loss': [],
        'acc': [],
        'entropy': [],
        'init_entropy': None,
    }

    # Measure entropy at init (before any training)
    x_test, _ = make_copy_batch(256, seq_len, vocab_size)
    history['init_entropy'] = model.attention_entropy(x_test)

    for step in range(steps):
        model.train()
        x, target = make_copy_batch(batch, seq_len, vocab_size)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0:
            model.eval()
            x_val, target_val = make_copy_batch(512, seq_len, vocab_size)
            with torch.no_grad():
                logits_val, _ = model(x_val)
                val_loss = F.cross_entropy(logits_val, target_val).item()
                acc = (logits_val.argmax(1) == target_val).float().mean().item()
            ent = model.attention_entropy(x_val)
            history['step'].append(step)
            history['loss'].append(val_loss)
            history['acc'].append(acc)
            history['entropy'].append(ent)

    return history

# ── Experiment A: d=64, scale vs no-scale ─────────────────────────────────────
print("Running Attention Scaling experiment...")
print("  Experiment A: d=64, with vs without √d scaling")
hist_scale   = train_attn(d=64, scale=True)
hist_noscale = train_attn(d=64, scale=False)

# ── Experiment B: vary d, measure init entropy ─────────────────────────────────
print("  Experiment B: varying d, measuring init entropy")
dims = [8, 16, 32, 64, 128, 256]
init_entropy_scale   = []
init_entropy_noscale = []
max_entropy = {}

for d in dims:
    T = 9  # seq_len+1
    max_ent = math.log(T)
    max_entropy[d] = max_ent
    # Measure entropy at init (average over 5 seeds)
    ents_s, ents_ns = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        m_s  = SingleHeadAttnModel(8, 8, d, 8, scale=True)
        m_ns = SingleHeadAttnModel(8, 8, d, 8, scale=False)
        for m in [m_s, m_ns]:
            for mod in m.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
        x_test, _ = make_copy_batch(512)
        ents_s.append(m_s.attention_entropy(x_test))
        ents_ns.append(m_ns.attention_entropy(x_test))
    init_entropy_scale.append(np.mean(ents_s))
    init_entropy_noscale.append(np.mean(ents_ns))
    print(f"    d={d:4d}: scale={np.mean(ents_s):.3f}  no-scale={np.mean(ents_ns):.3f}  max={max_ent:.3f}")

# ── Experiment C: final accuracy across dims ──────────────────────────────────
print("  Experiment C: accuracy across d for scale vs no-scale")
final_acc_scale, final_acc_noscale = [], []
for d in dims:
    print(f"    d={d}")
    torch.manual_seed(SEED)
    h_s  = train_attn(d=d, scale=True,  steps=600, batch=128)
    torch.manual_seed(SEED)
    h_ns = train_attn(d=d, scale=False, steps=600, batch=128)
    final_acc_scale.append(h_s['acc'][-1])
    final_acc_noscale.append(h_ns['acc'][-1])

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'hist_scale': hist_scale,
    'hist_noscale': hist_noscale,
    'dims': dims,
    'init_entropy_scale': init_entropy_scale,
    'init_entropy_noscale': init_entropy_noscale,
    'final_acc_scale': final_acc_scale,
    'final_acc_noscale': final_acc_noscale,
    'max_entropy_d64': math.log(9),
}
with open('experiments/attention-scaling/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Figure 1: Init entropy vs dimension ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
max_ent_line = [math.log(9)] * len(dims)
ax.plot(dims, max_ent_line, 'k--', linewidth=1.5, label=f'Maximum entropy (uniform), log(9)={math.log(9):.2f}')
ax.plot(dims, init_entropy_scale,   'o-', color='#2ecc71', linewidth=2, markersize=8, label='With √d scaling')
ax.plot(dims, init_entropy_noscale, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Without scaling')

for i, d in enumerate(dims):
    ax.annotate(f'{init_entropy_scale[i]:.2f}',   (d, init_entropy_scale[i]),   textcoords='offset points', xytext=(0,8),  ha='center', fontsize=9, color='#2ecc71')
    ax.annotate(f'{init_entropy_noscale[i]:.2f}', (d, init_entropy_noscale[i]), textcoords='offset points', xytext=(0,-14), ha='center', fontsize=9, color='#e74c3c')

ax.set_xlabel('Embedding dimension d', fontsize=12)
ax.set_ylabel('Attention entropy at initialization', fontsize=12)
ax.set_title('Softmax Entropy at Initialization vs Dimension\n(Higher = more uniform, better for learning)', fontsize=12, fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(dims)
ax.set_xticklabels(dims)
ax.legend(fontsize=10)
ax.set_ylim(0, math.log(9)*1.15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/attention-scaling/figures/init_entropy_vs_dim.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Training curves (d=64) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for hist, color, label in [
    (hist_scale,   '#2ecc71', f'With √d scaling (init entropy={hist_scale["init_entropy"]:.2f})'),
    (hist_noscale, '#e74c3c', f'Without scaling (init entropy={hist_noscale["init_entropy"]:.2f})'),
]:
    axes[0].plot(hist['step'], hist['loss'],              color=color, label=label, linewidth=2)
    axes[1].plot(hist['step'], [a*100 for a in hist['acc']],   color=color, label=label, linewidth=2)
    axes[2].plot(hist['step'], hist['entropy'],            color=color, label=label, linewidth=2)

axes[0].set_title('Validation Loss', fontweight='bold')
axes[0].set_xlabel('Step'); axes[0].set_ylabel('Cross-entropy loss'); axes[0].legend(fontsize=8)
axes[1].set_title('Task Accuracy (Selective Copy)', fontweight='bold')
axes[1].set_xlabel('Step'); axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(fontsize=8)
chance = 100/8
axes[1].axhline(chance, color='gray', linestyle='--', linewidth=1, label=f'Chance ({chance:.1f}%)')
axes[2].set_title('Attention Entropy Over Training', fontweight='bold')
axes[2].set_xlabel('Step'); axes[2].set_ylabel('Entropy')
axes[2].axhline(math.log(9), color='k', linestyle='--', linewidth=1, label=f'Max entropy={math.log(9):.2f}')
axes[2].legend(fontsize=8)

plt.suptitle('Attention Scaling Experiment (d=64, Selective Copy Task)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('experiments/attention-scaling/figures/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: Final accuracy vs dimension ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(dims))
w = 0.35
bars1 = ax.bar(x - w/2, [a*100 for a in final_acc_scale],   w, color='#2ecc71', alpha=0.85, label='With √d scaling')
bars2 = ax.bar(x + w/2, [a*100 for a in final_acc_noscale], w, color='#e74c3c', alpha=0.85, label='Without scaling')
ax.axhline(100/8, color='gray', linestyle='--', linewidth=1.5, label='Chance (12.5%)')
ax.set_xticks(x); ax.set_xticklabels([f'd={d}' for d in dims])
ax.set_ylabel('Final Accuracy (%)', fontsize=12)
ax.set_title('Final Task Accuracy vs Embedding Dimension\n(After 600 training steps)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 110)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.0f}%', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.0f}%', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('experiments/attention-scaling/figures/accuracy_vs_dim.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== RESULTS SUMMARY ===")
print(f"Max possible entropy (seq_len=8+1=9): {math.log(9):.3f}")
print(f"\nAt initialization (d=64):")
print(f"  With scaling:    entropy = {hist_scale['init_entropy']:.4f}")
print(f"  Without scaling: entropy = {hist_noscale['init_entropy']:.4f}")
print(f"\nAfter training (d=64):")
print(f"  With scaling:    acc = {hist_scale['acc'][-1]*100:.1f}%")
print(f"  Without scaling: acc = {hist_noscale['acc'][-1]*100:.1f}%")
print(f"  Chance:          acc = {100/8:.1f}%")
print(f"\nInit entropy vs dimension:")
for d, es, ens in zip(dims, init_entropy_scale, init_entropy_noscale):
    print(f"  d={d:4d}: scale={es:.3f}  no-scale={ens:.3f}")
print(f"\nFinal accuracy vs dimension:")
for d, as_, ans in zip(dims, final_acc_scale, final_acc_noscale):
    print(f"  d={d:4d}: scale={as_*100:.1f}%  no-scale={ans*100:.1f}%")
print("\nDone. Figures saved to experiments/attention-scaling/figures/")
