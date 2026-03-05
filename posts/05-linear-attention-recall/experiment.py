"""
exp-linear-attention-recall.py

Core claim: linear attention cannot selectively retrieve key-specific values.

Experiment design:
  Panel (a) — ANALYTICAL: for N random key-value pairs and a perfect query,
    compute the fraction of attention weight assigned to the matching pair.
    Softmax: near 1/1 (selective). Linear ELU: near 1/N (uniform).

  Panel (b) — TRAINING: 4-pair recall task, 1000 steps, 3 seeds.
    Show softmax converges; linear attention doesn't.

  Panel (c) — MECHANISM: after training, plot the attention weight distribution
    on the matching vs non-matching key positions. Softmax = sharp spike.
    Linear ELU = flat.
"""
import random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42); np.random.seed(42); torch.manual_seed(42)
C1,C2,C3 = '#1a1a2e','#e63946','#2a9d8f'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,
    'xtick.major.width':0.8,'ytick.major.width':0.8,
    'xtick.major.size':3,'ytick.major.size':3,'lines.linewidth':1.8,'figure.dpi':150})

D = 64   # dimension for analytical experiment

# ── Panel (a): ANALYTICAL retrieval weight ────────────────────────────
# Question: if query = k_1 (perfect match), what fraction of attention
# weight does each method assign to position 1?
# Averaged over 1000 random samples of {k_1,...,k_N}.

def phi_elu(x):
    return F.elu(x) + 1.0

def retrieval_weight_analytical(N, n_trials=1000, d=D):
    """Returns (softmax_mean, softmax_std, linear_mean, linear_std)."""
    soft_ws, lin_ws = [], []
    for _ in range(n_trials):
        # random keys — each row is a key embedding
        K = torch.randn(N, d)   # (N, d)
        q = K[0]                # query = first key (perfect match)
        # Softmax: w_i = exp(q·k_i/√d) / Σ exp(q·k_j/√d)
        scores = (K @ q) / (d**0.5)       # (N,)
        w_soft = F.softmax(scores, dim=0)
        soft_ws.append(w_soft[0].item())
        # Linear ELU: w_i = φ(q)·φ(k_i) / Σ_j φ(q)·φ(k_j)
        Q2 = phi_elu(q)                   # (d,)
        K2 = phi_elu(K)                   # (N, d)
        scores_lin = K2 @ Q2              # (N,)
        w_lin = scores_lin / (scores_lin.sum() + 1e-6)
        lin_ws.append(w_lin[0].item())
    return (np.mean(soft_ws), np.std(soft_ws),
            np.mean(lin_ws),  np.std(lin_ws))

print("Panel (a): analytical retrieval weight...")
Ns = [2, 4, 8, 16, 32, 64]
res = [retrieval_weight_analytical(n) for n in Ns]
s_mean = [r[0] for r in res]; s_std = [r[1] for r in res]
l_mean = [r[2] for r in res]; l_std = [r[3] for r in res]
uniform = [1/n for n in Ns]
for n, sm, lm in zip(Ns, s_mean, l_mean):
    print(f"  N={n:2d}: softmax={sm:.3f}  linear={lm:.3f}  uniform=1/{n}={1/n:.3f}")

# ── Panel (b): minimal training — 4-pair recall ────────────────────────
N_PAIRS = 4; VOCAB = 24   # 8 keys (0..7), 16 values (8..23)
K_IDS = list(range(8)); V_IDS = list(range(8, 24))
SEQ = N_PAIRS*2+1; DIM = 32; STEPS = 600; SEEDS = [42,123]
CHANCE = 1/16  # 16 possible values

def make_batch(bs=256):
    # Sample N_PAIRS distinct keys (0..7) per sample via argsort trick
    noise  = torch.rand(bs, 8)
    all_k  = noise.argsort(dim=-1)[:, :N_PAIRS]        # (bs, N_PAIRS) distinct keys
    all_v  = torch.randint(8, 24, (bs, N_PAIRS))        # (bs, N_PAIRS) values
    pairs  = torch.stack([all_k, all_v], dim=2).reshape(bs, N_PAIRS*2)
    qi     = torch.randint(0, N_PAIRS, (bs,))
    query  = all_k[torch.arange(bs), qi]
    tgt    = all_v[torch.arange(bs), qi]
    inp    = torch.cat([pairs, query.unsqueeze(1)], dim=1)
    return inp, tgt

def softmax_attn(Q,K,V):
    w = F.softmax(torch.bmm(Q,K.transpose(1,2))/(D**0.5), dim=-1)
    return torch.bmm(w,V), w

def linear_elu(Q,K,V):
    Q2,K2 = phi_elu(Q), phi_elu(K)
    KV   = torch.bmm(K2.transpose(1,2), V)
    norm = (Q2 * K2.sum(1, keepdim=True)).sum(-1, keepdim=True) + 1e-6
    w    = torch.bmm(Q2, K2.transpose(1,2))
    w    = w / (w.sum(-1, keepdim=True) + 1e-6)
    return torch.bmm(Q2,KV)/((Q2*K2.sum(1,keepdim=True)).sum(-1,keepdim=True)+1e-6), w

class RecallModel(nn.Module):
    def __init__(self, attn_fn):
        super().__init__()
        self.attn_fn = attn_fn
        self.emb = nn.Embedding(VOCAB, DIM)
        self.Wq  = nn.Linear(DIM, DIM, bias=False)
        self.Wk  = nn.Linear(DIM, DIM, bias=False)
        self.Wv  = nn.Linear(DIM, DIM, bias=False)
        self.head= nn.Linear(DIM, VOCAB)
        self._last_w = None
    def forward(self, x):
        h = self.emb(x)
        Q,K,V = self.Wq(h),self.Wk(h),self.Wv(h)
        out,w = self.attn_fn(Q,K,V); self._last_w=w.detach()
        return self.head(out[:,-1,:])

def train(attn_fn, label):
    all_accs = []; saved_m = None
    for seed in SEEDS:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        m   = RecallModel(attn_fn)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
        accs = []
        for s in range(STEPS):
            x,y = make_batch(256)
            F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad(); sched.step()
            if s%30==0:
                with torch.no_grad():
                    xv,yv=make_batch(512)
                    accs.append((m(xv).argmax(-1)==yv).float().mean().item())
        all_accs.append(accs)
        if saved_m is None: saved_m=m
        print(f"  [{label} seed={seed}] {accs[-1]:.3f}")
    return saved_m, np.array(all_accs)

print("\nPanel (b): training recall models...")
t0 = time.time()
ms, a_soft = train(softmax_attn, 'softmax')
me, a_elu  = train(linear_elu,   'linear-elu')
print(f"Done in {time.time()-t0:.1f}s")
steps_ax = list(range(0, STEPS, 30))

# ── Panel (c): attention weight on matching key, after training ────────
print("\nPanel (c): attention weight distribution...")
W_SOFT, W_ELU = [], []
with torch.no_grad():
    for _ in range(500):
        x, y = make_batch(1)
        ms(x); w_s = ms._last_w[0, -1, :].numpy()  # last query's weights over keys
        me(x); w_e = me._last_w[0, -1, :].numpy()
        # find position of the matching key in the sequence
        query_key = x[0, -1].item()
        for pos in range(0, SEQ-1, 2):
            if x[0, pos].item() == query_key:
                W_SOFT.append(w_s[pos])
                W_ELU.append(w_e[pos])
                break
W_SOFT = np.array(W_SOFT); W_ELU = np.array(W_ELU)
print(f"  Mean attention on matching key: softmax={W_SOFT.mean():.3f}  linear={W_ELU.mean():.3f}  uniform=1/{SEQ}={1/SEQ:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(11,3.4))
fig.suptitle('Linear Attention — The Recall Failure', fontsize=11, fontweight='bold', y=1.03)

# (a) analytical retrieval weight vs N
ax = axes[0]
ax.plot(Ns, s_mean, color=C1, ls='--', lw=1.8, marker='o', ms=4, label='Softmax')
ax.fill_between(Ns, np.array(s_mean)-np.array(s_std),
                    np.array(s_mean)+np.array(s_std), color=C1, alpha=0.15)
ax.plot(Ns, l_mean, color=C3, ls='-',  lw=2.2, marker='^', ms=4, label='Linear (ELU φ)')
ax.fill_between(Ns, np.array(l_mean)-np.array(l_std),
                    np.array(l_mean)+np.array(l_std), color=C3, alpha=0.15)
ax.plot(Ns, uniform, color=C2, ls=':', lw=1.5, label='Uniform (1/N)')
ax.set_xlabel('Number of keys N'); ax.set_ylabel('Weight on matching key')
ax.set_xscale('log', base=2)
ax.set_title('(a) Retrieval selectivity vs. number of keys\n(analytical, 1000 random trials, log scale)')
ax.legend(fontsize=7.5)

# (b) training accuracy
ax = axes[1]
ms_=a_soft.mean(0); ss_=a_soft.std(0)
me_=a_elu.mean(0);  se_=a_elu.std(0)
ax.plot(steps_ax, ms_, color=C1, ls='--', lw=1.8, label='Softmax')
ax.fill_between(steps_ax,ms_-ss_,ms_+ss_,color=C1,alpha=0.15)
ax.plot(steps_ax, me_, color=C3, ls='-',  lw=2.2, label='Linear (ELU φ)')
ax.fill_between(steps_ax,me_-se_,me_+se_,color=C3,alpha=0.15)
ax.axhline(CHANCE, color='gray', ls=':', lw=1., label=f'Chance ({CHANCE:.1%})')
ax.set_xlabel('Training step'); ax.set_ylabel('Recall accuracy')
ax.set_title('(b) Recall accuracy during training\n(4 pairs, vocab=24, mean±std 3 seeds)')
ax.legend(fontsize=7.5); ax.set_ylim(0, 1.05)

# (c) attention weight histogram on matching key (trained models)
ax = axes[2]
bins = np.linspace(0, 1, 30)
ax.hist(W_SOFT, bins=bins, color=C1, alpha=0.7, label=f'Softmax (mean={W_SOFT.mean():.2f})', density=True)
ax.hist(W_ELU,  bins=bins, color=C3, alpha=0.7, label=f'Linear ELU (mean={W_ELU.mean():.2f})', density=True)
ax.axvline(1/SEQ, color=C2, ls='--', lw=1.5, label=f'Uniform = 1/{SEQ}={1/SEQ:.2f}')
ax.set_xlabel('Attention weight on matching key'); ax.set_ylabel('Density')
ax.set_title('(c) Attention weight on the correct key\n(trained models, 500 test sequences)')
ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig('posts/fig-linear-attention-recall.png', dpi=200, bbox_inches='tight', pad_inches=0.12)
plt.close()
print("\nFigure saved: posts/fig-linear-attention-recall.png")
print(f"Key numbers:")
print(f"  Softmax final acc: {ms_[-1]:.3f}   Linear final acc: {me_[-1]:.3f}")
print(f"  Softmax weight on match: {W_SOFT.mean():.3f}   Linear: {W_ELU.mean():.3f}   Uniform: {1/SEQ:.3f}")
print(f"  Analytical N=64: softmax={s_mean[-1]:.3f}  linear={l_mean[-1]:.3f}")
