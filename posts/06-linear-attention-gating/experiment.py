"""
exp-linear-attention-gating.py — needle-in-haystack gating experiment

Task: [MARK, VALUE, noise*L, QUERY] → output VALUE
Tests whether gating protects memory across L noise tokens.

Conditions:
  1. Vanilla RNN (no gate)
  2. Fixed gate (g=0.9)
  3. Data-dependent gate g_t=σ(Wg·x_t)  [GLA style]
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

MARK=0; QUERY=1; VAL_START=2; N_VAL=8; NOISE_START=VAL_START+N_VAL
VOCAB=NOISE_START+N_VAL; DIM=16; NOISE_LEN=16; SEEDS=[42,123]; CHANCE=1/N_VAL

def make_batch(bs=256, nl=NOISE_LEN):
    T = 3+nl
    inp = torch.zeros(bs, T, dtype=torch.long)
    inp[:,0] = MARK
    inp[:,1] = torch.randint(VAL_START, VAL_START+N_VAL, (bs,))
    inp[:,2:2+nl] = torch.randint(NOISE_START, VOCAB, (bs, nl))
    inp[:,-1] = QUERY
    return inp, inp[:,1]

class VanillaRNN(nn.Module):
    def __init__(self, dim=DIM, vocab=VOCAB):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim); self.W=nn.Linear(dim,dim)
        self.U=nn.Linear(dim,dim,bias=False); self.head=nn.Linear(dim,vocab)
        self._gates=None
    def forward(self,x):
        B,T=x.shape; h=torch.zeros(B,DIM); gl=[]
        for t in range(T): h=torch.tanh(self.W(self.emb(x[:,t]))+self.U(h)); gl.append(torch.ones(B))
        self._gates=torch.stack(gl,1); return self.head(h)

class FixedGateRNN(nn.Module):
    def __init__(self, dim=DIM, vocab=VOCAB, g=0.9):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim); self.W=nn.Linear(dim,dim)
        self.head=nn.Linear(dim,vocab); self.g=g; self._gates=None
    def forward(self,x):
        B,T=x.shape; h=torch.zeros(B,DIM); gl=[]
        for t in range(T): h=self.g*h+(1-self.g)*torch.tanh(self.W(self.emb(x[:,t]))); gl.append(torch.full((B,),self.g))
        self._gates=torch.stack(gl,1); return self.head(h)

class GatedRNN(nn.Module):
    def __init__(self, dim=DIM, vocab=VOCAB):
        super().__init__()
        self.emb=nn.Embedding(vocab,dim); self.W=nn.Linear(dim,dim)
        self.Wg=nn.Linear(dim,1); self.head=nn.Linear(dim,vocab); self._gates=None
    def forward(self,x):
        B,T=x.shape; h=torch.zeros(B,DIM); gl=[]
        for t in range(T):
            e=self.emb(x[:,t]); g=torch.sigmoid(self.Wg(e)).squeeze(-1)
            h=g.unsqueeze(-1)*h+(1-g).unsqueeze(-1)*torch.tanh(self.W(e)); gl.append(g.detach())
        self._gates=torch.stack(gl,1); return self.head(h)

STEPS=300; LR=5e-3

def train(Cls, kw, label):
    all_accs=[]; saved=None
    for seed in SEEDS:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        m=Cls(**kw); opt=torch.optim.Adam(m.parameters(),lr=LR); accs=[]
        for s in range(STEPS):
            x,y=make_batch(256); F.cross_entropy(m(x),y).backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); opt.zero_grad()
            if s%15==0:
                with torch.no_grad(): xv,yv=make_batch(512); accs.append((m(xv).argmax(-1)==yv).float().mean().item())
        all_accs.append(accs); saved=saved or m
        print(f'  [{label} s={seed}] final={all_accs[-1][-1]:.3f}')
    return saved, np.array(all_accs)

print("Training RNN models (needle-in-haystack, noise_len=16)...")
t0=time.time()
mv, a_van = train(VanillaRNN,    {},        'vanilla')
mf, a_fix = train(FixedGateRNN,  {'g':0.9}, 'fixed')
md, a_dep = train(GatedRNN,      {},        'dep')
print(f"Done in {time.time()-t0:.1f}s")
steps_ax=list(range(0,STEPS,15))

# scaling
print("Scaling (noise_len)...")
noise_lens=[2,4,8,12,16]
def quick(Cls,kw,nl,steps=120):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    m=Cls(**kw); opt=torch.optim.Adam(m.parameters(),lr=LR)
    for _ in range(steps):
        x,y=make_batch(256,nl); F.cross_entropy(m(x),y).backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); opt.zero_grad()
    with torch.no_grad(): xv,yv=make_batch(512,nl); return (m(xv).argmax(-1)==yv).float().mean().item()
sv=[quick(VanillaRNN,{},nl) for nl in noise_lens]
sf=[quick(FixedGateRNN,{'g':0.9},nl) for nl in noise_lens]
sd=[quick(GatedRNN,{},nl) for nl in noise_lens]
for nl,v,f,d in zip(noise_lens,sv,sf,sd): print(f'  nl={nl}: van={v:.3f} fix={f:.3f} dep={d:.3f}')

# gate visualization
print("Gate visualization...")
torch.manual_seed(42); random.seed(42)
vis=GatedRNN(); vopt=torch.optim.Adam(vis.parameters(),lr=LR)
for _ in range(400):
    x,y=make_batch(256); F.cross_entropy(vis(x),y).backward()
    nn.utils.clip_grad_norm_(vis.parameters(),1.); vopt.step(); vopt.zero_grad()
with torch.no_grad():
    xv,_=make_batch(1); vis(xv)
    gates_vis=vis._gates[0].numpy(); T_vis=NOISE_LEN+3
    mark_pos=[0]; noise_pos=list(range(2,2+NOISE_LEN)); query_pos=[NOISE_LEN+2]

# ── plot ──────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(11,3.4))
fig.suptitle('Gated RNN — Needle-in-Haystack Memory',fontsize=11,fontweight='bold',y=1.03)

# (a) gate visualization
ax=axes[0]
t_ax=np.arange(T_vis)
ax.axhline(1.0,color=C1,ls='--',lw=1.5,alpha=0.7,label='Vanilla (no gate)')
ax.axhline(0.9,color=C2,ls='-.',lw=1.5,alpha=0.7,label='Fixed gate (g=0.9)')
ax.plot(t_ax,gates_vis,color=C3,lw=1.8,label='Data-dep gate g_t')
ax.axvline(0.5,color='gray',lw=0.5,alpha=0.4,ls=':')
ax.text(0,-0.08,'MARK',ha='center',fontsize=7,color='gray',transform=ax.get_xaxis_transform())
ax.text(2+NOISE_LEN//2,-0.08,'noise tokens',ha='center',fontsize=7,color='gray',transform=ax.get_xaxis_transform())
ax.text(T_vis-1,-0.08,'QUERY',ha='center',fontsize=7,color='gray',transform=ax.get_xaxis_transform())
ax.set_xlabel('Token position'); ax.set_ylabel('Gate value g_t')
ax.set_title('(a) Learned gate activity\n(trained data-dep model, one sequence)')
ax.legend(fontsize=7.5,loc='center right'); ax.set_ylim(-0.05,1.15)

# (b) training accuracy
ax=axes[1]
mv_=a_van.mean(0); sv_=a_van.std(0)
mf_=a_fix.mean(0); sf_=a_fix.std(0)
md_=a_dep.mean(0); sd_=a_dep.std(0)
ax.plot(steps_ax,mv_,color=C1,ls='--',lw=1.8,label='Vanilla RNN')
ax.fill_between(steps_ax,mv_-sv_,mv_+sv_,color=C1,alpha=0.15)
ax.plot(steps_ax,md_,color=C2,ls='-.',lw=1.8,label='Data-dep gate')
ax.fill_between(steps_ax,md_-sd_,md_+sd_,color=C2,alpha=0.15)
ax.plot(steps_ax,mf_,color=C3,ls='-', lw=2.2,label='Fixed gate (g=0.9)')
ax.fill_between(steps_ax,mf_-sf_,mf_+sf_,color=C3,alpha=0.15)
ax.axhline(CHANCE,color='gray',ls=':',lw=1.,label='Chance (1/8)')
ax.set_xlabel('Training step'); ax.set_ylabel('Accuracy')
ax.set_title('(b) Accuracy during training\n(noise_len=16, mean±std 2 seeds)')
ax.legend(fontsize=7.5); ax.set_ylim(0,1.05)

# (c) scaling
ax=axes[2]
ax.plot(noise_lens,sv,color=C1,ls='--',lw=1.8,marker='o',ms=4,label='Vanilla RNN')
ax.plot(noise_lens,sd,color=C2,ls='-.',lw=1.8,marker='s',ms=4,label='Data-dep gate')
ax.plot(noise_lens,sf,color=C3,ls='-', lw=2.2,marker='^',ms=4,label='Fixed gate (g=0.9)')
ax.axhline(CHANCE,color='gray',ls=':',lw=1.,label='Chance')
ax.set_xlabel('Noise length L (tokens between value and query)')
ax.set_ylabel('Accuracy')
ax.set_title('(c) Accuracy vs. number of noise tokens\n(120 training steps each)')
ax.legend(fontsize=7.5); ax.set_ylim(0,1.05)

plt.tight_layout()
plt.savefig('posts/fig-linear-attention-gating.png',dpi=200,bbox_inches='tight',pad_inches=0.12)
plt.close()
print("Figure saved: posts/fig-linear-attention-gating.png")
print(f"\nKey numbers:")
print(f"  Final acc (nl=16): van={mv_[-1]:.3f} fix={mf_[-1]:.3f} dep={md_[-1]:.3f}  chance={CHANCE:.3f}")
print(f"  Scaling nl=8: van={sv[2]:.3f} fix={sf[2]:.3f}")
