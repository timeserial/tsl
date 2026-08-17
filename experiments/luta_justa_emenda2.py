#!/usr/bin/env python3
"""AMENDMENT No. 2 (pre-registered before any result): a real fair fight,
now with the anti-basin recipe on BOTH sides.

Changes relative to Queue 13, all declared here:
- Training (both sides): 2 replicas per measurement, chosen by the TRAINING
  ERROR (never validation nor test) - the recipe that worked in Queue 12.
- Our training: lr annealing 0.1→0.02 (0.99^ep). Selection 150 ep; finals 400.
- Adversary: Adam as always; same 2 replicas per training run. 150 ep.
- Configuration selection: mean VALIDATION NRMSE over 2 seeds (1-seed
  selection is what collapsed in Queue 13).
- Reduced and declared spaces: us {(16,·),(24,·)} with lr already built into
  the annealing -> {top 16, top 24}; adversary {gru h16, gru h32} × lr
  {1e-3, 3e-3} (the winners of Queue 13's search; lstm and h64 were never
  close).
- Finals: winner of each side, 5 seeds, TEST set untouched. Decision at 1σ.
"""
import sys; sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, torch, torch.nn as nn
from dataclasses import replace
from pcnet import PCConfig, PCNetwork
from pcnet.dtypes import F
from pcnet.datasets import load_csv_column, load_uci_har_inertial
import os
S=os.environ.get("PCNET_DATA","/private/tmp/claude-501/-Users-brunofitascustodio-Code-nn/de5187af-3636-40ee-9044-c6e4e37ed175/scratchpad")
def nrmse(p,t): return float(np.sqrt(np.mean((p-t)**2)/np.mean(t**2)))
def sigm(a): return 1.0/(1.0+np.exp(-a))
DOM=[("ETTm1",load_csv_column(f"{S}/data/ETTm1.csv","OT")),
     ("acc",load_uci_har_inertial(f"{S}/har/UCI HAR Dataset","total_acc_x")),
     ("gyro",load_uci_har_inertial(f"{S}/har/UCI HAR Dataset","body_gyro_z"))]
NTR=min(len(d.train) for _,d in DOM); NV=max(24,int(0.15*NTR)); NT=NTR-NV
block=16
def inter(frames_list, n):
    nb=n//block
    return np.concatenate([fr[:n][b*block:(b+1)*block] for b in range(nb) for fr in frames_list])
tr_frames=[d.train for _,d in DOM]
mix_sel=inter([f[:NT] for f in tr_frames], NT)
mix_full=inter([f[:NTR] for f in tr_frames], NTR)
val_pairs=[(d.train[:NT], d.train[NT:NTR]) for _,d in DOM]
test_pairs=[(d.train[:NTR], d.test) for _,d in DOM]

def train_ours_once(top, data, sd, epochs):
    net=PCNetwork(PCConfig(seed=sd, fast_path=False, use_precision=True,
                           a_lr=0.0, sizes=(64,top), gated_transition=True))
    g_=net.gated; W0=net.layers[0].W
    for ep in range(epochs):
        lr=0.02+0.08*(0.99**ep)
        for fr in data:
            s=net._z_prev[net.L].copy(); net.step(fr, learn=False)
            c=np.tanh(g_.A@s); gg=sigm(g_.G@s+g_.b); prior=(1-gg)*s+gg*c
            e=fr-W0@prior; h=W0.T@e
            ns=float(s@s)+1e-6; npr=float(prior@prior)+1e-6
            W0+=F(lr/npr)*np.outer(e,prior).astype(F)
            g_.A+=F(lr/ns)*np.outer(h*gg*(1-c*c),s).astype(F)
            g_.G+=F(lr/ns)*np.outer(h*(c-s)*gg*(1-gg),s).astype(F)
            g_.b+=F(lr)*(h*(c-s)*gg*(1-gg)).astype(F)
            net.layers[0].refresh_device()
    net.reset(); errs=[]
    for fr in data[:300]:
        errs.append(float(np.mean((fr-net.predict_next())**2))); net.step(fr, learn=False)
    return net, float(np.mean(errs))

def train_ours(top, data, sd, epochs):
    cands=[train_ours_once(top, data, 1000*sd+k, epochs) for k in range(2)]
    return min(cands, key=lambda t:t[1])[0]

def eval_ours(net, pairs):
    out=[]; b=net.cfg; net.cfg=replace(b, max_iters=50, theta=0.0, settle_min_gain=0.0)
    for warm,target in pairs:
        snap=net.snapshot_state(); net.reset()
        for f in warm[-32:]: net.step(f, learn=False)
        P=[]
        for f in target: P.append(net.predict_next()); net.step(f, learn=False)
        out.append(nrmse(np.array(P,dtype=F),target)); net.restore_state(snap)
    net.cfg=b; return float(np.mean(out))

class R(nn.Module):
    def __init__(s,h):
        super().__init__(); s.rnn=nn.GRU(64,h,batch_first=True); s.head=nn.Linear(h,64)
    def forward(s,x): o,_=s.rnn(x.unsqueeze(0)); return s.head(o.squeeze(0))

def train_adv_once(h, lr, data, sd, epochs=150):
    torch.manual_seed(sd); m=R(h)
    opt=torch.optim.Adam(m.parameters(), lr=lr)
    x=torch.tensor(data,dtype=torch.float32)
    for _ in range(epochs):
        loss=((m(x[:-1])-x[1:])**2).mean(); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    with torch.no_grad():
        tr_err=float(((m(x[:-1])-x[1:])**2).mean())
    return m, tr_err

def train_adv(h, lr, data, sd, epochs=150):
    cands=[train_adv_once(h, lr, data, 1000*sd+k, epochs) for k in range(2)]
    return min(cands, key=lambda t:t[1])[0]

def eval_adv(m, pairs):
    out=[]
    with torch.no_grad():
        for warm,target in pairs:
            seq=torch.tensor(np.concatenate([warm[-32:],target]),dtype=torch.float32)
            out.append(nrmse(m(seq[:-1])[31:].numpy(),target))
    return float(np.mean(out))

if __name__=="__main__":
    print("AMENDMENT 2 - selection (mean val over 2 seeds, anti-basin recipe on both sides)", flush=True)
    best_o=(9,None)
    for top in (16,24):
        vs=[eval_ours(train_ours(top, mix_sel, sd, 150), val_pairs) for sd in (0,1)]
        v=float(np.mean(vs)); print(f"  us top={top}: val={v:.3f} {vs}", flush=True)
        if v<best_o[0]: best_o=(v,top)
    best_a=(9,None)
    for h in (16,32):
        for lr in (1e-3,3e-3):
            vs=[eval_adv(train_adv(h,lr,mix_sel,sd), val_pairs) for sd in (0,1)]
            v=float(np.mean(vs)); print(f"  adv h={h} lr={lr}: val={v:.3f} {vs}", flush=True)
            if v<best_a[0]: best_a=(v,(h,lr))
    print(f"winners: us top={best_o[1]} | adv {best_a[1]}\n", flush=True)
    print("FINALS - 5 seeds, test untouched", flush=True)
    ours=[eval_ours(train_ours(best_o[1], mix_full, sd, 400), test_pairs) for sd in range(5)]
    for sd,v in enumerate(ours): print(f"  us seed={sd}: {v:.3f}", flush=True)
    h,lr=best_a[1]
    adv=[eval_adv(train_adv(h,lr,mix_full,sd), test_pairs) for sd in range(5)]
    for sd,v in enumerate(adv): print(f"  adv seed={sd}: {v:.3f}", flush=True)
    ro,ra=np.array(ours),np.array(adv)
    print(f"\nUS: {ro.mean():.3f} ± {ro.std():.3f}", flush=True)
    print(f"ADVERSARY: {ra.mean():.3f} ± {ra.std():.3f}", flush=True)
    print(f"VERDICT: {'US' if ro.mean()+ro.std()<ra.mean()-ra.std() else ('ADVERSARY' if ra.mean()+ra.std()<ro.mean()-ro.std() else 'TIE at 1σ')}", flush=True)
