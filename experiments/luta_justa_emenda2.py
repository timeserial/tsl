#!/usr/bin/env python3
"""EMENDA Nº2 (pré-registada antes de qualquer resultado): luta justa real,
agora com a receita anti-bacia nos DOIS lados.

Mudanças face à Fila 13, todas declaradas aqui:
- Treino (ambos os lados): 2 réplicas por medição, escolhida pelo ERRO DE
  TREINO (nunca validação nem teste) — a receita que funcionou na Fila 12.
- Nosso treino: recozimento lr 0.1→0.02 (0.99^ep). Seleção 150 ép; finais 400.
- Adversário: Adam como sempre; mesmas 2 réplicas por treino. 150 ép.
- Seleção de configuração: NRMSE de VALIDAÇÃO médio de 2 seeds (a seleção
  de 1 seed foi o que colapsou na Fila 13).
- Espaços reduzidos e declarados: nós {(16,·),(24,·)} já com lr embutido no
  recozimento -> {topo 16, topo 24}; adversário {gru h16, gru h32} × lr
  {1e-3, 3e-3} (os vencedores da busca da Fila 13; lstm e h64 nunca
  estiveram perto).
- Finais: vencedor de cada lado, 5 seeds, TESTE intocado. Decisão a 1σ.
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
    print("EMENDA 2 — seleção (val médio de 2 seeds, receita anti-bacia nos dois lados)", flush=True)
    best_o=(9,None)
    for top in (16,24):
        vs=[eval_ours(train_ours(top, mix_sel, sd, 150), val_pairs) for sd in (0,1)]
        v=float(np.mean(vs)); print(f"  nós topo={top}: val={v:.3f} {vs}", flush=True)
        if v<best_o[0]: best_o=(v,top)
    best_a=(9,None)
    for h in (16,32):
        for lr in (1e-3,3e-3):
            vs=[eval_adv(train_adv(h,lr,mix_sel,sd), val_pairs) for sd in (0,1)]
            v=float(np.mean(vs)); print(f"  adv h={h} lr={lr}: val={v:.3f} {vs}", flush=True)
            if v<best_a[0]: best_a=(v,(h,lr))
    print(f"vencedores: nós topo={best_o[1]} | adv {best_a[1]}\n", flush=True)
    print("FINAIS — 5 seeds, teste intocado", flush=True)
    ours=[eval_ours(train_ours(best_o[1], mix_full, sd, 400), test_pairs) for sd in range(5)]
    for sd,v in enumerate(ours): print(f"  nós seed={sd}: {v:.3f}", flush=True)
    h,lr=best_a[1]
    adv=[eval_adv(train_adv(h,lr,mix_full,sd), test_pairs) for sd in range(5)]
    for sd,v in enumerate(adv): print(f"  adv seed={sd}: {v:.3f}", flush=True)
    ro,ra=np.array(ours),np.array(adv)
    print(f"\nNÓS: {ro.mean():.3f} ± {ro.std():.3f}", flush=True)
    print(f"ADVERSÁRIO: {ra.mean():.3f} ± {ra.std():.3f}", flush=True)
    print(f"VEREDICTO: {'NÓS' if ro.mean()+ro.std()<ra.mean()-ra.std() else ('ADVERSÁRIO' if ra.mean()+ra.std()<ro.mean()-ro.std() else 'EMPATE a 1σ')}", flush=True)
