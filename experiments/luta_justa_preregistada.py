import sys; sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, torch, torch.nn as nn
from dataclasses import replace
from pcnet import PCConfig, PCNetwork
from pcnet.dtypes import F
from pcnet.datasets import load_csv_column, load_uci_har_inertial
S="/private/tmp/claude-501/-Users-brunofitascustodio-Code-nn/de5187af-3636-40ee-9044-c6e4e37ed175/scratchpad"
def nrmse(p,t): return float(np.sqrt(np.mean((p-t)**2)/np.mean(t**2)))
def sigm(a): return 1.0/(1.0+np.exp(-a))
DOM=[("ETTm1",load_csv_column(f"{S}/data/ETTm1.csv","OT")),
     ("acc",load_uci_har_inertial(f"{S}/har/UCI HAR Dataset","total_acc_x")),
     ("gyro",load_uci_har_inertial(f"{S}/har/UCI HAR Dataset","body_gyro_z"))]
NTR=min(len(d.train) for _,d in DOM)
NV = max(24, int(0.15*NTR)); NT = NTR-NV
tr_tasks=[(d.train[:NT],) for _,d in DOM]
val_tasks=[(d.train[:NT], d.train[NT:NTR]) for _,d in DOM]
test_tasks=[(d.train[:NTR], d.test) for _,d in DOM]
block=16
def interleave(n):
    nb=n//block
    return np.concatenate([t[0][b*block:(b+1)*block] for b in range(nb) for t in tr_tasks]) if n==NT else \
           np.concatenate([tr[:n][b*block:(b+1)*block] for b in range(nb) for tr,_ in test_tasks])
mix_sel = interleave(NT)      # training for selection
mix_full = interleave(NTR)    # training for the final

def eval_ours(net, pairs):
    out=[]
    b=net.cfg; net.cfg=replace(b, max_iters=50, theta=0.0, settle_min_gain=0.0)
    for warm, target in pairs:
        snap=net.snapshot_state(); net.reset()
        for f in warm[-32:]: net.step(f, learn=False)
        P=[]
        for f in target: P.append(net.predict_next()); net.step(f, learn=False)
        out.append(nrmse(np.array(P,dtype=F), target)); net.restore_state(snap)
    net.cfg=b
    return float(np.mean(out))

def train_ours(top, lr, data, sd, epochs=80):
    net=PCNetwork(PCConfig(seed=sd, fast_path=False, use_precision=True,
                           a_lr=0.0, sizes=(64,top), gated_transition=True))
    g_=net.gated; W0=net.layers[0].W
    for ep in range(epochs):
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
    return net

class RNN(nn.Module):
    def __init__(self, cell, hidden):
        super().__init__()
        self.kind=cell
        self.rnn = (nn.GRU if cell=="gru" else nn.LSTM)(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 64)
    def forward(self, seq): o,_=self.rnn(seq.unsqueeze(0)); return self.head(o.squeeze(0))

def train_rnn(cell, hidden, lr, data, sd, epochs=150):
    torch.manual_seed(sd); m=RNN(cell,hidden)
    opt=torch.optim.Adam(m.parameters(), lr=lr)
    x=torch.tensor(data, dtype=torch.float32)
    for _ in range(epochs):
        loss=((m(x[:-1])-x[1:])**2).mean(); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m

def eval_rnn(m, pairs):
    out=[]
    with torch.no_grad():
        for warm,target in pairs:
            seq=torch.tensor(np.concatenate([warm[-32:],target]),dtype=torch.float32)
            P=m(seq[:-1])[31:]
            out.append(nrmse(P.numpy(), target))
    return float(np.mean(out))

print(f"FILA 13 - PRÉ-REGISTO: busca igual, seleção por validação ({NV} tramas/domínio), teste intocado\n", flush=True)
print("=== BUSCA: nosso lado (9 cfg, seed 0, val) ===", flush=True)
best_o=(9,None)
for top in (16,24,32):
    for lr in (0.05,0.1,0.2):
        v=eval_ours(train_ours(top,lr,mix_sel,0), val_tasks)
        print(f"  topo={top} lr={lr}: val={v:.3f}", flush=True)
        if v<best_o[0]: best_o=(v,(top,lr))
print(f"vencedor nosso: {best_o}\n", flush=True)
print("=== BUSCA: adversário (18 cfg, seed 0, val) ===", flush=True)
best_a=(9,None)
for cell in ("gru","lstm"):
    for hidden in (16,32,64):
        for lr in (1e-3,3e-3,1e-2):
            v=eval_rnn(train_rnn(cell,hidden,lr,mix_sel,0), val_tasks)
            print(f"  {cell} h={hidden} lr={lr}: val={v:.3f}", flush=True)
            if v<best_a[0]: best_a=(v,(cell,hidden,lr))
print(f"vencedor adversário: {best_a}\n", flush=True)
print("=== FINAL: 5 seeds, treino completo, TESTE ===", flush=True)
top,lr = best_o[1]; ours=[]
for sd in range(5):
    ours.append(eval_ours(train_ours(top,lr,mix_full,sd), test_tasks))
    print(f"  nós seed={sd}: {ours[-1]:.3f}", flush=True)
cell,hidden,alr = best_a[1]; adv=[]
for sd in range(5):
    adv.append(eval_rnn(train_rnn(cell,hidden,alr,mix_full,sd), test_tasks))
    print(f"  adv seed={sd}: {adv[-1]:.3f}", flush=True)
ro,ra=np.array(ours),np.array(adv)
print(f"\nNÓS ({top},{lr}): {ro.mean():.3f} ± {ro.std():.3f}", flush=True)
print(f"ADVERSÁRIO ({cell},h{hidden},lr{alr}): {ra.mean():.3f} ± {ra.std():.3f}", flush=True)
print(f"VEREDICTO: {'NÓS' if ro.mean()+ro.std()<ra.mean()-ra.std() else ('ADVERSÁRIO' if ra.mean()+ra.std()<ro.mean()-ro.std() else 'EMPATE a 1σ')}", flush=True)
