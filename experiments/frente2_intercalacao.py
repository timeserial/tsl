#!/usr/bin/env python3
"""FRONT 2 of the attack on local credit: does the gap shrink with fine interleaving?

    .venv/bin/python experiments/frente2_intercalacao.py

Hypothesis: the local-rule ≈ gradient equivalence only holds with small errors;
coarse blocks let the dormant task's errors grow between visits.
If the us-vs-GRU gap shrinks with fine blocks, the price of locality is
actually the price of the regime - and it is attacked with curriculum, not with mechanisms.

The target is re-measured with the same block: changing the block changes the task for both.
Takes ~40 min; run with the machine idle.
"""
import sys; sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, torch, torch.nn as nn
from continual import make_tasks, evaluate_task
from pcnet import PCConfig, PCNetwork
from pcnet.train import train
def nrmse(p,t): return float(np.sqrt(np.mean((p-t)**2)/np.mean(t**2)))
tasks = make_tasks(3, 64, 400, seed=1)
BASE = dict(fast_path=False, use_precision=True, a_lr=0.1, sizes=(64,24), gated_transition=True)

def mix(block):
    n_blocks = min(len(t[0]) for t in tasks)//block
    return np.concatenate([tr[b*block:(b+1)*block] for b in range(n_blocks) for tr,_ in tasks])

def ours(block, seeds=range(6)):
    m = mix(block); rows=[]
    for s in seeds:
        net = PCNetwork(PCConfig(seed=s, **BASE)); train(net, m, epochs=80)
        rows.append(np.mean([evaluate_task(net, te, tr) for tr,te in tasks]))
    return np.array(rows)

def gru0(block, seeds=range(3), hidden=16, epochs=150):
    m = torch.tensor(mix(block), dtype=torch.float32); out=[]
    for seed in seeds:
        torch.manual_seed(seed)
        cell = nn.GRUCell(64, hidden); head = nn.Linear(hidden, 64)
        opt = torch.optim.Adam(list(cell.parameters())+list(head.parameters()), lr=3e-3)
        for _ in range(epochs):
            h = torch.zeros(1, hidden); loss = 0.0
            for t in range(len(m)-1):
                h = h.detach(); h = cell(m[t:t+1], h)
                loss = loss + ((head(h) - m[t+1:t+2])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sc=[]
        with torch.no_grad():
            for tr,te in tasks:
                seq = torch.tensor(np.concatenate([tr[-32:], te]), dtype=torch.float32)
                h = torch.zeros(1, hidden); P=[]
                for t in range(len(seq)-1):
                    h = cell(seq[t:t+1], h); P.append(head(h))
                sc.append(nrmse(torch.cat(P)[31:].numpy(), te))
        out.append(float(np.mean(sc)))
    return np.array(out)

print("FRONT 2: does the gap shrink with fine interleaving?\n")
print(f"{'block':>6} {'us':>16} {'GRU-0':>16} {'gap':>10}")
for block in (16, 4, 1):
    o = ours(block); g = gru0(block)
    print(f"{block:>6} {o.mean():>8.3f}±{o.std():.3f} {g.mean():>8.3f}±{g.std():.3f} {o.mean()-g.mean():>10.3f}")
