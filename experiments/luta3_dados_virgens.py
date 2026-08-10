#!/usr/bin/env python3
"""LUTA 3 — PRÉ-REGISTO (commit antes de qualquer corrida; dados virgens).

Motivação declarada: as Lutas 1-2 correram num span de teste que a fase
exploratória já tinha avaliado (divulgado em §4.2 do paper). A Luta 3
responde: dados NUNCA avaliados por nenhum modelo deste projeto.

DADOS (virgens — zero referências no repositório até este commit):
- ETTh1, coluna OT (transformador, horário) — data/ETTh1.csv
- Apneia-ECG a01 (PhysioNet, 100 Hz, int16 LE) — data/apnea_a01.dat
Janelas de 64, split temporal 80/20 por domínio, normalização calculada
só no treino (via _split_and_normalize, como sempre). Intercalação em
blocos de 16. Validação = últimos 15% do treino (mín. 24 tramas).

LADOS DA LUTA (receita anti-bacia nos dois, como na Emenda 2):
- Nós: TSL, topo ∈ {16, 24}; lr recozido 0.02+0.08·0.99^ep; 2 réplicas
  por treino escolhidas pelo ERRO DE TREINO; seleção 150 ép, finais 400.
- Adversário: {GRU, LSTM} × h ∈ {16, 32} × lr ∈ {1e-3, 3e-3} = 8 configs
  (contra 2 nossas); Adam, clip 1.0, 2 réplicas por treino, 150 ép em
  seleção e finais (como na Emenda 2).
- Seleção de configuração: NRMSE de validação médio de 2 seeds.
- FINAIS: 20 seeds por lado, no teste (usado apenas aqui).

BASELINES (a hipótese nula do referee; avaliação prequencial — preveem a
trama seguinte ANTES de a ver, e adaptam-se online durante todo o span):
- Persistência: x̂(t+1) = x(t). Determinística.
- NLMS-AR: x̂(t+1) = W·x(t), W 64×64 atualizado online por NLMS;
  μ ∈ {0.1, 0.3, 0.5, 1.0} escolhido por validação. Determinístico.
- ESN: reservatório 128, leak e raio espectral ∈ {0.3,0.6}×{0.8,0.95}
  por validação, leitura RLS (esquecimento 0.999) online; 20 seeds.

REGRA DE DECISÃO (declarada aqui; substitui o 1σ das lutas anteriores):
o lado A vence o lado B sse o IC bootstrap 95% (10 000 reamostras) da
diferença das médias excluir 0. Reportar sempre: por-seed, média ± dp
(4 casas), IC 95%, Welch t, Cohen's d. Empates dizem-se empates.

Saída: correr com `python -u experiments/luta3_dados_virgens.py | tee
experiments/luta3_finais.txt` e arquivar o ficheiro no commit do veredicto.
"""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn as nn
from dataclasses import replace
from pcnet import PCConfig, PCNetwork
from pcnet.dtypes import F
from pcnet.datasets import load_csv_column, _split_and_normalize

def nrmse(p, t): return float(np.sqrt(np.mean((p - t) ** 2) / np.mean(t ** 2)))
def sigm(a): return 1.0 / (1.0 + np.exp(-a))

def load_apnea(path, frame_len=64, train_frac=0.8):
    x = np.fromfile(path, dtype="<i2").astype(np.float64)
    return _split_and_normalize(x, frame_len, "apnea_a01", train_frac,
                                units="ECG", rate_hz=100.0)

DOM = [("ETTh1", load_csv_column("data/ETTh1.csv", "OT")),
       ("apnea", load_apnea("data/apnea_a01.dat"))]
NTR = min(len(d.train) for _, d in DOM)
NV = max(24, int(0.15 * NTR)); NT = NTR - NV
block = 16

def inter(frames_list, n):
    nb = n // block
    return np.concatenate([fr[:n][b * block:(b + 1) * block]
                           for b in range(nb) for fr in frames_list])

tr_frames = [d.train for _, d in DOM]
mix_sel = inter([f[:NT] for f in tr_frames], NT)
mix_full = inter([f[:NTR] for f in tr_frames], NTR)
val_pairs = [(d.train[:NT], d.train[NT:NTR]) for _, d in DOM]
test_pairs = [(d.train[:NTR], d.test) for _, d in DOM]

# ---------------------------------------------------------------- o nosso lado
def train_ours_once(top, data, sd, epochs):
    net = PCNetwork(PCConfig(seed=sd, fast_path=False, use_precision=True,
                             a_lr=0.0, sizes=(64, top), gated_transition=True))
    g_ = net.gated; W0 = net.layers[0].W
    for ep in range(epochs):
        lr = 0.02 + 0.08 * (0.99 ** ep)
        for fr in data:
            s = net._z_prev[net.L].copy(); net.step(fr, learn=False)
            c = np.tanh(g_.A @ s); gg = sigm(g_.G @ s + g_.b)
            prior = (1 - gg) * s + gg * c
            e = fr - W0 @ prior; h = W0.T @ e
            ns = float(s @ s) + 1e-6; npr = float(prior @ prior) + 1e-6
            W0 += F(lr / npr) * np.outer(e, prior).astype(F)
            g_.A += F(lr / ns) * np.outer(h * gg * (1 - c * c), s).astype(F)
            g_.G += F(lr / ns) * np.outer(h * (c - s) * gg * (1 - gg), s).astype(F)
            g_.b += F(lr) * (h * (c - s) * gg * (1 - gg)).astype(F)
            net.layers[0].refresh_device()
    net.reset(); errs = []
    for fr in data[:300]:
        errs.append(float(np.mean((fr - net.predict_next()) ** 2)))
        net.step(fr, learn=False)
    return net, float(np.mean(errs))

def train_ours(top, data, sd, epochs):
    cands = [train_ours_once(top, data, 1000 * sd + k, epochs) for k in range(2)]
    return min(cands, key=lambda t: t[1])[0]

def eval_ours(net, pairs):
    out = []; b = net.cfg
    net.cfg = replace(b, max_iters=50, theta=0.0, settle_min_gain=0.0)
    for warm, target in pairs:
        snap = net.snapshot_state(); net.reset()
        for f in warm[-32:]: net.step(f, learn=False)
        P = []
        for f in target: P.append(net.predict_next()); net.step(f, learn=False)
        out.append(nrmse(np.array(P, dtype=F), target)); net.restore_state(snap)
    net.cfg = b; return float(np.mean(out))

# ---------------------------------------------------------------- adversário
class R(nn.Module):
    def __init__(s, cell, h):
        super().__init__()
        s.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(64, h, batch_first=True)
        s.head = nn.Linear(h, 64)
    def forward(s, x):
        o, _ = s.rnn(x.unsqueeze(0)); return s.head(o.squeeze(0))

def train_adv_once(cell, h, lr, data, sd, epochs=150):
    torch.manual_seed(sd); m = R(cell, h)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    x = torch.tensor(data, dtype=torch.float32)
    for _ in range(epochs):
        loss = ((m(x[:-1]) - x[1:]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        tr = float(((m(x[:-1]) - x[1:]) ** 2).mean())
    return m, tr

def train_adv(cell, h, lr, data, sd, epochs=150):
    cands = [train_adv_once(cell, h, lr, data, 1000 * sd + k, epochs)
             for k in range(2)]
    return min(cands, key=lambda t: t[1])[0]

def eval_adv(m, pairs):
    out = []
    with torch.no_grad():
        for warm, target in pairs:
            seq = torch.tensor(np.concatenate([warm[-32:], target]),
                               dtype=torch.float32)
            out.append(nrmse(m(seq[:-1])[31:].numpy(), target))
    return float(np.mean(out))

# ---------------------------------------------------------------- baselines
def eval_persistence(pairs):
    out = []
    for warm, target in pairs:
        prev = np.concatenate([warm[-1:], target[:-1]])
        out.append(nrmse(prev, target))
    return float(np.mean(out))

def eval_nlms_ar(mu, pairs):
    out = []
    for warm, target in pairs:
        W = np.zeros((64, 64), dtype=np.float64)
        stream = np.concatenate([warm, target]); n_warm = len(warm)
        preds = []
        for i in range(len(stream) - 1):
            x, y = stream[i], stream[i + 1]
            p = W @ x
            if i + 1 >= n_warm: preds.append(p)
            e = y - p
            W += mu * np.outer(e, x) / (float(x @ x) + 1e-6)
        out.append(nrmse(np.array(preds), target))
    return float(np.mean(out))

def eval_esn(leak, rho, sd, pairs, n_res=128, forget=0.999):
    rng = np.random.default_rng(sd)
    Win = rng.uniform(-0.5, 0.5, (n_res, 64))
    Wr = rng.normal(0, 1, (n_res, n_res))
    eig = np.max(np.abs(np.linalg.eigvals(Wr))); Wr *= rho / eig
    out = []
    for warm, target in pairs:
        r = np.zeros(n_res)
        Wout = np.zeros((64, n_res)); Pmat = np.eye(n_res) * 100.0
        stream = np.concatenate([warm, target]); n_warm = len(warm)
        preds = []
        for i in range(len(stream) - 1):
            r = (1 - leak) * r + leak * np.tanh(Win @ stream[i] + Wr @ r)
            p = Wout @ r
            if i + 1 >= n_warm: preds.append(p)
            # RLS com esquecimento
            y = stream[i + 1]
            Pr = Pmat @ r
            k = Pr / (forget + float(r @ Pr))
            Wout += np.outer(y - p, k)
            Pmat = (Pmat - np.outer(k, Pr)) / forget
        out.append(nrmse(np.array(preds), target))
    return float(np.mean(out))

# ---------------------------------------------------------------- estatística
def stats_block(a, b, la, lb):
    """IC bootstrap 95% de mean(b)-mean(a), Welch t, Cohen's d."""
    a, b = np.array(a), np.array(b)
    rng = np.random.default_rng(7)
    diffs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
             for _ in range(10000)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    va, vb = a.var(ddof=1), b.var(ddof=1)
    t = (b.mean() - a.mean()) / np.sqrt(va / len(a) + vb / len(b))
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    d = (b.mean() - a.mean()) / sp if sp > 0 else float("inf")
    print(f"  {la}: {a.mean():.4f} ± {a.std(ddof=1):.4f}   "
          f"{lb}: {b.mean():.4f} ± {b.std(ddof=1):.4f}", flush=True)
    print(f"  diff({lb}-{la}) IC95%=[{lo:.4f},{hi:.4f}]  Welch t={t:.2f}  "
          f"d={d:.2f}", flush=True)
    if lo > 0: print(f"  => {la} VENCE (IC exclui 0)", flush=True)
    elif hi < 0: print(f"  => {lb} VENCE (IC exclui 0)", flush=True)
    else: print("  => EMPATE (IC contém 0)", flush=True)

if __name__ == "__main__":
    for n, d in DOM: print(d.describe(), flush=True)
    print(f"NTR={NTR} NT={NT} NV={NV}\n", flush=True)

    print("=== SELEÇÃO (val médio de 2 seeds) ===", flush=True)
    best_o = (9, None)
    for top in (16, 24):
        vs = [eval_ours(train_ours(top, mix_sel, sd, 150), val_pairs)
              for sd in (0, 1)]
        v = float(np.mean(vs))
        print(f"  nós topo={top}: val={v:.4f} {[f'{x:.4f}' for x in vs]}", flush=True)
        if v < best_o[0]: best_o = (v, top)
    best_a = (9, None)
    for cell in ("gru", "lstm"):
        for h in (16, 32):
            for lr in (1e-3, 3e-3):
                vs = [eval_adv(train_adv(cell, h, lr, mix_sel, sd), val_pairs)
                      for sd in (0, 1)]
                v = float(np.mean(vs))
                print(f"  adv {cell} h={h} lr={lr}: val={v:.4f}", flush=True)
                if v < best_a[0]: best_a = (v, (cell, h, lr))
    best_mu = min(((eval_nlms_ar(mu, val_pairs), mu)
                   for mu in (0.1, 0.3, 0.5, 1.0)))
    best_esn = min(((eval_esn(lk, rho, 0, val_pairs), (lk, rho))
                    for lk in (0.3, 0.6) for rho in (0.8, 0.95)))
    print(f"vencedores: nós topo={best_o[1]} | adv {best_a[1]} | "
          f"NLMS mu={best_mu[1]} | ESN {best_esn[1]}\n", flush=True)

    print("=== FINAIS — 20 seeds, teste usado apenas aqui ===", flush=True)
    ours = [eval_ours(train_ours(best_o[1], mix_full, sd, 400), test_pairs)
            for sd in range(20)]
    for sd, v in enumerate(ours): print(f"  nós seed={sd}: {v:.4f}", flush=True)
    cell, h, lr = best_a[1]
    adv = [eval_adv(train_adv(cell, h, lr, mix_full, sd), test_pairs)
           for sd in range(20)]
    for sd, v in enumerate(adv): print(f"  adv seed={sd}: {v:.4f}", flush=True)
    pers = eval_persistence(test_pairs)
    nlms = eval_nlms_ar(best_mu[1], test_pairs)
    esn = [eval_esn(*best_esn[1], sd, test_pairs) for sd in range(20)]
    print(f"\n  persistência: {pers:.4f}", flush=True)
    print(f"  NLMS-AR (mu={best_mu[1]}): {nlms:.4f}", flush=True)
    print(f"  ESN: {np.mean(esn):.4f} ± {np.std(esn, ddof=1):.4f}", flush=True)

    print("\n=== VEREDICTOS (regra: IC bootstrap 95% da diferença) ===", flush=True)
    stats_block(ours, adv, "NÓS", "ADV")
    stats_block(ours, esn, "NÓS", "ESN")
    print(f"  NÓS vs NLMS-AR ({nlms:.4f}) e persistência ({pers:.4f}): "
          f"determinísticos — comparar com a média e o IC de NÓS.", flush=True)
