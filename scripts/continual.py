#!/usr/bin/env python3
"""Aprender uma coisa nova sem destruir a anterior.

    .venv/bin/python scripts/continual.py --tasks 5 --epochs 25

O esquecimento catastrófico é dos problemas mais antigos e mais reais do
campo: ensina-se uma tarefa a uma rede, depois outra, e a primeira desaparece.
Não é falta de capacidade — é o dilema plasticidade/estabilidade. Uma rede que
aprende depressa esquece depressa, e uma que não esquece não aprende.

Os transformers têm o problema todo, e é **estrutural**: são treinados uma vez
e congelados, e tudo o que parece aprendizagem depois disso (contexto, RAG,
fine-tuning) é andaime por fora. Não se resolve com escala.

A resposta do cérebro, e a que a secção 4 do `CONTEXTO.md` propõe, é não
escolher: duas memórias a velocidades diferentes. O hipocampo grava o episódio
na hora sem tocar nos pesos; o sono reproduz o que se repetiu e destila-o
devagar no córtex.

O protocolo aqui é o padrão da literatura de aprendizagem contínua:

  * N tarefas, cada uma um sinal com estrutura diferente;
  * treina-se **em sequência**, uma de cada vez, sem nunca voltar atrás;
  * no fim, mede-se em **todas**.

Duas medidas, e a segunda é a que interessa:

  * **NRMSE final médio** — quão bem ficou em tudo o que aprendeu.
  * **Esquecimento** — quanto piorou numa tarefa entre acabar de a aprender e
    chegar ao fim. É este número que separa quem acumula de quem substitui.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcnet import PCConfig, PCNetwork, make_signal  # noqa: E402
from pcnet.dtypes import F  # noqa: E402
from pcnet.episodic import EpisodicConfig  # noqa: E402
from pcnet.report import pm, rule, table  # noqa: E402
from pcnet.signals import DEFAULT_FREQS  # noqa: E402
from pcnet.train import train  # noqa: E402


def make_tasks(n_tasks: int, frame_len: int, n_frames: int, seed: int):
    """N sinais com conteúdo espectral distinto — N "mundos" diferentes."""
    tasks = []
    for k in range(n_tasks):
        scale = 0.6 + 0.35 * k  # cada tarefa vive noutra banda
        freqs = tuple(min(0.45, f * scale) for f in DEFAULT_FREQS)
        sig = make_signal(n_frames=n_frames, frame_len=frame_len,
                          freqs=freqs, seed=seed + 100 * k)
        cut = int(0.8 * sig.n_frames)
        tasks.append((sig.frames[:cut], sig.frames[cut:]))
    return tasks


def nrmse(pred, target) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2) / np.mean(target**2)))


def evaluate_task(net: PCNetwork, test: np.ndarray, warmup: np.ndarray) -> float:
    """Avalia sem aprender e sem deixar rasto no estado."""
    snapshot = net.snapshot_state()
    try:
        net.reset()
        for frame in warmup[-16:]:
            net.step(frame, learn=False)
        preds = []
        for frame in test:
            preds.append(net.predict_next())
            net.step(frame, learn=False)
        return nrmse(np.array(preds, dtype=F), test)
    finally:
        net.restore_state(snapshot)


def run_pcnet(tasks, epochs, seed, memory: bool, sleep: int,
              interleave: int = 0, **cfg_kw):
    net = PCNetwork(PCConfig.recommended(seed=seed, **cfg_kw))
    if memory:
        net.attach_memory(EpisodicConfig(n_slots=256))

    matrix = np.zeros((len(tasks), len(tasks)))
    for k, (train_frames, _) in enumerate(tasks):
        train(net, train_frames, epochs=epochs, replay_per_epoch=interleave)
        if sleep and net.memory is not None:
            # O sono: reproduz episódios de tudo o que já viu, não só do que
            # acabou de aprender. É isto que devia impedir o esquecimento.
            net.consolidate(n_episodes=32, passes=sleep)
        for j, (tr_j, te_j) in enumerate(tasks):
            matrix[k, j] = evaluate_task(net, te_j, tr_j)
    return matrix, (net.memory.stats() if net.memory else {})


def run_torch(tasks, epochs, seed, kind: str, frame_len: int):
    """Uma rede treinada com backprop, na mesma sequência. O contraste."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    if kind == "mlp":
        net = nn.Sequential(nn.Linear(frame_len, 48), nn.Tanh(), nn.Linear(48, frame_len))
    else:
        class G(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(frame_len, 32, batch_first=True)
                self.head = nn.Linear(32, frame_len)

            def forward(self, x):
                out, _ = self.rnn(x.unsqueeze(0))
                return self.head(out.squeeze(0))
        net = G()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    def fwd(seq):
        return net(seq) if kind == "gru" else net(seq)

    def score(test, warmup):
        with torch.no_grad():
            seq = torch.tensor(np.concatenate([warmup[-16:], test]), dtype=torch.float32)
            pred = fwd(seq[:-1])[15:]
        return nrmse(pred.numpy(), test)

    matrix = np.zeros((len(tasks), len(tasks)))
    for k, (train_frames, _) in enumerate(tasks):
        x = torch.tensor(np.asarray(train_frames), dtype=torch.float32)
        for _ in range(epochs * 8):
            loss = ((fwd(x[:-1]) - x[1:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        for j, (tr_j, te_j) in enumerate(tasks):
            matrix[k, j] = score(te_j, tr_j)
    return matrix, {}


def run_joint(tasks, epochs, seed):
    """Controlo decisivo: treinar em tudo ao mesmo tempo.

    Se nem assim a rede conseguir fazer as N tarefas, então o que a tabela
    mostra não é esquecimento — é falta de capacidade, e nenhuma memória
    episódica do mundo resolve isso. É o mesmo cuidado que evitou concluir
    que o modelo era mau no giroscópio do HAR, quando o que não tinha sinal
    era a tarefa.
    """
    net = PCNetwork(PCConfig.recommended(seed=seed))
    # Intercalar em blocos curtos, não concatenar. Concatenar seria treino
    # sequencial com outro nome — o modelo acabaria sempre na última tarefa,
    # que foi exatamente o que a primeira versão deste controlo fez, dando um
    # "teto" mais baixo que as variantes que devia limitar.
    block = 16
    chunks = []
    n_blocks = min(len(t[0]) for t in tasks) // block
    for b in range(n_blocks):
        for tr, _ in tasks:
            chunks.append(tr[b * block:(b + 1) * block])
    mixed = np.concatenate(chunks)
    train(net, mixed, epochs=epochs)
    row = np.array([evaluate_task(net, te, tr) for tr, te in tasks])
    return np.tile(row, (len(tasks), 1)), {}


def summarize(matrix: np.ndarray) -> dict:
    n = len(matrix)
    final = matrix[-1]
    # Esquecimento: quanto piorou cada tarefa entre acabar de a aprender e o fim
    forgetting = [float(matrix[-1, j] - matrix[j, j]) for j in range(n - 1)]
    return {
        "final_medio": float(final.mean()),
        "primeira_tarefa_no_fim": float(final[0]),
        "primeira_tarefa_acabada_de_aprender": float(matrix[0, 0]),
        "esquecimento_medio": float(np.mean(forgetting)) if forgetting else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--frame-len", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sleep", type=int, default=2, help="passagens de consolidação")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    tasks = make_tasks(args.tasks, args.frame_len, args.frames, seed=1)
    print(f"{args.tasks} tarefas, {args.epochs} passagens cada, "
          f"{len(tasks[0][0])} tramas de treino por tarefa")
    print("Treino sequencial: cada tarefa é vista uma vez e nunca mais.\n")

    variants = [
        ("TETO: treino conjunto (não sequencial)",
         lambda s: run_joint(tasks, args.epochs, s)),
        ("rede preditiva (sem memória)", lambda s: run_pcnet(tasks, args.epochs, s, False, 0)),
        ("+ memória episódica", lambda s: run_pcnet(tasks, args.epochs, s, True, 0)),
        ("+ memória + consolidação", lambda s: run_pcnet(tasks, args.epochs, s, True, args.sleep)),
        ("+ memória + replay intercalado", lambda s: run_pcnet(
            tasks, args.epochs, s, True, args.sleep, interleave=32)),
    ]
    try:
        import torch  # noqa: F401
        variants += [
            ("MLP (backprop)", lambda s: run_torch(tasks, args.epochs, s, "mlp", args.frame_len)),
            ("GRU (backprop)", lambda s: run_torch(tasks, args.epochs, s, "gru", args.frame_len)),
        ]
    except ImportError:
        pass

    rule("Aprender em sequência, medir em tudo")
    rows, results = [], {}
    for name, fn in variants:
        stats, mems = [], {}
        for seed in range(args.seeds):
            matrix, mems = fn(seed)
            stats.append(summarize(matrix))
        agg = {k: (float(np.mean([s[k] for s in stats])),
                   float(np.std([s[k] for s in stats]))) for k in stats[0]}
        results[name] = {"agg": agg, "memoria": mems}
        rows.append({
            "modelo": name,
            "NRMSE final (todas)": pm(*agg["final_medio"]),
            "1ª tarefa: acabada": f"{agg['primeira_tarefa_acabada_de_aprender'][0]:.3f}",
            "1ª tarefa: no fim": f"{agg['primeira_tarefa_no_fim'][0]:.3f}",
            "esquecimento": pm(*agg["esquecimento_medio"]),
        })
        print(f"  … {name}")
    print()
    table(rows)
    print("\n  Esquecimento = quanto piorou cada tarefa entre acabar de a aprender")
    print("  e chegar ao fim. Zero seria acumular; positivo grande é substituir.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"\n  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
