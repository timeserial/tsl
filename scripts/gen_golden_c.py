"""Gera os vetores dourados para validar o inferidor+treinador C.

Contrato (lições do caos registadas em c/README.md):
 1. UMA atualização de dois tempos a partir de estado fixo: exata a 1e-5.
 2. Trajetória completa (treino 200 janelas + avaliação 100): NRMSE agregado
    do C dentro de 2% do Python.
"""
import sys; sys.path.insert(0,"src")
import numpy as np
from pcnet import PCConfig, PCNetwork
from pcnet.dtypes import F
from pcnet.signals import make_signal

def sigm(a): return 1.0/(1.0+np.exp(-a))

def c_lit(v):
    t=f"{float(v):.9g}"
    if "." not in t and "e" not in t and "E" not in t: t+=".0"
    return t+"f"

def c_arr(name, a):
    a=np.asarray(a,dtype=F).ravel()
    body=",".join(c_lit(v) for v in a)
    return f"static const float {name}[{a.size}] = {{{body}}};\n"

net = PCNetwork(PCConfig(seed=0, fast_path=False, use_precision=False,
                         a_lr=0.0, sizes=(64,24), gated_transition=True))
g_=net.gated; W0=net.layers[0].W
sig = make_signal(n_frames=320, frame_len=64, seed=7)
frames = sig.frames.astype(F)

out=[]
out.append("/* Gerado por gen_golden_c.py - vetores dourados da célula de dois tempos */\n")
out.append("#define N_IN 64\n#define N_TOP 24\n#define LR 0.1f\n")
out.append(c_arr("g_W0_init", W0))
out.append(c_arr("g_A_init", g_.A))
out.append(c_arr("g_G_init", g_.G))
out.append(c_arr("g_b_init", g_.b))

# --- contrato 1: uma atualização isolada, estado fixo -------------------
s_fix = (np.random.default_rng(3).standard_normal(24)*0.4).astype(F)
x_fix = frames[10]
c=np.tanh(g_.A@s_fix); gg=sigm(g_.G@s_fix+g_.b); prior=(1-gg)*s_fix+gg*c
e=x_fix-W0@prior; h=W0.T@e
ns=float(s_fix@s_fix)+1e-6; npr=float(prior@prior)+1e-6
W0n=W0+F(0.1/npr)*np.outer(e,prior).astype(F)
An=g_.A+F(0.1/ns)*np.outer(h*gg*(1-c*c),s_fix).astype(F)
Gn=g_.G+F(0.1/ns)*np.outer(h*(c-s_fix)*gg*(1-gg),s_fix).astype(F)
bn=g_.b+F(0.1)*(h*(c-s_fix)*gg*(1-gg)).astype(F)
out.append(c_arr("g_s_fix", s_fix)); out.append(c_arr("g_x_fix", x_fix))
out.append(c_arr("g_W0_after1", W0n)); out.append(c_arr("g_A_after1", An))
out.append(c_arr("g_G_after1", Gn)); out.append(c_arr("g_b_after1", bn))

# --- contrato 2: trajetória agregada ------------------------------------
# treino online 200 janelas (estado via média móvel simples do prior — o C
# replica exatamente esta dinâmica reduzida, sem assentamento, para o
# contrato ser determinístico; o assentamento valida-se por agregado no
# repositório Python)
s=np.zeros(24,dtype=F)
err_tr=[]
for fr in frames[:200]:
    c=np.tanh(g_.A@s); gg=sigm(g_.G@s+g_.b); prior=((1-gg)*s+gg*c).astype(F)
    pred=(W0@prior).astype(F)
    e=(fr-pred).astype(F); err_tr.append(float(np.mean(e*e)))
    h=(W0.T@e).astype(F)
    ns=float(s@s)+1e-6; npr=float(prior@prior)+1e-6
    W0+=F(0.1/npr)*np.outer(e,prior).astype(F)
    g_.A+=F(0.1/ns)*np.outer(h*gg*(1-c*c),s).astype(F)
    g_.G+=F(0.1/ns)*np.outer(h*(c-s)*gg*(1-gg),s).astype(F)
    g_.b+=F(0.1)*(h*(c-s)*gg*(1-gg)).astype(F)
    s=prior  # dinâmica reduzida determinística
err_ev=[]
for fr in frames[200:300]:
    c=np.tanh(g_.A@s); gg=sigm(g_.G@s+g_.b); prior=((1-gg)*s+gg*c).astype(F)
    pred=(W0@prior).astype(F); e=fr-pred
    err_ev.append(float(np.mean(e*e))); s=prior
out.append(c_arr("g_frames", frames[:300]))
out.append(f"#define G_MSE_TRAIN {c_lit(np.mean(err_tr))}\n")
out.append(f"#define G_MSE_EVAL {c_lit(np.mean(err_ev))}\n")
open("c/golden_twostroke.h","w").write("".join(out))
print("golden ok  mse_train=%.6f mse_eval=%.6f"%(np.mean(err_tr),np.mean(err_ev)))
