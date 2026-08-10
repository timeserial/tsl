# Frentes abertas no problema do crédito local

Estado: campeão 0.659 ± 0.046 (regra local, portão+raso) vs alvo 0.575 ± 0.019
(gradiente exato, mesma célula). O intervalo ~13% é o preço medido da
localidade com toda a arquitetura igualada. Marcos e história: README raiz.

Por ordem:

1. ~~Assentamento completo~~ — nulo (0.672/0.664 vs 0.659), re-testado no campeão.
2. ~~Intercalação fina~~ — medida (blocos 16/4/1, alvo re-medido por bloco):
   o intervalo encolhe (0.084 → 0.057 → 0.013) mas por degradação mútua rumo
   ao colapso (a bloco 1 ambos ~1.0 — sem continuidade temporal não há
   tarefa). Miragem; o preço da localidade não se paga pelo ritmo dos dados.
   Nota menor: intercalação fina magoa o GRU mais do que a nós.
3. ~~Crítico de 1 bit~~ — implementado (`critic_retract`) e nulo em todas as
   doses (0.681/0.653/0.659 vs 0.659±0.046). Um bit de retração chega tarde e
   desfaz bom e mau por igual.
4. ~~Perturbação + dopamina~~ — REINFORCE antitético sobre A diverge (3.63).

## A REGRA DE DOIS TEMPOS (o resultado da sessão)

Creditar do erro em malha aberta retroprojetado (h = W0ᵀ·ε0, cadeia exata
pelo portão), ANTES de assentar. O assentamento infere; o crédito tira-se
antes. Numa rede rasa isto é 100% local — a transposta é a leitura que o
crossbar já faz.

Conjunto 3 mundos: 0.579±0.004 (Fila 1) e 0.551±0.020 (Fila 3, variante),
contra 0.575±0.019 do GRU-gradiente-exato. EMPATE. A regra local antiga
fazia 0.659. Ressalvas medidas: imposto em tarefa única (0.108 vs 0.087);
o sequencial ainda precisa da proteção acoplada à via nova.

Mortos nas filas seguintes: traços e-prop sobre a regra nova (0.595-0.833,
todos piores); híbrido acordado/dormir (0.910 — as regras lutam);
composição por componente W0-clássico+transição-exata (2.08 — diverge:
duas regras, dois alvos, um W0).

Progressão do conjunto (regra de dois tempos, SGD):
0.579 (80ép, lr 0.1) → 0.534 (200ép, lr 0.05) → **0.521 ± 0.024** com
assentamento fundo na inferência (mi=50, θ=0 — a vantagem que o GRU não
tem valeu 0.013). Alvo 0.490 a ~1σ.

Mortos adicionais: RMSProp local (1.09/0.67 — variância amplifica direções
raras do fluxo intercalado), crédito exato a 2 passos (1.05), topo 48
(0.63). Alerta aberto: variantes de implementação da mesma regra dão
0.551 vs 0.579 (~0.03 por explicar); canónico é o conservador.

Sequencial: neurogénese (agente) 0.818±0.046 — recorde, integrada em
src/pcnet/neurogenesis.py.

Em curso: Fila 8 (400ép, lr 0.02-0.03, eval mi=100), Fila 7 (multi-mundo
REAL: ETTm1+HAR acc+HAR gyro, clássica vs dois-tempos vs GRU-0 vs BPTT).

Banco de suplentes (por ordem de aposta): crédito híbrido acordado-local /
sono-gradiente sobre episódios (pragmático, alta probabilidade, quebra a
pureza); dendrites segregadas (Guerguiev-Richards, o ataque científico);
neurogénese por novidade (para o esquecimento); pesadelos ressuscitados no
substrato certo (slots episódicos discretos, ou Forward-Forward com misturas
como negativos).

Mortos com causa, não repetir: pesadelos (cego e dirigido), relé talâmico,
portões espaciais, traços de elegibilidade (neutro), θ↑/θ↓ no treino,
assentamento completo, capacidade, mistura por semelhança, esparsidade de
código, expansão à giro dentado.
