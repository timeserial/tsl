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

## SEQUENCIAL REAL: as descobertas transferem

3 domínios reais em sequência, nunca revisitar (o protocolo do sensor):
clássica 1.151±0.025 (esquecimento catastrófico) · neurogénese
0.812±0.013 · **dois-tempos 0.800±0.026** — ambas as descobertas
transferem; o crédito em malha aberta interfere menos com o passado por
natureza. Combinação dois-tempos+neurogénese ainda por medir.

## DADOS REAIS: a regra local ultrapassa o backprop

Multi-mundo real — ETTm1 (transformador, 15 min) + HAR acelerómetro +
HAR giroscópio (50 Hz), intercalados em blocos de 16, 327 tramas/domínio:

| regra | NRMSE |
|---|---|
| local clássica (erro assentado) | 0.713 ± 0.039 |
| GRU + BPTT completo | 0.650 ± 0.003 |
| GRU gradiente exato (sem tempo) | 0.646 ± 0.001 |
| **dois tempos (local)** | **0.636 ± 0.001** |

**VEREDICTO PRÉ-REGISTADO (Fila 13): ADVERSÁRIO.** Com busca igual dos
dois lados e seleção por validação, o GRU afinado faz 0.645±0.002 e a
configuração que a NOSSA validação escolheu colapsou no teste
(0.685±0.017). O claim "batemos o backprop no real" morre aqui. Nota
importante: a configuração fixada a priori (do sintético) continua a
fazer 0.636 — melhor que qualquer adversário — mas não foi a escolhida
pelo protocolo, e regra é regra. A lição nua: o défice atual é de
ROBUSTEZ (sensibilidade a config/seed/split; o GRU+Adam é ±0.002), não
de desempenho médio. É o próximo alvo do drawing board.

Fila 8 (paciência, lr 0.02, 400ép): seeds 0.474/0.438/0.592 → média
0.501±0.066. DUAS de três seeds abaixo do alvo 0.490 (uma a 0.438) —
capacidade demonstrada, consistência não: uma seed encalha em má bacia.
Fila 10 (recozimento, 6 seeds): **0.492 ± 0.037 — paridade com o BPTT
(0.490)**; 3/6 seeds abaixo, recorde absoluto do benchmark 0.431 (nosso;
nada de nenhum tamanho fez melhor — força bruta h64 2-camadas faz
0.60-0.64, capacidade afoga nesta tarefa). ### Profundidade por empilhamento (agente): compra pouco

Pilha de 2 tijolos rasos, interface residual (o de cima explica só o que o
de baixo deixou): 0.543±0.037 vs 0.564±0.019 do tijolo único (8 seeds) —
ganho −0.022, 6/8 seeds, ~1.5σ, não significativo a 5%. Interface de
estado-completo: 1.113 (disputa de crédito, o desastre da via rápida ao
nível latente). Mecânica: o andar de cima persegue um alvo móvel e o
resíduo de um tijolo bom já é quase ruído. O backprop compra profundidade
por coordenação global; a composição local empilha limpa-resíduos. O
problema do crédito PROFUNDO de 99 continua aberto — a nossa resposta
continua a ser achatar, não aprofundar. Detalhe de reprodução: o marco
0.579 exige refresh_device() após atualizações externas de W0.
Script: experiments/profundidade_empilhamento.py.

**Fila 12, FINAL: 0.466 ± 0.028 — a média da regra local ABAIXO do BPTT
(0.490), 5/6 seeds, via recozimento + restarts escolhidos pelo erro de
treino (sem fuga de teste).** Separação ~0.9σ. Com o recorde 0.431, a
frente sintética fecha: paridade robusta, superioridade média modesta.

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
