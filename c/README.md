# Inferência em C — passo 3

Ainda vazio. Este ficheiro fixa o contrato para que o passo 3 seja tradução e
não redesenho.

## O que o Python já garante

O núcleo de inferência não usa nada que não exista em C89 com inteiros:

- Sem autograd, sem grafo, sem ativações guardadas — a regra de aprendizagem
  é local, por isso a inferência só precisa dos pesos e dos estados atuais.
- Sem alocação: todos os arrays têm tamanho conhecido em tempo de compilação
  (`pc_sizes` em `model.h`).
- `float32` em todo o lado (`pcnet/dtypes.py`), nunca `float64`, para que o
  Python e o C não divirjam em bits difíceis de explicar depois.
- Uma única não-linearidade, `tanh`, que no destino é uma LUT de 256 entradas.
- Saturação explícita (`z_clip`, `z_max`) em vez de confiar no alcance —
  em int8 isto é o comportamento natural do hardware, não código extra.

## Artefactos gerados

`python3 scripts/run_phase0.py` escreve para `runs/phase0/`:

- `model.h` — `pc_sizes`, `pc_W0..pc_W2`, `pc_A` e os `#define` da config
  (`PC_MAX_ITERS`, `PC_Z_LR`, `PC_THETA`, `PC_Z_CLIP`, `PC_Z_MAX`,
  `PC_SETTLE_TOL`, `PC_SETTLE_MIN_GAIN`).
- `golden.h` — `PC_GOLDEN_N` tramas de entrada, a previsão em malha aberta,
  o estado do topo depois do assentamento e o nº de iterações por trama.

## O teste honesto

O C tem de reproduzir `pc_golden_pred` e `pc_golden_top` a partir de
`pc_golden_input`, com estado inicial a zeros e sem aprender.

**Mas não trama a trama, e não pelo nº de iterações.** O critério de paragem é
uma decisão discreta: uma diferença de 1e-4 no passo de assentamento — ou uma
ordem de soma diferente no C, que é garantido acontecer — chega para o laço
parar uma iteração antes ou depois. Uma iteração muda o estado do topo, que
atravessa para a trama seguinte. Medido: 0.01% de diferença no passo dá ~0.1
de diferença numa trama concreta, com a média intacta. Ver
`test_exit_decision_is_chaotic_but_aggregates_are_stable`.

O que se compara, então:

1. **Uma única trama, uma única iteração forçada** (`PC_MAX_ITERS = 1`, sem
   early exit): aí sim, exato a ~1e-5. É isto que valida a aritmética.
2. **Agregados sobre `PC_GOLDEN_N` tramas**: RMSE médio dentro de ~5%, nº
   médio de iterações dentro de ±1. É isto que valida o comportamento.

`pc_golden_iters` serve de referência para a média, não para comparação
elemento a elemento. Um C que bata a aritmética no ponto 1 e os agregados no
ponto 2 está correto; um que bata as tramas todas exatamente está a ter sorte.

Se a inferência não couber em C simples com inteiros, não cabe no crossbar.
Essa é a razão de existir deste passo; ver secção 6 do `CONTEXTO.md`.
