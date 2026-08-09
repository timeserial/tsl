# Rede preditiva hierárquica esparsa — Phase 0

Passos 1 e 2 do plano em `CONTEXTO.md`: hierarquia preditiva de 3 camadas
geradoras (64 → 32 → 16 → 8) a prever a próxima trama de um sinal, com
aprendizagem local (sem backprop), esparsidade por limiar e early exit; depois
os mesmos pesos ternarizados para {-1,0,1} e programados num crossbar com
defeitos.

O objetivo **não** é uma métrica de ML — é responder a perguntas com números
que possam falhar:

| # | pergunta | onde se responde |
|---|----------|------------------|
| 1 | a dinâmica de assentamento converge? | `test_settling_decreases_the_energy` |
| 2 | a aprendizagem local funciona? | `test_local_learning_beats_the_persistence_baseline` |
| 3 | o limiar de esparsidade compensa? | `test_higher_threshold_silences_more_and_converte_menos` |
| 4 | o compute é proporcional à surpresa? | `test_surprising_frames_cost_more_than_banal_ones` |
| 5 | os pesos aguentam ser ternários? | `test_training_on_the_crossbar_beats_quantising_at_the_end` |
| 6 | o ciclo tolera um dispositivo defeituoso? | secção 3 de `run_step2.py` |

## Correr

```sh
.venv/bin/python -m pytest            # 115 testes, ~60 s
.venv/bin/python scripts/run_phase0.py    # passo 1: ~60 s
.venv/bin/python scripts/run_step2.py     # passo 2: ~8 min
.venv/bin/python scripts/benchmark.py --data ETTm1.csv --column OT
.venv/bin/python scripts/ablate.py --har "UCI HAR Dataset"
.venv/bin/python scripts/demo.py --events 6   # ver funcionar no terminal
```

O pacote depende só de NumPy. O `.venv` traz também torch, e é só para as
linhas de base do `benchmark.py` serem as verdadeiras em vez de versões
escritas à mão por quem tem interesse no resultado.

`scripts/run_phase0.py` escreve `runs/phase0/` com `model.npz`, `model.h`,
`golden.h` e `results.json`; `run_step2.py` escreve `runs/step2/results.json`.

## Passo 1 — resultados (60 passagens, defaults, seed 0)

```
NRMSE de previsão da trama seguinte
  repetir a trama anterior (linha de base) : 1.229
  rede antes de treinar                    : 0.993
  rede depois de treinar                   : 0.219

Assentamento: 120/120 tramas com energia monótona decrescente,
              energia final = 6% da inicial.

Limiar de esparsidade θ:
    θ  NRMSE  silenciados_%  ADC_%  MACs_sobem_%  iters
  0.00  0.218            0.0   22.4          94.1   4.83
  0.02  0.219           73.9    5.9          35.4   4.84
  0.05  0.307           85.5    2.3          24.1   3.09
  0.10  0.417           89.1    1.3          21.0   2.19

Compute vs. surpresa (12 transitórios em 200 tramas):
  banal            surpresa 0.12   5.4 iters   207 conversões ADC
  com transitório  surpresa 1.53   7.4 iters   686 conversões ADC
  -> uma trama surpreendente custa 3.3× uma banal
```

O número que interessa é a coluna **ADC_%**: conversões efetivamente feitas
sobre as que uma rede densa faria a correr sempre até ao teto de iterações.
A θ = 0.02 são 5.9% — e o erro de previsão não piora (0.219 vs. 0.218).
Silenciar o que já estava previsto não custa exatidão; custa só a exatidão
que se estava a comprar com ruído.

Ressalva: nada disto é uma medição de energia. É contagem de operações e de
conversões numa máquina que não é a máquina certa. O argumento energético só
se demonstra no crossbar (passo 4 do plano); ver a secção 6 do `CONTEXTO.md`.

## Passo 2 — o substrato (3 seeds, média ± desvio)

Nada aqui é reportado sem dispersão: um dispositivo analógico é uma amostra de
uma distribuição, e medir um só seria medir a sorte.

```
Quantização                                  NRMSE      vs float
  float32                              0.281 ± 0.100         —
  ternário, quantizado no fim (PTQ)    0.658 ± 0.062     2.34×
  ternário, treinado no crossbar (QAT) 0.314 ± 0.006     1.12×
  (43% dos pesos ternários ficam a zero: dispositivos que nem
   é preciso programar nem ler)

Variabilidade de programação          programado    treinado no
                                        só no fim    dispositivo
  σ_rel = 0.10                      0.336 ± 0.005  0.331 ± 0.015
  σ_rel = 0.20                      0.674 ± 0.419  0.321 ± 0.012
  σ_rel = 0.40                      1.835 ± 1.081  0.341 ± 0.012

ADC (σ_rel = 0.1)   8 bits 0.337   6 bits 0.522   4 bits 0.770
                                                  3 bits 1.000

Limiar θ em ternário (σ_rel = 0.1)
    θ = 0.00   NRMSE 0.408    0% silenciados   ADC 47.6%
    θ = 0.02   NRMSE 0.336   58% silenciados   ADC 14.0%
```

Treinar com o quantizador dentro do laço custa quase nada de código: a regra é
local, não há gradiente para fazer passar através dele. O peso float é o
shadow weight, o ternário é o que está no crossbar. É a única parte do desenho
onde a ausência de backprop dá uma vantagem *de engenharia*, e não só de
energia.

### O ciclo é auto-corretivo — e há um limite exato para isso

| σ_rel | energia em malha aberta | depois de assentar | fechado pelo ciclo | NRMSE |
|-------|------------------------|--------------------|--------------------|-------|
| 0.0   | 0.38 | 0.022 | 94% | 0.314 ± 0.006 |
| 0.1   | 0.43 | 0.029 | 93% | 0.336 ± 0.004 |
| 0.2   | 2.39 | 0.182 | 92% | 0.674 ± 0.420 |
| 0.4   | 16.47 | 0.918 | 94% | 1.835 ± 1.081 |

Com um crossbar 40% fora de especificação a energia em malha aberta dispara
43× — e o assentamento continua a fechar 94% dela, com os estados sempre
limitados. **O ciclo corrige mesmo os defeitos do dispositivo.** O que não
corrige é a *previsão* da trama seguinte, e não pode: essa desce pelos mesmos
pesos partidos antes de existir qualquer erro que a guie. Auto-correção não é
omnisciência — e é exatamente por isso que o treino tem de acontecer no
dispositivo, como a secção 6 do `CONTEXTO.md` antecipava. Treinar no crossbar
recupera 98% da degradação a σ_rel = 0.2 e 0.4.

### Há um penhasco, e é por dispositivo

Sem retreinar, varrendo σ_rel finamente sobre três crossbars:

```
σ_rel   0.00  0.08  0.12  0.14  0.16  0.18  0.20  0.25  0.30
  #0    0.31  0.33  0.35  0.37  0.37  0.38  0.40  1.27  2.33
  #1    0.31  0.33  0.35  0.42  0.58  0.81  1.27  1.97  2.16
  #2    0.32  0.33  0.33  0.34  0.35  0.35  0.36  0.37  0.40
```

Degradação suave até ~15%, e depois *alguns* crossbars caem de um penhasco
enquanto outros continuam bem. A previsão que cai fica **descorrelacionada** do
alvo, não apenas mal escalada — reescalá-la pelo ganho ótimo não recupera nada
(NRMSE 0.97, tão bom como prever zero) — e os estados permanecem limitados. Não
é saturação nem divergência numérica: é uma bifurcação da dinâmica em malha
fechada. Treinar no dispositivo elimina o penhasco.

Consequência prática: **para esta arquitetura não se pode treinar-e-programar.**
Ou se treina no dispositivo, ou se calibra nele.

## Comparação com problemas reais

`scripts/benchmark.py` põe a rede preditiva contra o que realmente compete
neste nicho — persistência, AR linear, MLP, GRU e um transformer causal
pequeno — todos com orçamento de parâmetros comparável e todos no mesmo
protocolo de streaming: no instante *t* o modelo vê tudo até *t* e tem de
dizer a trama *t+1*. Ninguém vê o futuro, e a normalização usa só estatísticas
do troço de treino.

**Não com um LLM.** Um LLM não resolve este problema e este modelo não resolve
o do LLM; a secção 1 do `CONTEXTO.md` decidiu explicitamente não competir com
transformers em linguagem. Comparar 2752 parâmetros a prever um sensor com um
modelo de linguagem não é uma comparação difícil, é uma comparação sem eixo
comum.

### ETTm1, temperatura de óleo de transformador (dados reais, 15 min)

O benchmark padrão dos papers de forecasting com transformers. 1088 tramas de
64 amostras, 871 treino / 217 teste.

| modelo | NRMSE | parâmetros | MACs/trama |
|--------|-------|-----------|-----------|
| **rede preditiva + via rápida** | **0.212** | 6848 | — |
| AR linear (ordem 4) | 0.211 | 16448 | 16 448 |
| transformer causal (d=12, ctx=16) | 0.225 | 2896 | 2 928 |
| rede preditiva (passo 1) | 0.235 | 2752 | 42 015 |
| MLP | 0.243 | 2953 | 2 880 |
| GRU | 0.246 | 1449 | 1 355 |
| persistência | 0.285 | 0 | 0 |

Leitura honesta: **na versão do passo 1** a rede fica a meio da tabela — bate o
MLP, o GRU e a persistência, perde para o transformer e para uma regressão
linear de ordem 4 com seis vezes mais parâmetros. Com a via rápida (secção
seguinte) passa a empatar com o melhor, usando 2.4× menos parâmetros que o AR.

Nenhuma destas diferenças é grande. A leitura que interessa não é a ordem da
tabela — é que uma arquitetura desenhada para outras restrições, treinada sem
backprop, fica ao nível do que existe. É o que a estratégia da secção 1 do
`CONTEXTO.md` precisa que seja verdade, e é o mínimo que precisava.

### O sinal de brinquedo era fácil demais

O mesmo banco de ensaios aplicado ao sinal do passo 1 revelou algo que eu
devia ter verificado no início: **um AR linear de ordem 4 obtém 0.040 contra os
nossos 0.118.** Uma soma de três sinusóides é, por construção, um sistema
linear — e validar uma arquitetura não-linear hierárquica num problema que uma
regressão linear resolve melhor não valida grande coisa. Os números do passo 1
continuam corretos; o que muda é o peso que merecem. É por isso que a
comparação passou a correr sobre dados reais.

### HAR, giroscópio a 50 Hz: quando ninguém consegue

Um controlo que valia a pena correr antes de concluir seja o que for sobre o
modelo. Tramas de 64 amostras a 50 Hz são 1.28 s de movimento humano, e a
esse horizonte o sinal é praticamente imprevisível:

| modelo | NRMSE | parâmetros |
|--------|-------|-----------|
| **rede preditiva** | **0.986** | **2752** |
| transformer causal | 1.014 | 2896 |
| GRU | 1.044 | 1449 |
| MLP | 1.059 | 2953 |
| persistência | 1.179 | 0 |
| AR linear (ordem 4) | 1.983 | 16448 |

Toda a gente fica em ~1.0, que é o mesmo que dizer "prevejo zero". A tarefa é
que não tem sinal a esse horizonte — e o AR linear, com 16 448 parâmetros para
327 tramas de treino, fica *pior* que prever zero, que é o que o
sobreajustamento faz.

Ser o único abaixo de 1.0 num problema onde ninguém consegue é uma vitória
modesta, e é assim que deve ser lida. O que este controlo vale mesmo é como
antídoto: sem ele, o 0.986 lia-se como "o nosso modelo falhou aqui" quando o
que falhou foi o enquadramento do problema.

### O que melhorou o modelo, e o que não

Três ideias de neurociência foram implementadas e medidas isoladamente
(`scripts/ablate.py`). Duas não pagaram. Uma pagou muito.

**Escalas de tempo hierárquicas** (`temporal.py`) — Hasson mediu janelas
receptivas temporais que crescem ao longo do córtex; Kiebel, Daunizeau &
Friston mostram que isso cai de um modelo preditivo onde cada nível tem
dinâmica própria, progressivamente mais lenta.

O resultado **depende do problema**, e é isso que o torna interessante. No
sinal de brinquedo e em ETTm1 não ajuda nada — porque esses sinais não *têm*
hierarquia de escalas de tempo para modelar. No acelerómetro do HAR ajuda
(0.594 → 0.559), porque um sensor colado a uma pessoa tem mesmo estrutura a
várias escalas: a passada, o gesto, a mudança de atividade.

A lição não é sobre a ideia, é sobre o método: passei um bom bocado a
concluir que a ideia não servia, quando o que não servia era o problema onde a
estava a medir. Fica ligável (`level_transition`), desligada por omissão até
haver um demonstrador cuja estrutura a justifique.

**Precisão como atenção** (`precision.py`) — π = inverso da variância
esperada do erro, que Friston identifica com a atenção e com a modulação de
ganho pós-sináptico. Sozinha, é um empate em exatidão. **Mas paga quando há
uma via rápida**, e muito: corta as conversões ADC em 5-8× sem perder
exatidão. Faz sentido — só vale a pena estimar a fiabilidade de um canal
quando há mais do que um canal.

**Via rápida** (a secção 4 do `CONTEXTO.md`, que faltava) — uma via curta e
sempre ligada que prevê a trama seguinte diretamente da anterior, em paralelo
com a hierarquia profunda, à maneira das vias cortico-subcorticais da
*shallow brain hypothesis*:

```
ẑ_0 = A_0·x(t-1)  +  f(W_0·z_1)
      └ via rápida ┘  └ hierarquia ┘
```

A decomposição é aditiva de propósito — os gradientes de tudo o resto ficam
iguais — mas **cada via aprende com o seu próprio erro**, não com o mesmo. A
hierarquia deixa de ter de explicar o sinal e passa a explicar o resíduo.

| conjunto | variante | NRMSE | ADC/trama | iterações |
|----------|----------|-------|-----------|-----------|
| brinquedo¹ | passo 1 | 0.186 ± 0.005 | 567 | — |
| brinquedo¹ | via rápida + precisão | **0.074** | **99** | — |
| ETTm1¹ | passo 1 | 0.230 ± 0.005 | 583 | — |
| ETTm1¹ | via rápida (μ=0.05) | **0.212** | 913 | — |
| ETTm1¹ | via rápida + precisão | 0.218 | **112** | — |
| HAR acc_x² | passo 1 | 0.594 ± 0.012 | 1 370 | 14.9 |
| HAR acc_x² | via rápida | **0.532 ± 0.007** | 1 500 | 15.5 |
| HAR acc_x² | via rápida + precisão | 0.539 ± 0.000 | **141** | **1.2** |

¹ 2 seeds  ² 3 seeds. A ablação completa de ETTm1 com 3 seeds ficou por
correr; reproduz-se com
`scripts/ablate.py --data ETTm1.csv --column OT --seeds 3`.

Melhor exatidão *e* 5-10× menos conversões, em três conjuntos diferentes. Com
a via rápida a rede iguala o melhor de todos os baselines em ETTm1 (AR de
ordem 4, 0.211) com 2.4× menos parâmetros.

O número mais revelador é o das iterações no HAR: **1.2**. A via rápida
explica quase tudo, a precisão desconta corretamente o que sobra, e a
hierarquia cara só acorda quando é mesmo precisa. É exatamente o mecanismo que
a secção 4 do `CONTEXTO.md` descreve — a via grosseira sempre ligada que só
chama o circuito lento quando não chega — e apareceu sem ser programado como
tal: sai do critério de paragem existente assim que há uma via rápida a
baixar a surpresa inicial.

**Falhou duas vezes antes de funcionar**, e as duas falhas ensinam mais que o
sucesso. Primeiro, treinar as duas vias com o mesmo resíduo: perseguem o mesmo
sinal sem nada que atribua crédito, e disputam-no (ETTm1 0.230 → 0.724).
Depois, taxa de aprendizagem acima do limite de estabilidade (0.208 → 0.899).

### Escolher o problema é metade do trabalho

Uma coisa que só se vê depois de correr o banco de ensaios em vários sítios:
**os problemas ao nosso alcance são ou lineares demais ou impossíveis.**

- No sinal de brinquedo, um AR de ordem 4 obtém 0.040 contra os nossos 0.118.
- Em ETTm1, um AR de **ordem 1** — uma única matriz 64×64 ajustada por mínimos
  quadrados — obtém 0.216, melhor que os nossos 0.230. A tarefa é dominada por
  estrutura linear de uma trama para a seguinte.
- No giroscópio do HAR, ninguém fica abaixo de 1.0 (nós ficamos em 0.986).

Uma hierarquia preditiva não tem nada a ganhar num problema linear nem num
problema sem sinal. Isto não é uma desculpa, é uma restrição de desenho: o
demonstrador do passo 4 tem de ser escolhido com este cuidado, e a afirmação a
defender não é "somos mais exatos que um transformer" — é a que a secção 1 do
`CONTEXTO.md` já tinha escolhido: **igualar a exatidão do que existe, a uma
fração das conversões, num nicho onde o transformer nem entra.** Nesse eixo os
números existem e estão acima. No eixo da exatidão pura, a resposta honesta
neste momento é "equivalente, não superior".

### Os três eixos de custo, e o que cada um esconde

Uma coluna só mente. Em **MACs digitais** a rede preditiva perde por uma
margem grande — assentar iterativamente faz várias passagens onde os outros
fazem uma, e 42 015 contra 2 880 do MLP não é uma diferença que se argumente.
Em **conversões ADC** — o que se paga num crossbar analógico, onde as
multiplicações são feitas pela física — a conta inverte-se, porque só o erro
acima do limiar é convertido.

É essa segunda coluna que carrega a tese inteira. Se a arquitetura não ganhar
aí, não ganha em lado nenhum, porque em operações digitais já perdeu. E medir
isso a sério exige o crossbar, não o Mac.

## Aprendizagem contínua: memória episódica e consolidação

`src/pcnet/episodic.py` implementa a peça que faltava da secção 4 do
`CONTEXTO.md` — o hipocampo e o sono. Um armazém key-value de tamanho fixo,
endereçado pelo estado do topo (o resumo que a rede faz do contexto), que
grava quando há surpresa e é reproduzido offline para destilar nos pesos.

`scripts/continual.py` mede o que isso devia comprar: N tarefas aprendidas
**em sequência**, sem nunca voltar atrás, e no fim mede-se em todas.

```
3 tarefas, 25 passagens cada, treino sequencial

                              NRMSE final   esquecimento
TETO: treino conjunto               0.823          0.000
rede preditiva (sem memória)        0.950          1.069
+ memória episódica                 0.950          1.067
+ memória + consolidação            0.891          0.519
+ memória + replay intercalado      0.983          0.463
MLP (backprop)                      1.007          1.153
GRU (backprop)                      1.040          1.398
```

**Duas leituras, e a segunda anula a primeira.**

A boa: a consolidação **corta o esquecimento a metade** (1.069 → 0.52), e as
redes treinadas com backprop esquecem mais do que nós (1.15 e 1.40). A memória
episódica sozinha não faz nada — é o *replay* que conta, não o armazém.

A que interessa: **o teto é 0.823.** Treinar nas três tarefas ao mesmo tempo,
em condições ideais e intercaladas, dá quase o mesmo que treinar em sequência
e esquecer tudo. Ou seja: **a arquitetura não consegue reter três tarefas nem
quando as vê todas juntas.** O que a tabela mede não é esquecimento — é falta
de capacidade. Não se mede esquecimento em tarefas que o modelo nunca reteve.

Foi a terceira vez neste projeto que medi uma boa ideia no problema errado
(depois do sinal de brinquedo e do giroscópio do HAR). Desta vez o controlo
apanhou-o antes de virar conclusão. **A primeira versão do controlo estava
errada** — concatenava as tarefas em vez de as intercalar, portanto era treino
sequencial com outro nome, e dava um "teto" mais baixo do que as variantes que
devia limitar. Um controlo mal feito é pior que nenhum, porque parece rigor.

### Porque é que o teto não sobe

**Aviso de contaminação, e a auditoria.** Uma primeira ronda inteira de
hipóteses (capacidade, mistura de dinâmicas, aquecimento, oráculo, esparsidade,
expansão) foi corrida com a via rápida ligada — e a via rápida, sendo um
preditor linear partilhado, fixa o resultado no compromisso (~0.82) e mascara
qualquer efeito da hierarquia. Descobriu-se quando duas forças de
metaplasticidade separadas por um fator de 100 davam resultados idênticos.
Tudo o que se segue foi **refeito sem via rápida**.

Resultado da repetição limpa (treino conjunto, 3 tarefas):

| variante | NRMSE médio |
|----------|------------|
| base (64,32,16,8) | **0.778** |
| metaplasticidade λ=10 | 0.833 |
| mistura de 3 dinâmicas | 0.842 |
| esparsidade 10% | 1.001 (= prever zero) |
| capacidade (64,64,48,32) | 1.103 |
| expandir+esparsificar (giro dentado) | 1.048 |
| *AR(1) linear, mínimos quadrados* | *0.741* |
| *GRU com backprop* | ***0.397*** |

A repetição não salvou nenhum mecanismo — pelo contrário: sem a máscara, eles
são *ativamente prejudiciais*, não neutros. E a base simples continua pior que
uma única matriz linear.

### Quanto custa não usar backprop: 2×

O teste que faltava não era outro botão da arquitetura — era mudar a *regra de
aprendizagem*. As mesmas três tarefas, o mesmo treino conjunto intercalado, um
modelo de capacidade comparável, treinado com backprop:

| método | parâmetros | NRMSE médio (3 tarefas) |
|--------|-----------|------------------------|
| GRU, backprop | 5024 | **0.397** |
| GRU, backprop | 2352 | 0.509 |
| **AR(1) linear, mínimos quadrados** | 4096 | **0.741** |
| rede preditiva, regra local | 6848 | 0.778 – 0.826 |

*(numa tarefa só, a rede preditiva fica em 0.07 e o GRU em ~0.09)*

E é aqui que a coisa fica séria: **a nossa rede é pior do que uma única matriz
linear.** Todas as variantes que testei — capacidade, mistura de dinâmicas,
esparsidade, expansão, oráculo, mais iterações — aterram entre 0.78 e 0.83, à
volta do compromisso linear. Não é que a hierarquia ajude pouco: não ajuda
nada. Todo o resultado estava a ser fixado por uma parte linear do modelo, e as
minhas experiências arquitetónicas mediam ruído em cima disso.

Duas conclusões, e a primeira corrige uma afirmação anterior deste README que
era forte demais.

**Não é a arquitetura, é a regra.** Com backprop, capacidade comparável faz
0.397. A representação existe; o que falha é encontrá-la sem atribuição de
crédito global.

**E o cérebro não nos salvou.** Tentei, por esta ordem, as respostas
biológicas ao esquecimento catastrófico: sistemas complementares de memória
(hipocampo rápido + córtex lento), replay durante o sono, replay intercalado,
separação de padrões por códigos esparsos, expansão antes de esparsificar
(giro dentado, corpo pedunculado da mosca), e dinâmicas comutadas por
contexto. **Nenhuma moveu o número.** Só o replay mexeu — e mexeu no
esquecimento, não no teto.

A leitura honesta é que estes mecanismos pressupõem que a aprendizagem
subjacente consegue formar representações distintas quando lhe damos espaço
para isso. A regra local não forma: converge para o compromisso, e nenhuma
quantidade de espaço, esparsidade ou memória a faz mudar de ideias.

Isto é o muro da atribuição de crédito, medido no nosso próprio sistema em vez
de citado. Não é fatal para *uma* tarefa — aí a rede é boa e continua a ser o
que o projeto precisa. É fatal para a ambição de acumular vários mundos, e não
o vou disfarçar.

## Como está organizado

```
src/pcnet/
  config.py      PCConfig — também o contrato com o C
  layer.py       PCLayer  — predict / modulated_error / backward / learn
  network.py     PCNetwork — o laço de assentamento e a transição temporal
  metrics.py     StepTrace, RunStats — a instrumentação
  signals.py     o sinal de brinquedo e a linha de base de persistência
  device.py      ternarização, variabilidade, ruído de leitura, ADC
  train.py       laço de treino, varrimento de θ
  export.py      npz, model.h, golden.h — a fronteira treino/inferência
  report.py      tabelas de texto para os scripts
scripts/run_phase0.py   passo 1
scripts/run_step2.py    passo 2
tests/
c/               inferência em C (passo 3) — ver c/README.md
```

## O algoritmo

Por trama, três fases:

**1. A previsão desce.** O topo propaga-se no tempo, `ẑ_L = tanh(A·z_L(t-1))`,
e daí desce nível a nível, `ẑ_l = f(W_l·z_{l+1})`, até gerar uma trama
esperada. Isto acontece antes de ver os dados: o primeiro `ε_0` é o erro de
previsão em malha aberta.

**2. A surpresa sobe.** `ε_l = z_l − ẑ_l`; o que tiver `|ε| < θ` não é
transmitido; o resto corrige os estados de cima,
`Δz_l ∝ W_{l-1}ᵀ(ε̃_{l-1} ⊙ f') − ε̃_l`. Repete-se enquanto valer a pena.

**3. Aprende-se localmente.** `ΔW_l ∝ (ε_l ⊙ f') ⊗ z_{l+1}` e
`ΔA ∝ ε_L ⊗ z_L(t-1)`. Só quantidades presentes na sinapse: sem grafo, sem
ativações guardadas, sem backprop.

### Decisões que não são óbvias

**A surpresa mede-se pela energia `½Σ‖ε_l‖²`, não pela soma dos `|ε|`.**
A energia é a função que a dinâmica de assentamento desce, logo é a única que
serve de critério de convergência. A soma L1 não serve: os níveis latentes
arrancam com `ε = 0` por construção (foram inicializados pela previsão de
cima) e crescem à medida que assumem a sua parte da explicação, o que faz a
soma L1 oscilar mesmo quando a energia desce de forma monótona. Foi este
detalhe que, na primeira versão, fez as tramas *mais* surpreendentes parecer
*mais* baratas.

**O early exit tem três razões, e só uma é boa.**
`explicado` (nada passou o limiar — silêncio, custo zero), `estagnado`
(outra iteração não paga o que custa) e `teto` (gastou o orçamento). A
distinção interessa porque `estagnado` é a rede a desistir de algo que não
sabe explicar: poupa energia, mas é uma admissão de ignorância, não uma
vitória.

**O critério de paragem é absoluto, não relativo.** `settle_min_gain` compara
a redução de energia com o custo de mais uma iteração — joules contra joules.
Com um critério *relativo*, uma trama cujo resíduo é irredutível estagna de
imediato e a rede gasta *menos* onde há *mais* surpresa, que é exatamente o
contrário do que a arquitetura promete. O critério relativo continua
disponível (`settle_rel_tol`), desligado por omissão.

**Todas as regras delta deste projeto têm o mesmo limite de estabilidade, e
esqueci-o três vezes.** Uma regra `ΔW ∝ ε·xᵀ` só é estável para
`lr < 1/‖x‖²`. Aconteceu no `z_lr` do assentamento (a 0.5 diverge), aconteceu
outra vez quando os pesos ternários inflaram `σ_max`, e aconteceu uma terceira
com a via rápida — onde `lr = 0.02` estava acima do limite `1/‖x‖² ≈ 0.016` e
transformava um modelo de 0.208 num de 0.899. A correção definitiva não é
afinar o número, é **normalizar o passo** (`ΔA ∝ ε·xᵀ/‖x‖²`, que em filtragem
adaptativa se chama NLMS): a taxa passa a ser adimensional — "que fração do
erro corrigir por trama" — e deixa de depender da escala do sinal, portanto
deixa de ser preciso reafiná-la por conjunto de dados.

**`z_lr` tem de ser pequeno, e o limite é calculável.** O assentamento é
descida de gradiente com passo fixo e só é estável para
`z_lr < 2/(1 + σ_max(W)²)`; como `σ_max` cresce com a aprendizagem, o valor
seguro desce ao longo do treino. A 0.5 a rede diverge. Por omissão o passo é
limitado por esse bound (`adaptive_z_lr`), com `σ_max` estimado por iteração de
potência — duas leituras do crossbar por atualização de pesos. Não é grátis: a
σ_rel = 0 custa um pouco (0.302 → 0.315) e a σ_rel = 0.4 poupa muito
(0.417 → 0.341). `z_clip` e `z_max` são a rede de segurança que resta — e são
de graça no destino, onde a aritmética int8 satura sozinha.

**O erro sensorial sobe um nível por iteração.** Com um teto de assentamento
abaixo de `n_levels - 1`, o estado do topo nunca chega a ser corrigido, nunca
sai de zero, e a rede prevê zero — NRMSE exatamente 1.0, indistinguível de não
existir. Os "3-10 passos" do plano não são um número arbitrário: 3 é a
profundidade da hierarquia. Fica preso em
`test_settling_budget_below_the_depth_predicts_nothing`.

**A decisão de saída é discreta, logo o comportamento por trama é caótico.**
Uma diferença de 0.01% no passo de assentamento chega para o laço parar uma
iteração antes ou depois; uma iteração muda o estado do topo, que atravessa
para a trama seguinte. Medido: 1e-4 de perturbação no passo dá ~0.1 de
diferença numa trama concreta, com a média intacta. Isto **muda o contrato com
o C**: validar o inferidor trama a trama, ou pelo número de iterações, é
validar ruído. Ver `c/README.md`.

**Avaliar não pode perturbar o treino.** A rede é online e o estado do topo
atravessa as tramas, por isso `evaluate()` guarda e repõe o estado dinâmico.
Sem isso, medir a meio do treino altera o treino (custava ~0.02 de NRMSE).

## Limitações conhecidas

- **A qualidade da previsão não é monótona no número de passagens**, e a
  dispersão entre seeds do modelo float é alta (0.281 ± 0.100). A regra
  Hebbiana tem taxa fixa e nunca recozimento, por isso continua a oscilar em
  torno da solução. Testei quatro esquemas de anelagem: os agressivos congelam
  a rede antes de convergir (0.65), o suave empata. `weight_decay` estabiliza
  mas piora o mínimo. Fica por resolver. Entretanto o protocolo de medição
  contorna-o — tudo o que é comparação de dispositivos é feito sobre os
  *mesmos* pesos, e o resto leva ≥3 seeds com dispersão reportada.
  Curiosamente o modelo ternário é bastante mais estável (0.314 ± 0.006): a
  quantização atua como regularizador.
- **O ruído de dispositivo é um modelo genérico**, não o do PoC AIMC. Os
  parâmetros (`sigma_rel`, `sigma_abs`, `stuck_frac`, `read_sigma`) estão
  desenhados para receber valores do ngspice, mas os números só valem quando
  ligados ao circuito real. A gama que interessa — variabilidade de ReRAM/PCM
  a 5-20% — é onde o penhasco começa a aparecer, por isso essa calibração não
  é um detalhe.
- **A transição temporal A fica fora do crossbar** por omissão
  (`include_transition=False`): são 64 pesos contra 2688, e no desenho é a via
  rápida, que faz sentido manter digital. A suposição é testável ligando a
  flag; não a testei.
- **O limiar durante o treino custa alguma exatidão** face a treinar com
  θ = 0 e só o aplicar na inferência. Não está explorado. Em contrapartida, com
  ruído de dispositivo o limiar *ajuda* (0.408 → 0.336): silenciar erros
  pequenos suprime também o ruído do substrato.
- **A "via rápida" ainda não existe como tal.** A transição do topo é a sua
  forma mais pobre; falta o gate de confiança que decide quando acordar a via
  lenta (secção 4 do `CONTEXTO.md`).
- **Não há memória episódica nem consolidação offline** (secção 4).

## Próximos passos

3. Inferência em C, validada contra `golden.h` — mas com o contrato revisto em
   `c/README.md`: aritmética exata numa iteração forçada, agregados com
   tolerância no resto.
4. ESP32 + crossbar.

E, atravessado a ambos: ligar `DeviceModel` ao modelo ngspice do PoC AIMC,
para o penhasco da secção 3b ser medido em silício e não em suposições.
