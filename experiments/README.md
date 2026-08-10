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
4. **Perturbação + dopamina** — a aposta de fundo: regra local (enviesada,
   estável) + perturbação de nós (não-enviesada, ruidosa) combinadas como
   estimador com controlo de variância. Por desenhar.

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
