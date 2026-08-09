"""Memória episódica: gravar agora, consolidar depois.

A secção 4 do `CONTEXTO.md` pede duas memórias, e a distinção entre elas é o
ponto todo:

  * **Episódica** — um armazém key-value gravável na hora. Um facto novo entra
    aqui *sem tocar nos pesos*. É por isso que aprender uma coisa nova não
    destrói o que já se sabia.
  * **Consolidação offline** — mais tarde, em repouso, os episódios são
    reproduzidos e destilados nos pesos geradores pela mesma regra local de
    sempre. O que se repete fica; o que foi acidental é esquecido.

É o hipocampo e o sono. E é a resposta do cérebro ao esquecimento
catastrófico: uma rede que aprende só nos pesos tem de escolher entre plástica
e estável — plástica esquece, estável não aprende. Duas memórias com
velocidades diferentes resolvem o dilema em vez de o negociar.

Três decisões de desenho que valem a pena explicar:

**A chave é o estado do topo.** O que endereça a memória não é a entrada crua,
é o resumo que a própria rede faz do contexto. Recuperação por conteúdo, como
a completação de padrões no hipocampo — um pedaço do contexto chega para
trazer o episódio inteiro.

**O valor é o erro, não a observação.** A memória guarda aquilo em que a
hierarquia se enganou, não o que aconteceu. Assim a memória corrige o resíduo
em vez de competir com os pesos — é a mesma divisão de trabalho que fez a via
rápida funcionar, depois de ter falhado por ser treinada no sinal todo.

**Escreve-se quando há surpresa.** O mesmo número que decide quanto pensar
decide o que vale a pena gravar. No cérebro é o que a novidade faz às sinapses
do hipocampo; aqui sai de graça, porque a surpresa já estava calculada.

O armazém tem tamanho fixo, de propósito. Memória que cresce com o tempo não
cabe num dispositivo — e é exatamente a limitação da solução da Google
("Memory Caching", arXiv:2602.24281), que resolve o mesmo problema gastando
memória proporcional ao comprimento da sequência. Aqui o orçamento é fixo e a
pergunta interessante passa a ser o que *deitar fora*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import F


@dataclass(frozen=True)
class EpisodicConfig:
    n_slots: int = 128  # orçamento fixo: nº de episódios que cabem
    write_threshold: float = 1.5  # surpresa (× a média corrente) para gravar
    read_threshold: float = 0.6  # semelhança mínima para a memória opinar
    temperature: float = 0.1  # suavidade da recuperação (0 = vencedor único)
    read_gain: float = 1.0  # quanto da correção recuperada se aplica
    decay: float = 0.9999  # esquecimento passivo dos episódios não usados
    # Amostragem por reservatório: quando o armazém está cheio, cada episódio
    # novo entra com probabilidade n_slots/n_vistos e expulsa um ao acaso.
    # Isso mantém uma amostra aproximadamente uniforme de *tudo* o que foi
    # visto, em vez de encher com o mais recente. Sem isto, uma tarefa nova
    # varre as anteriores do armazém em minutos — e a consolidação fica sem
    # nada de antigo para reproduzir, que foi exatamente o que aconteceu.
    reservoir: bool = True


class EpisodicMemory:
    """Armazém key-value de tamanho fixo, endereçável por conteúdo."""

    __slots__ = ("cfg", "keys", "values", "frames", "prev_frames", "strength",
                 "n_written", "_surprise_avg", "_reads", "_hits", "_seen", "_rng")

    def __init__(self, key_dim: int, value_dim: int, cfg: EpisodicConfig | None = None):
        self.cfg = cfg or EpisodicConfig()
        self.keys = np.zeros((self.cfg.n_slots, key_dim), dtype=F)
        # O valor é o resíduo — o que a hierarquia falhou — porque é isso que
        # a memória tem de corrigir sem competir com os pesos.
        self.values = np.zeros((self.cfg.n_slots, value_dim), dtype=F)
        # A observação crua guarda-se à parte, só para o replay do sono: a
        # consolidação precisa da trama tal como aconteceu, não do resíduo
        # (que foi medido contra uma previsão que entretanto já mudou).
        self.frames = np.zeros((self.cfg.n_slots, value_dim), dtype=F)
        # E a trama *anterior*, porque um episódio não é uma observação — é
        # uma observação num contexto. Reproduzir a trama sozinha, fora da
        # sequência de que veio, é recordar uma palavra sem a frase: a rede
        # recebe um erro enorme e os pesos vão parar a sítios errados. Medido:
        # com replay descontextualizado a rede nem sequer aprendia a primeira
        # tarefa (NRMSE 0.97, o mesmo que prever zero).
        self.prev_frames = np.zeros((self.cfg.n_slots, value_dim), dtype=F)
        # Força de cada traço: sobe quando é gravado ou útil, desce sozinha.
        # Slots com força nula estão vazios.
        self.strength = np.zeros(self.cfg.n_slots, dtype=F)
        self.n_written = 0
        self._surprise_avg = 0.0
        self._reads = 0
        self._hits = 0
        self._seen = 0  # candidatos vistos, para o reservatório
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    @property
    def n_occupied(self) -> int:
        return int(np.count_nonzero(self.strength))

    @property
    def hit_rate(self) -> float:
        return self._hits / self._reads if self._reads else 0.0

    def clear(self) -> None:
        self.keys.fill(0.0)
        self.values.fill(0.0)
        self.frames.fill(0.0)
        self.prev_frames.fill(0.0)
        self.strength.fill(0.0)
        self.n_written = 0
        self._surprise_avg = 0.0
        self._reads = self._hits = 0
        self._seen = 0

    # ------------------------------------------------------------------
    def _similarity(self, key: np.ndarray) -> np.ndarray:
        """Cosseno entre a chave e cada traço ocupado; -1 nos vazios."""
        occupied = self.strength > 0
        sim = np.full(self.cfg.n_slots, -1.0, dtype=F)
        if not np.any(occupied):
            return sim
        kn = float(np.linalg.norm(key))
        if kn < 1e-8:
            return sim
        norms = np.linalg.norm(self.keys[occupied], axis=1)
        norms = np.maximum(norms, 1e-8)
        sim[occupied] = self.keys[occupied].dot(key) / (norms * kn)
        return sim

    def read(self, key: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Recupera a correção associada ao contexto atual.

        Devolve (correção, confiança). Confiança é a semelhança do melhor
        traço; abaixo do limiar devolve None, porque uma memória que opina
        sempre é uma memória que injeta ruído em todo o lado.
        """
        self._reads += 1
        sim = self._similarity(key)
        best = float(sim.max())
        if best < self.cfg.read_threshold:
            return None, best

        # Média dos traços próximos, pesada por semelhança. Com temperatura
        # baixa aproxima-se do vencedor único.
        mask = sim >= self.cfg.read_threshold
        w = np.exp((sim[mask] - best) / max(self.cfg.temperature, 1e-6))
        w = (w / w.sum()).astype(F)
        value = (w[:, None] * self.values[mask]).sum(axis=0).astype(F)

        # Usar um traço reforça-o: o que serve sobrevive à limpeza.
        self.strength[mask] += F(0.1) * w
        self._hits += 1
        return value, best

    def write(self, key: np.ndarray, value: np.ndarray, frame: np.ndarray,
              prev_frame: np.ndarray, surprise: float) -> bool:
        """Grava um episódio se ele foi suficientemente surpreendente.

        Devolve True se gravou. O limiar é *relativo* à surpresa média
        corrente: o que conta como digno de nota depende do que é habitual,
        não de uma constante afinada à mão.
        """
        self._surprise_avg = 0.99 * self._surprise_avg + 0.01 * surprise
        if self._surprise_avg <= 0:
            return False
        if surprise < self.cfg.write_threshold * self._surprise_avg:
            return False

        # Se já existe um traço praticamente idêntico, reforça-o em vez de
        # duplicar. O limiar é apertado de propósito: com 0.95 o armazém
        # *congelava* — assim que os traços cobriam o espaço de chaves, tudo o
        # que chegava era considerado repetição e nunca mais entrava nada.
        # Uma memória que deixa de gravar é um buffer com boas maneiras.
        self._seen += 1
        sim = self._similarity(key)
        best_idx = int(np.argmax(sim))
        if float(sim[best_idx]) > 0.995:
            self.values[best_idx] = 0.5 * (self.values[best_idx] + value)
            self.frames[best_idx] = 0.5 * (self.frames[best_idx] + frame)
            self.prev_frames[best_idx] = 0.5 * (self.prev_frames[best_idx] + prev_frame)
            self.strength[best_idx] += F(0.5)
            return True

        slot = self._victim()
        if slot < 0:
            return False
        self.keys[slot] = key
        self.values[slot] = value
        self.frames[slot] = frame
        self.prev_frames[slot] = prev_frame
        self.strength[slot] = F(1.0)
        self.n_written += 1
        return True

    def _victim(self) -> int:
        """Que slot sacrificar. Devolve -1 para recusar a escrita.

        Com reservatório, um armazém cheio aceita o episódio novo só com
        probabilidade n_slots/n_vistos, e nesse caso expulsa um ao acaso. É o
        algoritmo clássico para manter uma amostra uniforme de um fluxo sem
        saber de antemão o comprimento dele — e é o que faz com que memórias
        antigas sobrevivam à chegada de uma tarefa nova.
        """
        empty = np.flatnonzero(self.strength <= 0)
        if len(empty):
            return int(empty[0])
        if not self.cfg.reservoir:
            return int(np.argmin(self.strength))
        if self._rng.random() < self.cfg.n_slots / self._seen:
            return int(self._rng.integers(self.cfg.n_slots))
        return -1

    def decay(self) -> None:
        """Esquecimento passivo. Chamado uma vez por trama."""
        if self.cfg.decay < 1.0:
            self.strength *= F(self.cfg.decay)
            self.strength[self.strength < 1e-3] = 0.0

    # ------------------------------------------------------------------
    def replay(self, n: int, rng: np.random.Generator) -> list[int]:
        """Escolhe episódios para consolidar, favorecendo os mais fortes.

        Os fortes são os que se repetiram ou os que voltaram a ser úteis — o
        que é exatamente o critério que se quer destilar nos pesos. O que
        aconteceu uma vez e nunca mais fica na memória rápida até se apagar.
        """
        occupied = np.flatnonzero(self.strength > 0)
        if len(occupied) == 0:
            return []
        p = self.strength[occupied].astype(np.float64)
        p = p / p.sum()
        n = min(n, len(occupied))
        return [int(i) for i in rng.choice(occupied, size=n, replace=False, p=p)]

    def stats(self) -> dict:
        return {
            "ocupados": self.n_occupied,
            "escritos": self.n_written,
            "taxa_de_acerto": round(self.hit_rate, 3),
            "força_média": round(float(self.strength[self.strength > 0].mean())
                                 if self.n_occupied else 0.0, 3),
        }
