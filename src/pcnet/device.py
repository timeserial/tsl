"""O substrato: pesos ternários e ruído de dispositivo.

Passo 2 do plano. A questão é uma só: quanto é que a rede perde quando os
pesos deixam de ser float32 e passam a ser dispositivos físicos imperfeitos?

Três imperfeições, e a distinção entre elas é o que interessa:

  * **Quantização** — o peso passa a {-1,0,1} com um ganho por linha. É
    determinística e conhecida.
  * **Variabilidade de programação** — cada dispositivo fica com uma
    condutância ligeiramente diferente da pedida, e *fica assim*. É amostrada
    uma vez, na programação, e não se pode tirar a média por cima: é o erro
    que realmente testa a tolerância da arquitetura.
  * **Ruído de leitura** — muda a cada MAC. Este sim, o ciclo de assentamento
    tem hipótese de o diluir ao longo das iterações.

Confundir as duas últimas é a forma mais fácil de exagerar a tolerância a
ruído de um sistema analógico, por isso estão separadas no modelo.

A quantização do ADC entra noutro sítio: em `PCNetwork._threshold`, que é
onde o erro atravessa a fronteira analógico→digital.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dtypes import F


@dataclass(frozen=True)
class DeviceModel:
    """Parâmetros do substrato. Tudo a 0/False = float32 ideal."""

    # --- quantização ------------------------------------------------------
    ternary: bool = False
    # Limiar TWN: Δ = ternary_threshold × média(|W|). 0.75 é o valor de
    # Li & Liu (2016); acima disso zera-se demasiado, abaixo perde-se o
    # benefício de esparsidade.
    ternary_threshold: float = 0.75
    # Ganho por linha (um amplificador de leitura por neurónio de saída) em
    # vez de um ganho único por matriz. Fisicamente realizável e bastante
    # melhor, porque as linhas têm escalas muito diferentes.
    per_row_scale: bool = True

    # --- variabilidade de programação (estática) --------------------------
    sigma_rel: float = 0.0  # desvio proporcional ao peso programado
    sigma_abs: float = 0.0  # desvio absoluto (fuga, offset)
    stuck_frac: float = 0.0  # fração de dispositivos presos a zero

    # --- ruído de leitura (dinâmico, por MAC) -----------------------------
    read_sigma: float = 0.0

    # --- ADC ---------------------------------------------------------------
    adc_bits: int = 0  # 0 = sem quantização
    adc_range: float = 2.0  # fundo de escala, ±adc_range

    # A transição temporal A é 64 pesos contra 2688 dos geradores, e no
    # desenho é a via rápida — um SSM pequeno que faz sentido manter digital.
    # Por omissão fica fora do crossbar; pôr a True testa essa suposição.
    include_transition: bool = False

    seed: int = 0

    @property
    def is_ideal(self) -> bool:
        return not (
            self.ternary
            or self.sigma_rel
            or self.sigma_abs
            or self.stuck_frac
            or self.read_sigma
            or self.adc_bits
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# quantização
# ---------------------------------------------------------------------------
def ternarize(
    W: np.ndarray, threshold: float = 0.75, per_row: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """W ≈ escala · T, com T em {-1, 0, 1}.

    Devolve (T, escala). A escala é por linha se `per_row`, senão um escalar
    replicado — em qualquer caso com forma (n_linhas, 1) para multiplicar
    diretamente.
    """
    W = np.asarray(W, dtype=F)
    absW = np.abs(W)
    axis_mean = absW.mean(axis=1, keepdims=True) if per_row else absW.mean()
    delta = (threshold * axis_mean).astype(F)

    T = np.where(absW >= delta, np.sign(W), 0.0).astype(F)
    kept = T != 0

    if per_row:
        n_kept = kept.sum(axis=1, keepdims=True)
        total = (absW * kept).sum(axis=1, keepdims=True)
        scale = np.divide(total, n_kept, out=np.zeros_like(total), where=n_kept > 0)
    else:
        n_kept = kept.sum()
        s = float((absW * kept).sum() / n_kept) if n_kept else 0.0
        scale = np.full((W.shape[0], 1), s, dtype=F)

    return T, scale.astype(F)


# ---------------------------------------------------------------------------
# uma matriz de pesos tal como existe no substrato
# ---------------------------------------------------------------------------
class AnalogArray:
    """Os desvios fixos de um crossbar concreto, amostrados uma única vez.

    `program(W)` pega nos pesos que o treino quer e devolve os que o
    dispositivo realmente tem. Os desvios não são re-amostrados a cada
    chamada — é isso que os torna variabilidade de programação e não ruído.
    """

    __slots__ = ("model", "shape", "_rel", "_abs", "_alive", "_rng")

    def __init__(self, shape: tuple[int, int], model: DeviceModel, seed: int) -> None:
        self.model = model
        self.shape = shape
        rng = np.random.default_rng(seed)

        self._rel = (
            (rng.standard_normal(shape) * model.sigma_rel).astype(F)
            if model.sigma_rel
            else None
        )
        self._abs = (
            (rng.standard_normal(shape) * model.sigma_abs).astype(F)
            if model.sigma_abs
            else None
        )
        self._alive = (
            (rng.random(shape) >= model.stuck_frac) if model.stuck_frac else None
        )
        # RNG separado para o ruído de leitura: o dinâmico não pode partilhar
        # a sequência com o estático, senão reprogramar muda o ruído.
        self._rng = np.random.default_rng(seed + 1_000_003)

    def program(self, W: np.ndarray) -> np.ndarray:
        """Pesos pedidos -> pesos que o dispositivo tem."""
        m = self.model
        if m.ternary:
            T, scale = ternarize(W, m.ternary_threshold, m.per_row_scale)
            W_eff = (T * scale).astype(F)
        else:
            W_eff = np.asarray(W, dtype=F).copy()

        if self._rel is not None:
            W_eff = (W_eff * (1.0 + self._rel)).astype(F)
        if self._abs is not None:
            W_eff = (W_eff + self._abs).astype(F)
        if self._alive is not None:
            W_eff = np.where(self._alive, W_eff, F(0.0)).astype(F)
        return W_eff

    def read(self, out: np.ndarray) -> np.ndarray:
        """Ruído de leitura, re-amostrado a cada MAC."""
        s = self.model.read_sigma
        if not s:
            return out
        return (out + self._rng.standard_normal(out.shape) * s).astype(F)


def quantize_adc(x: np.ndarray, bits: int, full_scale: float) -> np.ndarray:
    """Quantização uniforme com saturação em ±full_scale.

    `bits` conta o sinal: 8 bits -> 256 níveis sobre 2·full_scale.
    """
    if bits <= 0:
        return x
    levels = (1 << bits) - 1
    step = (2.0 * full_scale) / levels
    clipped = np.clip(x, -full_scale, full_scale)
    return (np.round(clipped / step) * step).astype(F)
