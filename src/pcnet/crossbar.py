"""Crossbar analógico transparente: uma matriz de pesos como par diferencial
de condutâncias, com as não-idealidades clássicas da literatura de
dispositivos, cada uma independente e com magnitude própria.

O mapeamento (padrão desde Burr et al. 2015, "Experimental demonstration ...
using ... PCM"; Yu 2018, Proc. IEEE):

    W_ij = (G+_ij − G−_ij) / k,   G± ∈ [G_min, G_max],   k = (G_max−G_min)/w_max

com G_max normalizado a 1 e G_min = 1/on_off. O par arranca no ponto médio
(G± = (G_min+G_max)/2 ± w·k/2): peso zero = par equilibrado, e cada
atualização Δw divide-se em +Δw·k/2 no ramo positivo e −Δw·k/2 no negativo —
o esquema de escrita diferencial habitual (Gokmen & Vlasov 2016).

As não-idealidades, cada uma com o modelo canónico citado no sítio onde é
aplicada:

  1. Ruído de escrita (ciclo-a-ciclo):  ΔG ← ΔG + N(0, (σ_w·ΔG_range)²)
     por impulso e por dispositivo (modelo de NeuroSim, Chen et al. 2018).
  2. Não-linearidade + assimetria de escrita: saturação exponencial —
     potenciação comprimida perto de G_max, depressão perto de G_min, com
     fatores diferentes α_p/α_d (forma por-passo do modelo A_p/A_d do
     NeuroSim; limite α→0 = linear; generaliza o soft-bounds de
     Fusi & Abbott 2007).
  3. Ruído de leitura: multiplicativo por dispositivo, G_lido = G·(1+ξ),
     ξ~N(0,σ_r) i.i.d. por leitura (RTN/1f agregados; Joshi et al. 2020,
     Nat. Comm.). Propagado analiticamente para a saída — exato em
     distribuição, sem gerar uma matriz de ruído por MAC.
  4. ADC: quantização uniforme com saturação em ±fundo-de-escala, n_bits,
     aplicada às leituras nos dois sentidos.
  5. Deriva de condutância (PCM): G(t) = G(t₀)·(t/t₀)^(−ν), ν por
     dispositivo ~ N(ν̄, σ_ν) (Ielmini et al. 2007; Le Gallo & Sebastian
     2020). É a dispersão de ν que impede o par diferencial de a cancelar.
  6. Queda IR (1ª ordem): atenuação efetiva que cresce com a posição na
     linha e com a corrente total do array,
         a_ij = 1 − γ·p_ij·u,  p_ij = ((i+1)/N + (j+1)/M)/2,
         u = média(G+ + G−)/(2·G_max)
     com γ = queda máxima no canto mais distante a carga plena. Modelo
     declaradamente simples: a queda real resolve a rede resistiva
     (ver ISAAC, Shafiee et al. 2016); aqui fica o termo de 1ª ordem —
     posição × utilização — dobrado sobre as duas leituras.
  7. Comporta de escrita por endurance (o θ do projeto): |Δw| < θ_w não gera
     impulso físico nenhum. Conta-se cada impulso de dispositivo emitido e
     poupado — a moeda do trade-off endurance/exatidão.

Fidelidade numérica: com tudo desligado o objeto degenera num acumulador
float32 idêntico bit-a-bit ao treino em float — é o portão de sanidade que a
experiência exige (reproduzir 0.579±0.004 antes de ligar seja o que for).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .dtypes import F


@dataclass(frozen=True)
class CrossbarModel:
    """Física partilhada por todos os arrays. Tudo a 0 / on_off=inf = ideal."""

    # razão on/off r = G_max/G_min. G_min = 1/r (G_max ≡ 1). inf → G_min = 0.
    # Literatura: ReRAM filamentar ~10–100; PCM ~10²–10³ (Yu 2018).
    on_off: float = math.inf
    # ruído de escrita ciclo-a-ciclo, fração da gama (G_max−G_min) por impulso.
    # Medições típicas: 0.5–5% da gama; >10% é dispositivo mau (Chen 2018).
    sigma_write: float = 0.0
    # não-linearidade de escrita: ΔG_pot = ΔG·exp(−α_p·x̂),
    # ΔG_dep = ΔG·exp(−α_d·(1−x̂)), x̂ = (G−G_min)/(G_max−G_min) ∈ [0,1].
    # α = ln(razão entre o passo maior e o menor ao longo da gama):
    # α≈0.5 quase linear, α≈1–2 típico de TaOx/HfOx, α≥5 severo (PCMO).
    alpha_p: float = 0.0
    alpha_d: float = 0.0
    # ruído de leitura multiplicativo por dispositivo, por leitura.
    # ~1% (arrays PCM calibrados, Joshi 2020) a ~5–10% (RTN forte).
    sigma_read: float = 0.0
    # ADC nas saídas (0 = sem quantização). 8 bits é o ponto de desenho
    # habitual (ISAAC); 6 é agressivo; 12 é quase transparente.
    adc_bits: int = 0
    # deriva PCM: expoente médio ν̄ (amorfo ~0.05–0.1, cristalino ~0.005,
    # célula projetada p/ deriva baixa ~0.02) e dispersão relativa entre
    # dispositivos (Le Gallo 2018 mede ~30%).
    drift_nu: float = 0.0
    drift_nu_spread: float = 1.0 / 3.0
    # t/t₀ entre programar e ler: 1e4 (t₀=1 s → leitura ~3 h depois).
    drift_t_ratio: float = 1e4
    # queda IR: γ = atenuação máxima (canto mais distante, array carregado).
    # Com R_fio ~1–5 Ω/célula e R_dispositivo ~10–100 kΩ em arrays de 64+:
    # poucos % (mild) a dezenas de % (severe) — Shafiee 2016, Ielmini 2018.
    ir_drop: float = 0.0
    # comporta de endurance: |Δw| < theta_write ⇒ nenhum impulso físico.
    theta_write: float = 0.0

    @property
    def analog_ideal(self) -> bool:
        """Sem física analógica nenhuma (a comporta θ_w é permitida: opera
        sobre o pedido de escrita, antes de existir dispositivo)."""
        return (
            math.isinf(self.on_off)
            and not self.sigma_write
            and not self.alpha_p
            and not self.alpha_d
            and not self.sigma_read
            and not self.adc_bits
            and not self.drift_nu
            and not self.ir_drop
        )

    def to_dict(self) -> dict:
        return asdict(self)


class Crossbar:
    """Uma matriz de pesos a viver num array analógico concreto.

    Operações — exatamente as três do TSL e mais nada:
      read(x)    — leitura direta   y = W·x      (previsão)
      read_T(e)  — leitura transposta h = Wᵀ·e   (retroprojeção)
      update(dW) — escrita rank-1 (o chamador passa o outer product)
    e as duas de ciclo de vida: program(W) inicial e apply_drift() entre
    treino e avaliação.

    `w_max` é o fundo de escala de peso do array (fixa k); pesos pedidos além
    dele saturam nas condutâncias — como no silício. `adc_range` é o fundo de
    escala do ADC deste array (a periferia é dimensionada por array).
    """

    G_MAX = 1.0

    def __init__(self, shape: tuple[int, int], w_max: float,
                 model: CrossbarModel, seed: int,
                 adc_range: float = 4.0, adc_range_T: float | None = None,
                 use_adc: bool = True, use_ir: bool = True) -> None:
        self.shape = tuple(shape)
        self.model = model
        self.w_max = float(w_max)
        # ADCs distintos nos dois sentidos: a leitura direta sai nas colunas,
        # a transposta nas linhas — periferias separadas, fundos de escala
        # dimensionados ao sinal de cada sentido (calibração de projeto).
        self.adc_range = float(adc_range)
        self.adc_range_T = float(adc_range_T if adc_range_T is not None
                                 else adc_range)
        # bias/colunas únicas: sem ADC nem IR (offset somado na periferia
        # digital; uma coluna não tem linha comprida para cair tensão).
        self.use_adc = bool(use_adc) and model.adc_bits > 0
        self.use_ir = bool(use_ir) and model.ir_drop > 0.0
        self.rng = np.random.default_rng(seed)

        self.g_min = 0.0 if math.isinf(model.on_off) else self.G_MAX / model.on_off
        self.g_range = self.G_MAX - self.g_min
        self.k = self.g_range / self.w_max  # condutância por unidade de peso
        self._g0 = 0.5 * (self.G_MAX + self.g_min)  # ponto médio (peso 0)

        # p_ij ∈ (0,1]: distância elétrica normalizada aos drivers (canto
        # (0,0) perto, canto (N,M) longe) — o termo de posição da queda IR.
        n, m = self.shape
        self._pos = 0.5 * ((np.arange(1, n + 1)[:, None] / n)
                           + (np.arange(1, m + 1)[None, :] / m))

        # modo exato: acumulador float32 bit-idêntico ao treino float.
        self._exact = model.analog_ideal
        self._Gp: np.ndarray | None = None  # condutâncias (float64)
        self._Gm: np.ndarray | None = None
        self._Wf = np.zeros(self.shape, dtype=F)  # pesos efetivos (leitura)
        self._S2: np.ndarray | None = None  # (a²·(G+²+G−²))/k², p/ ruído

        # contabilidade de endurance e diagnóstico
        self.pulses_issued = 0   # impulsos de dispositivo emitidos
        self.pulses_gated = 0    # impulsos poupados pela comporta θ_w
        self.updates = 0         # chamadas a update()
        self.clip_events = 0     # elementos que bateram no fundo de escala

    # -- programação inicial ------------------------------------------------
    def program(self, W: np.ndarray) -> None:
        """Programa o array a partir de W (program-and-verify: a
        não-linearidade não entra — itera-se até acertar — mas o ruído de
        escrita residual do último impulso entra)."""
        W = np.asarray(W, dtype=F)
        if self._exact:
            self._Wf = W.copy()  # acumulador float32: bit-exato
            return
        Wc = np.clip(W.astype(np.float64), -self.w_max, self.w_max)
        self.clip_events += int(np.sum(np.abs(W) > self.w_max))
        half = 0.5 * self.k * Wc
        self._Gp = self._g0 + half
        self._Gm = self._g0 - half
        if self.model.sigma_write > 0.0:
            s = self.model.sigma_write * self.g_range
            self._Gp += self.rng.standard_normal(self.shape) * s
            self._Gm += self.rng.standard_normal(self.shape) * s
        self._clip_rails()
        self._refresh()

    # -- leituras -----------------------------------------------------------
    def read(self, x: np.ndarray) -> np.ndarray:
        """y = W·x, com a física de leitura em cima."""
        y = self._Wf.dot(x)
        if self.model.sigma_read > 0.0 and self._S2 is not None:
            # Var(y_i) = σ_r²·Σ_j (G+²+G−²)_ij x_j² / k²  — a propagação
            # exata do ruído multiplicativo i.i.d. por dispositivo. É aqui
            # que a razão on/off morde: G_min alto = modo comum grande que a
            # subtração cancela no sinal mas não no ruído.
            y = y + self.rng.standard_normal(y.shape) * (
                self.model.sigma_read * np.sqrt(self._S2.dot(x * x)))
        return self._adc(y)

    def read_T(self, e: np.ndarray) -> np.ndarray:
        """h = Wᵀ·e: os mesmos dispositivos, lidos no sentido inverso —
        logo os mesmos desvios estáticos e outra dose de ruído de leitura."""
        y = self._Wf.T.dot(e)
        if self.model.sigma_read > 0.0 and self._S2 is not None:
            y = y + self.rng.standard_normal(y.shape) * (
                self.model.sigma_read * np.sqrt(self._S2.T.dot(e * e)))
        return self._adc(y, self.adc_range_T)

    def _adc(self, y: np.ndarray, fs: float | None = None) -> np.ndarray:
        if not self.use_adc:
            return y.astype(F, copy=False)
        # quantização uniforme com saturação (mid-tread), n_bits com sinal.
        fs = self.adc_range if fs is None else fs
        step = 2.0 * fs / ((1 << self.model.adc_bits) - 1)
        return (np.round(np.clip(y, -fs, fs) / step) * step).astype(F)

    # -- escrita ------------------------------------------------------------
    def update(self, dW: np.ndarray) -> None:
        """Escrita rank-1 (ou qualquer ΔW): impulsos diferenciais ±ΔG/2.

        A comporta θ_w decide ANTES da física: elementos abaixo do limiar não
        geram impulso (endurance poupada), os restantes atravessam
        não-linearidade → ruído → rails, por esta ordem.
        """
        self.updates += 1
        m = self.model
        if m.theta_write > 0.0:
            mask = np.abs(dW) >= m.theta_write
            n_write = int(mask.sum())
            self.pulses_gated += 2 * (dW.size - n_write)
        else:
            mask = None
            n_write = dW.size
        self.pulses_issued += 2 * n_write  # dois dispositivos por par
        if n_write == 0:
            return

        if self._exact:
            # acumulador float32: mesma aritmética do treino float (o += de
            # numpy sobre float32), logo o portão de sanidade é bit-exato.
            self._Wf += (dW if mask is None else np.where(mask, dW, F(0.0))).astype(F, copy=False)
            return

        dG = 0.5 * self.k * np.asarray(dW, dtype=np.float64)
        if mask is not None:
            dG = np.where(mask, dG, 0.0)
        pulsed = dG != 0.0

        # ramo positivo recebe +dG, negativo −dG; em cada dispositivo o
        # sentido do impulso decide se é potenciação (α_p) ou depressão (α_d).
        for G, sgn in ((self._Gp, 1.0), (self._Gm, -1.0)):
            step = sgn * dG
            if m.alpha_p > 0.0 or m.alpha_d > 0.0:
                x_hat = np.clip((G - self.g_min) / self.g_range, 0.0, 1.0)
                gain = np.where(
                    step > 0.0,
                    np.exp(-m.alpha_p * x_hat),          # saturação p/ G_max
                    np.exp(-m.alpha_d * (1.0 - x_hat)),  # saturação p/ G_min
                )
                step = step * gain
            if m.sigma_write > 0.0:
                step = step + pulsed * (
                    self.rng.standard_normal(self.shape)
                    * (m.sigma_write * self.g_range))
            G += step
        self._clip_rails()
        self._refresh()

    # -- deriva (entre treino e avaliação) ----------------------------------
    def apply_drift(self) -> None:
        """G(t) = G(t₀)·(t/t₀)^(−ν), ν_ij ~ N(ν̄, σ_ν) por dispositivo."""
        m = self.model
        if m.drift_nu <= 0.0 or self._exact:
            return
        s = m.drift_nu * m.drift_nu_spread
        for G in (self._Gp, self._Gm):
            nu = np.clip(self.rng.normal(m.drift_nu, s, self.shape), 0.0, None)
            G *= m.drift_t_ratio ** (-nu)
        # a deriva desce a condutância absoluta; pode passar abaixo de G_min
        # (a célula "esfria" para lá do estado programável) — só o chão físico
        # zero se impõe.
        np.clip(self._Gp, 0.0, self.G_MAX, out=self._Gp)
        np.clip(self._Gm, 0.0, self.G_MAX, out=self._Gm)
        self._refresh()

    # -- internos -----------------------------------------------------------
    def _clip_rails(self) -> None:
        np.clip(self._Gp, self.g_min, self.G_MAX, out=self._Gp)
        np.clip(self._Gm, self.g_min, self.G_MAX, out=self._Gm)

    def _refresh(self) -> None:
        """Recalcula o operador efetivo depois de qualquer escrita/deriva."""
        if self.use_ir:
            # a_ij = 1 − γ·p_ij·u: posição × utilização média de corrente.
            u = float((self._Gp.mean() + self._Gm.mean()) / (2.0 * self.G_MAX))
            att = 1.0 - self.model.ir_drop * self._pos * u
        else:
            att = 1.0
        self._Wf = ((att * (self._Gp - self._Gm)) / self.k).astype(F)
        if self.model.sigma_read > 0.0:
            self._S2 = (att * att) * (self._Gp**2 + self._Gm**2) / (self.k * self.k)

    # -- inspeção -----------------------------------------------------------
    @property
    def W_eff(self) -> np.ndarray:
        """Os pesos que o array realmente aplica (float32, sem ruído)."""
        return self._Wf

    def rail_frac(self) -> float:
        """Fração de dispositivos encostados aos rails (1% da gama)."""
        if self._exact:
            return 0.0
        tol = 0.01 * self.g_range
        at = ((self._Gp <= self.g_min + tol) | (self._Gp >= self.G_MAX - tol)
              | (self._Gm <= self.g_min + tol) | (self._Gm >= self.G_MAX - tol))
        return float(at.mean())

    def stats(self) -> dict:
        return {
            "pulses_issued": self.pulses_issued,
            "pulses_gated": self.pulses_gated,
            "updates": self.updates,
            "rail_frac": round(self.rail_frac(), 4),
            "clip_events": self.clip_events,
        }
