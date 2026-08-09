"""Instrumentação.

O que interessa medir não é a loss — é o custo. Três números por trama:
surpresa por iteração de assentamento, fração de erros silenciados pelo
limiar, e nº de iterações até o early exit disparar. A partir deles sai a
contabilidade que o argumento energético precisa: conversões ADC (o gargalo
caro do crossbar) proporcionais à surpresa e não à dimensão da rede.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Porque é que o assentamento parou. A distinção interessa: "explicado" é o
# early exit que queremos (previsão certa, silêncio, custo zero); "estagnado"
# é a rede a desistir de algo que não sabe explicar — também poupa energia,
# mas é uma admissão de ignorância, não uma vitória; "teto" é o caso caro.
EXIT_EXPLAINED = "explicado"
EXIT_STALLED = "estagnado"
EXIT_CEILING = "teto"


@dataclass
class StepTrace:
    """Tudo o que uma trama gastou e revelou."""

    # Surpresa = energia ½Σ‖ε_l‖² sobre todos os níveis, *antes* do limiar,
    # uma entrada por avaliação de erro. O elemento 0 é a surpresa em malha
    # aberta: o quanto a previsão descida do passo anterior falhou.
    surprise: list[float] = field(default_factory=list)
    # A mesma coisa depois do limiar: o que efetivamente subiu.
    transmitted: list[float] = field(default_factory=list)

    iters: int = 0  # atualizações de estado realizadas
    exit_reason: str = EXIT_CEILING

    eps_total: int = 0  # componentes de erro avaliadas
    eps_silenced: int = 0  # ... das quais ficaram abaixo do limiar

    macs_up: int = 0  # MACs no caminho ascendente (esparso)
    macs_up_dense: int = 0  # ... se nada fosse silenciado
    macs_down: int = 0  # MACs no caminho descendente (denso; é o crossbar)

    # ε_0 em malha aberta: o erro de previsão da trama, o número "de ML".
    pred_error: np.ndarray | None = None
    target_rms: float = 0.0

    @property
    def early_exit(self) -> bool:
        """Saiu antes de gastar o teto de iterações (por qualquer razão)."""
        return self.exit_reason != EXIT_CEILING

    @property
    def open_loop_surprise(self) -> float:
        return self.surprise[0] if self.surprise else 0.0

    @property
    def final_surprise(self) -> float:
        return self.surprise[-1] if self.surprise else 0.0

    @property
    def silenced_frac(self) -> float:
        return self.eps_silenced / self.eps_total if self.eps_total else 0.0

    @property
    def adc_conversions(self) -> int:
        """Erros que passaram o limiar = conversões ADC necessárias."""
        return self.eps_total - self.eps_silenced

    @property
    def pred_rmse(self) -> float:
        if self.pred_error is None:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.pred_error))))

    @property
    def pred_nrmse(self) -> float:
        """RMSE normalizado pelo RMS do alvo: 1.0 = tão bom como prever zero."""
        return self.pred_rmse / self.target_rms if self.target_rms > 0 else 0.0


@dataclass
class RunStats:
    """Agregado sobre uma sequência de tramas."""

    n_steps: int = 0
    mean_iters: float = 0.0
    mean_open_loop_surprise: float = 0.0
    mean_final_surprise: float = 0.0
    silenced_frac: float = 0.0
    early_exit_frac: float = 0.0
    explained_frac: float = 0.0
    stalled_frac: float = 0.0
    mac_up_frac: float = 0.0  # MACs ascendentes efetivos / densos
    adc_frac: float = 0.0  # conversões ADC / (rede densa, max_iters)
    pred_nrmse: float = 0.0

    def as_row(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "iters": round(self.mean_iters, 3),
            "surprise_open_loop": round(self.mean_open_loop_surprise, 4),
            "surprise_final": round(self.mean_final_surprise, 4),
            "silenced_%": round(100 * self.silenced_frac, 1),
            "early_exit_%": round(100 * self.early_exit_frac, 1),
            "explained_%": round(100 * self.explained_frac, 1),
            "stalled_%": round(100 * self.stalled_frac, 1),
            "macs_up_%": round(100 * self.mac_up_frac, 1),
            "adc_%": round(100 * self.adc_frac, 1),
            "pred_nrmse": round(self.pred_nrmse, 4),
        }


def summarize(traces: list[StepTrace], max_iters: int, eps_per_pass: int) -> RunStats:
    """Reduz uma lista de tramas a uma linha de tabela.

    `eps_per_pass` é o nº de componentes de erro de uma passagem densa; serve
    de denominador para a fração de ADC no pior caso (rede densa a correr
    sempre até ao teto de iterações).
    """
    if not traces:
        return RunStats()

    n = len(traces)
    dense_adc = max(1, (max_iters + 1) * eps_per_pass * n)
    macs_up_dense = sum(t.macs_up_dense for t in traces)
    eps_total = sum(t.eps_total for t in traces)

    return RunStats(
        n_steps=n,
        mean_iters=sum(t.iters for t in traces) / n,
        mean_open_loop_surprise=sum(t.open_loop_surprise for t in traces) / n,
        mean_final_surprise=sum(t.final_surprise for t in traces) / n,
        silenced_frac=(sum(t.eps_silenced for t in traces) / eps_total)
        if eps_total
        else 0.0,
        early_exit_frac=sum(1 for t in traces if t.early_exit) / n,
        explained_frac=sum(1 for t in traces if t.exit_reason == EXIT_EXPLAINED) / n,
        stalled_frac=sum(1 for t in traces if t.exit_reason == EXIT_STALLED) / n,
        mac_up_frac=(sum(t.macs_up for t in traces) / macs_up_dense)
        if macs_up_dense
        else 0.0,
        adc_frac=sum(t.adc_conversions for t in traces) / dense_adc,
        pred_nrmse=float(np.mean([t.pred_nrmse for t in traces])),
    )
