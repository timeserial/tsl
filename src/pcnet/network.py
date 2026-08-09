"""A hierarquia preditiva esparsa.

Um passo de tempo (uma trama do sensor) corre assim:

  1. A previsão desce. O topo propaga-se no tempo (ẑ_L = tanh(A·z_L(t-1)))
     e daí desce nível a nível até gerar uma trama esperada ẑ_0. Isto
     acontece *antes* de ver os dados: ε_0 na primeira avaliação é o erro de
     previsão em malha aberta.
  2. A surpresa sobe. Compara-se, aplica-se o limiar (|ε| < θ não é
     transmitido — silêncio é custo zero) e o que sobra corrige os estados de
     cima. Repete-se até a surpresa total convergir: *early exit* para
     entradas banais, mais iterações para as difíceis. O compute é
     proporcional à surpresa, não à capacidade da rede.
  3. Aprende-se localmente. ΔW ∝ ε ⊗ z: só quantidades presentes na sinapse.
     Sem backprop, sem grafo, sem ativações guardadas.

O estado latente do topo persiste entre tramas — é a memória de trabalho.
"""

from __future__ import annotations

import numpy as np

from . import activations
from .config import PCConfig
from .device import AnalogArray, DeviceModel, quantize_adc
from .dtypes import F
from .episodic import EpisodicConfig, EpisodicMemory
from .gated import GatedTransition
from .layer import PCLayer
from .mixture import TopMixture
from .metrics import EXIT_CEILING, EXIT_EXPLAINED, EXIT_STALLED, StepTrace
from .precision import Precision, UnitPrecision
from .temporal import Transition, timescales


class PCNetwork:
    def __init__(self, config: PCConfig | None = None) -> None:
        self.cfg = config or PCConfig()
        rng = np.random.default_rng(self.cfg.seed)

        sizes = self.cfg.sizes
        self.L = len(sizes) - 1  # índice do nível de topo

        # layers[l] gera o nível l a partir do nível l+1.
        self.layers: list[PCLayer] = []
        for l in range(self.L):
            act = self.cfg.sensory_activation if l == 0 else self.cfg.latent_activation
            self.layers.append(
                PCLayer(sizes[l], sizes[l + 1], act, rng, self.cfg.init_scale,
                        gated=self.cfg.gated_layers)
            )

        # Transição temporal do topo: uma recorrência linear + tanh.
        n_top = sizes[-1]
        self.A = (
            np.eye(n_top, dtype=F)
            + (rng.standard_normal((n_top, n_top)) * (0.1 / np.sqrt(n_top))).astype(F)
            if self.cfg.use_transition
            else np.eye(n_top, dtype=F)
        )
        self._f_top, self._fprime_top = activations.get(self.cfg.latent_activation)
        # Mistura de dinâmicas no topo. Com n_dynamics=1 fica exatamente a
        # matriz única de antes e nada muda.
        self.mixture = (
            TopMixture(n_top, self.cfg.n_dynamics, rng, self.cfg.dynamics_tau)
            if self.cfg.n_dynamics > 1
            else None
        )
        self.gated = (
            GatedTransition(n_top, rng) if self.cfg.gated_transition else None
        )

        # Estados, pré-alocados (espelham os arrays estáticos do C).
        self.z: list[np.ndarray] = [np.zeros(n, dtype=F) for n in sizes]
        self.prior_top = np.zeros(n_top, dtype=F)
        self.a_top = np.zeros(n_top, dtype=F)
        # O estado de cada nível na trama anterior. Só os latentes precisam:
        # o nível sensorial é fixado pelos dados e não se prevê a si próprio.
        self._z_prev: list[np.ndarray] = [np.zeros(n, dtype=F) for n in sizes]

        # Escalas de tempo: um modelo temporal por nível latente, cada vez mais
        # lento à medida que se sobe (ver temporal.py). O topo continua a usar
        # `A`, que já é denso e já sabe viver num crossbar.
        # λ decresce com a altura: o topo integra sobre a janela mais longa.
        lams = timescales(self.L, self.cfg.timescale_base, self.cfg.timescale_ratio)
        self._top_lam = (
            lams[self.L - 1] if self.cfg.level_transition != "none" else 1.0
        )
        self.transitions: list[Transition | None] = [None] * (self.L + 1)
        if self.cfg.level_transition != "none":
            for l in range(1, self.L):
                self.transitions[l] = Transition(
                    sizes[l],
                    self.cfg.level_transition,
                    lams[l - 1],
                    rng,
                    activations.get(self.cfg.latent_activation),
                )
        self.lams = lams

        # Precisão: o ganho com que o erro de cada canal conta (ver precision.py).
        # Dois canais por nível latente — o que vem de cima e o que vem do
        # próprio passado — porque são erros de naturezas diferentes.
        def _prec(n: int):
            if not self.cfg.use_precision:
                return UnitPrecision()
            return Precision(n, self.cfg.precision_per_unit, self.cfg.precision_lr)

        self.prec_hier = [_prec(sizes[l]) for l in range(self.L)]
        self.prec_temp = [None] + [_prec(sizes[l]) for l in range(1, self.L + 1)]

        # Via rápida: prevê a trama seguinte diretamente da anterior. Linear e
        # sem não-linearidade nenhuma — é a via "grosseira mas imediata".
        n0 = sizes[0]
        self.A0 = (
            (rng.standard_normal((n0, n0)) * (0.1 / np.sqrt(n0))).astype(F)
            if self.cfg.fast_path
            else None
        )
        self._fast = np.zeros(n0, dtype=F)

        # Memória episódica: desligada até `attach_memory`. A chave é o estado
        # do topo (o resumo do contexto), o valor é o erro que a hierarquia
        # cometeu — a memória corrige o resíduo, não compete com os pesos.
        self.memory: EpisodicMemory | None = None
        self._recall = np.zeros(n0, dtype=F)
        self._recall_conf = 0.0
        self._replay_rng = np.random.default_rng(self.cfg.seed + 31337)

        # Diagnóstico: fixar dimensões do estado de topo a um valor dado, para
        # perguntar "e se a rede *soubesse* em que mundo está?". Sem isto, um
        # indicador posto na entrada é 3 dimensões em 67 e quase não empurra o
        # latente — o oráculo sai fraco e a conclusão sai errada.
        self.top_clamp: np.ndarray | None = None

        # Constantes de contabilidade.
        self.macs_down_per_pass = sum(lay.macs_down() for lay in self.layers)
        self.eps_per_pass = sum(sizes)  # um ε por unidade, topo incluído

        # Substrato: None = float32 ideal.
        self.device_model: DeviceModel | None = None
        self._A_device = None
        self.A_eff = self.A

    # ------------------------------------------------------------------
    # estado
    # ------------------------------------------------------------------
    @property
    def _z_top_prev(self) -> np.ndarray:
        """O estado do topo na trama anterior."""
        return self._z_prev[self.L]

    def reset(self) -> None:
        """Zera os estados latentes. Os pesos ficam."""
        for z in self.z:
            z.fill(0.0)
        for z in self._z_prev:
            z.fill(0.0)
        self.prior_top.fill(0.0)

    def snapshot_state(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Guarda o estado dinâmico (não os pesos).

        A rede é online e o estado atravessa o tempo, logo *avaliar* mexe no
        estado. Sem isto, medir o modelo a meio do treino altera o treino —
        e a curva de aprendizagem passa a medir a régua.
        """
        return ([z.copy() for z in self.z], [z.copy() for z in self._z_prev])

    def restore_state(self, snapshot: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        zs, prevs = snapshot
        for dst, src in zip(self.z, zs):
            dst[:] = src
        for dst, src in zip(self._z_prev, prevs):
            dst[:] = src

    # ------------------------------------------------------------------
    # substrato
    # ------------------------------------------------------------------
    def attach_device(self, model: DeviceModel) -> None:
        """Programa os pesos num crossbar concreto (com os defeitos dele).

        Os desvios são amostrados uma vez, aqui — reprogramar com os mesmos
        parâmetros e a mesma seed dá o *mesmo* dispositivo. Um crossbar
        diferente é outra seed, e é por isso que qualquer medição de
        tolerância tem de ser feita sobre vários.
        """
        self.device_model = model
        for l, lay in enumerate(self.layers):
            lay.attach_device(AnalogArray(lay.W.shape, model, model.seed + 7919 * l))
        if model.include_transition:
            self._A_device = AnalogArray(self.A.shape, model, model.seed + 104729)
            self.A_eff = self._A_device.program(self.A)
        else:
            self._A_device = None
            self.A_eff = self.A

    def detach_device(self) -> None:
        self.device_model = None
        self._A_device = None
        self.A_eff = self.A
        for lay in self.layers:
            lay.detach_device()

    def _refresh_transition(self) -> None:
        if self._A_device is not None:
            self.A_eff = self._A_device.program(self.A)

    # ------------------------------------------------------------------
    # o caminho descendente
    # ------------------------------------------------------------------
    def _temporal_prior(self) -> np.ndarray:
        """ẑ_L(t) = f(A · z_L(t-1)): o que o topo espera de si próprio."""
        if not self.cfg.use_transition:
            self.a_top[:] = self._z_top_prev
            self.prior_top[:] = self._z_top_prev
            return self.prior_top
        z_prev = self._z_top_prev
        if self.cfg.state_leak:
            z_prev = (F(1.0 - self.cfg.state_leak) * z_prev).astype(F, copy=False)
        if self.gated is not None:
            # A mistura (1-g)z + g·tanh(Az) já inclui a não-linearidade e a
            # retenção: devolve-se o prior diretamente, sem tanh nem λ extra.
            self.prior_top[:] = self.gated.predict(z_prev)
            self.a_top[:] = self.prior_top
            return self.prior_top
        if self.mixture is not None:
            self.a_top[:] = self.mixture.predict(z_prev)
        else:
            np.dot(self.A_eff, z_prev, out=self.a_top)
        if self._A_device is not None:
            self.a_top[:] = self._A_device.read(self.a_top)
        # A escala de tempo do topo. Sem isto o topo é um A denso sem retenção
        # nenhuma — ou seja, o nível *mais rápido* da rede, exatamente ao
        # contrário do que uma hierarquia de janelas temporais pede.
        if self._top_lam < 1.0:
            self.a_top *= F(self._top_lam)
            self.a_top += F(1.0 - self._top_lam) * z_prev
        self.prior_top[:] = self._f_top(self.a_top)
        return self.prior_top

    def _combine_priors(self, level: int, from_above: np.ndarray) -> np.ndarray:
        """Junta o que o nível de cima prevê com o que o próprio passado prevê.

        Duas fontes de informação independentes sobre a mesma quantidade, cada
        uma com a sua precisão: a média ponderada pelas precisões é a resposta
        Bayesiana, e é também o mínimo da energia se mais nada puxasse o
        estado. Sem transição no nível, sobra só a previsão de cima — que é
        exatamente o comportamento do passo 1.
        """
        trans = self.transitions[level]
        if trans is None:
            return from_above
        from_past = trans.predict(self._z_prev[level])
        ph = np.asarray(self.prec_hier[level].value, dtype=F)
        pt = np.asarray(self.prec_temp[level].value, dtype=F)
        return ((ph * from_above + pt * from_past) / (ph + pt)).astype(F, copy=False)

    def predict_next(self) -> np.ndarray:
        """A trama que a rede espera ver a seguir, sem olhar para os dados.

        Previsão em malha aberta, útil para avaliar e para comparar contra o
        inferidor em C. Não altera z nem o estado que atravessa tramas (só
        mexe em buffers de rascunho, que o passo seguinte reescreve).
        """
        z = self._temporal_prior().copy()
        for l in range(self.L - 1, 0, -1):
            z = self._combine_priors(l, self.layers[l].predict(z)).copy()
        deep = self.layers[0].predict(z)
        out = (deep + self._fast_prediction()).astype(F, copy=False)
        if self.memory is not None:
            value, _ = self.memory.read(self._z_prev[self.L])
            if value is not None:
                out = (out + F(self.memory.cfg.read_gain) * value).astype(F)
        return out

    def _fast_prediction(self) -> np.ndarray:
        """A via rápida: A_0 · (trama anterior). Zero se estiver desligada."""
        if self.A0 is None:
            return F(0.0)
        np.dot(self.A0, self._z_prev[0], out=self._fast)
        return self._fast

    # ------------------------------------------------------------------
    # memória episódica
    # ------------------------------------------------------------------
    def attach_memory(self, cfg: EpisodicConfig | None = None) -> EpisodicMemory:
        """Liga um armazém episódico de tamanho fixo (ver episodic.py)."""
        self.memory = EpisodicMemory(
            key_dim=self.cfg.sizes[-1], value_dim=self.cfg.sizes[0], cfg=cfg
        )
        return self.memory

    def detach_memory(self) -> None:
        self.memory = None
        self._recall.fill(0.0)
        self._recall_conf = 0.0

    def _memory_recall(self) -> np.ndarray:
        """O que a memória diz que vai falhar, a partir do contexto atual.

        Endereçada pelo estado do topo da trama anterior — é esse o resumo do
        contexto que a rede tem antes de ver os dados novos.
        """
        self._recall.fill(0.0)
        self._recall_conf = 0.0
        if self.memory is None:
            return self._recall
        value, conf = self.memory.read(self._z_prev[self.L])
        self._recall_conf = conf
        if value is not None:
            self._recall[:] = F(self.memory.cfg.read_gain) * value
        return self._recall

    def dream(self, n_dreams: int = 16, noise: float = 0.15) -> int:
        """REM: o córtex gera os seus próprios padrões e treina sobre eles.

        Semeia o topo com um contexto guardado (mais ruído) ou com ruído puro,
        desce pelo modelo generativo para *sonhar* uma trama, e aprende sobre
        ela. Parece circular — treinar no que o próprio modelo gera — e é
        exatamente essa a função: ancorar o mapeamento que já existe, para que
        a aprendizagem nova o deforme menos. É o pseudorehearsal de Robins
        (1995), que o sono REM já fazia primeiro: ensaiar o conhecimento
        antigo sem guardar os dados antigos.
        """
        snapshot = self.snapshot_state()
        dreamed = 0
        try:
            for _ in range(n_dreams):
                if self.memory is not None and self.memory.n_occupied:
                    slot = self.memory.replay(1, self._replay_rng)
                    z = self.memory.keys[slot[0]].copy()
                else:
                    z = np.zeros(self.cfg.sizes[-1], dtype=F)
                z += F(noise) * self._replay_rng.standard_normal(z.size).astype(F)

                # sonhar: descer o modelo generativo a partir desse contexto
                self._z_prev[self.L][:] = z
                zz = self._f_top(self.A_eff.dot(z)).astype(F, copy=False)
                for l in range(self.L - 1, 0, -1):
                    zz = self.layers[l].predict(zz).copy()
                frame = self.layers[0].predict(zz).copy()
                self._z_prev[0][:] = frame  # o sonho é o seu próprio contexto

                self.step(frame, learn=True, use_memory=False)
                dreamed += 1
        finally:
            self.restore_state(snapshot)
        return dreamed

    def downscale(self, factor: float = 0.01) -> None:
        """Renormalização sináptica (Tononi & Cirelli): o sono encolhe todas
        as sinapses multiplicativamente, preservando os rácios. A vigília
        potencia; o sono repõe o orçamento."""
        f = F(1.0 - factor)
        for lay in self.layers:
            lay.W *= f
            lay.refresh_device()
        self.A *= f
        if self.A0 is not None:
            self.A0 *= f
        self._refresh_transition()

    def sleep(self, cycles: int = 3, sws: int = 16, rem: int = 8,
              downscale: float = 0.0, lr_scale: float = 0.1) -> dict:
        """Uma noite: alternância SWS (replay episódico) / REM (sonho).

        A ordem importa e é a fisiológica — primeiro o episódico entra
        (hipocampo -> córtex), depois o sonho re-ancora o conjunto.
        """
        # A plasticidade do sono não é a da vigília: a acetilcolina muda de
        # nível e o replay atua com ganho reduzido. Sonhar à taxa plena
        # arrasta os pesos para o que o modelo gera mal (medido: média 1.01
        # contra 0.85 sem sono).
        from dataclasses import replace as _replace
        original = self.cfg
        self.cfg = _replace(original, w_lr=original.w_lr * lr_scale,
                            a_lr=original.a_lr * lr_scale,
                            fast_path_lr=original.fast_path_lr * lr_scale)
        replayed = dreamed = 0
        try:
            for _ in range(cycles):
                replayed += self.consolidate(n_episodes=sws)
                dreamed += self.dream(n_dreams=rem)
        finally:
            self.cfg = original
        if downscale > 0.0:
            self.downscale(downscale)
        return {"replayed": replayed, "dreamed": dreamed}

    def consolidate(self, n_episodes: int = 16, passes: int = 1) -> int:
        """O sono: reproduz episódios e destila-os nos pesos.

        Corre a mesma regra local de sempre sobre as tramas guardadas, como se
        estivessem a acontecer agora. O que se repetiu — os traços fortes — é
        escolhido primeiro. Devolve quantos episódios foram reproduzidos.

        Isto é offline de propósito: não acontece enquanto se observa, e por
        isso não compete com a inferência nem gasta energia no sensor.
        """
        if self.memory is None:
            return 0
        snapshot = self.snapshot_state()
        replayed = 0
        try:
            for _ in range(passes):
                for slot in self.memory.replay(n_episodes, self._replay_rng):
                    # Repor o contexto antes de reviver o episódio: o estado do
                    # topo que a memória guardou, e a trama anterior de que a
                    # via rápida precisa. É a completação de padrões — a chave
                    # traz de volta o estado, não só o conteúdo.
                    self._z_prev[self.L][:] = self.memory.keys[slot]
                    self._z_prev[0][:] = self.memory.prev_frames[slot]
                    for l in range(1, self.L):
                        self._z_prev[l].fill(0.0)
                    self.step(self.memory.frames[slot], learn=True, use_memory=False)
                    replayed += 1
        finally:
            self.restore_state(snapshot)
        return replayed

    def _reconstruct(self, slot: int) -> np.ndarray:
        """A observação tal como aconteceu, guardada no episódio."""
        assert self.memory is not None
        return self.memory.frames[slot]

    # ------------------------------------------------------------------
    # um passo de tempo
    # ------------------------------------------------------------------
    def step(self, x: np.ndarray, learn: bool = True,
             use_memory: bool = True) -> StepTrace:
        """Processa uma trama. Devolve o traço de instrumentação."""
        cfg = self.cfg
        x = np.asarray(x, dtype=F)
        if x.shape != (cfg.sizes[0],):
            raise ValueError(
                f"trama com forma {x.shape}, esperado ({cfg.sizes[0]},)"
            )

        trace = StepTrace()
        trace.target_rms = float(np.sqrt(np.mean(np.square(x))))

        # --- 1. a previsão desce ---------------------------------------
        # Cada nível latente recebe duas previsões: a do nível de cima e a do
        # seu próprio passado, à sua escala de tempo. Combinam-se pela
        # precisão de cada uma.
        fast = np.asarray(self._fast_prediction(), dtype=F).copy()
        # A memória opina antes de se ver os dados, a partir do contexto: "da
        # última vez que o mundo estava assim, a previsão falhou por isto".
        key = self._z_prev[self.L].copy()
        recall = (self._memory_recall().copy() if (use_memory and self.memory)
                  else np.zeros_like(fast))
        fast = (fast + recall).astype(F, copy=False)
        prior_top = self._temporal_prior()
        temporal_prior: list[np.ndarray | None] = [None] * (self.L + 1)
        temporal_prior[self.L] = prior_top
        self.z[self.L][:] = prior_top
        if self.top_clamp is not None:
            self.z[self.L][:len(self.top_clamp)] = self.top_clamp
        for l in range(self.L - 1, 0, -1):
            from_above = self.layers[l].predict(self.z[l + 1])
            trans = self.transitions[l]
            if trans is not None:
                temporal_prior[l] = trans.predict(self._z_prev[l]).copy()
            self.z[l][:] = self._combine_priors(l, from_above)
        self.z[0][:] = x  # o nível sensorial é fixado pelos dados

        # --- 2. a surpresa sobe (assentamento iterativo) ----------------
        eps_thr: list[np.ndarray] = [None] * (self.L + 1)  # type: ignore[list-item]
        eps_mod: list[np.ndarray] = [None] * self.L  # type: ignore[list-item]
        eps_temp: list[np.ndarray] = [None] * (self.L + 1)  # type: ignore[list-item]
        prev_surprise = np.inf

        for k in range(cfg.max_iters + 1):
            # A "surpresa" é a energia F = ½Σ‖ε_l‖², somada sobre os níveis.
            # É esta a quantidade que a dinâmica de assentamento desce, logo é
            # a única que se pode usar como critério de convergência. (A soma
            # dos |ε| não serve: os níveis latentes arrancam com ε=0 por
            # construção — foram inicializados pela previsão de cima — e
            # crescem à medida que assumem a sua parte da explicação, o que
            # faz a soma L1 oscilar mesmo quando a energia desce.)
            surprise = 0.0
            transmitted = 0.0

            # erros hierárquicos: ε^h_l = z_l − f(W_l z_{l+1})
            # No nível sensorial soma-se a via rápida: a hierarquia só tem de
            # explicar o que ela não conseguiu.
            for l in range(self.L):
                zhat = self.layers[l].predict(self.z[l + 1])
                eps = self.z[l] - zhat
                if l == 0 and (self.A0 is not None or self.memory is not None):
                    eps = (eps - fast).astype(F, copy=False)
                if k == 0 and l == 0:
                    # ε_0 antes de qualquer correção: o erro de previsão honesto.
                    trace.pred_error = eps.copy()
                pi = self.prec_hier[l]
                surprise += 0.5 * float(np.dot(eps, pi.weight(eps)))
                et = self._threshold(eps, trace, pi)
                transmitted += 0.5 * float(np.dot(et, pi.weight(et)))
                eps_thr[l] = et
                eps_mod[l] = self.layers[l].modulated_error(pi.weight(et))

            # erros temporais: ε^t_l = z_l − f(A_l z_l(t-1)), por nível latente.
            # No topo é o único erro que existe (não há nível acima).
            for l in range(1, self.L + 1):
                prior = temporal_prior[l]
                if prior is None:
                    continue
                eps = self.z[l] - prior
                pi = self.prec_temp[l]
                surprise += 0.5 * float(np.dot(eps, pi.weight(eps)))
                et = self._threshold(eps, trace, pi)
                transmitted += 0.5 * float(np.dot(et, pi.weight(et)))
                eps_temp[l] = et
            eps_thr[self.L] = eps_temp[self.L]

            trace.macs_down += self.macs_down_per_pass
            trace.surprise.append(surprise)
            trace.transmitted.append(transmitted)

            # --- quando parar ------------------------------------------
            # Nada passou o limiar: a previsão explicou a entrada. Silêncio.
            if transmitted <= cfg.settle_tol:
                trace.exit_reason = EXIT_EXPLAINED
                break
            # A iterar sem lucro: mais uma iteração não paga o que custa.
            if k > 0:
                gain = prev_surprise - surprise
                if gain < cfg.settle_min_gain or (
                    cfg.settle_rel_tol > 0.0
                    and gain < cfg.settle_rel_tol * max(prev_surprise, 1e-12)
                ):
                    trace.exit_reason = EXIT_STALLED
                    break
            if k == cfg.max_iters:
                trace.exit_reason = EXIT_CEILING
                break
            prev_surprise = surprise

            # --- correção dos estados de cima --------------------------
            # Δz_l = z_lr · ( W_{l-1}ᵀ(π ε̃^h_{l-1} ⊙ f') − π ε̃^h_l − π ε̃^t_l )
            #
            # Três forças sobre cada estado latente: o erro que sobe de baixo
            # (o que o nível ainda não explicou), o erro contra a previsão do
            # nível de cima, e o erro contra o que ele próprio previu no
            # instante anterior. É a última que a versão do passo 1 só tinha
            # no topo.
            for l in range(1, self.L + 1):
                lay = self.layers[l - 1]
                up = lay.backward(eps_mod[l - 1], eps_thr[l - 1])
                nnz = int(np.count_nonzero(eps_thr[l - 1]))
                trace.macs_up += lay.macs_up(nnz)
                trace.macs_up_dense += lay.macs_down()

                pull = up
                if l < self.L and eps_thr[l] is not None:
                    pull = pull - self.prec_hier[l].weight(eps_thr[l])
                if eps_temp[l] is not None:
                    pull = pull - self.prec_temp[l].weight(eps_temp[l])
                dz = F(self._z_lr_for(l)) * pull
                if cfg.z_clip > 0.0:
                    np.clip(dz, -cfg.z_clip, cfg.z_clip, out=dz)
                self.z[l] += dz
                if cfg.z_max > 0.0:
                    np.clip(self.z[l], -cfg.z_max, cfg.z_max, out=self.z[l])
                if cfg.state_sparsity > 0.0:
                    self._compete(self.z[l], cfg.state_sparsity)
                if l == self.L and self.top_clamp is not None:
                    k = len(self.top_clamp)
                    self.z[l][:k] = self.top_clamp
            trace.iters += 1

        # --- 3. aprendizagem local -------------------------------------
        if learn and cfg.plasticity_gate > 0.0:
            # Neuromodulação: a plasticidade só abre quando há novidade.
            self._surprise_avg = (0.99 * getattr(self, "_surprise_avg", 0.0)
                                  + 0.01 * trace.open_loop_surprise)
            if trace.open_loop_surprise < cfg.plasticity_gate * self._surprise_avg:
                learn = False
        if learn:
            for l in range(self.L):
                self.layers[l].learn(
                    eps_mod[l],
                    self.z[l + 1],
                    cfg.w_lr,
                    cfg.weight_decay,
                    cfg.grad_clip,
                    cfg.metaplasticity,
                    cfg.meta_decay,
                )
                self.prec_hier[l].learn(self.z[l] - self.layers[l].zhat)
                if eps_thr[l] is not None or l == 0:
                    self.layers[l].learn_gate(
                        eps_thr[l] if l > 0 else eps_thr[0],
                        self.z[l + 1], cfg.w_lr, cfg.grad_clip,
                    )
            if cfg.use_transition:
                self._learn_transition(eps_thr[self.L])
            # cada transição aprende com o seu próprio erro temporal e com o
            # seu próprio passado — nada mais local do que isto
            for l in range(1, self.L):
                trans = self.transitions[l]
                if trans is not None and eps_temp[l] is not None:
                    trans.learn(eps_temp[l], self._z_prev[l], cfg.b_lr, cfg.grad_clip)
            for l in range(1, self.L + 1):
                if temporal_prior[l] is not None:
                    self.prec_temp[l].learn(self.z[l] - temporal_prior[l])
            self._anchor_precisions()
            if self.A0 is not None:
                # A via rápida aprende com o *seu próprio* erro (x − A_0·x_ant),
                # não com o resíduo que sobra depois da hierarquia.
                #
                # Treinar as duas vias com o mesmo resíduo parece elegante e é
                # um desastre: ambas perseguem o mesmo sinal ao mesmo tempo,
                # sem nada que atribua crédito, e disputam-no. Medido: ETTm1
                # passou de 0.230 para 0.724. Com alvos separados cada via faz
                # o que sabe — a rápida prevê o que consegue sozinha, a
                # profunda explica o que sobrou — que é a divisão de trabalho
                # que a hipótese das duas vias afirma existir.
                # Passo normalizado (NLMS). A regra delta ΔA ∝ ε·xᵀ só é
                # estável para lr < 1/‖x‖², e ‖x‖ depende da trama e da escala
                # do sinal — um lr fixo está sempre errado para algum sinal.
                # Dividir por ‖x‖² torna o passo adimensional: `fast_path_lr`
                # passa a significar "que fração do erro corrigir por trama",
                # entre 0 e ~2, e deixa de ser um número a afinar por dataset.
                # É a mesma armadilha do z_lr, e a mesma correção.
                #
                # A normalização vem *antes* do clip, de propósito: clipar
                # primeiro reintroduz a dependência da escala que ela existe
                # para eliminar (com o sinal 100× maior, todas as entradas
                # saturam no clip e a direção do passo perde-se).
                # A via rápida aprende contra o seu próprio alvo, sem contar
                # com o que a memória recuperou — senão as duas voltam a
                # disputar o mesmo resíduo.
                own_error = (self.z[0] - fast + recall).astype(F, copy=False)
                norm = float(np.dot(self._z_prev[0], self._z_prev[0])) + 1e-6
                dA0 = np.outer(own_error / norm, self._z_prev[0]).astype(F, copy=False)
                if cfg.grad_clip > 0.0:
                    np.clip(dA0, -cfg.grad_clip, cfg.grad_clip, out=dA0)
                self.A0 += F(cfg.fast_path_lr) * dA0

        # --- 4. memória episódica --------------------------------------
        # Grava-se o que surpreendeu, com a chave que estava em vigor quando a
        # previsão foi feita — para que da próxima vez que o contexto se
        # pareça com este, a correção esteja lá.
        if self.memory is not None:
            if learn and trace.pred_error is not None:
                self.memory.write(
                    key, trace.pred_error, self.z[0], self._z_prev[0],
                    trace.open_loop_surprise,
                )
            self.memory.decay()

        # o estado atravessa para a trama seguinte
        self._z_prev[0][:] = self.z[0]
        for l in range(1, self.L + 1):
            self._z_prev[l][:] = self.z[l]
        return trace

    # ------------------------------------------------------------------
    def _learn_transition(self, eps_top_thr: np.ndarray) -> None:
        """ΔA ∝ (ε_topo ⊙ f') ⊗ z_topo(t-1). Também local."""
        cfg = self.cfg
        if self.gated is not None:
            self.gated.learn(eps_top_thr, cfg.a_lr, cfg.grad_clip)
            return
        eps_mod = eps_top_thr * self._fprime_top(self.a_top)
        if self.mixture is not None:
            self.mixture.learn(eps_mod, cfg.a_lr, cfg.proto_lr, cfg.grad_clip)
            return
        dA = np.outer(eps_mod, self._z_top_prev).astype(F, copy=False)
        if cfg.grad_clip > 0.0:
            np.clip(dA, -cfg.grad_clip, cfg.grad_clip, out=dA)
        if cfg.weight_decay > 0.0:
            self.A *= F(1.0 - cfg.a_lr * cfg.weight_decay)
        self.A += F(cfg.a_lr) * dA
        self._refresh_transition()

    def _anchor_precisions(self) -> None:
        """Fixa a escala das precisões no canal sensorial.

        π do nível 0 fica sempre em 1 e todos os outros passam a significar
        "este erro vale k vezes um erro sensorial". Sem esta âncora, todas as
        precisões sobem juntas à medida que a rede aprende (π → 1/⟨ε²⟩), o
        Hessiano infla, o passo de assentamento adaptativo encolhe para não
        divergir, e a inferência deixa de acontecer.
        """
        if not self.cfg.use_precision:
            return
        offset = float(np.mean(self.prec_hier[0].log_pi))
        for p in self.prec_hier:
            p.rescale(offset)
        for p in self.prec_temp[1:]:
            p.rescale(offset)

    @staticmethod
    def _compete(z: np.ndarray, frac: float) -> None:
        """Só as k unidades mais fortes sobrevivem; as outras vão a zero.

        Inibição lateral, in-place. É o mecanismo pelo qual contextos
        diferentes passam a usar populações diferentes — e dois mundos que
        usam neurónios diferentes não se apagam um ao outro.
        """
        k = max(1, int(round(frac * z.size)))
        if k >= z.size:
            return
        cut = np.partition(np.abs(z), z.size - k)[z.size - k]
        z[np.abs(z) < cut] = 0.0

    def _z_lr_for(self, level: int) -> float:
        """Passo de assentamento do nível `level`, limitado pela estabilidade.

        O Hessiano da energia em ordem a z_l é

            π^h_l·I  +  π^t_l·I  +  π^h_{l-1}·W_{l-1}ᵀW_{l-1}

        (o termo temporal contribui identidade porque z_l(t-1) está fixo),
        logo o maior valor próprio é π^h_l + π^t_l + π^h_{l-1}·σ_max(W)² e o
        passo estável é 2 sobre isso. Com precisão ligada, os ganhos entram
        aqui — é o que impede um nível de precisão alta de desestabilizar o
        assentamento.
        """
        if not self.cfg.adaptive_z_lr:
            return self.cfg.z_lr
        lam = self.prec_hier[level - 1].scalar * self.layers[level - 1].sigma_max**2
        if level < self.L:
            lam += self.prec_hier[level].scalar
        if self.transitions[level] is not None or level == self.L:
            lam += self.prec_temp[level].scalar
        return min(self.cfg.z_lr, self.cfg.z_lr_safety * 2.0 / max(lam, 1e-12))

    def _threshold(self, eps: np.ndarray, trace: StepTrace, prec=None) -> np.ndarray:
        """|ε| < θ não é transmitido. É aqui que nasce a esparsidade real.

        O limiar aplica-se ao erro *normalizado* √π·ε, não ao erro cru: assim
        θ significa "meio desvio-padrão de surpresa" em qualquer nível, em vez
        de significar coisas diferentes no sinal cru e no latente abstrato.
        Sem precisão, √π = 1 e volta a ser o limiar do passo 1.

        E é aqui que o ADC entra: só o que passa o limiar é convertido, por
        isso a quantização aplica-se depois de silenciar, não antes.
        """
        trace.eps_total += eps.size
        if self.cfg.theta <= 0.0:
            out = eps.astype(F, copy=False)
        else:
            scale = eps if prec is None else prec.normalize(eps)
            mask = np.abs(scale) >= self.cfg.theta
            trace.eps_silenced += int(eps.size - np.count_nonzero(mask))
            out = np.where(mask, eps, F(0.0)).astype(F, copy=False)

        m = self.device_model
        if m is not None and m.adc_bits > 0:
            out = quantize_adc(out, m.adc_bits, m.adc_range)
        return out

    # ------------------------------------------------------------------
    # sequências
    # ------------------------------------------------------------------
    def run(
        self, frames: np.ndarray, learn: bool = True, reset: bool = False
    ) -> list[StepTrace]:
        """Corre uma sequência de tramas (n_frames, n_sensory)."""
        frames = np.asarray(frames, dtype=F)
        if frames.ndim != 2:
            raise ValueError("frames tem de ter forma (n_frames, n_sensory)")
        if reset:
            self.reset()
        return [self.step(frame, learn=learn) for frame in frames]

    # ------------------------------------------------------------------
    # pesos (fronteira com o C)
    # ------------------------------------------------------------------
    @property
    def weights(self) -> list[np.ndarray]:
        return [lay.W for lay in self.layers]

    def state_dict(self) -> dict:
        d = {f"W{l}": lay.W.copy() for l, lay in enumerate(self.layers)}
        d["A"] = self.A.copy()
        return d

    def load_state_dict(self, d: dict) -> None:
        for l, lay in enumerate(self.layers):
            W = np.asarray(d[f"W{l}"], dtype=F)
            if W.shape != lay.W.shape:
                raise ValueError(f"W{l}: forma {W.shape}, esperado {lay.W.shape}")
            lay.W[:] = W
            lay.refresh_device()
        A = np.asarray(d["A"], dtype=F)
        if A.shape != self.A.shape:
            raise ValueError(f"A: forma {A.shape}, esperado {self.A.shape}")
        self.A[:] = A
        self._refresh_transition()
