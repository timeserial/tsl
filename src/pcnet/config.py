"""Configuração da rede preditiva.

Um único dataclass, serializável, que é ao mesmo tempo o contrato com o
inferidor em C: tudo o que aqui está tem de ser exprimível como `#define` ou
como array estático.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PCConfig:
    # Hierarquia. sizes[0] é o nível sensorial (a trama observada), o último
    # é o topo. Default: 64 -> 32 -> 16 -> 8, como no plano.
    sizes: tuple[int, ...] = (64, 32, 16, 8)

    # Ativação do gerador por nível. O nível 0 é o sinal real, logo identidade;
    # os níveis latentes usam tanh (limita o estado, dá derivada barata).
    sensory_activation: str = "identity"
    latent_activation: str = "tanh"

    # --- dinâmica de assentamento -----------------------------------------
    # Teto de passos de assentamento. É um teto, não um custo: o early exit
    # mantém a média em ~8. Note-se que há um mínimo estrutural — o erro
    # sensorial sobe um nível por iteração, logo abaixo de `n_levels - 1` o
    # topo nunca chega a ser corrigido e a rede prevê zero.
    max_iters: int = 25
    # z_lr é um passo de descida no gradiente da energia ½Σ‖ε‖²: só é estável
    # para z_lr < 2/(1 + σ_max(W)²). Como σ_max cresce com a aprendizagem, o
    # valor seguro é baixo — a 0.5 a rede diverge. Ver também `z_clip`.
    z_lr: float = 0.2
    # Com isto ligado, cada nível usa min(z_lr, z_lr_safety · 2/(1+σ_max²)) com
    # σ_max estimado por iteração de potência sobre os pesos que o dispositivo
    # realmente tem. Custa duas leituras do crossbar por atualização de pesos
    # e é o que impede a variabilidade de dispositivo de atravessar o limite
    # de estabilidade sem se dar por isso.
    adaptive_z_lr: bool = True
    z_lr_safety: float = 0.9
    # Saída "explicado": nada passa o limiar, não há o que transmitir.
    settle_tol: float = 1e-6
    # Saída "estagnado", critério absoluto: outra iteração custa energia, por
    # isso só compensa se reduzir a energia do erro em pelo menos isto. Troca
    # joules por joules — é a leitura física, e é a que faz o compute crescer
    # com a surpresa (uma trama difícil ainda tem muito a ganhar por iteração).
    settle_min_gain: float = 0.003
    # Critério relativo alternativo (0 = desligado). Cuidado: para uma trama
    # cujo resíduo é irredutível, a melhoria relativa colapsa de imediato e a
    # rede desiste — poupa energia, mas gasta *menos* onde há *mais* surpresa,
    # que é o contrário do que a arquitetura promete.
    settle_rel_tol: float = 0.0
    # Saturação, o equivalente à aritmética saturante do int8. Sem isto, uma
    # trama muito surpreendente faz o estado disparar e a surpresa *sobe*.
    z_clip: float = 0.5  # teto de |Δz| por iteração (0 = desligado)
    z_max: float = 2.0  # teto de |z| nos níveis latentes (0 = desligado)

    # --- esparsidade de código (separação de padrões) ----------------------
    # Fração de unidades latentes que ficam ativas em cada trama (0 = todas).
    # No córtex andam 1-5%; no giro dentado, menos ainda, e de propósito:
    # entradas parecidas são forçadas a códigos esparsos e quase disjuntos
    # para não se apagarem umas às outras. É a defesa principal do cérebro
    # contra o esquecimento catastrófico, e a nossa rede não a tinha — todas
    # as unidades participavam em todas as tramas, logo todos os pesos eram
    # puxados por todas as tarefas.
    # Implementada por competição: ficam as k maiores, as outras vão a zero.
    # É inibição lateral, e sai de graça em conversões ADC.
    state_sparsity: float = 0.0

    # --- esparsidade de erro -----------------------------------------------
    # |ε| < theta  ->  o erro não é transmitido (silêncio = custo zero).
    theta: float = 0.02

    # --- aprendizagem local ------------------------------------------------
    w_lr: float = 0.1  # ΔW ∝ ε · zᵀ
    a_lr: float = 0.1  # ΔA ∝ ε_topo · z_prevᵀ  (transição temporal)
    # --- metaplasticidade: sinapses que endurecem -------------------------
    # Cada peso guarda a média do quadrado do seu próprio gradiente local — a
    # sua "importância" — e a taxa efetiva dele passa a ser lr/(1+λ·imp). Uma
    # sinapse que já provou importar resiste a ser reescrita, e a tarefa
    # seguinte é obrigada a usar sinapses diferentes: união em vez de
    # compromisso.
    #
    # É o análogo do alargamento da espinha dendrítica e dos modelos em
    # cascata de Fusi. E note-se o que isto tem de particular: em ML,
    # proteger pesos importantes (EWC) exige a informação de Fisher, que
    # exige backprop. Aqui o gradiente local já está na sinapse — a
    # importância sai de graça, e é uma coisa que uma rede com backprop não
    # consegue fazer sem maquinaria extra.
    metaplasticity: float = 0.0
    meta_decay: float = 0.999  # esquecimento da importância acumulada

    # Plasticidade aberta pela surpresa (acetilcolina/noradrenalina): não se
    # aprende quando o mundo é previsível. Protege o passado e poupa energia.
    plasticity_gate: float = 0.0  # 0 = sempre a aprender

    weight_decay: float = 0.0
    grad_clip: float = 1.0  # teto no |ΔW| por passo, estabilidade

    # --- estado temporal ---------------------------------------------------
    # O topo mantém memória entre tramas: ẑ_L(t) = tanh(A · z_L(t-1)).
    # É a "via rápida" em forma mínima: uma recorrência linear.
    use_transition: bool = True
    # Nº de dinâmicas no topo (ver mixture.py). 1 = uma matriz só, que é o
    # comportamento anterior. >1 dá à rede regimes alternativos e um selector
    # por contexto — sem isto, tarefas com dinâmicas diferentes puxam a única
    # matriz para uma média que não serve nenhuma, e aumentar o tamanho da
    # rede não resolve (medido: dobrar os parâmetros não muda nada).
    # Transição com portões multiplicativos (ver gated.py): a dinâmica do
    # topo passa a ser escolhida pelo erro, unidade a unidade, em função do
    # contexto — o mecanismo que a decomposição do muro apontou como a maior
    # parte da distância para o backprop.
    gated_transition: bool = False
    # Traços de elegibilidade na transição portada (e-prop / synaptic
    # tagging): crédito através do tempo com memória por sinapse, sem BPTT.
    eligibility: bool = False
    # O relé talâmico: o candidato e o portão da transição veem também a
    # trama anterior, em paralelo com o estado recorrente (ver gated.py).
    thalamic: bool = False
    # Portões também nas camadas geradoras: ẑ_l = g⊙f(W·z), g = σ(G·z + b).
    # O crédito continua local — ΔW ganha o fator g, ΔG usa ε⊙f(Wz)⊙g(1-g) —
    # e o caminho ascendente ganha o termo do portão. É o análogo espacial do
    # portão temporal: decidir por unidade *que parte* da previsão se aplica
    # neste contexto.
    gated_layers: bool = False
    n_dynamics: int = 1
    dynamics_tau: float = 0.25  # nitidez do selector (baixo = mais exclusivo)
    proto_lr: float = 0.02
    state_leak: float = 0.0  # fuga aplicada a z_L(t-1) antes da transição

    # --- hierarquia de escalas de tempo (ver temporal.py) ------------------
    # "none" reproduz o passo 1: só o topo tem memória, os níveis intermédios
    # são instantâneos. "diagonal" dá a cada unidade a sua constante de tempo
    # (n parâmetros por nível — a versão barata e literalmente biológica).
    # "dense" deixa cada nível misturar-se consigo próprio (n² por nível).
    level_transition: str = "none"
    # --- via rápida (secção 4 do CONTEXTO) --------------------------------
    # Uma via curta e sempre ligada que prevê a trama seguinte diretamente da
    # anterior, sem passar pela hierarquia. A hierarquia deixa de ter de
    # explicar o sinal e passa a explicar só o que a via rápida falha:
    #
    #     ẑ_0 = A_0·z_0(t-1)  +  f(W_0·z_1)
    #           └─ via rápida ┘  └ hierarquia ┘
    #
    # A decomposição é aditiva de propósito: os gradientes de tudo o resto
    # ficam exatamente iguais. Mas cada via aprende com o *seu próprio* erro,
    # não com o mesmo — treinar ambas no mesmo resíduo fá-las disputar o mesmo
    # sinal sem nada que atribua crédito (medido: ETTm1 0.230 -> 0.724).
    # É a "shallow brain hypothesis" na sua forma mais pobre — uma via
    # cortico-subcortical curta em paralelo com a via profunda.
    fast_path: bool = False
    # Fração do erro que a via rápida corrige por trama (passo normalizado,
    # ver network.py). Adimensional: 1.0 corrigiria o erro todo de uma vez.
    fast_path_lr: float = 0.1
    # λ do nível latente mais baixo, e quanto abranda por degrau. ratio=2:
    # cada nível integra sobre o dobro do tempo do de baixo.
    timescale_base: float = 1.0
    timescale_ratio: float = 2.0
    b_lr: float = 0.05  # taxa de aprendizagem das transições por nível

    # --- precisão (ver precision.py) ---------------------------------------
    # π por nível: o ganho com que cada erro conta. Com isto ligado, o limiar
    # θ passa a ser aplicado ao erro normalizado √π·ε, e a mesma constante quer
    # dizer a mesma coisa no nível sensorial e no topo.
    use_precision: bool = False
    precision_per_unit: bool = False
    precision_lr: float = 0.01

    seed: int = 0
    init_scale: float = 1.0  # multiplicador do init ~ 1/sqrt(fan_in)

    def __post_init__(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError("é preciso pelo menos um nível sensorial e um latente")
        if any(n <= 0 for n in self.sizes):
            raise ValueError("todos os tamanhos têm de ser positivos")
        if not 0.0 < self.z_lr <= 1.0:
            raise ValueError("z_lr tem de estar em ]0, 1]")
        if self.z_clip < 0.0 or self.z_max < 0.0:
            raise ValueError("z_clip e z_max não podem ser negativos")
        if self.theta < 0.0:
            raise ValueError("theta não pode ser negativo")
        if self.max_iters < 0:
            raise ValueError("max_iters não pode ser negativo")
        if self.level_transition not in ("none", "diagonal", "dense"):
            raise ValueError(f"level_transition desconhecida: {self.level_transition!r}")
        if self.timescale_ratio < 1.0:
            raise ValueError("timescale_ratio < 1 poria o topo mais rápido que a base")

    # -- utilidades ---------------------------------------------------------
    @classmethod
    def recommended(cls, **overrides) -> "PCConfig":
        """A melhor configuração medida até agora.

        Via rápida + precisão: melhor exatidão e 5-10× menos conversões ADC
        do que os defaults, em três conjuntos diferentes (ver README).

        Porque é que não são os defaults: os defaults reproduzem exatamente os
        passos 1 e 2 do plano, e há artefactos publicados contra eles —
        `runs/phase0/model.h`, os vetores de referência do C, e todos os
        números do README. Mudar os defaults invalidaria esse registo sem
        acrescentar nada que este método não dê. É uma decisão a tomar quando
        houver um demonstrador escolhido, não antes.
        """
        base = dict(fast_path=True, fast_path_lr=0.05, use_precision=True)
        return cls(**{**base, **overrides})

    @property
    def n_levels(self) -> int:
        return len(self.sizes)

    @property
    def n_weights(self) -> int:
        """Número de matrizes geradoras W_l (uma por par de níveis)."""
        return len(self.sizes) - 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PCConfig":
        d = dict(d)
        d["sizes"] = tuple(d["sizes"])
        return cls(**d)
