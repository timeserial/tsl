import pytest

from pcnet import PCConfig


def test_defaults_match_the_plan():
    cfg = PCConfig()
    assert cfg.sizes == (64, 32, 16, 8)
    assert cfg.n_levels == 4
    assert cfg.n_weights == 3
    # Teto, não custo médio: o early exit mantém a média em ~8, e o mínimo
    # estrutural é n_levels-1 (o erro sobe um nível por iteração).
    assert cfg.max_iters >= cfg.n_levels - 1
    assert cfg.adaptive_z_lr


def test_roundtrip_through_dict():
    cfg = PCConfig(sizes=(8, 4), theta=0.05, max_iters=3)
    assert PCConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sizes": (8,)}, "pelo menos"),
        ({"sizes": (8, 0)}, "positivos"),
        ({"z_lr": 0.0}, "z_lr"),
        ({"z_lr": 1.5}, "z_lr"),
        ({"theta": -0.1}, "theta"),
        ({"max_iters": -1}, "max_iters"),
        ({"z_clip": -1.0}, "z_clip"),
    ],
)
def test_invalid_configs_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PCConfig(**kwargs)


def test_recommended_turns_on_what_the_measurements_justify():
    cfg = PCConfig.recommended()
    assert cfg.fast_path and cfg.use_precision
    assert not PCConfig().fast_path  # os defaults reproduzem os passos 1 e 2
    assert PCConfig.recommended(seed=7, theta=0.05).theta == 0.05
    assert PCConfig.recommended(fast_path=False).fast_path is False
