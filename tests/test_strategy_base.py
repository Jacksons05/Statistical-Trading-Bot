import pytest

from sttbot.strategies.base import Param, Signal, Strategy


class Dummy(Strategy):
    alpha = Param(1.0, low=0.0, high=2.0, kind="float")
    n = Param(3, low=1, high=5, kind="int")
    mode = Param("a", choices=("a", "b", "c"), kind="categorical")

    def generate_signal(self, x):
        return Signal.LONG if x > self.alpha else Signal.FLAT


def test_defaults_applied():
    d = Dummy()
    assert d.alpha == 1.0 and d.n == 3 and d.mode == "a"


def test_override_and_config():
    d = Dummy(alpha=1.5)
    assert d.alpha == 1.5
    assert d.config() == {"alpha": 1.5, "n": 3, "mode": "a"}


def test_unknown_param_rejected():
    with pytest.raises(TypeError):
        Dummy(beta=9)


def test_params_introspection():
    assert set(Dummy.params()) == {"alpha", "n", "mode"}


def test_int_param_grid_is_deduped_and_integer():
    grid = Dummy.params()["n"].grid(points=5)
    assert grid == [1, 2, 3, 4, 5]
    assert all(isinstance(v, int) for v in grid)


def test_categorical_grid():
    assert Dummy.params()["mode"].grid() == ["a", "b", "c"]


def test_grid_product_and_override():
    combos = list(Dummy.grid(alpha=[0.0, 2.0], n=[3], mode=["a"]))
    assert combos == [
        {"alpha": 0.0, "n": 3, "mode": "a"},
        {"alpha": 2.0, "n": 3, "mode": "a"},
    ]


def test_step_grid():
    p = Param(0.0, low=0.0, high=1.0, step=0.5)
    assert p.grid() == [0.0, 0.5, 1.0]


def test_signal_is_signed():
    assert int(Signal.LONG) == 1 and int(Signal.SHORT) == -1 and int(Signal.FLAT) == 0
