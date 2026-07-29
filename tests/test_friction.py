import random

import pytest

from sttbot.economics.friction import (
    FrictionModel,
    should_use_maker,
    stealth_delay,
    stealth_size,
)


def test_taker_pays_spread_and_fees():
    m = FrictionModel(taker_fee=0.001, slippage=0.0005, gas=0.0002)
    net = m.net_edge(0.01, spread=0.002, maker=True)
    taker = m.net_edge(0.01, spread=0.002, maker=False)
    # Maker keeps the spread and pays no taker fee/slippage, so it nets more.
    assert net > taker
    assert taker == pytest.approx(0.01 - 0.001 - 0.0005 - 0.0002 - 0.002)


def test_maker_rebate_adds_edge():
    m = FrictionModel(maker_fee=-0.0002)  # rebate
    assert m.net_edge(0.01, spread=0.003, maker=True) == pytest.approx(0.0102)


def test_should_use_maker_when_spread_large_vs_edge():
    assert should_use_maker(gross_edge=0.01, spread=0.005) is True  # 50% of edge
    assert should_use_maker(gross_edge=0.01, spread=0.001) is False  # 10% of edge


def test_should_use_maker_on_nonpositive_edge():
    assert should_use_maker(gross_edge=0.0, spread=0.001) is True


def test_stealth_size_is_perturbed_but_bounded():
    rng = random.Random(42)
    sizes = [stealth_size(250, jitter=0.03, rng=rng) for _ in range(200)]
    assert any(abs(s - 250) > 1e-6 for s in sizes)  # actually perturbed
    assert all(s >= 125 for s in sizes)  # never below the 50% floor
    assert abs(sum(sizes) / len(sizes) - 250) < 15  # roughly unbiased


def test_stealth_size_rejects_nonpositive():
    with pytest.raises(ValueError):
        stealth_size(0)


def test_stealth_delay_non_negative_and_bounded():
    rng = random.Random(7)
    delays = [stealth_delay(0.1, 1.5, rng=rng) for _ in range(100)]
    assert all(0.1 <= d <= 1.6 for d in delays)
