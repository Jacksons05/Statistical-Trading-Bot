import math

import pytest

from sttbot.strategies.amm import (
    Pool,
    impermanent_loss,
    no_arb_band,
    optimal_arbitrage,
)


@pytest.fixture
def pool():
    # 1,000 tokens against 100,000 numeraire -> spot 100.0, 30bp fee.
    return Pool(token_reserve=1_000.0, numeraire_reserve=100_000.0, fee=0.003)


def test_invariants(pool):
    assert pool.spot_price == pytest.approx(100.0)
    assert pool.k == pytest.approx(1e8)
    assert pool.gamma == pytest.approx(0.997)


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        Pool(0, 100)
    with pytest.raises(ValueError):
        Pool(100, 100, fee=1.0)


def test_swap_preserves_constant_product_after_fees(pool):
    after = pool.with_swap(numeraire_in=1_000.0)
    # Fees accrue to the pool, so k grows rather than staying exactly constant.
    assert after.k > pool.k
    assert after.spot_price > pool.spot_price  # buying token raises its price


def test_buy_and_sell_are_lossy_round_trip(pool):
    tokens = pool.buy_token(1_000.0)
    back = pool.with_swap(numeraire_in=1_000.0).sell_token(tokens)
    # Two 30bp fees plus curvature: you cannot round-trip to breakeven.
    assert back < 1_000.0


def test_price_impact_grows_with_size(pool):
    small = pool.price_impact_buy(100.0)
    large = pool.price_impact_buy(10_000.0)
    assert 0 < small < large
    # A trade worth 10% of the numeraire reserve should hurt materially.
    assert large > 0.05


def test_no_arb_band_brackets_spot(pool):
    low, high = no_arb_band(pool)
    assert low < pool.spot_price < high
    # Inside the band, no direction is worth trading at any size.
    assert optimal_arbitrage(pool, (low + pool.spot_price) / 2) is None
    assert optimal_arbitrage(pool, (high + pool.spot_price) / 2) is None


def test_arb_buys_when_pool_is_cheap(pool):
    trade = optimal_arbitrage(pool, external_price=110.0)
    assert trade is not None
    assert trade.direction == "buy_dex"
    assert trade.profitable
    # The trade should move the pool toward, but not past, the external price.
    assert pool.spot_price < trade.pool_price_after <= 110.0


def test_arb_sells_when_pool_is_expensive(pool):
    trade = optimal_arbitrage(pool, external_price=90.0)
    assert trade is not None
    assert trade.direction == "sell_dex"
    assert trade.profitable
    assert 90.0 <= trade.pool_price_after < pool.spot_price


@pytest.mark.parametrize("external", [105.0, 110.0, 130.0, 200.0])
def test_analytic_optimum_matches_brute_force_buy(pool, external):
    """The closed form must beat every nearby size, not merely be positive."""
    trade = optimal_arbitrage(pool, external)
    assert trade is not None

    def profit(size: float) -> float:
        return pool.buy_token(size) * external - size

    best = max((profit(s) for s in _grid(trade.size_in)), default=0.0)
    assert trade.gross_profit >= best - 1e-6
    assert trade.gross_profit == pytest.approx(profit(trade.size_in))


@pytest.mark.parametrize("external", [95.0, 90.0, 70.0, 50.0])
def test_analytic_optimum_matches_brute_force_sell(pool, external):
    trade = optimal_arbitrage(pool, external)
    assert trade is not None

    def profit(size: float) -> float:
        return pool.sell_token(size) - size * external

    best = max((profit(s) for s in _grid(trade.size_in)), default=0.0)
    assert trade.gross_profit >= best - 1e-6


def _grid(centre: float, span: float = 0.5, points: int = 401):
    """Sizes bracketing ``centre`` by +/-``span`` fraction, excluding zero."""
    lo, hi = centre * (1 - span), centre * (1 + span)
    return [lo + (hi - lo) * i / (points - 1) for i in range(points) if lo > 0]


def test_trading_to_price_equality_overshoots(pool):
    """Sizing to equalise prices earns less than the profit-maximising size."""
    external = 130.0
    trade = optimal_arbitrage(pool, external)
    assert trade is not None
    # Size that would drive the pool spot exactly to the external price.
    equalising = math.sqrt(pool.k * external) - pool.numeraire_reserve
    naive_profit = pool.buy_token(equalising) * external - equalising
    assert equalising > trade.size_in
    assert naive_profit < trade.gross_profit


def test_gas_can_make_an_optimal_trade_unprofitable(pool):
    trade = optimal_arbitrage(pool, 100.5, gas_cost=0.0)
    assert trade is not None and trade.profitable
    # Same edge, but gas exceeds it: still returned, flagged unprofitable.
    expensive = optimal_arbitrage(pool, 100.5, gas_cost=trade.gross_profit * 2)
    assert expensive is not None
    assert not expensive.profitable
    assert expensive.net_profit < 0


def test_capital_cap_shrinks_the_trade(pool):
    full = optimal_arbitrage(pool, 130.0)
    capped = optimal_arbitrage(pool, 130.0, max_capital=500.0)
    assert full is not None and capped is not None
    assert capped.size_in == pytest.approx(500.0)
    assert capped.size_in < full.size_in
    assert 0 < capped.gross_profit < full.gross_profit


def test_capital_cap_on_sell_side_is_denominated_in_numeraire(pool):
    capped = optimal_arbitrage(pool, 50.0, max_capital=500.0)
    assert capped is not None
    assert capped.direction == "sell_dex"
    # 500 numeraire buys 10 tokens externally at 50.
    assert capped.size_in == pytest.approx(10.0)


def test_deeper_pools_absorb_more(pool):
    deep = Pool(10_000.0, 1_000_000.0, fee=0.003)  # same price, 10x depth
    shallow_trade = optimal_arbitrage(pool, 110.0)
    deep_trade = optimal_arbitrage(deep, 110.0)
    assert shallow_trade is not None and deep_trade is not None
    assert deep_trade.size_in > shallow_trade.size_in
    assert deep_trade.gross_profit > shallow_trade.gross_profit


def test_zero_fee_pool_has_no_band():
    free = Pool(1_000.0, 100_000.0, fee=0.0)
    low, high = no_arb_band(free)
    assert low == pytest.approx(high) == pytest.approx(free.spot_price)
    # Any divergence at all is then actionable.
    assert optimal_arbitrage(free, 100.01) is not None


def test_impermanent_loss_symmetry_and_magnitude():
    assert impermanent_loss(1.0) == pytest.approx(0.0)
    # A 2x move costs an LP ~5.72%, and halving costs exactly the same.
    assert impermanent_loss(2.0) == pytest.approx(-0.0572, abs=1e-4)
    assert impermanent_loss(0.5) == pytest.approx(impermanent_loss(2.0))
    assert impermanent_loss(4.0) < impermanent_loss(2.0) < 0
    with pytest.raises(ValueError):
        impermanent_loss(0)


def test_rejects_bad_arb_arguments(pool):
    with pytest.raises(ValueError):
        optimal_arbitrage(pool, -1.0)
    with pytest.raises(ValueError):
        optimal_arbitrage(pool, 110.0, gas_cost=-1)
    with pytest.raises(ValueError):
        optimal_arbitrage(pool, 110.0, max_capital=0)
