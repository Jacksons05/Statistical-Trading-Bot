import pytest

from sttbot.backtest.mm_simulator import random_walk, simulate, trending_path
from sttbot.strategies.market_making import MarketMaker, QuoteParams
from sttbot.venues.prediction import KALSHI_FEES, POLYMARKET_FEES, FeeModel


def maker(**overrides):
    params = dict(target_edge=0.01, max_inventory=100.0, quote_size=10.0,
                  risk_aversion=0.05)
    params.update(overrides)
    return MarketMaker(QuoteParams(**params), fees=FeeModel())


def test_flat_path_never_fills():
    """No price movement, no crossings, no trades."""
    result = simulate(maker(), [0.50] * 50)
    assert result.num_fills == 0
    assert result.net_pnl == pytest.approx(0.0)
    assert result.mean_markout is None


def test_sub_tick_noise_never_fills():
    """With a 1c tick you cannot scalp sub-cent wobble.

    Quotes snap to 0.49/0.51 around a 0.50 mid, and a +/-0.3c path never
    reaches either. The tick is a hard floor on the noise amplitude a maker can
    monetise, independent of fees.
    """
    path = [0.50 + (0.003 if i % 2 else -0.003) for i in range(60)]
    assert simulate(maker(), path).num_fills == 0


def test_excursion_and_reversion_is_profitable():
    """The regime a maker actually wants: price leaves the mid and comes back.

    0.50 -> 0.51 -> 0.50 -> 0.49 -> 0.50 lifts the ask at 0.51 and hits the bid
    at 0.49, both around a stable fair value. This is uninformed flow, and it
    is what the spread is compensation for.
    """
    cycle = [0.50, 0.51, 0.50, 0.49]
    path = cycle * 15
    result = simulate(maker(), path, settlement=0.50)
    assert result.num_fills > 0
    assert result.net_pnl > 0


def test_oscillation_wider_than_the_spread_loses():
    """Quoting around the last mid on a swinging series is a losing trade.

    With a +/-3c oscillation and a 0.5c half-spread, the maker quotes an ask of
    0.475 while fair value is 0.50 and gets lifted 2.5c too cheap, every time.
    A maker needs a fair-value estimate better than "the last mid" whenever the
    price swings by more than its own spread — this is a property of the
    strategy, not a simulator artifact.
    """
    path = [0.50 + (0.03 if i % 2 else -0.03) for i in range(60)]
    result = simulate(maker(), path, settlement=0.50)
    assert result.num_fills > 0
    assert result.net_pnl < 0
    # Mark-out is not asserted here: on a pure oscillation the price returns to
    # where it started, so the post-fill move is ambiguous by construction. The
    # P&L is the finding.


def test_slow_drift_does_not_fill_a_requoting_maker():
    """A maker that re-quotes every tick outruns a gradual trend.

    Steps of 0.002 against a 0.005 half-spread never cross, because the quote
    re-centres on each new mid. Adverse selection comes from jumps that outpace
    re-quoting, not from slow drift — which is why the trend tests below need
    volatility, not just drift.
    """
    gradual = [0.40 + 0.002 * i for i in range(60)]
    assert simulate(maker(), gradual).num_fills == 0


def test_random_walk_is_deterministic_by_seed():
    a = random_walk(0.5, 100, 0.01, seed=7)
    b = random_walk(0.5, 100, 0.01, seed=7)
    c = random_walk(0.5, 100, 0.01, seed=8)
    assert a == b and a != c


def test_random_walk_stays_in_bounds():
    path = random_walk(0.5, 500, 0.2, seed=3)  # deliberately violent
    assert all(0.01 <= p <= 0.99 for p in path)


def test_markout_detects_adverse_selection_on_a_trend():
    """The core check: a strong trend must show up as negative mark-out.

    A maker quoting both sides into a rally is repeatedly lifted just before
    the price runs further away. If this passes with positive mark-out, the
    simulator is not measuring adverse selection and every result it produces
    is untrustworthy.
    """
    path = trending_path(0.30, 300, drift=0.002, volatility=0.004, seed=11)
    result = simulate(maker(), path, markout_steps=20)
    assert result.num_fills > 0
    assert result.mean_markout is not None
    assert result.mean_markout < 0
    assert result.adverse_fill_rate > 0.5


def test_markout_is_better_on_a_trendless_path_than_a_trending_one():
    calm = simulate(maker(), random_walk(0.5, 300, 0.004, seed=5), markout_steps=20)
    trend = simulate(
        maker(), trending_path(0.30, 300, 0.002, 0.004, seed=5), markout_steps=20
    )
    assert calm.mean_markout is not None and trend.mean_markout is not None
    assert calm.mean_markout > trend.mean_markout


def test_markout_sign_convention_buy_and_sell():
    """Being lifted into a rally, or hit into a slide, are both unfavourable.

    Steps of 0.015 exceed the 0.005 half-spread, so the move outpaces
    re-quoting and the quote is actually crossed.
    """
    rising = [0.40 + 0.015 * i for i in range(30)]
    falling = [0.60 - 0.015 * i for i in range(30)]
    # Into a rise the maker is lifted (sells) -> unfavourable.
    up = simulate(maker(), rising, markout_steps=15)
    assert all(f.side == "sell" for f in up.fills)
    assert up.mean_markout < 0
    # Into a fall the maker is hit (buys) -> also unfavourable.
    down = simulate(maker(), falling, markout_steps=15)
    assert all(f.side == "buy" for f in down.fills)
    assert down.mean_markout < 0


def test_inventory_limit_is_respected():
    """A one-way market must not accumulate past the cap."""
    falling = [0.80 - 0.003 * i for i in range(200)]
    result = simulate(maker(max_inventory=50.0, quote_size=10.0), falling)
    assert abs(result.final_inventory) <= 50.0


def test_higher_risk_aversion_carries_less_inventory():
    falling = [0.80 - 0.002 * i for i in range(200)]
    timid = simulate(maker(risk_aversion=8.0), falling)
    bold = simulate(maker(risk_aversion=0.1), falling)
    assert abs(timid.final_inventory) <= abs(bold.final_inventory)


def test_fees_reduce_pnl_and_are_accounted():
    path = [0.50 + (0.05 if i % 2 else -0.05) for i in range(60)]
    free = simulate(MarketMaker(QuoteParams(), POLYMARKET_FEES), path, settlement=0.5)
    paid = simulate(MarketMaker(QuoteParams(), KALSHI_FEES), path, settlement=0.5)
    assert free.total_fees == 0.0
    if paid.num_fills:
        assert paid.total_fees > 0
        # gross_pnl adds fees back, so it must exceed net whenever fees are paid.
        assert paid.gross_pnl > paid.net_pnl
        assert paid.gross_pnl - paid.net_pnl == pytest.approx(paid.total_fees)


def test_settlement_values_leftover_inventory():
    falling = [0.80 - 0.004 * i for i in range(120)]
    won = simulate(maker(), falling, settlement=1.0)
    lost = simulate(maker(), falling, settlement=0.0)
    # Same fills either way; only the terminal value of inventory differs.
    assert won.num_fills == lost.num_fills
    assert won.final_inventory == lost.final_inventory
    if won.final_inventory > 0:
        assert won.net_pnl > lost.net_pnl


def test_settlement_defaults_to_final_mid():
    path = random_walk(0.5, 100, 0.01, seed=2)
    result = simulate(maker(), path)
    assert result.settlement_value == pytest.approx(result.final_inventory * path[-1])


def test_summary_reports_markout():
    result = simulate(maker(), random_walk(0.5, 200, 0.01, seed=4))
    assert "mean_markout" in result.summary()
    assert "fills=" in result.summary()


def test_simulate_validates_inputs():
    with pytest.raises(ValueError):
        simulate(maker(), [0.5])
    with pytest.raises(ValueError):
        simulate(maker(), [0.5, 0.5], markout_steps=-1)
