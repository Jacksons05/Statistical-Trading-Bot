import pytest

from sttbot.strategies.market_making import (
    MarketMaker,
    QuoteParams,
    breakeven_spread,
    contract_variance,
    is_tradeable,
    untradeable_band,
)
from sttbot.venues.prediction import KALSHI_FEES, POLYMARKET_FEES, FeeModel


def test_contract_variance_peaks_at_the_coin_flip():
    assert contract_variance(0.5) == pytest.approx(0.25)
    assert contract_variance(0.02) < contract_variance(0.2) < contract_variance(0.5)
    assert contract_variance(0.0) == 0.0
    assert contract_variance(0.9) == pytest.approx(contract_variance(0.1))
    with pytest.raises(ValueError):
        contract_variance(1.5)


def test_breakeven_spread_is_two_fees():
    assert breakeven_spread(0.5, KALSHI_FEES) == pytest.approx(2 * 0.07 * 0.25)
    # The headline constraint: 3.5c round trip at mid-book, 0.67c at the tail.
    assert breakeven_spread(0.50, KALSHI_FEES) == pytest.approx(0.035)
    assert breakeven_spread(0.05, KALSHI_FEES) == pytest.approx(0.00665)
    assert breakeven_spread(0.5, POLYMARKET_FEES) == 0.0


def test_untradeable_band_is_the_middle_of_the_book():
    """A quadratic fee blocks a contiguous block around 0.50, not the tails."""
    band = untradeable_band(KALSHI_FEES, available_spread=0.01)
    assert band is not None
    low, high = band
    assert low < 0.50 < high
    # Tails stay workable, middle does not.
    assert is_tradeable(0.05, KALSHI_FEES, 0.01)
    assert not is_tradeable(0.50, KALSHI_FEES, 0.01)


def test_wider_spread_shrinks_the_blocked_band():
    narrow = untradeable_band(KALSHI_FEES, 0.01)
    wide = untradeable_band(KALSHI_FEES, 0.03)
    assert narrow is not None and wide is not None
    assert (wide[1] - wide[0]) < (narrow[1] - narrow[0])


def test_whole_book_blocked_when_spread_below_every_fee():
    band = untradeable_band(KALSHI_FEES, available_spread=0.0001)
    assert band == (0.01, 0.99)


def test_zero_fee_venue_blocks_nothing():
    assert untradeable_band(POLYMARKET_FEES, 0.001) is None
    assert is_tradeable(0.50, POLYMARKET_FEES, 0.001)


def test_untradeable_band_validates_spread():
    with pytest.raises(ValueError):
        untradeable_band(KALSHI_FEES, 0.0)


# --- quoting ---------------------------------------------------------------


@pytest.fixture
def maker():
    return MarketMaker(
        QuoteParams(target_edge=0.01, max_inventory=100.0, quote_size=10.0),
        fees=FeeModel(),  # zero-fee, so tests isolate inventory behaviour
    )


def test_flat_inventory_quotes_symmetrically(maker):
    quote = maker.quote(0.50, inventory=0.0)
    assert quote.is_two_sided
    assert quote.reservation_price == pytest.approx(0.50)
    assert quote.bid == pytest.approx(0.50 - quote.half_spread)
    assert quote.ask == pytest.approx(0.50 + quote.half_spread)


def test_long_inventory_skews_quotes_down(maker):
    flat = maker.quote(0.50, inventory=0.0)
    long = maker.quote(0.50, inventory=50.0)
    # Both sides move down: less likely to buy more, more likely to be lifted.
    assert long.reservation_price < flat.reservation_price
    assert long.bid < flat.bid
    assert long.ask < flat.ask


def test_short_inventory_skews_quotes_up(maker):
    flat = maker.quote(0.50, inventory=0.0)
    short = maker.quote(0.50, inventory=-50.0)
    assert short.reservation_price > flat.reservation_price
    assert short.bid > flat.bid


def test_skew_scales_with_inventory_and_risk_aversion(maker):
    small = maker.quote(0.50, inventory=10.0).reservation_price
    large = maker.quote(0.50, inventory=80.0).reservation_price
    assert large < small < 0.50

    timid = MarketMaker(QuoteParams(risk_aversion=0.1), fees=FeeModel())
    bold = MarketMaker(QuoteParams(risk_aversion=5.0), fees=FeeModel())
    assert bold.quote(0.5, 50.0).reservation_price < timid.quote(0.5, 50.0).reservation_price


def test_skew_vanishes_at_expiry(maker):
    """Inventory risk is proportional to time left to carry it."""
    early = maker.quote(0.50, inventory=50.0, time_remaining=1.0)
    late = maker.quote(0.50, inventory=50.0, time_remaining=0.0)
    assert early.reservation_price < late.reservation_price
    assert late.reservation_price == pytest.approx(0.50)


def test_skew_is_smaller_near_the_tails(maker):
    """p(1-p) is small at the extremes, so the same inventory is less risky."""
    mid_skew = 0.50 - maker.quote(0.50, inventory=50.0).reservation_price
    tail_skew = 0.05 - maker.quote(0.05, inventory=50.0).reservation_price
    assert 0 < tail_skew < mid_skew


def test_long_limit_withdraws_the_bid_only(maker):
    quote = maker.quote(0.50, inventory=100.0)
    assert quote.bid is None      # stop accumulating
    assert quote.ask is not None  # keep unwinding
    assert not quote.is_two_sided


def test_short_limit_withdraws_the_ask_only(maker):
    quote = maker.quote(0.50, inventory=-100.0)
    assert quote.ask is None
    assert quote.bid is not None


def test_quotes_stay_inside_the_tradeable_band(maker):
    for fair in (0.01, 0.02, 0.98, 0.99):
        quote = maker.quote(fair)
        for side in (quote.bid, quote.ask):
            if side is not None:
                assert 0.01 <= side <= 0.99


def test_quote_is_never_crossed(maker):
    for fair in (0.01, 0.1, 0.5, 0.9, 0.99):
        for inv in (-100.0, -50.0, 0.0, 50.0, 100.0):
            quote = maker.quote(fair, inventory=inv)
            if quote.is_two_sided:
                assert quote.bid < quote.ask


def test_spread_widens_to_cover_fees():
    free = MarketMaker(QuoteParams(target_edge=0.01), fees=POLYMARKET_FEES)
    kalshi = MarketMaker(QuoteParams(target_edge=0.01), fees=KALSHI_FEES)
    # Same target edge, but Kalshi's fee forces a materially wider quote.
    assert kalshi.half_spread(0.50) > free.half_spread(0.50)
    assert kalshi.quote(0.50).spread > breakeven_spread(0.50, KALSHI_FEES)


def test_spread_covers_breakeven_across_the_book():
    kalshi = MarketMaker(QuoteParams(target_edge=0.005), fees=KALSHI_FEES)
    for price in (0.05, 0.15, 0.30, 0.50, 0.70, 0.95):
        quote = kalshi.quote(price)
        assert quote.spread > breakeven_spread(price, KALSHI_FEES)


def test_can_quote_profitably_respects_the_market_spread():
    kalshi = MarketMaker(fees=KALSHI_FEES)
    # A 2c market at mid-book is tighter than the 3.5c round-trip fee.
    assert not kalshi.can_quote_profitably(0.50, market_spread=0.02)
    assert kalshi.can_quote_profitably(0.50, market_spread=0.05)
    # The same 2c market is comfortably workable at the tail.
    assert kalshi.can_quote_profitably(0.05, market_spread=0.02)


def test_quote_validates_inputs(maker):
    with pytest.raises(ValueError):
        maker.quote(1.5)
    with pytest.raises(ValueError):
        maker.quote(0.5, inventory=0.0, time_remaining=2.0)


def test_params_validate():
    for bad in (
        dict(max_inventory=0),
        dict(quote_size=0),
        dict(risk_aversion=-1),
    ):
        with pytest.raises(ValueError):
            QuoteParams(**bad)
