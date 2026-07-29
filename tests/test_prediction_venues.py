import pytest

from sttbot.venues.prediction import (
    KALSHI_FEES,
    POLYMARKET_FEES,
    FeeModel,
    VenueQuote,
    find_cross_venue_arb,
    implied_from_book,
    kelly_fraction,
)


def quote(venue="poly", bid=0.40, ask=0.42, fees=None, bid_size=1000, ask_size=1000):
    return VenueQuote(
        venue=venue,
        market_id="EVENT-X",
        best_bid=bid,
        best_ask=ask,
        fees=fees or POLYMARKET_FEES,
        bid_size=bid_size,
        ask_size=ask_size,
    )


# --- fee models ------------------------------------------------------------


def test_kalshi_fee_peaks_at_the_middle():
    mid = KALSHI_FEES.cost(0.50)
    assert mid == pytest.approx(0.07 * 0.25)
    # Symmetric, and vanishing toward both extremes.
    assert KALSHI_FEES.cost(0.10) == pytest.approx(KALSHI_FEES.cost(0.90))
    assert KALSHI_FEES.cost(0.05) < KALSHI_FEES.cost(0.25) < mid
    assert KALSHI_FEES.cost(0.0) == pytest.approx(0.0)
    assert KALSHI_FEES.cost(1.0) == pytest.approx(0.0)


def test_kalshi_fee_scales_with_contracts():
    assert KALSHI_FEES.cost(0.5, contracts=100) == pytest.approx(
        100 * KALSHI_FEES.cost(0.5)
    )


def test_proportional_fee_model_distinguishes_maker_and_taker():
    fees = FeeModel(taker=0.02, maker=0.005)
    assert fees.cost(0.50) == pytest.approx(0.01)
    assert fees.cost(0.50, maker=True) == pytest.approx(0.0025)


def test_polymarket_fees_are_zero():
    assert POLYMARKET_FEES.cost(0.5, contracts=1000) == 0.0


def test_fee_validates_price():
    with pytest.raises(ValueError):
        KALSHI_FEES.cost(1.5)


# --- cross-venue arbitrage -------------------------------------------------


def test_requires_explicit_equivalence_declaration():
    a = quote("poly", 0.40, 0.42)
    b = quote("kalshi", 0.50, 0.52, fees=KALSHI_FEES)
    with pytest.raises(ValueError, match="equivalent"):
        find_cross_venue_arb(a, b)


def test_finds_arb_and_orients_the_legs():
    cheap = quote("poly", 0.40, 0.42)
    rich = quote("kalshi", 0.50, 0.52, fees=POLYMARKET_FEES)
    arb = find_cross_venue_arb(cheap, rich, equivalent=True)
    assert arb is not None
    assert arb.buy_venue == "poly" and arb.sell_venue == "kalshi"
    assert arb.gross_edge == pytest.approx(0.08)  # sell 0.50 - buy 0.42
    assert arb.profitable


def test_argument_order_does_not_change_the_result():
    a = quote("poly", 0.40, 0.42)
    b = quote("kalshi", 0.50, 0.52, fees=POLYMARKET_FEES)
    forward = find_cross_venue_arb(a, b, equivalent=True)
    reverse = find_cross_venue_arb(b, a, equivalent=True)
    assert forward is not None and reverse is not None
    assert forward.buy_venue == reverse.buy_venue
    assert forward.net_edge == pytest.approx(reverse.net_edge)


def test_no_arb_when_books_overlap():
    a = quote("poly", 0.40, 0.42)
    b = quote("kalshi", 0.41, 0.43, fees=POLYMARKET_FEES)
    assert find_cross_venue_arb(a, b, equivalent=True) is None


def test_identical_gross_edge_flips_sign_across_the_kalshi_curve():
    """A 1c gross edge dies mid-book and survives at the tail.

    This is why the quadratic fee cannot be approximated by a flat rate: at
    P=0.50 Kalshi takes 1.75c, more than the edge; at P=0.05 it takes 0.33c.
    """
    mid = find_cross_venue_arb(
        quote("poly", 0.47, 0.49),
        quote("kalshi", 0.50, 0.52, fees=KALSHI_FEES),
        equivalent=True,
    )
    assert mid is None  # 0.01 gross - 0.0175 fee < 0

    tail = find_cross_venue_arb(
        quote("poly", 0.03, 0.04),
        quote("kalshi", 0.05, 0.06, fees=KALSHI_FEES),
        equivalent=True,
    )
    assert tail is not None and tail.profitable
    assert tail.gross_edge == pytest.approx(0.01)  # same edge, opposite verdict
    assert tail.net_edge == pytest.approx(0.01 - 0.07 * 0.05 * 0.95)


def test_size_is_bounded_by_the_thinner_leg():
    cheap = quote("poly", 0.40, 0.42, ask_size=500)
    rich = quote("kalshi", 0.50, 0.52, fees=POLYMARKET_FEES, bid_size=120)
    arb = find_cross_venue_arb(cheap, rich, equivalent=True)
    assert arb is not None
    assert arb.contracts == 120
    assert arb.net_profit == pytest.approx(arb.net_edge * 120)


def test_max_contracts_caps_size():
    cheap = quote("poly", 0.40, 0.42)
    rich = quote("kalshi", 0.50, 0.52, fees=POLYMARKET_FEES)
    arb = find_cross_venue_arb(cheap, rich, equivalent=True, max_contracts=50)
    assert arb is not None and arb.contracts == 50


def test_min_net_edge_filters():
    cheap = quote("poly", 0.40, 0.42)
    rich = quote("kalshi", 0.43, 0.45, fees=POLYMARKET_FEES)
    assert find_cross_venue_arb(cheap, rich, equivalent=True, min_net_edge=0.10) is None


def test_cannot_arb_a_market_against_itself():
    a = quote("poly")
    with pytest.raises(ValueError, match="itself"):
        find_cross_venue_arb(a, a, equivalent=True)


def test_crossed_book_rejected():
    with pytest.raises(ValueError, match="crossed"):
        VenueQuote("poly", "X", best_bid=0.6, best_ask=0.4)


def test_price_outside_unit_interval_rejected():
    with pytest.raises(ValueError):
        VenueQuote("poly", "X", best_bid=-0.1, best_ask=0.5)


# --- sizing and consensus --------------------------------------------------


def test_kelly_is_zero_without_edge():
    assert kelly_fraction(0.50, 0.50) == 0.0
    assert kelly_fraction(0.40, 0.50) == 0.0  # -EV


def test_kelly_grows_with_edge():
    small = kelly_fraction(0.55, 0.50)
    large = kelly_fraction(0.70, 0.50)
    assert 0 < small < large
    # p=0.60 at price 0.50 -> (0.60-0.50)/(1-0.50) = 0.20
    assert kelly_fraction(0.60, 0.50) == pytest.approx(0.20)


def test_kelly_certainty_stakes_everything():
    assert kelly_fraction(1.0, 0.50) == pytest.approx(1.0)


def test_kelly_validates_inputs():
    with pytest.raises(ValueError):
        kelly_fraction(1.5, 0.5)
    for bad_price in (0.0, 1.0):
        with pytest.raises(ValueError):
            kelly_fraction(0.6, bad_price)


def test_consensus_is_depth_weighted():
    thin = quote("thin", 0.20, 0.22, bid_size=1, ask_size=1)      # mid 0.21
    deep = quote("deep", 0.60, 0.62, bid_size=10_000, ask_size=10_000)  # mid 0.61
    consensus = implied_from_book([thin, deep])
    assert consensus > 0.60  # the deep book dominates


def test_consensus_falls_back_to_equal_weight_without_sizes():
    a = quote("a", 0.20, 0.22, bid_size=0, ask_size=0)
    b = quote("b", 0.60, 0.62, bid_size=0, ask_size=0)
    assert implied_from_book([a, b]) == pytest.approx((0.21 + 0.61) / 2)


def test_consensus_requires_a_quote():
    with pytest.raises(ValueError):
        implied_from_book([])
