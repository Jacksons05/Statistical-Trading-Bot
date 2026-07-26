import pytest

from sttbot.strategies.prob_arbitrage import Outcome, find_boundary_arbitrage


def test_underround_buy_basket():
    # Asks sum to 0.95 < 1.0 -> buy all legs for a guaranteed 1.00 payout.
    outcomes = [
        Outcome("X", best_bid=0.30, best_ask=0.32),
        Outcome("Y", best_bid=0.28, best_ask=0.30),
        Outcome("Z", best_bid=0.31, best_ask=0.33),
    ]
    opp = find_boundary_arbitrage(outcomes)
    assert opp is not None
    assert opp.side == "buy_basket"
    assert opp.gross_edge == pytest.approx(0.05)
    assert opp.net_edge == pytest.approx(0.05)


def test_overround_sell_basket():
    # Bids sum to 1.05 > 1.0 -> sell all legs, collect the excess.
    outcomes = [
        Outcome("X", best_bid=0.36, best_ask=0.38),
        Outcome("Y", best_bid=0.34, best_ask=0.36),
        Outcome("Z", best_bid=0.35, best_ask=0.37),
    ]
    opp = find_boundary_arbitrage(outcomes)
    assert opp is not None
    assert opp.side == "sell_basket"
    assert opp.gross_edge == pytest.approx(0.05)


def test_fees_can_erase_thin_edge():
    outcomes = [
        Outcome("X", best_bid=0.30, best_ask=0.33),
        Outcome("Y", best_bid=0.28, best_ask=0.33),
        Outcome("Z", best_bid=0.31, best_ask=0.33),
    ]
    # Ask sum 0.99, gross edge 0.01; a 2% per-leg fee wipes it out.
    assert find_boundary_arbitrage(outcomes, fee_rate=0.02) is None


def test_no_opportunity_when_book_balanced():
    outcomes = [
        Outcome("X", best_bid=0.49, best_ask=0.51),
        Outcome("Y", best_bid=0.49, best_ask=0.51),
    ]
    assert find_boundary_arbitrage(outcomes) is None


def test_min_net_edge_filter():
    outcomes = [
        Outcome("X", best_bid=0.30, best_ask=0.32),
        Outcome("Y", best_bid=0.28, best_ask=0.30),
        Outcome("Z", best_bid=0.31, best_ask=0.33),
    ]
    assert find_boundary_arbitrage(outcomes, min_net_edge=0.10) is None


def test_crossed_book_rejected():
    with pytest.raises(ValueError):
        Outcome("bad", best_bid=0.6, best_ask=0.4)


def test_price_out_of_range_rejected():
    with pytest.raises(ValueError):
        Outcome("bad", best_bid=-0.1, best_ask=0.4)


def test_requires_two_outcomes():
    with pytest.raises(ValueError):
        find_boundary_arbitrage([Outcome("only", 0.4, 0.5)])
