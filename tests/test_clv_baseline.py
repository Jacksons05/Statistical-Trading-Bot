"""Tests for the CLV baseline control.

The point of this module is to stop a mechanically-positive CLV being read as
skill, so the tests are built around exactly that failure mode.
"""

import datetime as dt

import pytest

from helpers import FakeMatch
from sttbot.backtest.clv import (
    clv_excess_per_bet,
    clv_baseline,
    clv_odds_matched_baseline,
    clv_skill,
)
from sttbot.backtest.walk_forward import Bet, best_of_book_odds, single_book_odds


def _match(entry, close, best=None):
    return FakeMatch(
        date=dt.date(2024, 1, 1),
        home="A",
        away="B",
        home_goals=1,
        away_goals=0,
        odds_home=entry[0],
        odds_draw=entry[1],
        odds_away=entry[2],
        max_home=(best or entry)[0],
        max_draw=(best or entry)[1],
        max_away=(best or entry)[2],
        close_home=close[0],
        close_draw=close[1],
        close_away=close[2],
    )


def test_baseline_is_negative_against_a_single_vigged_book():
    """Betting a random side of one book and settling at the same book's close
    loses the margin on average -- the honest null."""
    entry = (2.30, 3.30, 3.10)
    close = (2.20, 3.20, 3.00)  # every price shortens
    matches = [_match(entry, close) for _ in range(200)]
    summary = clv_baseline(matches, odds_selector=single_book_odds)
    assert summary.n == 200
    assert summary.mean_clv > 0  # close shorter than entry => we beat the close

    # And with the close longer than entry, CLV must go the other way.
    worse = [_match(entry, (2.50, 3.60, 3.40)) for _ in range(200)]
    assert clv_baseline(worse).mean_clv < 0


def test_best_of_book_entry_manufactures_positive_clv_with_no_model():
    """The exact artifact this module exists to expose."""
    entry = (2.30, 3.30, 3.10)
    best = (2.45, 3.50, 3.30)  # max across books, always better
    matches = [_match(entry, entry, best=best) for _ in range(300)]

    at_single = clv_baseline(matches, odds_selector=single_book_odds)
    at_best = clv_baseline(matches, odds_selector=best_of_book_odds)

    assert at_single.mean_clv == pytest.approx(0.0, abs=1e-12)
    assert at_best.mean_clv > 0
    assert at_best.positive_rate == pytest.approx(1.0)


def test_clv_skill_subtracts_the_free_lunch():
    baseline = clv_baseline(
        [_match((2.30, 3.30, 3.10), (2.30, 3.30, 3.10), best=(2.45, 3.50, 3.30))
         for _ in range(300)],
        odds_selector=best_of_book_odds,
    )
    # A strategy matching the baseline has demonstrated no selection skill.
    assert clv_skill(baseline.mean_clv, baseline) == pytest.approx(0.0)
    assert clv_skill(baseline.mean_clv + 0.01, baseline) == pytest.approx(0.01)
    # Healthy-looking positive CLV can still be negative skill.
    assert clv_skill(baseline.mean_clv / 2, baseline) < 0


def test_baseline_is_deterministic_for_a_seed():
    matches = [_match((2.0, 3.5, 4.0), (1.9, 3.6, 4.2)) for _ in range(100)]
    a = clv_baseline(matches, seed=7)
    b = clv_baseline(matches, seed=7)
    assert (a.mean_clv, a.positive_rate) == (b.mean_clv, b.positive_rate)


def test_baseline_skips_matches_without_a_closing_line():
    matches = [_match((2.0, 3.5, 4.0), (None, None, None)) for _ in range(50)]
    summary = clv_baseline(matches)
    assert summary == type(summary)(0, 0.0, 0.0)


def test_more_picks_per_match_tightens_the_estimate():
    matches = [_match((2.0, 3.5, 4.0), (1.95, 3.4, 3.9)) for _ in range(100)]
    assert clv_baseline(matches, picks_per_match=5).n == 500


# --- odds-matched baseline --------------------------------------------------


def _bet(odds: float, clv: float = 0.0) -> Bet:
    return Bet(
        date=dt.date(2024, 1, 1),
        home="A",
        away="B",
        selection="H",
        odds=odds,
        model_prob=1.0 / odds,
        gross_edge=0.0,
        net_edge=0.0,
        stake=1.0,
        pnl=0.0,
        won=False,
        clv=clv,
    )


def test_odds_matched_baseline_exposes_a_longshot_preference():
    """The confound this control exists for.

    Here only longshots drift, and the "strategy" does nothing cleverer than
    always bet longshots. The uniform baseline reports large apparent skill;
    the odds-matched baseline correctly reports none.
    """
    # Home is the longshot and its price shortens; the other two do not move.
    matches = [
        _match((6.00, 3.60, 1.60), (5.00, 3.60, 1.60)) for _ in range(300)
    ]
    bets = [_bet(6.00) for _ in range(50)]

    strategy_clv = 1.0 / 5.00 - 1.0 / 6.00  # what every longshot bet earns

    uniform = clv_baseline(matches, odds_selector=single_book_odds)
    matched = clv_odds_matched_baseline(matches, bets, odds_selector=single_book_odds)

    assert clv_skill(strategy_clv, uniform) > 0.02  # looks like real skill
    assert clv_skill(strategy_clv, matched) == pytest.approx(0.0, abs=1e-9)


def test_odds_matched_baseline_still_credits_genuine_selection():
    """Picking the mover from among same-priced selections is real skill."""
    # Two selections priced identically; only one of them shortens.
    movers = [_match((3.00, 3.00, 3.00), (2.50, 3.00, 3.00)) for _ in range(200)]
    bets = [_bet(3.00) for _ in range(50)]

    strategy_clv = 1.0 / 2.50 - 1.0 / 3.00  # always picked the one that moved
    matched = clv_odds_matched_baseline(movers, bets, odds_selector=single_book_odds)

    # The bucket average is dragged down by the two non-movers, so beating it
    # is a real result rather than a pricing artifact.
    assert clv_skill(strategy_clv, matched) > 0.0


def test_odds_matched_baseline_handles_empty_inputs():
    matches = [_match((2.0, 3.5, 4.0), (1.9, 3.6, 4.2)) for _ in range(10)]
    assert clv_odds_matched_baseline(matches, []).n == 0
    assert clv_odds_matched_baseline([], [_bet(2.0)]).n == 0


def test_excess_per_bet_is_the_paired_form_of_the_baseline():
    """Mean per-bet excess must equal skill against the matched baseline."""
    matches = [_match((2.10, 3.40, 3.80), (2.00, 3.50, 3.90)) for _ in range(200)]
    bets = [_bet(2.10, clv=1.0 / 2.00 - 1.0 / 2.10) for _ in range(40)]

    excess = clv_excess_per_bet(matches, bets, odds_selector=single_book_odds)
    matched = clv_odds_matched_baseline(matches, bets, odds_selector=single_book_odds)

    assert excess.size == len(bets)
    assert float(excess.mean()) == pytest.approx(
        clv_skill(bets[0].clv, matched), abs=1e-12
    )


def test_excess_per_bet_supports_a_standard_error():
    """The reason this returns per-bet values rather than an aggregate."""
    matches = [_match((3.00, 3.00, 3.00), (2.50, 3.00, 3.00)) for _ in range(200)]
    bets = [_bet(3.00, clv=1.0 / 2.50 - 1.0 / 3.00) for _ in range(60)]

    excess = clv_excess_per_bet(matches, bets, odds_selector=single_book_odds)
    assert excess.size == 60
    assert float(excess.mean()) > 0
    # Constant CLV here, so the paired spread is zero and the edge is certain.
    assert float(excess.std()) == pytest.approx(0.0, abs=1e-12)


def test_excess_per_bet_empty_for_unusable_input():
    assert clv_excess_per_bet([], [_bet(2.0)]).size == 0
    matches = [_match((2.0, 3.5, 4.0), (1.9, 3.6, 4.2)) for _ in range(10)]
    assert clv_excess_per_bet(matches, []).size == 0
    # Bets with no closing line contribute nothing.
    assert clv_excess_per_bet(matches, [_bet(2.0, clv=None)]).size == 0


def test_odds_matched_baseline_survives_a_degenerate_price_distribution():
    """Every price identical collapses the quantile edges; must not crash."""
    matches = [_match((3.0, 3.0, 3.0), (2.9, 2.9, 2.9)) for _ in range(50)]
    summary = clv_odds_matched_baseline(matches, [_bet(3.0) for _ in range(5)])
    assert summary.n > 0
    assert summary.mean_clv > 0
