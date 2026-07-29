"""Tests for the CLV baseline control.

The point of this module is to stop a mechanically-positive CLV being read as
skill, so the tests are built around exactly that failure mode.
"""

import datetime as dt

import pytest

from helpers import FakeMatch
from sttbot.backtest.clv import clv_baseline, clv_skill
from sttbot.backtest.walk_forward import best_of_book_odds, single_book_odds


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
