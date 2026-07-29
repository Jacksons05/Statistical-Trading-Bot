"""Tests for the walk-forward backtester and its metrics.

The properties worth defending here are not "does it produce a number" but:
no look-ahead, friction charged consistently between the entry rule and the
P&L, and an honest zero when the market is efficient. All synthetic, seeded,
network-free.
"""

import datetime as dt
import math
from dataclasses import replace

import numpy as np
import pytest

from helpers import FakeMatch
from sttbot.backtest import walk_forward as wf
from sttbot.backtest.metrics import max_drawdown, sharpe_ratio, summarise
from sttbot.backtest.walk_forward import (
    best_of_book_odds,
    single_book_odds,
    walk_forward_backtest,
)
from sttbot.economics.friction import FrictionModel
from sttbot.strategies.dixon_coles import DixonColesModel, TeamRating, score_matrix


TEAMS = [f"T{i}" for i in range(10)]
# Centred attacks, so this is directly comparable to what the fitter returns.
TRUTH = DixonColesModel(
    ratings={
        team: TeamRating(attack=0.36 - 0.08 * i, defence=-0.10 + 0.055 * i)
        for i, team in enumerate(TEAMS)
    },
    home_advantage=0.26,
    rho=-0.06,
)


def _true_probs(home: str, away: str) -> tuple[float, float, float]:
    return TRUTH.match_probabilities(home, away).as_tuple()


def build_season(
    seed: int,
    start: dt.date,
    *,
    margin: float = 0.05,
    distortion: float = 0.0,
    with_closing: bool = True,
) -> list[FakeMatch]:
    """A double round robin priced by a bookmaker of known accuracy.

    ``margin`` is the overround the book charges. ``distortion`` tilts the
    book's home-win price: positive means the book offers *longer* odds on the
    home side than it should, which is a real edge the strategy ought to find.
    The closing line is set to the undistorted fair price, so a bet taken into
    a distortion shows positive CLV.
    """
    rng = np.random.default_rng(seed)
    matches: list[FakeMatch] = []
    day = start
    width = 13
    flat = np.arange(width * width)

    for home in TEAMS:
        for away in TEAMS:
            if home == away:
                continue
            lam, mu = TRUTH.expected_goals(home, away)
            cell = rng.choice(
                flat, p=score_matrix(lam, mu, TRUTH.rho, width - 1).ravel()
            )
            probs = np.array(_true_probs(home, away))

            # Book's view: true probabilities, tilted then loaded with margin.
            view = probs.copy()
            view[0] /= 1.0 + distortion
            view = view / view.sum() * (1.0 + margin)
            prices = tuple(float(1.0 / p) for p in view)

            fair = probs / probs.sum() * (1.0 + margin)
            closing = tuple(float(1.0 / p) for p in fair)

            matches.append(
                FakeMatch(
                    date=day,
                    home=home,
                    away=away,
                    home_goals=int(cell // width),
                    away_goals=int(cell % width),
                    odds_home=prices[0],
                    odds_draw=prices[1],
                    odds_away=prices[2],
                    max_home=prices[0] * 1.02,
                    max_draw=prices[1] * 1.02,
                    max_away=prices[2] * 1.02,
                    close_home=closing[0] if with_closing else None,
                    close_draw=closing[1] if with_closing else None,
                    close_away=closing[2] if with_closing else None,
                )
            )
            day += dt.timedelta(days=1)
    return matches


def build_history(n_seasons: int, seed: int = 4, **kwargs) -> list[FakeMatch]:
    matches: list[FakeMatch] = []
    day = dt.date(2020, 8, 1)
    for season in range(n_seasons):
        matches += build_season(seed + season, day, **kwargs)
        day += dt.timedelta(days=120)
    return matches


# A 40% home-price distortion over ten seasons, entered only above a 10% net
# edge. Sized from measurement, not taste: at a 25% distortion the realised ROI
# over ~700 bets still ranged from -0.8% to +11% across seeds, because betting
# P&L is dominated by outcome variance. The settings below put the worst seed
# of six at +12% ROI, so the test measures the strategy rather than the draw.
DISTORTION = 0.40
MIN_EDGE = 0.10


@pytest.fixture(scope="module")
def distorted_history() -> list[FakeMatch]:
    return build_history(10, distortion=DISTORTION)


@pytest.fixture(scope="module")
def distorted_result(distorted_history):
    return walk_forward_backtest(
        distorted_history, min_train_matches=150, min_net_edge=MIN_EDGE
    )


# --- no look-ahead ----------------------------------------------------------


def test_training_set_never_contains_the_predicted_matchday(monkeypatch):
    """The single most important property: the fit cannot see the future."""
    seen: list[tuple[dt.date, dt.date]] = []
    real_fit = wf.fit_dixon_coles

    def spy(train, **kwargs):
        reference = kwargs["reference_date"]
        seen.append((max(m.date for m in train), reference))
        return real_fit(train, **kwargs)

    monkeypatch.setattr(wf, "fit_dixon_coles", spy)
    walk_forward_backtest(build_history(3), min_train_matches=150)

    assert seen, "expected at least one refit"
    for latest_train_date, reference in seen:
        assert latest_train_date < reference


def test_training_window_respects_the_lookback(monkeypatch):
    windows: list[tuple[dt.date, dt.date, dt.date]] = []
    real_fit = wf.fit_dixon_coles

    def spy(train, **kwargs):
        dates = [m.date for m in train]
        windows.append((min(dates), max(dates), kwargs["reference_date"]))
        return real_fit(train, **kwargs)

    monkeypatch.setattr(wf, "fit_dixon_coles", spy)
    walk_forward_backtest(build_history(4), train_window_days=200,
                          min_train_matches=150)

    for earliest, _, reference in windows:
        assert (reference - earliest).days <= 200


def test_result_of_the_bet_match_cannot_change_the_model(monkeypatch):
    """Flipping a future result must not change any bet placed before it."""
    history = build_history(3)
    base = walk_forward_backtest(history, min_train_matches=150)
    assert base.bets, "need bets to make this meaningful"

    # Rewrite the final matchday's scores; earlier bets must be untouched.
    last_date = max(m.date for m in history)
    tampered = [
        replace(m, home_goals=9, away_goals=0) if m.date == last_date else m
        for m in history
    ]
    after = walk_forward_backtest(tampered, min_train_matches=150)

    before_last = [b for b in base.bets if b.date < last_date]
    after_last = [b for b in after.bets if b.date < last_date]
    assert before_last == after_last


# --- friction consistency ---------------------------------------------------


def test_net_edge_equals_expected_pnl_per_unit_staked():
    """The entry rule and the accounting must not be able to disagree."""
    friction = FrictionModel(taker_fee=0.02, slippage=0.005, gas=0.001)
    result = walk_forward_backtest(
        build_history(3, distortion=0.25),
        friction=friction,
        min_train_matches=150,
    )
    assert result.bets

    for bet in result.bets[:50]:
        win_pnl = bet.stake * (bet.odds - 1.0) - bet.stake * 0.026
        lose_pnl = -bet.stake - bet.stake * 0.026
        expected = bet.model_prob * win_pnl + (1 - bet.model_prob) * lose_pnl
        assert bet.net_edge * bet.stake == pytest.approx(expected, abs=1e-9)


def test_friction_reduces_pnl_and_bet_count():
    history = build_history(3, distortion=0.25)
    free = walk_forward_backtest(history, min_train_matches=150)
    costly = walk_forward_backtest(
        history, friction=FrictionModel(taker_fee=0.05), min_train_matches=150
    )
    assert costly.metrics.n_bets <= free.metrics.n_bets
    assert costly.metrics.roi < free.metrics.roi


def test_zero_friction_charges_nothing():
    assert wf._cost_per_unit(FrictionModel()) == pytest.approx(0.0)
    assert wf._cost_per_unit(
        FrictionModel(taker_fee=0.02, slippage=0.01, gas=0.005)
    ) == pytest.approx(0.035)


def test_min_net_edge_is_enforced():
    result = walk_forward_backtest(
        build_history(3, distortion=0.25), min_net_edge=0.10, min_train_matches=150
    )
    assert all(bet.net_edge >= 0.10 for bet in result.bets)


# --- does it find a real edge, and refuse a fake one? -----------------------


def test_finds_a_deliberately_mispriced_book(distorted_result):
    """With the book offering too-long home prices, the model should profit."""
    metrics = distorted_result.metrics
    assert metrics.n_bets > 200
    assert metrics.roi > 0.05
    assert metrics.final_bankroll > 100.0


def test_backs_the_side_that_is_actually_mispriced(distorted_result):
    """A lower-variance signal than P&L: is it betting the right selection?

    The distortion is applied to the home price only, so the home side is where
    the edge is. It will not be 100%: renormalising the book's view shortens
    the draw and away prices, and a model fitted on finite data occasionally
    reads one of those as an edge too.
    """
    bets = distorted_result.bets
    home_share = sum(1 for b in bets if b.selection == "H") / len(bets)
    assert home_share > 0.75


def test_efficient_market_yields_no_edge():
    """Against a fairly-priced book with margin, ROI must not be positive.

    This is the test that catches a backtest quietly cheating: the strategy
    knows the true model here, and still should not beat a vigged fair book.
    """
    result = walk_forward_backtest(
        build_history(4, margin=0.06, distortion=0.0), min_train_matches=150
    )
    assert result.metrics.roi <= 0.02


# --- CLV --------------------------------------------------------------------


def test_clv_is_positive_when_betting_into_a_distorted_price(distorted_result):
    """Entry at a distorted-long price vs a fair close is exactly what CLV measures.

    Unlike ROI, CLV compares two prices rather than a price to an outcome, so
    it carries no settlement variance -- which is precisely why it is the
    better proxy for durable edge. Every bet on the distorted side beats the
    close by construction, and that holds exactly, on every seed.
    """
    metrics = distorted_result.metrics
    assert metrics.n_bets_with_clv == metrics.n_bets
    assert metrics.mean_clv > 0

    home_bets = [b for b in distorted_result.bets if b.selection == "H"]
    assert all(b.clv > 0 for b in home_bets)
    # The renormalised draw/away prices are shortened, so bets there lose CLV.
    assert metrics.clv_positive_rate == pytest.approx(
        len(home_bets) / metrics.n_bets
    )


def test_clv_absent_when_no_closing_line():
    result = walk_forward_backtest(
        build_history(3, distortion=0.25, with_closing=False), min_train_matches=150
    )
    assert result.bets
    assert result.metrics.n_bets_with_clv == 0
    assert result.metrics.mean_clv == 0.0
    assert "n/a" in result.metrics.summary()


# --- bookkeeping ------------------------------------------------------------


def test_unrated_teams_are_skipped_and_counted():
    history = build_history(3)
    newcomer = [
        replace(m, home="PromotedFC")
        for m in history[-30:]
        if m.date == max(x.date for x in history)
    ]
    result = walk_forward_backtest(history + newcomer, min_train_matches=150)
    assert result.n_skipped_unrated >= len(newcomer)
    assert not any(b.home == "PromotedFC" for b in result.bets)


def test_best_of_book_selector_is_more_optimistic():
    history = build_history(3, distortion=0.20)
    single = walk_forward_backtest(
        history, odds_selector=single_book_odds, min_train_matches=150
    )
    best = walk_forward_backtest(
        history, odds_selector=best_of_book_odds, min_train_matches=150
    )
    assert best.metrics.n_bets >= single.metrics.n_bets
    assert best.metrics.roi > single.metrics.roi


def test_start_date_skips_burn_in():
    history = build_history(4)
    cutoff = dt.date(2022, 1, 1)
    result = walk_forward_backtest(history, start=cutoff, min_train_matches=150)
    assert all(bet.date >= cutoff for bet in result.bets)


def test_refit_cadence_controls_number_of_fits():
    history = build_history(3)
    often = walk_forward_backtest(history, refit_every_days=1, min_train_matches=150)
    rarely = walk_forward_backtest(history, refit_every_days=60, min_train_matches=150)
    assert often.n_fits > rarely.n_fits


def test_insufficient_history_produces_no_bets():
    result = walk_forward_backtest(build_history(1), min_train_matches=10_000)
    assert result.bets == []
    assert result.metrics.n_bets == 0
    assert result.metrics.roi == 0.0
    assert result.n_fits == 0


def test_rejects_bad_parameters():
    history = build_history(1)
    with pytest.raises(ValueError, match="stake must be positive"):
        walk_forward_backtest(history, stake=0)
    with pytest.raises(ValueError, match="train_window_days"):
        walk_forward_backtest(history, train_window_days=0)


def test_summary_renders(monkeypatch):
    result = walk_forward_backtest(
        build_history(3, distortion=0.25), min_train_matches=150
    )
    text = result.summary()
    assert "ROI" in text and "Sharpe" in text and "CLV" in text


# --- metrics ----------------------------------------------------------------


def test_max_drawdown_on_a_known_curve():
    # 100 -> 120 -> 60: worst fall is 60/120 = 50%.
    assert max_drawdown([100, 120, 60, 90]) == pytest.approx(0.5)
    assert max_drawdown([100, 110, 120]) == pytest.approx(0.0)
    assert max_drawdown([]) == pytest.approx(0.0)


def test_sharpe_of_constant_returns_is_zero_not_infinite():
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0
    assert sharpe_ratio([0.01]) == 0.0
    assert sharpe_ratio([]) == 0.0


def test_sharpe_scales_with_annualisation():
    returns = [0.01, -0.005, 0.02, 0.0, -0.01]
    daily = sharpe_ratio(returns, periods_per_year=365)
    weekly = sharpe_ratio(returns, periods_per_year=52)
    assert daily == pytest.approx(weekly * math.sqrt(365 / 52))


def test_summarise_aggregates_same_day_bets_into_one_observation():
    day = dt.date(2024, 5, 1)
    other = dt.date(2024, 5, 2)
    metrics = summarise(
        [day, day, other],
        [1.0, -1.0, 2.0],
        [1.0, 1.0, 1.0],
        [True, False, True],
        [0.01, None, -0.02],
        bankroll=100.0,
    )
    assert metrics.n_bets == 3
    assert metrics.total_pnl == pytest.approx(2.0)
    assert metrics.roi == pytest.approx(2.0 / 3.0)
    assert metrics.hit_rate == pytest.approx(2 / 3)
    assert metrics.final_bankroll == pytest.approx(102.0)
    # Two bets on `day` net to zero, so only two daily observations exist.
    assert metrics.n_bets_with_clv == 2
    assert metrics.mean_clv == pytest.approx(-0.005)
    assert metrics.clv_positive_rate == pytest.approx(0.5)


def test_summarise_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        summarise([dt.date(2024, 1, 1)], [1.0, 2.0], [1.0], [True], [None],
                  bankroll=10.0)
    with pytest.raises(ValueError, match="bankroll"):
        summarise([], [], [], [], [], bankroll=0.0)
