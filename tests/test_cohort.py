"""Tests for birth-cohort return measurement.

This module exists to replace an assumed tail index with a measured one, so
the tests focus on the things that would quietly corrupt that measurement:
anchoring entry to the wrong price, letting unmeasurable pools count as
outcomes, and estimating a tail index from too few points.
"""


import pytest

from sttbot.backtest.cohort import (
    detect_clone_groups,
    CohortMember,
    Outcome,
    best_take_profit,
    expected_value_with_take_profit,
    hill_alpha,
    measure_outcomes,
    summarise,
)


def candle(ts, o, h, l, c, v=100.0):
    return (ts, o, h, l, c, v)


def member(pool="P1", price=1.0):
    return CohortMember(
        pool=pool, base_token="T", created_at="2026-07-30T03:00:00Z",
        price_at_capture=price, reserve_at_capture=5_000.0,
    )


# --- outcome resolution -----------------------------------------------------


def test_entry_is_the_first_candle_open_not_the_capture_price():
    """Capture happens seconds after birth, so its price can already be stale.

    Anchoring entry to the recorded capture price while exiting on the candle
    series measures the two ends with different rulers.
    """
    series = [candle(0, 2.0, 4.0, 1.0, 3.0), candle(300, 3.0, 5.0, 2.0, 4.0)]
    (outcome,) = measure_outcomes([member(price=99.0)], lambda p: series).outcomes
    assert outcome.entry_price == 2.0
    assert outcome.max_price == 5.0
    assert outcome.final_price == 4.0
    assert outcome.max_multiple == pytest.approx(2.5)
    assert outcome.final_multiple == pytest.approx(2.0)


def test_fetch_failure_is_never_confused_with_never_traded():
    """The bug that deleted a third of a cohort and flattered the survivors.

    None means the request failed; [] means the API answered and the pool
    genuinely never traded. Only the second is data. Merging them made 164 of
    300 pools look like tokens that never traded, when re-querying them
    politely returned full history for every one.
    """
    failed = measure_outcomes([member("A")], lambda p: None)
    assert failed.outcomes == ()
    assert failed.fetch_failed == ("A",)
    assert failed.no_candles == ()

    never = measure_outcomes([member("B")], lambda p: [])
    assert never.no_candles == ("B",)
    assert never.fetch_failed == ()


def test_report_accounts_for_every_member():
    series = {"A": [candle(0, 1.0, 3.0, 1.0, 2.0)], "B": [], "C": None,
              "D": [candle(0, 0.0, 1.0, 0.0, 0.5)]}
    report = measure_outcomes([member(k) for k in "ABCD"], lambda p: series[p])
    assert len(report.outcomes) == 1
    assert report.no_candles == ("B",)
    assert report.fetch_failed == ("C",)
    assert report.bad_entry == ("D",)
    assert report.resolution_rate == pytest.approx(0.25)
    assert "1 resolved" in report.summary()


def test_skipping_the_birth_candle_removes_the_initialisation_sweep():
    """70.6% of a real cohort had its maximum high inside candle zero."""
    series = [candle(0, 1e-9, 1e-6, 1e-10, 2e-9), candle(60, 2e-9, 3e-9, 1e-9, 2.5e-9)]
    raw = measure_outcomes([member()], lambda p: series).outcomes[0]
    assert raw.max_multiple == pytest.approx(1000.0)  # the bootstrap sweep

    clean = measure_outcomes(
        [member()], lambda p: series, skip_first_candle=True
    ).outcomes[0]
    assert clean.max_multiple == pytest.approx(1.5)


def test_a_peak_on_no_volume_is_not_a_price():
    """The largest peak in a real cohort was 152,363x on $0.00074 of volume."""
    series = [
        candle(0, 1.0, 1.1, 0.9, 1.0, v=500.0),
        candle(60, 1.0, 1000.0, 1.0, 1.0, v=0.0007),  # dust tick
    ]
    raw = measure_outcomes([member()], lambda p: series).outcomes[0]
    assert raw.max_multiple == pytest.approx(1000.0)

    gated = measure_outcomes(
        [member()], lambda p: series, min_peak_volume=100.0
    ).outcomes[0]
    assert gated.max_multiple == pytest.approx(1.1)


def test_an_unconfirmed_wick_is_not_a_level_you_could_sell():
    series = [
        candle(0, 1.0, 1.0, 1.0, 1.0, v=500.0),
        candle(60, 1.0, 50.0, 1.0, 1.0, v=500.0),   # spike, closes back at 1
        candle(120, 1.0, 3.0, 1.0, 2.8, v=500.0),   # real move, holds
    ]
    raw = measure_outcomes([member()], lambda p: series).outcomes[0]
    assert raw.max_multiple == pytest.approx(50.0)

    confirmed = measure_outcomes(
        [member()], lambda p: series, confirm_peak_fraction=0.5
    ).outcomes[0]
    # The 50x wick is never closed near; the 3x high is.
    assert confirmed.max_multiple == pytest.approx(3.0)


def test_death_is_a_collapse_not_merely_a_loss():
    alive = Outcome("P", entry_price=1.0, max_price=1.2, final_price=0.5, candles=3)
    dead = Outcome("P", entry_price=1.0, max_price=1.2, final_price=0.001, candles=3)
    assert not alive.died
    assert dead.died


def test_measure_outcomes_handles_many_members():
    series = {"A": [candle(0, 1.0, 3.0, 1.0, 2.0)], "B": [candle(0, 1.0, 1.0, 0.0, 0.0)]}
    report = measure_outcomes([member("A"), member("B")], lambda p: series[p])
    assert {o.pool for o in report.outcomes} == {"A", "B"}


# --- tail index -------------------------------------------------------------


def test_hill_alpha_recovers_a_known_pareto_tail():
    """Sanity: draw from a known alpha and check it comes back."""
    import numpy as np

    rng = np.random.default_rng(0)
    true_alpha = 1.5
    sample = (1.0 + rng.pareto(true_alpha, 20_000))
    estimate = hill_alpha(sample, tail_fraction=0.05)
    assert estimate is not None
    assert estimate == pytest.approx(true_alpha, rel=0.15)


def test_fatter_tails_estimate_a_lower_alpha():
    import numpy as np

    rng = np.random.default_rng(1)
    fat = hill_alpha(1.0 + rng.pareto(0.8, 20_000), tail_fraction=0.05)
    thin = hill_alpha(1.0 + rng.pareto(4.0, 20_000), tail_fraction=0.05)
    assert fat is not None and thin is not None
    assert fat < thin


def test_hill_alpha_refuses_a_tail_too_short_to_mean_anything():
    """A tail index from three points is a number with no information in it."""
    assert hill_alpha([1.0, 2.0, 3.0]) is None
    assert hill_alpha([1.0] * 20) is None  # no dispersion in the tail
    assert hill_alpha([]) is None


def test_hill_alpha_validates_tail_fraction():
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError):
            hill_alpha([1.0, 2.0], tail_fraction=bad)


# --- summary ----------------------------------------------------------------


def outcomes_from(multiples, final=None):
    final = final if final is not None else multiples
    return [
        Outcome(f"P{i}", entry_price=1.0, max_price=m, final_price=f, candles=5)
        for i, (m, f) in enumerate(zip(multiples, final))
    ]


def test_summary_separates_peak_from_achievable():
    """Most tokens print a high they never close at."""
    stats = summarise(outcomes_from([5.0, 4.0, 3.0], final=[0.1, 0.2, 0.05]))
    assert stats.median_max_multiple == pytest.approx(4.0)
    assert stats.median_final_multiple == pytest.approx(0.1)  # median of .05/.1/.2
    assert stats.survival_rate == pytest.approx(1.0)  # 0.1, 0.2, 0.05 > 0.01


def test_summary_counts_collapses_as_deaths():
    stats = summarise(outcomes_from([2.0] * 4, final=[0.001, 0.002, 1.5, 0.0005]))
    assert stats.survival_rate == pytest.approx(0.25)


def test_empty_cohort_summarises_to_zeros_not_an_error():
    stats = summarise([])
    assert stats.n == 0 and stats.tail_alpha is None
    assert "n=0" in stats.summary()


# --- take profit ------------------------------------------------------------


def test_take_profit_exits_winners_at_the_trigger():
    # One token peaks at 10x, three die. TP at 5x catches the winner.
    outcomes = outcomes_from([10.0, 1.1, 1.0, 0.9], final=[0.01, 0.01, 0.01, 0.01])
    ev = expected_value_with_take_profit(outcomes, 5.0, cost_fraction=0.0)
    # (5 - 1) + 3 * (0.01 - 1), all over 4
    assert ev == pytest.approx((4.0 + 3 * -0.99) / 4)


def test_costs_and_exit_slippage_reduce_expected_value():
    outcomes = outcomes_from([10.0, 1.0], final=[0.01, 0.01])
    clean = expected_value_with_take_profit(outcomes, 5.0, cost_fraction=0.0)
    costly = expected_value_with_take_profit(outcomes, 5.0, cost_fraction=0.05)
    slipped = expected_value_with_take_profit(
        outcomes, 5.0, cost_fraction=0.0, slippage_on_exit=0.10
    )
    assert costly < clean
    assert slipped < clean


def test_a_take_profit_above_every_peak_captures_nothing():
    outcomes = outcomes_from([3.0, 2.0], final=[0.01, 0.01])
    ev = expected_value_with_take_profit(outcomes, 100.0, cost_fraction=0.0)
    assert ev == pytest.approx(0.01 - 1.0)


def test_best_take_profit_finds_the_optimum_in_sample():
    outcomes = outcomes_from([50.0, 1.0, 1.0, 1.0], final=[0.01] * 4)
    tp, ev = best_take_profit(outcomes, cost_fraction=0.0, candidates=(2, 10, 25, 100))
    assert tp == 25  # 50x peak, so 25 is the largest trigger that fills
    assert ev > 0


def test_best_take_profit_can_report_that_nothing_worked():
    """A cohort where every rule loses should say so, not pick a least-bad."""
    outcomes = outcomes_from([1.2] * 20, final=[0.01] * 20)
    tp, ev = best_take_profit(outcomes, cost_fraction=0.05)
    assert ev < 0


def test_take_profit_validates_input():
    with pytest.raises(ValueError):
        expected_value_with_take_profit(outcomes_from([2.0]), 0.0, cost_fraction=0.0)
    assert expected_value_with_take_profit([], 2.0, cost_fraction=0.0) == 0.0


# --- wash / clone detection -------------------------------------------------


def test_clone_families_are_detected_by_near_identical_volume():
    """Found live: three XAU/SOL mints created within 18 seconds, lifetime
    volumes $6529.84 / $6529.86 / $6529.98 -- four significant figures apart.

    Counting those as three independent winners triples a hit rate and breaks
    the independence every confidence interval here assumes.
    """
    members = [member("X1"), member("X2"), member("X3"), member("OTHER")]
    volumes = {"X1": 6529.84, "X2": 6529.86, "X3": 6529.98, "OTHER": 500.0}
    names = {"X1": "XAU/SOL", "X2": "XAU/SOL", "X3": "XAU/SOL", "OTHER": "REAL/SOL"}

    groups = detect_clone_groups(
        members, volumes, name_of=lambda m: names[m.pool]
    )
    assert len(groups) == 1
    assert set(groups[0]) == {"X1", "X2", "X3"}


def test_same_name_with_genuinely_different_volume_is_not_a_clone():
    members = [member("A"), member("B")]
    volumes = {"A": 100.0, "B": 8000.0}
    groups = detect_clone_groups(
        members, volumes, name_of=lambda m: "SAME/SOL"
    )
    assert groups == []


def test_distinct_names_are_never_grouped():
    members = [member("A"), member("B")]
    volumes = {"A": 1000.0, "B": 1000.0}
    names = {"A": "ONE/SOL", "B": "TWO/SOL"}
    assert detect_clone_groups(members, volumes, name_of=lambda m: names[m.pool]) == []


# --- liquidity death --------------------------------------------------------


def test_price_survival_is_meaningless_when_trading_stops():
    """Measured on a real cohort at 3.6 days: 98.3% "survived" on price while
    94.8% had not traded in 24 hours.

    A token whose market disappears stops producing candles, so its last close
    freezes at whatever it printed on the way out. The price series cannot show
    you the absence of trading.
    """
    now = 1_000_000
    frozen = Outcome("P", entry_price=1.0, max_price=2.0, final_price=1.0,
                     candles=1, last_trade_ts=now - 80 * 3600)
    assert not frozen.died          # price says it is fine
    assert frozen.stopped_trading(now)  # liquidity says it is gone


def test_a_still_trading_pool_is_not_counted_as_dead():
    now = 1_000_000
    live = Outcome("P", entry_price=1.0, max_price=2.0, final_price=1.5,
                   candles=40, last_trade_ts=now - 1800)
    assert not live.stopped_trading(now)
    assert live.hours_since_last_trade(now) == pytest.approx(0.5)


def test_silence_threshold_is_tunable():
    now = 1_000_000
    o = Outcome("P", 1.0, 2.0, 1.0, 1, last_trade_ts=now - 10 * 3600)
    assert not o.stopped_trading(now, max_silence_hours=24)
    assert o.stopped_trading(now, max_silence_hours=6)


def test_last_trade_timestamp_is_carried_through_measurement():
    series = [candle(0, 1.0, 2.0, 1.0, 1.5), candle(3600, 1.5, 1.6, 1.4, 1.5)]
    (outcome,) = measure_outcomes([member()], lambda p: series).outcomes
    assert outcome.last_trade_ts == 3600
