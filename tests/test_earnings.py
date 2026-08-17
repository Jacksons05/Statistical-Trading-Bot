"""Tests for the earnings pipeline and the PEAD event study.

PEAD is easy to fake, and every way of faking it is a timing error. These tests
target the timing directly: a post-market report must not be tradeable the same
day, the SUE denominator must not see the future, and drift must be measured
net of the market.
"""

import datetime as dt

import pytest

from sttbot.backtest.event_study import (
    build_observations,
    bucket_by_sue,
    long_short,
)
from sttbot.data.earnings import (
    POST_MARKET,
    PRE_MARKET,
    EarningsEvent,
    PriceSeries,
    parse_earnings,
    parse_yahoo_chart,
    trailing_sue,
)

D = dt.date


def event(reported, actual, estimate, when=PRE_MARKET, symbol="X"):
    return EarningsEvent(
        symbol=symbol,
        fiscal_end=reported - dt.timedelta(days=20),
        reported_date=reported,
        reported_eps=actual,
        estimated_eps=estimate,
        report_time=when,
    )


def series(start, closes, symbol="X"):
    """Consecutive daily sessions from `start`, skipping weekends."""
    dates, d = [], start
    while len(dates) < len(closes):
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    return PriceSeries(symbol, tuple(dates), tuple(closes))


# --- parsing ----------------------------------------------------------------


def test_parse_earnings_sorts_oldest_first_and_keeps_report_time():
    payload = {
        "symbol": "IBM",
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-04-22",
             "reportedEPS": "1.91", "estimatedEPS": "1.81", "reportTime": "post-market"},
            {"fiscalDateEnding": "2025-12-31", "reportedDate": "2026-01-28",
             "reportedEPS": "4.52", "estimatedEPS": "4.29", "reportTime": "pre-market"},
        ],
    }
    events = parse_earnings(payload)
    assert [e.reported_date for e in events] == [D(2026, 1, 28), D(2026, 4, 22)]
    assert events[0].report_time == PRE_MARKET
    assert events[1].surprise == pytest.approx(0.10)


def test_rows_without_an_estimate_are_dropped():
    """No estimate means no surprise; there is nothing to standardise."""
    payload = {"symbol": "X", "quarterlyEarnings": [
        {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-04-22",
         "reportedEPS": "1.91", "estimatedEPS": "None"},
        {"fiscalDateEnding": "2025-12-31", "reportedDate": "2026-01-28",
         "reportedEPS": "4.52", "estimatedEPS": "4.29"},
    ]}
    assert len(parse_earnings(payload)) == 1


def test_parse_earnings_tolerates_junk():
    assert parse_earnings(None) == []
    assert parse_earnings({"quarterlyEarnings": None}) == []
    assert parse_earnings({"quarterlyEarnings": ["nonsense"]}) == []


def test_parse_yahoo_prefers_adjusted_closes():
    """A split inside the window would otherwise read as a return."""
    payload = {"chart": {"result": [{
        "meta": {"symbol": "X"},
        "timestamp": [1735689600, 1735776000],
        "indicators": {
            "quote": [{"close": [200.0, 202.0]}],
            "adjclose": [{"adjclose": [100.0, 101.0]}],
        },
    }]}}
    s = parse_yahoo_chart(payload)
    assert s.closes == (100.0, 101.0)


def test_parse_yahoo_falls_back_to_close_and_drops_nulls():
    payload = {"chart": {"result": [{
        "meta": {"symbol": "X"},
        "timestamp": [1735689600, 1735776000, 1735862400],
        "indicators": {"quote": [{"close": [100.0, None, 102.0]}]},
    }]}}
    s = parse_yahoo_chart(payload)
    assert s.closes == (100.0, 102.0)
    assert parse_yahoo_chart({"chart": {"result": []}}) is None
    assert parse_yahoo_chart(None) is None


# --- the timing rule that decides the result --------------------------------


def test_post_market_reports_are_not_tradeable_until_the_next_day():
    """The single most important line in this pipeline.

    Entering at the announcement day's close captures the overnight repricing
    and reports it as drift.
    """
    assert event(D(2026, 4, 22), 2, 1, PRE_MARKET).first_tradeable_date() == D(2026, 4, 22)
    assert event(D(2026, 4, 22), 2, 1, POST_MARKET).first_tradeable_date() == D(2026, 4, 23)


def test_entry_skips_weekends_and_missing_sessions():
    """A Friday post-market report is not tradeable until Monday."""
    prices = series(D(2026, 4, 20), [100.0] * 10)  # Mon 20 Apr onwards
    friday = event(D(2026, 4, 24), 2, 1, POST_MARKET)
    idx = prices.index_on_or_after(friday.first_tradeable_date())
    assert prices.dates[idx] == D(2026, 4, 27)  # the following Monday


def test_index_on_or_after_returns_none_past_the_series():
    prices = series(D(2026, 4, 20), [100.0] * 3)
    assert prices.index_on_or_after(D(2027, 1, 1)) is None


# --- SUE must not see the future --------------------------------------------


def test_trailing_sue_uses_only_prior_surprises():
    """Standardising by the full sample tells a 2015 trade about 2024's vol."""
    # Four quiet quarters (surprise 0.01), then one big one, then a wild one.
    events = [event(D(2024, 1, 1) + dt.timedelta(days=90 * i), 1.01, 1.00)
              for i in range(4)]
    events.append(event(D(2025, 1, 1), 1.10, 1.00))   # the event under test
    events.append(event(D(2025, 4, 1), 5.00, 1.00))   # a huge later surprise

    sue = trailing_sue(events, 4, window=8, min_history=4)
    # Denominator comes from the four quiet quarters only. Their surprises are
    # identical, so dispersion is zero and SUE is undefined rather than infinite.
    assert sue is None

    varied = [
        event(D(2024, 1, 1), 1.02, 1.00),
        event(D(2024, 4, 1), 0.98, 1.00),
        event(D(2024, 7, 1), 1.03, 1.00),
        event(D(2024, 10, 1), 0.99, 1.00),
        event(D(2025, 1, 1), 1.10, 1.00),
        event(D(2025, 4, 1), 9.00, 1.00),  # must not affect the previous SUE
    ]
    with_future = trailing_sue(varied, 4)
    del varied[5]
    without_future = trailing_sue(varied, 4)
    assert with_future == pytest.approx(without_future)


def test_trailing_sue_needs_enough_history():
    events = [event(D(2024, 1, 1) + dt.timedelta(days=90 * i), 1.0 + i * 0.01, 1.0)
              for i in range(6)]
    assert trailing_sue(events, 0) is None          # nothing prior
    assert trailing_sue(events, 3, min_history=4) is None
    assert trailing_sue(events, 4, min_history=4) is not None


def test_trailing_sue_rejects_a_bad_index():
    with pytest.raises(IndexError):
        trailing_sue([event(D(2024, 1, 1), 1, 1)], 5)


def test_sue_sign_and_scale():
    events = [
        event(D(2024, 1, 1), 1.02, 1.00),
        event(D(2024, 4, 1), 0.98, 1.00),
        event(D(2024, 7, 1), 1.02, 1.00),
        event(D(2024, 10, 1), 0.98, 1.00),
        event(D(2025, 1, 1), 1.04, 1.00),
    ]
    sue = trailing_sue(events, 4)
    assert sue > 0                       # a beat is positive
    # Prior surprises are +-0.02, sd ~0.023; a +0.04 surprise is ~1.7 sd.
    assert 1.0 < sue < 2.5


# --- drift measurement -------------------------------------------------------


def base_events():
    return [
        event(D(2024, 1, 1), 1.02, 1.00),
        event(D(2024, 4, 1), 0.98, 1.00),
        event(D(2024, 7, 1), 1.02, 1.00),
        event(D(2024, 10, 1), 0.98, 1.00),
        event(D(2025, 1, 6), 1.06, 1.00, POST_MARKET),
    ]


def test_market_adjustment_removes_beta():
    """A stock that rose exactly with the index has drifted nowhere."""
    stock = series(D(2025, 1, 2), [100.0] * 2 + [110.0] * 30)
    bench = series(D(2025, 1, 2), [100.0] * 2 + [110.0] * 30, symbol="SPY")
    obs = build_observations(base_events(), stock, bench, horizon=5)
    assert len(obs) == 1
    assert obs[0].raw_return == pytest.approx(0.0)
    assert obs[0].excess_return == pytest.approx(0.0)


def test_excess_return_isolates_the_stock_specific_move():
    # Sessions: 0=Thu 2 Jan, 1=Fri 3, 2=Mon 6 (report, post-market),
    # 3=Tue 7 (entry at 100), ... 7=Mon 13 (exit at 110). The drift happens
    # after entry, so it is the trader's to earn.
    stock = series(D(2025, 1, 2), [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0])
    bench = series(D(2025, 1, 2), [100.0] * 8, symbol="SPY")
    obs = build_observations(base_events(), stock, bench, horizon=4)
    assert obs[0].entry_gap == pytest.approx(0.0)
    assert obs[0].excess_return == pytest.approx(0.10)


def test_the_announcement_gap_is_reported_but_not_earned():
    """The overnight repricing is exactly what a naive study counts as drift."""
    # Session 0 = 2 Jan, 1 = 3 Jan, 2 = 6 Jan (report, post-market),
    # 3 = 7 Jan (entry, gaps up 20%), flat thereafter.
    stock = series(D(2025, 1, 2), [100.0, 100.0, 100.0, 120.0, 120.0, 120.0])
    bench = series(D(2025, 1, 2), [100.0] * 6, symbol="SPY")
    obs = build_observations(base_events(), stock, bench, horizon=2)
    assert len(obs) == 1
    assert obs[0].entry_gap == pytest.approx(0.20)   # visible
    assert obs[0].raw_return == pytest.approx(0.0)   # and not counted


def test_events_without_enough_forward_prices_are_dropped():
    stock = series(D(2025, 1, 2), [100.0] * 4)
    bench = series(D(2025, 1, 2), [100.0] * 4, symbol="SPY")
    assert build_observations(base_events(), stock, bench, horizon=20) == []


# --- aggregation -------------------------------------------------------------


def obs_with(sue, excess):
    from sttbot.backtest.event_study import DriftObservation
    return DriftObservation("X", D(2025, 1, 1), sue, 0.0, excess, 0.0)


def test_buckets_are_ordered_and_counted():
    observations = [obs_with(-2.0, -0.05), obs_with(-1.0, -0.01),
                    obs_with(0.0, 0.0), obs_with(1.0, 0.01), obs_with(2.0, 0.05)]
    buckets = bucket_by_sue(observations)
    assert len(buckets) == 5
    assert buckets[0].n == 1 and buckets[0].mean_excess == pytest.approx(-0.05)
    assert buckets[-1].mean_excess == pytest.approx(0.05)


def test_long_short_shorts_the_negative_leg():
    observations = [obs_with(2.0, 0.04), obs_with(-2.0, -0.06)]
    n_long, n_short, mean, _ = long_short(observations, threshold=1.5)
    assert (n_long, n_short) == (1, 1)
    assert mean == pytest.approx(0.05)  # (0.04 + 0.06) / 2


def test_long_short_needs_both_legs():
    n_long, n_short, mean, t = long_short([obs_with(2.0, 0.04)], threshold=1.5)
    assert (n_long, n_short) == (1, 0)
    assert mean != mean  # NaN rather than a one-legged "spread"
