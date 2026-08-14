"""Tests for pricing Polymarket's beat-earnings contracts.

The failure mode this file exists for is the base rate. It is a single number
that silently sets the answer, it is easy to measure on the wrong population,
and getting it wrong produces a confident trade rather than an error.
"""

import datetime as dt

import pytest

from sttbot.data.earnings import EarningsEvent
from sttbot.strategies.earnings_market import (
    beat_probability,
    brier_score,
    edge_against_market,
    extract_ticker,
    measure_base_rate,
    skill_vs_base_rate,
)
from sttbot.venues.prediction import FeeModel

FEES = FeeModel(quadratic_taker=0.05, quadratic_maker=-0.0125)


def history(pattern, symbol="X"):
    """`pattern` is a string of B (beat), M (miss) and T (exact tie)."""
    out = []
    for i, ch in enumerate(pattern):
        surprise = {"B": 0.05, "M": -0.05, "T": 0.0}[ch]
        out.append(EarningsEvent(
            symbol=symbol,
            fiscal_end=dt.date(2020, 1, 1) + dt.timedelta(days=90 * i),
            reported_date=dt.date(2020, 2, 1) + dt.timedelta(days=90 * i),
            reported_eps=1.0 + surprise,
            estimated_eps=1.0,
            report_time="pre-market",
        ))
    return out


# --- ticker extraction -------------------------------------------------------


def test_extract_ticker_matches_the_full_sentence_shape():
    assert extract_ticker("Will Home Depot (HD) beat quarterly earnings?") == "HD"
    assert extract_ticker("Will Lowe's (LOW) beat quarterly earnings?") == "LOW"
    assert extract_ticker("Will Deere & Co (DE) beat quarterly earnings?") == "DE"


def test_extract_ticker_rejects_other_market_families():
    """Pricing the wrong contract against a company's history is silent."""
    assert extract_ticker("Walmart (WMT) Q2 US comparable sales growth?") is None
    assert extract_ticker("Japan GDP growth in Q2 2026 (QoQ Annualized)?") is None
    assert extract_ticker("Will anyone be jailed over (ABC) disclosures?") is None
    assert extract_ticker("") is None
    assert extract_ticker("Will Something beat quarterly earnings?") is None


# --- the base rate -----------------------------------------------------------


def test_ties_count_as_misses_because_the_contract_says_beat():
    """"Beat" is not "meet or beat", and large caps meet exactly often."""
    assert measure_base_rate([history("BBBB")]) == pytest.approx(1.0)
    assert measure_base_rate([history("BBTT")]) == pytest.approx(0.5)
    assert measure_base_rate([history("TTTT")]) == pytest.approx(0.0)


def test_base_rate_pools_across_companies():
    rate = measure_base_rate([history("BBBB"), history("MMMM")])
    assert rate == pytest.approx(0.5)
    assert measure_base_rate([]) == 0.0


def test_a_contaminated_base_rate_manufactures_a_trade():
    """The bug this whole module is arranged to prevent.

    Reproduces the measured case: a large cap beating ~86% of the time, priced
    at 0.74/0.79. Under the correct prior there is no trade. Pooling in one
    unlisted micro-cap that beats ~53% drags the prior down and turns the same
    market into a confident BUY NO.
    """
    large = history("BBBBBBBBBBBBBBMMMMBB")      # 16/20 = 80%
    micro = history("BMBMBMBMBMBMBMBMBMBM")      # 10/20 = 50%

    correct = measure_base_rate([large])
    pooled = measure_base_rate([large, micro])
    assert correct > pooled

    subject = history("BBBBBBBBMMMMBB")  # the company being priced
    p_correct = beat_probability(subject, len(subject), base_rate=correct).probability
    p_pooled = beat_probability(subject, len(subject), base_rate=pooled).probability
    assert p_correct > p_pooled

    edge_correct = edge_against_market(p_correct, 0.74, 0.79, FEES)
    edge_pooled = edge_against_market(p_pooled, 0.74, 0.79, FEES)
    # The contaminated prior sees a NO edge the correct one does not.
    assert edge_pooled.no_edge > edge_correct.no_edge


# --- the forecast ------------------------------------------------------------


def test_shrinkage_stops_a_short_streak_implying_certainty():
    """Eight straight beats is not a 100% shot."""
    f = beat_probability(history("BBBBBBBB"), 8, base_rate=0.80, prior_weight=6.0)
    assert f.trailing_beat_rate == pytest.approx(1.0)
    assert f.probability < 1.0
    assert f.probability == pytest.approx((8 + 6 * 0.80) / 14)


def test_shrinkage_pulls_toward_the_prior_from_both_directions():
    up = beat_probability(history("BBBBBBBB"), 8, base_rate=0.50)
    down = beat_probability(history("MMMMMMMM"), 8, base_rate=0.50)
    assert up.probability < 1.0 and up.probability > 0.5
    assert down.probability > 0.0 and down.probability < 0.5


def test_forecast_reads_only_prior_reports():
    """Walk-forward: the report being priced must not price itself."""
    events = history("BBBBBBBB" + "M")
    at_index = beat_probability(events, 8, base_rate=0.5)
    without_the_future = beat_probability(events[:8], 8, base_rate=0.5)
    assert at_index.probability == pytest.approx(without_the_future.probability)


def test_forecast_needs_a_minimum_history():
    assert beat_probability(history("BB"), 2, min_history=4) is None
    assert beat_probability(history("BBBB"), 4, min_history=4) is not None


def test_forecast_validates_inputs():
    with pytest.raises(IndexError):
        beat_probability(history("BBBB"), 99)
    with pytest.raises(ValueError):
        beat_probability(history("BBBB"), 4, base_rate=1.5)
    with pytest.raises(ValueError):
        beat_probability(history("BBBB"), 4, prior_weight=-1)


# --- the skill control -------------------------------------------------------


def test_brier_score_rewards_confident_correctness():
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)
    assert brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        brier_score([0.5], [True, False])


def test_a_model_that_only_repeats_the_base_rate_scores_zero_skill():
    """The control: matching the free forecast is not knowing anything."""
    outcomes = [True] * 8 + [False] * 2
    assert skill_vs_base_rate([0.8] * 10, outcomes, 0.8) == pytest.approx(0.0)


def test_a_model_that_separates_outcomes_scores_positive_skill():
    outcomes = [True, True, False, False]
    informed = [0.9, 0.9, 0.1, 0.1]
    assert skill_vs_base_rate(informed, outcomes, 0.5) > 0


def test_a_worse_than_constant_model_scores_negative_skill():
    outcomes = [True, True, False, False]
    backwards = [0.1, 0.1, 0.9, 0.9]
    assert skill_vs_base_rate(backwards, outcomes, 0.5) < 0


# --- edge against a real quote ------------------------------------------------


def test_edge_uses_the_ask_to_buy_and_the_bid_to_sell():
    """Scoring against the mid finds edge wherever the model merely disagrees
    with the midpoint by less than half the spread."""
    edge = edge_against_market(0.90, 0.70, 0.80, FEES)
    assert edge.yes_edge == pytest.approx(0.90 - 0.80 - FEES.cost(0.80))
    assert edge.no_edge == pytest.approx((1 - 0.90) - 0.30 - FEES.cost(0.30))
    assert edge.best_side == "YES"


def test_a_wide_spread_kills_both_sides():
    """With a 16c spread neither side clears, whatever the model thinks."""
    edge = edge_against_market(0.74, 0.66, 0.82, FEES)
    assert edge.yes_edge < 0
    assert edge.no_edge < 0
    assert edge.best_side is None


def test_the_no_side_is_the_complement_of_the_bid():
    """Polymarket mirrors one book: the NO ask is 1 - YES bid, not a quote."""
    edge = edge_against_market(0.10, 0.70, 0.80, FEES)
    assert edge.best_side == "NO"
    assert edge.no_edge == pytest.approx(0.90 - 0.30 - FEES.cost(0.30))


def test_edge_rejects_a_broken_quote():
    with pytest.raises(ValueError):
        edge_against_market(0.5, 0.8, 0.7, FEES)   # crossed
    with pytest.raises(ValueError):
        edge_against_market(1.5, 0.4, 0.5, FEES)   # not a probability
