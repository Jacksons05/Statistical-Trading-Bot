"""Closing-line value, measured against the baseline it has to beat.

Positive CLV is the best available proxy for durable edge, but it is trivially
easy to manufacture and therefore easy to misread. If entry prices are
best-of-book maxima across ~17 books and the closing line is a single book,
then *any* selection beats the close most of the time -- the comparison is
between the best of seventeen prices and one price, and the model never enters
into it.

Measured on 2018/19-2024/25 English and Scottish league data, random selections
at best-of-book entry returned mean CLV of +0.0044 to +0.0067 with a 54-63%
positive rate. A strategy reporting "+0.007 mean CLV, 64% positive" has
therefore demonstrated nothing at all.

So a CLV number is only meaningful next to the baseline for the same matches
and the same entry-price rule. :func:`clv_baseline` computes that baseline and
:func:`clv_skill` reports the excess, which is the part attributable to
selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..strategies.dixon_coles import closing_line_value
from .walk_forward import SELECTIONS, BacktestMatch, Bet, OddsSelector, single_book_odds


@dataclass(frozen=True)
class ClvSummary:
    n: int
    mean_clv: float
    positive_rate: float


def _universe(
    matches: Sequence[BacktestMatch], odds_selector: OddsSelector
) -> tuple[np.ndarray, np.ndarray]:
    """Every priceable (match, selection) pair as (implied_prob, clv) arrays."""
    probs: list[float] = []
    values: list[float] = []
    for match in matches:
        prices = odds_selector(match)
        closes = (match.close_home, match.close_draw, match.close_away)
        for entry, close in zip(prices, closes):
            if entry and entry > 1.0 and close and close > 1.0:
                probs.append(1.0 / entry)
                values.append(closing_line_value(entry, close))
    return np.asarray(probs), np.asarray(values)


def clv_baseline(
    matches: Sequence[BacktestMatch],
    *,
    odds_selector: OddsSelector = single_book_odds,
    seed: int = 0,
    picks_per_match: int = 1,
) -> ClvSummary:
    """CLV obtained by choosing selections uniformly at random.

    This is the null hypothesis for a CLV claim: whatever this returns is
    available with no model, no data and no skill.
    """
    rng = random.Random(seed)
    values: list[float] = []
    for match in matches:
        prices = odds_selector(match)
        closes = (match.close_home, match.close_draw, match.close_away)
        for _ in range(picks_per_match):
            i = rng.randrange(len(SELECTIONS))
            entry, close = prices[i], closes[i]
            if entry and entry > 1.0 and close and close > 1.0:
                values.append(closing_line_value(entry, close))

    if not values:
        return ClvSummary(0, 0.0, 0.0)
    return ClvSummary(
        n=len(values),
        mean_clv=float(np.mean(values)),
        positive_rate=sum(1 for v in values if v > 0) / len(values),
    )


def clv_odds_matched_baseline(
    matches: Sequence[BacktestMatch],
    bets: Sequence[Bet],
    *,
    odds_selector: OddsSelector = single_book_odds,
    n_buckets: int = 10,
) -> ClvSummary:
    """CLV of an average selection *priced like the ones the strategy chose*.

    The uniform baseline in :func:`clv_baseline` controls for the entry-price
    rule but not for which prices the strategy likes. A model that mostly backs
    longshots would inherit whatever CLV longshots carry as a group, and that
    would show up as skill under the uniform baseline while reflecting nothing
    but a preference for a region of the price range.

    This bucket-matches instead: each bet is compared against the mean CLV of
    every selection in the same implied-probability bucket, so what survives is
    the ability to pick *which* selection at a given price, which is the part
    that is actually forecasting.

    Buckets are quantiles of the price distribution, so they stay populated
    even though real odds are heavily skewed toward favourites.
    """
    matched = _match_buckets(matches, bets, odds_selector, n_buckets)
    if matched is None:
        return ClvSummary(0, 0.0, 0.0)

    means, rates, _ = matched
    return ClvSummary(
        n=int(means.size),
        mean_clv=float(means.mean()),
        positive_rate=float(rates.mean()),
    )


def _match_buckets(
    matches: Sequence[BacktestMatch],
    bets: Sequence[Bet],
    odds_selector: OddsSelector,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-bet (bucket mean CLV, bucket positive rate, own CLV).

    Returns ``None`` when the sample cannot support bucketing at all.
    """
    probs, values = _universe(matches, odds_selector)
    own = np.array([b.clv for b in bets if b.clv is not None])
    if probs.size == 0 or own.size == 0:
        return None

    n_buckets = max(1, min(n_buckets, probs.size))
    # Deduplicate: a heavily-tied price distribution collapses quantile edges.
    edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, n_buckets + 1)))
    if edges.size < 2:
        # Every price identical, so price-matching is vacuous: the whole
        # universe is one bucket. Comparing against its mean is still the right
        # answer, and keeps this consistent with clv_excess_per_bet.
        return (
            np.full(own.size, values.mean()),
            np.full(own.size, float((values > 0).mean())),
            own,
        )

    def bucket_of(x: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)

    membership = bucket_of(probs)
    means = np.full(edges.size - 1, np.nan)
    rates = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        in_bucket = values[membership == b]
        if in_bucket.size:
            means[b] = in_bucket.mean()
            rates[b] = (in_bucket > 0).mean()

    bet_bucket = bucket_of(
        np.array([1.0 / b.odds for b in bets if b.clv is not None])
    )
    matched_means = means[bet_bucket]
    matched_rates = rates[bet_bucket]
    usable = ~np.isnan(matched_means)
    if not usable.any():
        return None
    return matched_means[usable], matched_rates[usable], own[usable]


def clv_excess_per_bet(
    matches: Sequence[BacktestMatch],
    bets: Sequence[Bet],
    *,
    odds_selector: OddsSelector = single_book_odds,
    n_buckets: int = 10,
) -> np.ndarray:
    """Each bet's CLV minus the mean CLV of same-priced selections.

    The paired form of :func:`clv_odds_matched_baseline`. Because it returns
    one number per bet rather than an aggregate, the mean carries a standard
    error, which is what makes "is this skill real?" answerable rather than
    merely arguable.
    """
    matched = _match_buckets(matches, bets, odds_selector, n_buckets)
    if matched is None:
        return np.array([])
    means, _, own = matched
    return own - means


def clv_skill(strategy_mean_clv: float, baseline: ClvSummary) -> float:
    """Excess CLV over the random-selection baseline.

    This, not raw CLV, is the number that reflects selection skill. It can
    easily be negative for a strategy showing healthy-looking positive CLV.
    """
    return strategy_mean_clv - baseline.mean_clv
