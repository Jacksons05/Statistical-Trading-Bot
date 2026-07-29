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
from .walk_forward import SELECTIONS, BacktestMatch, OddsSelector, single_book_odds


@dataclass(frozen=True)
class ClvSummary:
    n: int
    mean_clv: float
    positive_rate: float


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


def clv_skill(strategy_mean_clv: float, baseline: ClvSummary) -> float:
    """Excess CLV over the random-selection baseline.

    This, not raw CLV, is the number that reflects selection skill. It can
    easily be negative for a strategy showing healthy-looking positive CLV.
    """
    return strategy_mean_clv - baseline.mean_clv
