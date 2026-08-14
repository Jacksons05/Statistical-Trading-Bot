"""Measure post-earnings drift honestly.

PEAD is one of the most-published anomalies in finance and one of the easiest
to fake. Three things decide whether a drift number means anything, and all
three are enforced here rather than left to the caller:

**Entry timing.** A company reporting after the close cannot be traded that
day. Entering at the announcement day's close captures the overnight gap --
the market's instant repricing -- and reports it as drift. That single error
can manufacture most of the effect. Entry here is the first session strictly
on or after :meth:`EarningsEvent.first_tradeable_date`, which already accounts
for post-market reports.

**Market adjustment.** A stock that rose 4% over 20 days while the index rose
4% has drifted nowhere. Raw returns measure beta plus drift; only the excess
is the anomaly. Every return here is reported net of a benchmark over the
identical sessions.

**Look-ahead in the signal.** Handled upstream by
:func:`sttbot.data.earnings.trailing_sue`, which standardises by prior
surprises only.

What is deliberately *not* modelled: borrow cost and short availability. The
short leg of a PEAD book is the expensive half in exactly the micro-caps where
the drift is supposed to live, so a long-short number here is an upper bound.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass
from typing import Sequence

from ..data.earnings import EarningsEvent, PriceSeries, trailing_sue


@dataclass(frozen=True)
class DriftObservation:
    """One tradeable earnings event with its realised excess return."""

    symbol: str
    reported_date: object
    sue: float
    entry_gap: float  # the announcement reaction, before entry -- not earned
    raw_return: float
    benchmark_return: float

    @property
    def excess_return(self) -> float:
        return self.raw_return - self.benchmark_return


def build_observations(
    events: Sequence[EarningsEvent],
    prices: PriceSeries,
    benchmark: PriceSeries,
    *,
    horizon: int = 20,
    sue_window: int = 8,
    min_history: int = 4,
) -> list[DriftObservation]:
    """Align events to prices and measure excess return over ``horizon`` sessions.

    ``entry_gap`` records the move from the last close before the announcement
    to the entry price. It is reported but never counted as strategy return --
    it is the repricing a trader reading the release could not have captured,
    and keeping it visible is what stops it quietly leaking into the result.
    """
    out: list[DriftObservation] = []
    for i, event in enumerate(events):
        sue = trailing_sue(events, i, window=sue_window, min_history=min_history)
        if sue is None:
            continue

        entry = prices.index_on_or_after(event.first_tradeable_date())
        if entry is None or entry == 0:
            continue
        stock = prices.forward_return(entry, horizon)
        if stock is None:
            continue

        # Benchmark over the same calendar window, aligned by date so a
        # holiday in one series cannot shift the other.
        b_start = benchmark.index_on_or_after(prices.dates[entry])
        if b_start is None:
            continue
        b_end_date = prices.dates[entry + horizon]
        b_end = benchmark.index_on_or_after(b_end_date)
        if b_end is None or b_end <= b_start:
            continue
        bench = benchmark.closes[b_end] / benchmark.closes[b_start] - 1.0

        gap = prices.closes[entry] / prices.closes[entry - 1] - 1.0
        out.append(
            DriftObservation(
                symbol=event.symbol,
                reported_date=event.reported_date,
                sue=sue,
                entry_gap=gap,
                raw_return=stock,
                benchmark_return=bench,
            )
        )
    return out


@dataclass(frozen=True)
class BucketResult:
    label: str
    n: int
    mean_excess: float
    t_stat: float
    hit_rate: float
    mean_gap: float


def _t_stat(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    sd = st.stdev(values)
    if sd <= 0:
        return float("nan")
    return st.mean(values) / (sd / math.sqrt(len(values)))


def bucket_by_sue(
    observations: Sequence[DriftObservation], edges: Sequence[float] = (-1.5, -0.5, 0.5, 1.5)
) -> list[BucketResult]:
    """Excess return by surprise bucket.

    PEAD predicts a monotone rise across buckets: the strongest positive
    surprises drift up most, the strongest negative ones down most. A single
    significant bucket with no monotonicity is noise wearing a result's hat.
    """
    labels = []
    bounds = [-math.inf, *edges, math.inf]
    for lo, hi in zip(bounds, bounds[1:]):
        lo_s = "-inf" if lo == -math.inf else f"{lo:+.1f}"
        hi_s = "+inf" if hi == math.inf else f"{hi:+.1f}"
        labels.append(f"SUE {lo_s}..{hi_s}")

    results = []
    for (lo, hi), label in zip(zip(bounds, bounds[1:]), labels):
        group = [o for o in observations if lo <= o.sue < hi]
        if not group:
            continue
        excess = [o.excess_return for o in group]
        results.append(
            BucketResult(
                label=label,
                n=len(group),
                mean_excess=st.mean(excess),
                t_stat=_t_stat(excess),
                hit_rate=sum(1 for e in excess if e > 0) / len(excess),
                mean_gap=st.mean(o.entry_gap for o in group),
            )
        )
    return results


def long_short(
    observations: Sequence[DriftObservation], threshold: float = 1.5
) -> tuple[int, int, float, float]:
    """(n_long, n_short, mean excess of the spread, t-stat).

    Long high-SUE, short low-SUE. The short leg's borrow cost is not modelled,
    so treat this as the optimistic bound.
    """
    longs = [o.excess_return for o in observations if o.sue >= threshold]
    shorts = [o.excess_return for o in observations if o.sue <= -threshold]
    if not longs or not shorts:
        return len(longs), len(shorts), float("nan"), float("nan")
    combined = longs + [-s for s in shorts]
    return len(longs), len(shorts), st.mean(combined), _t_stat(combined)
