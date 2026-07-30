"""Measure the realised return distribution of a token birth cohort.

The breadth strategy's answer is decided entirely by the fatness of the payoff
tail, and that parameter was assumed rather than measured. This measures it.

The method is forward, not retrospective, and that is not a stylistic choice.
Solana creates roughly 1,600 pools an hour against a 200-row discovery feed, so
by the time you ask about yesterday the cohort is unreachable. Worse, any
cohort assembled later from whatever is still indexed has already had its
failures removed -- exactly the survivorship bias that makes this asset class
look survivable. So a cohort is captured *at birth* and measured forward.

Two return measures are reported and they mean different things:

* ``final_multiple`` is buy-at-capture, hold-to-horizon. Achievable.
* ``max_multiple`` uses candle highs, so it is what a perfectly-timed exit
  would have made. **Not achievable** -- nobody sells the high -- but it bounds
  what any take-profit rule could extract, and if even the bound falls short of
  breakeven then no exit rule saves the strategy.

The tail index is estimated with the Hill estimator, which is the standard
tool for exactly this question and reports the alpha the Monte Carlo needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

# Below this a pool is treated as dead rather than merely cheap: the price has
# collapsed so far that the position is unexitable in practice.
DEAD_MULTIPLE = 0.01

Candles = Sequence[tuple[int, float, float, float, float, float]]
OhlcvFn = Callable[[str], Candles]


@dataclass(frozen=True)
class CohortMember:
    """A pool recorded at birth, before its outcome is known."""

    pool: str
    base_token: str
    created_at: str
    price_at_capture: float | None
    reserve_at_capture: float | None


@dataclass(frozen=True)
class Outcome:
    """What actually happened to one cohort member."""

    pool: str
    entry_price: float
    max_price: float
    final_price: float
    candles: int

    @property
    def max_multiple(self) -> float:
        return self.max_price / self.entry_price if self.entry_price > 0 else 0.0

    @property
    def final_multiple(self) -> float:
        return self.final_price / self.entry_price if self.entry_price > 0 else 0.0

    @property
    def died(self) -> bool:
        return self.final_multiple <= DEAD_MULTIPLE


@dataclass(frozen=True)
class MeasurementReport:
    """Outcomes plus an account of everything that did not resolve.

    The three buckets are kept apart because they mean different things and
    only one of them is data. A pool that could not be fetched is a hole in the
    pipeline; a pool that never traded is a result. Merging them silently
    deleted a third of one cohort and made the survivors look far better than
    the population.
    """

    outcomes: tuple[Outcome, ...]
    no_candles: tuple[str, ...]  # API answered, pool genuinely never traded
    fetch_failed: tuple[str, ...]  # could not fetch: rate limit or transport
    bad_entry: tuple[str, ...]  # candles present but entry price unusable

    @property
    def resolution_rate(self) -> float:
        total = (
            len(self.outcomes)
            + len(self.no_candles)
            + len(self.fetch_failed)
            + len(self.bad_entry)
        )
        return len(self.outcomes) / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{len(self.outcomes)} resolved, {len(self.no_candles)} never traded, "
            f"{len(self.fetch_failed)} unfetchable, {len(self.bad_entry)} bad entry"
        )


def measure_outcomes(
    members: Iterable[CohortMember],
    ohlcv: OhlcvFn,
    *,
    skip_first_candle: bool = False,
    min_peak_volume: float = 0.0,
    confirm_peak_fraction: float = 0.0,
) -> MeasurementReport:
    """Resolve each member's outcome from its candle history.

    Entry is the *first candle's open* rather than the price recorded at
    capture, so entry and exit are read off the same series.

    Three defences against microstructure artifacts, all off by default so the
    raw measurement stays visible:

    ``skip_first_candle`` ignores the birth candle entirely. In one measured
    cohort **70.6% of pools had their maximum high inside candle zero** -- the
    pool-initialisation sweep, where the first minute spans several hundred x
    between low and high before any real liquidity exists. Those highs are not
    prices anyone traded at.

    ``min_peak_volume`` requires the peak to sit on a candle carrying real
    volume. The largest peak in that cohort was 152,363x on a candle with
    **$0.00074 of volume**, in a pool whose entire lifetime volume was $6.65.

    ``confirm_peak_fraction`` requires a later candle to close at that fraction
    of the peak, which is a size-neutral way to reject a single unrepeated
    wick without deleting pools for being small.

    The measured tail is also unstable in time, which is the strongest reason
    to treat any single reading sceptically. Two measurements of one cohort 40
    minutes apart gave best peaks of 1,802x and 152,363x, because a single
    $0.00074 tick landed in between. (It is *not* an aggregation effect --
    re-bucketing minute bars into five-minute bars reproduces the peak exactly,
    since a maximum of highs is invariant to bucketing.)
    """
    outcomes: list[Outcome] = []
    no_candles: list[str] = []
    fetch_failed: list[str] = []
    bad_entry: list[str] = []

    for member in members:
        candles = ohlcv(member.pool)
        if candles is None:
            fetch_failed.append(member.pool)
            continue
        if not candles:
            no_candles.append(member.pool)
            continue

        entry_candle = 0
        series = candles
        if skip_first_candle:
            if len(candles) < 2:
                bad_entry.append(member.pool)
                continue
            series = candles[1:]
        entry = series[entry_candle][1]
        if not entry or entry <= 0:
            bad_entry.append(member.pool)
            continue

        eligible = [c for c in series if c[5] >= min_peak_volume]
        if not eligible:
            # Nothing traded enough to be a real print; the position is flat.
            outcomes.append(
                Outcome(member.pool, entry, entry, series[-1][4], len(series))
            )
            continue

        peak = max(c[2] for c in eligible)
        if confirm_peak_fraction > 0:
            peak = _confirmed_peak(series, peak, confirm_peak_fraction, entry)

        outcomes.append(
            Outcome(
                pool=member.pool,
                entry_price=entry,
                max_price=max(peak, entry * 0.0),
                final_price=series[-1][4],
                candles=len(series),
            )
        )

    return MeasurementReport(
        tuple(outcomes), tuple(no_candles), tuple(fetch_failed), tuple(bad_entry)
    )


def _confirmed_peak(
    series: Candles, peak: float, fraction: float, entry: float
) -> float:
    """Highest high that some close sits within ``fraction`` of.

    A wick nobody closed near is not a level you could have sold into. The
    candle's *own* close counts: a bar that runs up and holds there is a real
    move, and requiring a strictly later confirmation would reject every peak
    on the most recent bar purely for being recent.
    """
    best = entry
    for i, candle in enumerate(series):
        high = candle[2]
        if high <= best or high > peak:
            continue
        floor = high * fraction
        if candle[4] >= floor or any(later[4] >= floor for later in series[i + 1 :]):
            best = high
    return best


def detect_clone_groups(
    members: Sequence[CohortMember],
    lifetime_volumes: dict[str, float],
    *,
    name_of: Callable[[CohortMember], str],
    volume_tolerance: float = 0.001,
) -> list[list[str]]:
    """Pools that look like one bot running several mints of the same token.

    Found in a real cohort: three separate mints named XAU/SOL, created within
    18 seconds, with lifetime volumes of $6529.84, $6529.86 and $6529.98 --
    agreeing to four significant figures. Counting those as three independent
    winners triples a hit rate and breaks the independence every confidence
    interval here assumes. It also shows that a volume threshold cannot
    establish that trading was real, because wash volume clears any threshold
    you set.
    """
    by_name: dict[str, list[CohortMember]] = {}
    for member in members:
        by_name.setdefault(name_of(member), []).append(member)

    groups: list[list[str]] = []
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        volumes = [(m.pool, lifetime_volumes.get(m.pool, 0.0)) for m in group]
        for i, (pool_a, vol_a) in enumerate(volumes):
            matches = [pool_a]
            for pool_b, vol_b in volumes[i + 1 :]:
                if vol_a > 0 and abs(vol_a - vol_b) / vol_a <= volume_tolerance:
                    matches.append(pool_b)
            if len(matches) > 1:
                groups.append(matches)
                break
    return groups


def hill_alpha(multiples: Sequence[float], *, tail_fraction: float = 0.2) -> float | None:
    """Hill estimate of the Pareto tail index over the top ``tail_fraction``.

    Lower alpha means a fatter tail. Returns ``None`` when the tail holds fewer
    than five observations, because a tail index from three points is a number
    with no information in it.
    """
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must be in (0, 1)")
    values = np.sort(np.asarray([m for m in multiples if m > 0], dtype=float))
    if values.size < 10:
        return None

    k = max(int(values.size * tail_fraction), 1)
    tail = values[-k:]
    threshold = tail[0]
    if threshold <= 0:
        return None
    logs = np.log(tail[1:] / threshold)
    if logs.size < 5 or logs.mean() <= 0:
        return None
    return float(1.0 / logs.mean())


@dataclass(frozen=True)
class CohortStats:
    """Realised distribution for one cohort at one horizon."""

    n: int
    survival_rate: float
    median_final_multiple: float
    mean_final_multiple: float
    median_max_multiple: float
    p90_max_multiple: float
    p99_max_multiple: float
    best_max_multiple: float
    tail_alpha: float | None

    def fraction_above(self, multiples: Sequence[float], threshold: float) -> float:
        if not multiples:
            return 0.0
        return sum(1 for m in multiples if m >= threshold) / len(multiples)

    def summary(self) -> str:
        alpha = f"{self.tail_alpha:.2f}" if self.tail_alpha is not None else "n/a"
        return (
            f"n={self.n}  survived={self.survival_rate:.1%}  "
            f"median final={self.median_final_multiple:.2f}x  "
            f"median max={self.median_max_multiple:.2f}x  "
            f"p99 max={self.p99_max_multiple:.1f}x  "
            f"best={self.best_max_multiple:.1f}x  alpha={alpha}"
        )


def summarise(outcomes: Sequence[Outcome]) -> CohortStats:
    """Collapse outcomes into the distribution the strategy math needs."""
    if not outcomes:
        return CohortStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)

    final = np.array([o.final_multiple for o in outcomes])
    peak = np.array([o.max_multiple for o in outcomes])
    return CohortStats(
        n=len(outcomes),
        survival_rate=float(np.mean([not o.died for o in outcomes])),
        median_final_multiple=float(np.median(final)),
        mean_final_multiple=float(final.mean()),
        median_max_multiple=float(np.median(peak)),
        p90_max_multiple=float(np.quantile(peak, 0.90)),
        p99_max_multiple=float(np.quantile(peak, 0.99)),
        best_max_multiple=float(peak.max()),
        tail_alpha=hill_alpha(peak),
    )


def expected_value_with_take_profit(
    outcomes: Sequence[Outcome],
    take_profit: float,
    *,
    cost_fraction: float,
    slippage_on_exit: float = 0.0,
) -> float:
    """Expected return per unit staked under a take-profit rule.

    A position that touched ``take_profit`` is assumed to exit there; anything
    else is marked at its final price. That is generous -- it assumes the
    take-profit order actually filled at the trigger on the way past, which in
    a thin pool it may not -- so ``slippage_on_exit`` shaves the realised
    multiple to let that optimism be priced rather than ignored.
    """
    if take_profit <= 0:
        raise ValueError("take_profit must be positive")
    if not outcomes:
        return 0.0

    total = 0.0
    for outcome in outcomes:
        if outcome.max_multiple >= take_profit:
            realised = take_profit * (1.0 - slippage_on_exit)
        else:
            realised = outcome.final_multiple * (1.0 - slippage_on_exit)
        total += realised - 1.0 - cost_fraction
    return total / len(outcomes)


def best_take_profit(
    outcomes: Sequence[Outcome],
    *,
    cost_fraction: float,
    slippage_on_exit: float = 0.0,
    candidates: Sequence[float] = (1.25, 1.5, 2, 3, 5, 10, 25, 50, 100),
) -> tuple[float, float]:
    """``(take_profit, expected_value)`` for the best rule in ``candidates``.

    Fitted on the same data it is evaluated on, so the expected value it
    reports is an in-sample upper bound and not a forecast. It answers "did any
    exit rule work here?", which is worth knowing before asking whether one
    generalises.
    """
    scored = [
        (
            tp,
            expected_value_with_take_profit(
                outcomes, tp, cost_fraction=cost_fraction,
                slippage_on_exit=slippage_on_exit,
            ),
        )
        for tp in candidates
    ]
    return max(scored, key=lambda pair: pair[1])
