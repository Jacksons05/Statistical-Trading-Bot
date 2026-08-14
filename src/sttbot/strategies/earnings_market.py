"""Price Polymarket's "Will X beat quarterly earnings?" contracts.

Polymarket runs a family of binary markets on whether a company beats its
consensus EPS estimate. That is a forecasting question, which matters: the
measured conclusion elsewhere in this repo is that this venue pays for a view
and not for arbitrage, and a company's own history of beats is the cheapest
view available.

The model is deliberately plain -- a company's trailing beat rate, shrunk
toward the cross-sectional base rate. Two things make the shrinkage
non-optional rather than a refinement:

**Companies beat consensus far more often than not.** Analysts are guided down
before the print, so the unconditional base rate sits far above 50%. A model
that does not start from that number is wrong before it sees any data, and a
market priced near 50c on a 70%-base-rate event is either a gift or a warning
that the question is not what you think.

**Eight quarters is not enough to believe an extreme.** A company that beat all
of its last eight is not a 100% shot. Raw frequency on a short window produces
exactly the confident, wrong probabilities that make a book blow up on the
first miss.

The honest test of any such model is not whether it looks sensible but whether
it beats the base rate it was shrunk toward, which is what
:func:`skill_vs_base_rate` measures. A model that cannot is a complicated way
of quoting a constant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..data.earnings import EarningsEvent
from ..venues.prediction import FeeModel

# Cross-sectional share of reports that beat consensus. Overridden by a
# measured value wherever one is available; this is only the fallback prior.
DEFAULT_BASE_RATE = 0.70


@dataclass(frozen=True)
class BeatForecast:
    """A probability with the evidence behind it kept visible."""

    probability: float
    n_history: int
    trailing_beat_rate: float
    base_rate: float

    @property
    def is_shrunk_toward_prior(self) -> bool:
        """True when the prior moved the estimate more than the data did."""
        return abs(self.probability - self.base_rate) < abs(
            self.trailing_beat_rate - self.base_rate
        )


def beat_probability(
    events: Sequence[EarningsEvent],
    index: int,
    *,
    window: int = 12,
    prior_weight: float = 6.0,
    base_rate: float = DEFAULT_BASE_RATE,
    min_history: int = 4,
) -> BeatForecast | None:
    """P(the report at ``index`` beats), from prior reports only.

    Walk-forward by construction: only ``events[:index]`` is read, so this can
    be evaluated at any point in history without leaking the answer.

    ``prior_weight`` is in units of pseudo-observations. At the default of 6, a
    company with 8 quarters of history has its own record weighted 8/14 against
    the base rate -- enough to move the estimate, not enough to let a short run
    of beats imply certainty.
    """
    if index < 0 or index > len(events):
        raise IndexError("index outside events")
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_rate must be a probability")
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative")

    prior = events[max(0, index - window):index]
    if len(prior) < min_history:
        return None

    beats = sum(1 for e in prior if e.surprise > 0)
    n = len(prior)
    probability = (beats + prior_weight * base_rate) / (n + prior_weight)
    return BeatForecast(
        probability=probability,
        n_history=n,
        trailing_beat_rate=beats / n,
        base_rate=base_rate,
    )


def measure_base_rate(histories: Sequence[Sequence[EarningsEvent]]) -> float:
    """Unconditional share of reports that beat, across every company given.

    Ties (surprise exactly zero) count as misses, matching the usual contract
    wording of "beat" rather than "meet or beat". Polymarket's resolution
    criteria should be checked per market; where they differ, this is the
    conservative side.
    """
    total = beats = 0
    for events in histories:
        for e in events:
            total += 1
            if e.surprise > 0:
                beats += 1
    return beats / total if total else 0.0


# --- the control ------------------------------------------------------------


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must correspond")
    if not probabilities:
        return float("nan")
    return sum((p - float(o)) ** 2 for p, o in zip(probabilities, outcomes)) / len(
        probabilities
    )


def skill_vs_base_rate(
    probabilities: Sequence[float], outcomes: Sequence[bool], base_rate: float
) -> float:
    """Brier skill score against always quoting the base rate.

    Positive means the model knows something beyond "companies usually beat";
    zero or negative means it does not, however plausible its inputs look. This
    is the same discipline as the closing-line-value control used for sports:
    measure the edge against what you would get for free.
    """
    model = brier_score(probabilities, outcomes)
    naive = brier_score([base_rate] * len(outcomes), outcomes)
    if naive <= 0:
        return float("nan")
    return 1.0 - model / naive


# --- turning a probability into a trade -------------------------------------


@dataclass(frozen=True)
class MarketEdge:
    """Net edge on each side of a binary, after fees."""

    yes_edge: float
    no_edge: float

    @property
    def best_side(self) -> str | None:
        if self.yes_edge <= 0 and self.no_edge <= 0:
            return None
        return "YES" if self.yes_edge >= self.no_edge else "NO"

    @property
    def best_edge(self) -> float:
        return max(self.yes_edge, self.no_edge)


def edge_against_market(
    probability: float, yes_bid: float, yes_ask: float, fees: FeeModel
) -> MarketEdge:
    """Edge per contract on both sides, net of the venue's fee.

    Buying YES costs ``yes_ask`` and pays 1 on a beat. Buying NO costs
    ``1 - yes_bid`` -- Polymarket mirrors one book, so the NO ask is the
    complement of the YES bid rather than an independent quote -- and pays 1 on
    a miss.

    Both sides cross the spread, which is why the bid and the ask are both
    required. Scoring against the mid would report an edge on every market
    where the model merely disagrees with the midpoint by less than half the
    spread.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 < yes_bid <= yes_ask < 1.0:
        raise ValueError("need a sane two-sided quote")

    yes_edge = probability - yes_ask - fees.cost(yes_ask)
    no_price = 1.0 - yes_bid
    no_edge = (1.0 - probability) - no_price - fees.cost(no_price)
    return MarketEdge(yes_edge=yes_edge, no_edge=no_edge)


# --- matching a Polymarket contract to a company ----------------------------

_TICKER = re.compile(
    r"^Will\s+.+?\(([A-Z][A-Z0-9.\-]{0,6})\)\s+beat quarterly earnings", re.I
)


def extract_ticker(title: str) -> str | None:
    """Ticker from a "Will Company (TICK) beat quarterly earnings?" title.

    Deliberately strict. Pricing a contract against the wrong company's
    earnings history is silent and total -- the numbers all look reasonable and
    every one of them is about somebody else -- so this matches the full
    sentence shape rather than hunting for any parenthesised capital letters.
    A title that does not fit returns ``None`` and is skipped.
    """
    if not title:
        return None
    match = _TICKER.match(title.strip())
    return match.group(1).upper() if match else None
