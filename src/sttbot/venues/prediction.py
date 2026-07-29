"""Cross-venue prediction-market pricing and arbitrage.

Two venues can quote the same real-world event at different prices for reasons
that have nothing to do with disagreement about the outcome: separate capital
pools, KYC boundaries, and no shared settlement layer. That gap is the trade.

Two things make it harder than "buy low, sell high":

**Fees are not proportional.** Kalshi charges a fee that peaks at a 50c
contract and vanishes at the extremes, so the *same* cent of gross edge can be
profitable at 5c and unprofitable at 50c. Modelling this as a flat percentage
gets the sign of the trade wrong near the middle of the book.

**Contracts are not always fungible.** Two markets can share a headline and
settle differently — different sources, cutoffs, or tie handling. This module
will not net two legs against each other unless they are explicitly declared
equivalent, because a resolution-criteria mismatch turns a "risk-free" hedge
into an unhedged directional bet on a technicality.

Prices are probabilities in [0, 1]; a winning contract pays exactly 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    """Per-venue trading cost.

    ``maker``/``taker`` are proportional to notional. ``kalshi_style`` switches
    to Kalshi's published quadratic form, which dominates the proportional
    terms when set.
    """

    taker: float = 0.0
    maker: float = 0.0
    kalshi_style: bool = False
    kalshi_rate: float = 0.07

    def cost(self, price: float, contracts: float = 1.0, *, maker: bool = False) -> float:
        """Fee in numeraire for trading ``contracts`` at ``price``."""
        if not 0.0 <= price <= 1.0:
            raise ValueError("price must be a probability in [0, 1]")
        if contracts < 0:
            raise ValueError("contracts must be non-negative")
        if self.kalshi_style:
            # fee = rate * C * P * (1 - P): maximal at P=0.50, zero at the ends.
            return self.kalshi_rate * contracts * price * (1.0 - price)
        rate = self.maker if maker else self.taker
        return rate * contracts * price


# Published venue fee schedules. Polymarket has charged 0 on trades, taking its
# economics elsewhere; Kalshi's quadratic is the one that actually reshapes
# which trades are viable.
KALSHI_FEES = FeeModel(kalshi_style=True, kalshi_rate=0.07)
POLYMARKET_FEES = FeeModel(taker=0.0, maker=0.0)


@dataclass(frozen=True)
class VenueQuote:
    """One venue's two-sided market in a single binary outcome."""

    venue: str
    market_id: str
    best_bid: float
    best_ask: float
    fees: FeeModel = FeeModel()
    bid_size: float = 0.0
    ask_size: float = 0.0

    def __post_init__(self) -> None:
        for label, price in (("best_bid", self.best_bid), ("best_ask", self.best_ask)):
            if not 0.0 <= price <= 1.0:
                raise ValueError(f"{self.venue}: {label} {price} outside [0, 1]")
        if self.best_bid > self.best_ask:
            raise ValueError(f"{self.venue}: crossed book")

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


@dataclass(frozen=True)
class CrossVenueArb:
    """A matched pair: buy the outcome on one venue, sell it on the other."""

    buy_venue: str
    sell_venue: str
    buy_price: float
    sell_price: float
    contracts: float
    gross_edge: float  # per contract, before fees
    net_edge: float  # per contract, after both venues' fees
    net_profit: float  # net_edge * contracts

    @property
    def profitable(self) -> bool:
        return self.net_edge > 0


def find_cross_venue_arb(
    a: VenueQuote,
    b: VenueQuote,
    *,
    equivalent: bool = False,
    min_net_edge: float = 0.0,
    max_contracts: float | None = None,
) -> CrossVenueArb | None:
    """Best cross-venue trade on a single outcome, or ``None``.

    ``equivalent`` must be set explicitly to confirm the two markets settle on
    identical criteria. Without it this raises, rather than silently pairing
    legs that may not offset — a same-name market with a different resolution
    source is the single most expensive mistake available here.

    Size is capped by the resting depth on both legs, so the returned trade is
    one that could actually fill rather than a headline number against the top
    of an empty book.
    """
    if not equivalent:
        raise ValueError(
            "refusing to pair markets not declared equivalent: verify both "
            "resolve on identical criteria, then pass equivalent=True"
        )
    if a.venue == b.venue and a.market_id == b.market_id:
        raise ValueError("cannot arbitrage a market against itself")

    best: CrossVenueArb | None = None
    for buy, sell in ((a, b), (b, a)):
        gross = sell.best_bid - buy.best_ask
        if gross <= 0:
            continue
        # Depth on both legs bounds a hedged trade; either side unfilled leaves
        # naked directional risk.
        size = min(buy.ask_size, sell.bid_size)
        if max_contracts is not None:
            size = min(size, max_contracts)
        fees = buy.fees.cost(buy.best_ask) + sell.fees.cost(sell.best_bid)
        net = gross - fees
        if net < min_net_edge:
            continue
        candidate = CrossVenueArb(
            buy_venue=buy.venue,
            sell_venue=sell.venue,
            buy_price=buy.best_ask,
            sell_price=sell.best_bid,
            contracts=size,
            gross_edge=gross,
            net_edge=net,
            net_profit=net * size,
        )
        if best is None or candidate.net_edge > best.net_edge:
            best = candidate
    return best


def kelly_fraction(probability: float, price: float) -> float:
    """Kelly stake as a fraction of bankroll for a binary contract.

    Buying at ``price`` pays ``1/price`` gross, i.e. net odds
    ``b = (1 - price)/price``. Kelly is ``(p*b - q)/b``, which simplifies to
    ``(p - price)/(1 - price)``. Returns 0.0 when the bet is not +EV.

    Full Kelly is the growth-optimal stake under *known* probabilities. Model
    probabilities are estimates, and overstating an edge overbets it
    superlinearly, so scale this down (quarter-Kelly is common) in practice.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 < price < 1.0:
        raise ValueError("price must be strictly inside (0, 1)")
    edge = probability - price
    return max(0.0, edge / (1.0 - price))


def implied_from_book(quotes: list[VenueQuote]) -> float:
    """Depth-weighted consensus mid across venues.

    Weighting by resting size lets a deep venue dominate a thin one, instead of
    a 100-contract book moving the consensus as much as a 100,000 one. Falls
    back to an equal-weighted mean when no sizes are reported.
    """
    if not quotes:
        raise ValueError("need at least one quote")
    weights = [q.bid_size + q.ask_size for q in quotes]
    total = math.fsum(weights)
    if total <= 0:
        return math.fsum(q.mid for q in quotes) / len(quotes)
    return math.fsum(q.mid * w for q, w in zip(quotes, weights)) / total
