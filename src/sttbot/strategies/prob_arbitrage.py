"""Multi-outcome probability-boundary arbitrage.

For a set of mutually exclusive, collectively exhaustive outcomes, prices
expressed as probabilities must sum to 1. When independent traders quote each
leg, the book can drift off that boundary:

* ``sum(best_ask) < 1`` -> buy every outcome for a guaranteed 1.00 payout
  (a synthetic long basket) at a cost below par.
* ``sum(best_bid) > 1`` -> sell every outcome; you collect more than the
  1.00 you must pay out on resolution.

Fees are charged per leg and erode the raw boundary gap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Outcome:
    """One leg of a categorical market, priced in probability units (0..1)."""

    name: str
    best_bid: float
    best_ask: float

    def __post_init__(self) -> None:
        for label, price in (("best_bid", self.best_bid), ("best_ask", self.best_ask)):
            if not 0.0 <= price <= 1.0:
                raise ValueError(f"{self.name}: {label} {price} outside [0, 1]")
        if self.best_bid > self.best_ask:
            raise ValueError(f"{self.name}: crossed book (bid > ask)")


@dataclass(frozen=True)
class ArbitrageOpportunity:
    side: str  # "buy_basket" | "sell_basket"
    legs: tuple[str, ...]
    gross_edge: float  # boundary gap before fees
    net_edge: float  # per 1.00 basket notional, after fees
    prices: tuple[float, ...]


def find_boundary_arbitrage(
    outcomes: list[Outcome], fee_rate: float = 0.0, min_net_edge: float = 0.0
) -> ArbitrageOpportunity | None:
    """Return the best boundary arbitrage, or ``None`` if none clears fees.

    ``fee_rate`` is a proportional taker fee applied to each leg's traded
    price. ``min_net_edge`` filters out opportunities too thin to bother with.
    """
    if len(outcomes) < 2:
        raise ValueError("need at least two outcomes to form a basket")

    ask_sum = sum(o.best_ask for o in outcomes)
    bid_sum = sum(o.best_bid for o in outcomes)
    # Fees scale with the notional actually transacted on each leg.
    buy_fees = fee_rate * ask_sum
    sell_fees = fee_rate * bid_sum

    buy_net = (1.0 - ask_sum) - buy_fees  # pay ask_sum + fees, receive 1.00
    sell_net = (bid_sum - 1.0) - sell_fees  # receive bid_sum, pay out 1.00 + fees

    best: ArbitrageOpportunity | None = None
    if buy_net >= min_net_edge and buy_net >= sell_net:
        best = ArbitrageOpportunity(
            side="buy_basket",
            legs=tuple(o.name for o in outcomes),
            gross_edge=1.0 - ask_sum,
            net_edge=buy_net,
            prices=tuple(o.best_ask for o in outcomes),
        )
    elif sell_net >= min_net_edge:
        best = ArbitrageOpportunity(
            side="sell_basket",
            legs=tuple(o.name for o in outcomes),
            gross_edge=bid_sum - 1.0,
            net_edge=sell_net,
            prices=tuple(o.best_bid for o in outcomes),
        )
    return best
