"""Market-making simulator with explicit adverse-selection measurement.

The failure mode this is built to avoid: a simulator that fills your quote
whenever the price touches it, against a path that does not react to your
presence, will report a profit almost regardless of the strategy. Every fill
looks like captured spread. Real market making loses to *informed* flow — you
are bought from just before the price rises — and that cost is invisible unless
it is measured directly.

So fills are marked out here. For each fill, :attr:`Fill.markout` records the
mid ``markout_steps`` later minus the fill price, signed by trade direction:
positive means the price moved in your favour after you traded, negative means
you were run over. Averaged across fills this is the adverse-selection cost, and
it is reported alongside P&L rather than folded into it. A strategy with
positive gross spread capture and worse-negative mark-out is losing money while
appearing to work.

Fill model: a resting bid fills when the path trades at or below it, an ask when
the path trades at or above it. This is optimistic — it assumes queue priority
and no partial fills — so treat the output as an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..strategies.market_making import MarketMaker
from ..venues.prediction import FeeModel


@dataclass(frozen=True)
class Fill:
    step: int
    side: str  # "buy" | "sell"
    price: float
    size: float
    fee: float
    mid_at_fill: float
    markout: float | None = None  # signed favourable move after the fill


@dataclass
class SimulationResult:
    fills: list[Fill] = field(default_factory=list)
    final_inventory: float = 0.0
    cash: float = 0.0
    settlement_value: float = 0.0
    total_fees: float = 0.0

    @property
    def num_fills(self) -> int:
        return len(self.fills)

    @property
    def gross_pnl(self) -> float:
        """P&L before fees: cash flow plus whatever inventory settles for."""
        return self.cash + self.settlement_value + self.total_fees

    @property
    def net_pnl(self) -> float:
        return self.cash + self.settlement_value

    @property
    def mean_markout(self) -> float | None:
        """Average post-fill move in your favour. Negative = adverse selection."""
        marked = [f.markout for f in self.fills if f.markout is not None]
        if not marked:
            return None
        return sum(marked) / len(marked)

    @property
    def adverse_fill_rate(self) -> float | None:
        """Share of fills where the price moved against you afterwards."""
        marked = [f.markout for f in self.fills if f.markout is not None]
        if not marked:
            return None
        return sum(1 for m in marked if m < 0) / len(marked)

    def summary(self) -> str:
        markout = self.mean_markout
        markout_text = "n/a" if markout is None else f"{markout:+.5f}"
        return (
            f"fills={self.num_fills} net={self.net_pnl:+.4f} "
            f"fees={self.total_fees:.4f} inventory={self.final_inventory:+.1f} "
            f"mean_markout={markout_text}"
        )


def simulate(
    maker: MarketMaker,
    path: list[float],
    *,
    settlement: float | None = None,
    markout_steps: int = 10,
    fees: FeeModel | None = None,
) -> SimulationResult:
    """Run ``maker`` against a mid-price ``path``.

    ``settlement`` is the binary outcome (1.0 or 0.0) used to value leftover
    inventory; when ``None`` the final mid is used instead, which values the
    book at market rather than assuming a resolution.

    Time remaining is derived from position in the path, so inventory skew
    tightens as expiry approaches — carrying size into settlement is what turns
    a spread capture into a directional bet.
    """
    if len(path) < 2:
        raise ValueError("path needs at least two points")
    if markout_steps < 0:
        raise ValueError("markout_steps must be non-negative")

    fee_model = fees or maker.fees
    result = SimulationResult()
    inventory = 0.0
    cash = 0.0
    n = len(path)

    for step in range(n - 1):
        mid = path[step]
        nxt = path[step + 1]
        time_remaining = 1.0 - step / (n - 1)
        quote = maker.quote(mid, inventory, time_remaining)

        # A resting bid is hit when the market trades down through it; an ask is
        # lifted when the market trades up through it. Both can occur in a step,
        # but filling both from one move would be double-counting a single
        # directional trade, so the move's direction picks one.
        if quote.bid is not None and nxt <= quote.bid:
            fee = fee_model.cost(quote.bid, quote.size)
            inventory += quote.size
            cash -= quote.bid * quote.size + fee
            result.total_fees += fee
            result.fills.append(
                Fill(step, "buy", quote.bid, quote.size, fee, mid)
            )
        elif quote.ask is not None and nxt >= quote.ask:
            fee = fee_model.cost(quote.ask, quote.size)
            inventory -= quote.size
            cash += quote.ask * quote.size - fee
            result.total_fees += fee
            result.fills.append(
                Fill(step, "sell", quote.ask, quote.size, fee, mid)
            )

    # Mark out every fill against the price some steps later. Buying is
    # favourable if the price rose; selling if it fell.
    marked: list[Fill] = []
    for fill in result.fills:
        future_index = min(fill.step + markout_steps, n - 1)
        future_mid = path[future_index]
        move = future_mid - fill.price
        markout = move if fill.side == "buy" else -move
        marked.append(
            Fill(fill.step, fill.side, fill.price, fill.size, fill.fee,
                 fill.mid_at_fill, markout)
        )
    result.fills = marked

    terminal = settlement if settlement is not None else path[-1]
    result.final_inventory = inventory
    result.cash = cash
    result.settlement_value = inventory * terminal
    return result


def random_walk(
    start: float, steps: int, volatility: float, seed: int, *, drift: float = 0.0
) -> list[float]:
    """A bounded random walk for uninformed-flow testing.

    Deliberately trendless by default: a maker should be profitable here, and
    a strategy that is *not* profitable against uninformed flow has a problem
    that informed flow will only worsen.
    """
    import random

    rng = random.Random(seed)
    path, price = [start], start
    for _ in range(steps - 1):
        price = max(0.01, min(0.99, price + drift + rng.gauss(0.0, volatility)))
        path.append(price)
    return path


def trending_path(
    start: float, steps: int, drift: float, volatility: float, seed: int
) -> list[float]:
    """A directional path standing in for informed flow.

    A maker quoting both sides into a trend is repeatedly filled on the losing
    side — bought from as the price falls away, sold to as it runs up. This is
    the adverse-selection scenario, and mark-out should go clearly negative.
    """
    return random_walk(start, steps, volatility, seed, drift=drift)
