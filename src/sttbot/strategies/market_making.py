"""Market making (scalping) for binary prediction-market contracts.

The edge here is liquidity provision, not speed. You post a two-sided quote,
earn the spread when uninformed flow trades against you, and lose when informed
flow picks you off. Three things decide whether that nets out positive:

**Fees set a hard floor on where you can quote at all.** A round trip pays the
fee twice, and Kalshi's is quadratic — 3.50c at a 50c contract versus 0.67c at
5c. Quoting mid-book therefore needs a spread wider than 3.5c *before* any
profit, which most books do not offer. :func:`breakeven_spread` makes that
explicit and :func:`untradeable_band` reports the block of the book a given
spread cannot support. This is the single most important constraint on the
strategy and the reason tails are more tradeable than the middle.

**Inventory is directional risk.** A binary settles at 0 or 1, so unhedged
inventory is a bet, not a spread capture. Quotes are skewed away from the side
that would add to an existing position, following Avellaneda-Stoikov: shift the
reservation price against inventory so the book naturally unwinds you. Variance
per contract is ``p(1-p)`` — maximal at 0.50, vanishing at the extremes — which
is a second, independent reason the middle of the book is the dangerous place
to carry size.

**Adverse selection is the real cost, not fees.** You are filled precisely when
someone knows something. No quoting rule fixes that; it has to be *measured*,
which is what the simulator in ``backtest.mm_simulator`` does via fill mark-out.
A market-making backtest that fills you at your quote against an uninformed
price path will always look profitable and always be wrong.

Prices are probabilities in [0, 1] throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..venues.prediction import FeeModel

# Venues quote in whole cents; quoting outside this band is not accepted and
# 0.00/1.00 are settlement values rather than tradeable prices.
MIN_PRICE = 0.01
MAX_PRICE = 0.99


def contract_variance(price: float) -> float:
    """Terminal payoff variance of one binary contract, ``p(1-p)``.

    Maximal at 0.50 and zero at either extreme: a contract at 0.02 is nearly
    settled, one at 0.50 is a coin flip. This is the natural risk scale for
    inventory held to expiry.
    """
    if not 0.0 <= price <= 1.0:
        raise ValueError("price must be a probability in [0, 1]")
    return price * (1.0 - price)


def breakeven_spread(price: float, fees: FeeModel) -> float:
    """Minimum bid-ask spread that covers a round trip's fees.

    Both legs are priced as **maker** fills, because that is what a market
    maker is: passive on the buy and passive on the sell. Using the taker rate
    here would be wrong on any venue that splits the two — on Polymarket it is
    the difference between paying 5% and earning a rebate.

    Can be negative where makers are paid: the venue subsidises the round trip,
    so even a zero spread is profitable before adverse selection.
    """
    return fees.cost(price, maker=True) * 2.0


def untradeable_band(
    fees: FeeModel, available_spread: float
) -> tuple[float, float] | None:
    """Prices where ``available_spread`` cannot cover the round-trip fee.

    Returns the ``(low, high)`` band to avoid, or ``None`` when the whole book
    is workable. For a quadratic fee the unviable region is a contiguous block
    around 0.50 and the viable region is the two tails — which is why this
    reports the *exclusion* rather than a "viable range": the viable set is
    disjoint, so its min and max are always ~0.01 and ~0.99 and convey nothing.
    """
    if available_spread <= 0:
        raise ValueError("available_spread must be positive")
    blocked = [
        p / 100.0
        for p in range(int(MIN_PRICE * 100), int(MAX_PRICE * 100) + 1)
        if breakeven_spread(p / 100.0, fees) >= available_spread
    ]
    if not blocked:
        return None
    return (min(blocked), max(blocked))


def is_tradeable(price: float, fees: FeeModel, available_spread: float) -> bool:
    """Whether a single price clears the round-trip fee at ``available_spread``."""
    return breakeven_spread(price, fees) < available_spread


@dataclass(frozen=True)
class QuoteParams:
    """Market-maker configuration."""

    # Desired profit per round trip, on top of fees. The actual spread is the
    # larger of this-plus-fees and min_half_spread*2.
    target_edge: float = 0.01
    min_half_spread: float = 0.005
    # Avellaneda-Stoikov risk aversion. Higher skews quotes harder against
    # inventory, unwinding faster at the cost of fewer round trips.
    #
    # Scale matters more than it looks: the skew at full inventory, mid-book,
    # is `risk_aversion * 0.25`, so 1.0 would shift quotes by 25c — dwarfing a
    # ~1c spread and making the maker buy back its own inventory far above
    # fair value. The default gives ~1.25c at full inventory, comparable to
    # the spread it is trying to earn. Keep `risk_aversion * 0.25` on the order
    # of a few half-spreads.
    risk_aversion: float = 0.05
    max_inventory: float = 100.0
    # Contracts offered per side.
    quote_size: float = 10.0
    # Venues quote on a discrete grid (1c on both Kalshi and Polymarket). Once
    # fees are zero the tick becomes the binding floor on spread: you cannot
    # quote inside one tick, so the tightest possible round trip earns exactly
    # one tick minus whatever adverse selection takes.
    tick_size: float = 0.01

    def __post_init__(self) -> None:
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive")
        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")


@dataclass(frozen=True)
class Quote:
    """A two-sided quote. Either side may be ``None`` when suppressed."""

    bid: float | None
    ask: float | None
    size: float
    reservation_price: float
    half_spread: float

    @property
    def is_two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None

    @property
    def spread(self) -> float | None:
        if not self.is_two_sided:
            return None
        return self.ask - self.bid


class MarketMaker:
    """Generates inventory-aware, fee-aware two-sided quotes."""

    def __init__(self, params: QuoteParams | None = None, fees: FeeModel | None = None):
        self.params = params or QuoteParams()
        self.fees = fees or FeeModel()

    def reservation_price(
        self, fair_value: float, inventory: float, time_remaining: float = 1.0
    ) -> float:
        """Fair value shifted against current inventory.

        Long inventory pushes the reservation price down, so both quotes fall
        and the book is more likely to lift the ask than hit the bid — the
        position unwinds itself. The shift scales with ``p(1-p)`` and with time
        left to settlement, because both increase the risk of carrying it.
        """
        if not 0.0 <= time_remaining <= 1.0:
            raise ValueError("time_remaining must be a fraction in [0, 1]")
        skew = (
            inventory
            * self.params.risk_aversion
            * contract_variance(fair_value)
            * time_remaining
            / self.params.max_inventory
        )
        return fair_value - skew

    def half_spread(self, price: float) -> float:
        """Half-spread covering round-trip maker fees plus the target edge.

        Floored at half a tick so the two sides can never land on the same
        grid point. Where makers are paid, ``breakeven_spread`` is negative and
        the fee term legitimately *reduces* the required spread.
        """
        required = (breakeven_spread(price, self.fees) + self.params.target_edge) / 2.0
        return max(required, self.params.min_half_spread, self.params.tick_size / 2.0)

    def _to_tick(self, price: float, *, up: bool) -> float:
        """Snap to the venue's grid, always away from the mid.

        Rounding a bid down and an ask up keeps the realised spread at least as
        wide as intended; rounding to nearest could quietly tighten it below
        the level that was priced.
        """
        tick = self.params.tick_size
        ticks = price / tick
        snapped = (math.ceil(ticks) if up else math.floor(ticks)) * tick
        return round(snapped, 10)

    def quote(
        self, fair_value: float, inventory: float = 0.0, time_remaining: float = 1.0
    ) -> Quote:
        """Produce a quote, suppressing sides that would breach inventory limits.

        At the long limit the bid is withdrawn (stop buying) while the ask stays
        up (keep selling), and vice versa. A one-sided quote is the correct
        response to a full book, not an error.
        """
        if not 0.0 <= fair_value <= 1.0:
            raise ValueError("fair_value must be a probability in [0, 1]")

        reservation = self.reservation_price(fair_value, inventory, time_remaining)
        half = self.half_spread(fair_value)
        limit = self.params.max_inventory

        bid: float | None = None if inventory >= limit else reservation - half
        ask: float | None = None if inventory <= -limit else reservation + half

        # Snap to the venue grid (away from the mid), then clamp into the
        # tradeable band. A quote pinned to the boundary is dropped rather than
        # posted at a price that cannot earn the spread.
        if bid is not None:
            bid = round(max(MIN_PRICE, min(MAX_PRICE, self._to_tick(bid, up=False))), 10)
            if bid >= MAX_PRICE:
                bid = None
        if ask is not None:
            ask = round(max(MIN_PRICE, min(MAX_PRICE, self._to_tick(ask, up=True))), 10)
            if ask <= MIN_PRICE:
                ask = None
        # A crossed or locked quote is never valid; drop the passive side.
        if bid is not None and ask is not None and bid >= ask:
            bid, ask = None, None

        return Quote(
            bid=bid,
            ask=ask,
            size=self.params.quote_size,
            reservation_price=reservation,
            half_spread=half,
        )

    def can_quote_profitably(self, price: float, market_spread: float) -> bool:
        """Whether the market's own spread is wide enough to work inside.

        If the book is tighter than the round-trip fee, joining it guarantees a
        loss on every completed round trip.
        """
        return market_spread > breakeven_spread(price, self.fees)
