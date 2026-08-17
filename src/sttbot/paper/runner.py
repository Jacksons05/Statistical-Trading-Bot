"""The paper trading loop: propose, price, guard, record.

A strategy proposes :class:`Basket` objects. A basket is **all-or-nothing**,
which is the whole reason this abstraction exists rather than a flat list of
orders. The one strategy in this repo with a measured positive edge is
multi-outcome arbitrage, and a partially filled arbitrage is not a smaller
arbitrage -- it is an unhedged directional bet on whichever outcomes did fill.
Filling three legs of a four-leg basket converts a risk-free trade into the
exact position the trade existed to avoid.

The runner refuses to trade when the circuit breaker has tripped, refuses to
spend cash the account does not have, and prices every leg through a fill
model rather than assuming the quoted price is executable. Each of those is a
way a paper book flatters itself if left out.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..execution.oms import OMS
from ..risk.circuit_breaker import RiskCircuitBreaker
from .account import BUY, SELL, Fill, PaperAccount, Snapshot


@dataclass(frozen=True)
class Intent:
    """A proposed trade, before anyone checks whether it can be done."""

    symbol: str
    side: str
    qty: float
    limit_price: float
    strategy: str = ""
    ref: str = ""
    note: str = ""
    # Precomputed fee in numeraire, for venues whose fee is not a flat
    # proportional rate (Polymarket's is quadratic in price). None falls back
    # to basket.fee_rate * qty * fill_price.
    fee: float | None = None

    def __post_init__(self) -> None:
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be {BUY} or {SELL}")
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.fee is not None and self.fee < 0:
            raise ValueError("fee must be non-negative")


@dataclass(frozen=True)
class Basket:
    """Intents that must all execute or none of them.

    ``rationale`` is recorded on every resulting fill so a book can be audited
    back to the reason it was traded, months later, without the strategy code
    still being in the shape it was.
    """

    intents: tuple[Intent, ...]
    rationale: str = ""
    fee_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.intents:
            raise ValueError("a basket needs at least one intent")


@dataclass(frozen=True)
class RunResult:
    filled: tuple[Fill, ...]
    rejected: tuple[tuple[str, str], ...]  # (rationale or ref, reason)
    halted: bool
    snapshot: Snapshot

    def summary(self) -> str:
        return (
            f"{len(self.filled)} fills, {len(self.rejected)} baskets rejected, "
            f"equity ${self.snapshot.equity:,.2f}"
            + ("  [HALTED]" if self.halted else "")
        )


# Returns the executable average price for an intent, or None if it cannot be
# filled at all. Real implementations walk an order book.
FillModel = Callable[[Intent], float | None]


def limit_fill_model(intent: Intent) -> float | None:
    """Trivial model: everything fills at its limit price.

    Only appropriate for tests and for venues where the quoted size genuinely
    covers the order. Anywhere else it overstates fills.
    """
    return intent.limit_price


class _AccountOms:
    """Adapts a :class:`PaperAccount` to the OMS surface the breaker needs.

    The breaker's contract is that tripping *flattens the book*, so this
    actually closes positions at the supplied marks rather than only setting a
    flag. A kill switch that leaves positions open is not a kill switch.
    """

    def __init__(self, account: PaperAccount, marks: dict[str, float],
                 now: dt.datetime):
        self.account = account
        self.marks = marks
        self.now = now
        self.trading_enabled = True
        self.flattened: list[str] = []

    def get_orderbook(self, symbol: str) -> dict:  # pragma: no cover - unused
        return {"bids": [], "asks": []}

    def place_limit_order(self, symbol, side, price, qty) -> str:  # pragma: no cover
        raise NotImplementedError("the paper runner records fills directly")

    def get_order_status(self, order_id: str) -> str:  # pragma: no cover
        return "FILLED"

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover
        return None

    def cancel_all_open_orders(self) -> None:
        return None

    def flatten_all_positions(self) -> None:
        for symbol, position in self.account.open_positions().items():
            mark = self.marks.get(symbol)
            if mark is None:
                # Cannot flatten what nobody will quote. Left open and
                # reported rather than pretended closed at a made-up price.
                continue
            self.account.record(
                Fill(
                    ts=self.now,
                    symbol=symbol,
                    side=SELL if position.qty > 0 else BUY,
                    qty=abs(position.qty),
                    price=mark,
                    strategy="circuit-breaker",
                    ref=f"flatten:{symbol}:{self.now.isoformat()}",
                    note="flattened by circuit breaker",
                )
            )
            self.flattened.append(symbol)

    def disable_trading_loop(self) -> None:
        self.trading_enabled = False


@dataclass
class PaperRunner:
    """Executes strategy proposals against a durable paper account."""

    account: PaperAccount
    breaker: RiskCircuitBreaker | None = None
    fill_model: FillModel = limit_fill_model
    # Refuse to let cash go negative: a paper account that can overdraw is not
    # modelling anything.
    allow_negative_cash: bool = False
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)
    last_oms: _AccountOms | None = field(default=None, init=False)

    def run_once(
        self, baskets: Sequence[Basket], marks: dict[str, float]
    ) -> RunResult:
        now = self.clock()
        filled: list[Fill] = []
        rejected: list[tuple[str, str]] = []

        oms = _AccountOms(self.account, marks, now)
        self.last_oms = oms

        halted = False
        if self.breaker is not None:
            equity = self.account.snapshot(marks, ts=now).equity
            halted = not self.breaker.evaluate(max(equity, 0.0), oms)

        if halted:
            for basket in baskets:
                rejected.append((basket.rationale, "circuit breaker tripped"))
            return RunResult(
                tuple(filled), tuple(rejected), True,
                self.account.snapshot(marks, ts=now),
            )

        for basket in baskets:
            outcome = self._price_basket(basket, now)
            if isinstance(outcome, str):
                rejected.append((basket.rationale, outcome))
                continue
            recorded = [f for f in outcome if self.account.record(f)]
            filled.extend(recorded)

        return RunResult(
            tuple(filled), tuple(rejected), False,
            self.account.snapshot(marks, ts=now),
        )

    def _price_basket(self, basket: Basket, now: dt.datetime):
        """Return the basket's fills, or a string reason it was rejected."""
        fills: list[Fill] = []
        cash_needed = 0.0
        for intent in basket.intents:
            price = self.fill_model(intent)
            if price is None:
                return f"no fill available for {intent.symbol}"
            if intent.side == BUY and price > intent.limit_price:
                return f"{intent.symbol} would fill at {price:.4f} through its limit"
            if intent.side == SELL and price < intent.limit_price:
                return f"{intent.symbol} would fill at {price:.4f} through its limit"

            fee = (
                intent.fee if intent.fee is not None
                else basket.fee_rate * intent.qty * price
            )
            fill = Fill(
                ts=now, symbol=intent.symbol, side=intent.side, qty=intent.qty,
                price=price, fee=fee, strategy=intent.strategy,
                ref=intent.ref, note=basket.rationale or intent.note,
            )
            fills.append(fill)
            cash_needed -= fill.cash_delta

        if not self.allow_negative_cash and cash_needed > self.account.cash:
            return (
                f"needs ${cash_needed:,.2f} but only ${self.account.cash:,.2f} free"
            )
        return fills
