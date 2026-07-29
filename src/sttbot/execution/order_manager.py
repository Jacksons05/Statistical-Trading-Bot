"""Latency-aware order execution with slippage guards and a Time-To-Live.

Networks and feeds spike; a limit order left resting on a stale quote can be
adversely filled. :class:`DynamicOrderManager` rejects orders whose pre-trade
slippage already exceeds a cap, then polls for a fill and cancels the order if
it does not complete within its TTL.

The clock and sleep functions are injectable so the TTL loop is deterministic
under test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .oms import OMS


@dataclass
class ExecutionResult:
    filled: bool
    order_id: str | None
    reason: str


def slippage_bps(reference_price: float, target_price: float) -> float:
    """Signed slippage of the current reference vs. our target, in basis points.

    Positive means the market has moved *against* a buyer (reference above the
    price we wanted to pay).
    """
    if target_price <= 0:
        raise ValueError("target_price must be positive")
    return (reference_price - target_price) / target_price * 10_000


class DynamicOrderManager:
    def __init__(
        self,
        max_allowed_slippage_bps: float = 10.0,
        ttl_seconds: float = 2.5,
        poll_interval: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.max_slippage_bps = max_allowed_slippage_bps
        self.ttl_seconds = ttl_seconds
        self.poll_interval = poll_interval
        self._clock = clock
        self._sleep = sleep

    def execute_with_ttl(
        self, oms: OMS, symbol: str, side: str, target_price: float, qty: float
    ) -> ExecutionResult:
        side = side.upper()
        book = oms.get_orderbook(symbol)
        # Compare against the price we'd cross: best ask for a buy, best bid for a sell.
        if side == "BUY":
            reference = book["asks"][0][0]
            slip = slippage_bps(reference, target_price)
        elif side == "SELL":
            reference = book["bids"][0][0]
            slip = slippage_bps(target_price, reference)
        else:
            raise ValueError(f"invalid side {side!r}")

        if slip > self.max_slippage_bps:
            return ExecutionResult(
                filled=False,
                order_id=None,
                reason=f"slippage {slip:.2f}bps exceeds cap {self.max_slippage_bps}bps",
            )

        order_id = oms.place_limit_order(symbol, side, target_price, qty)
        start = self._clock()
        while self._clock() - start < self.ttl_seconds:
            if oms.get_order_status(order_id) == "FILLED":
                return ExecutionResult(True, order_id, "filled")
            self._sleep(self.poll_interval)

        oms.cancel_order(order_id)
        return ExecutionResult(False, order_id, "ttl_expired")
