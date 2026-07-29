"""Order Management System (OMS) interface and a simulated paper broker.

The real ecosystem plugs concrete adapters (``ib_insync`` for Interactive
Brokers, ``ccxt`` for crypto, custom REST clients for prediction markets)
behind the :class:`OMS` protocol. :class:`PaperBroker` implements the same
surface in-memory so strategies, the TTL order manager, and the circuit
breaker can be exercised deterministically in tests and paper trading.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

OrderStatus = str  # "OPEN" | "FILLED" | "CANCELLED"


@runtime_checkable
class OMS(Protocol):
    """Minimal execution surface every broker adapter must provide."""

    def get_orderbook(self, symbol: str) -> dict: ...
    def place_limit_order(self, symbol: str, side: str, price: float, qty: float) -> str: ...
    def get_order_status(self, order_id: str) -> OrderStatus: ...
    def cancel_order(self, order_id: str) -> None: ...
    def cancel_all_open_orders(self) -> None: ...
    def flatten_all_positions(self) -> None: ...
    def disable_trading_loop(self) -> None: ...


@dataclass
class _Order:
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    status: OrderStatus = "OPEN"


@dataclass
class PaperBroker:
    """In-memory broker with a static, per-symbol order book.

    A limit order fills immediately if it is marketable against the current
    book (buy price >= best ask, or sell price <= best bid); otherwise it rests
    as OPEN until cancelled or explicitly filled via :meth:`fill`.
    """

    books: dict[str, dict] = field(default_factory=dict)
    orders: dict[str, _Order] = field(default_factory=dict)
    positions: dict[str, float] = field(default_factory=dict)
    trading_enabled: bool = True
    _ids: "itertools.count" = field(default_factory=lambda: itertools.count(1))

    def set_orderbook(
        self, symbol: str, bids: list[list[float]], asks: list[list[float]]
    ) -> None:
        """Set a book as sorted ``[[price, size], ...]`` levels (best first)."""
        self.books[symbol] = {"bids": bids, "asks": asks}

    def get_orderbook(self, symbol: str) -> dict:
        if symbol not in self.books:
            raise KeyError(f"no orderbook for {symbol}")
        return self.books[symbol]

    def place_limit_order(self, symbol: str, side: str, price: float, qty: float) -> str:
        if not self.trading_enabled:
            raise RuntimeError("trading loop is disabled")
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"invalid side {side!r}")
        if qty <= 0:
            raise ValueError("qty must be positive")
        order = _Order(str(next(self._ids)), symbol, side, price, qty)
        book = self.books.get(symbol, {"bids": [], "asks": []})
        best_ask = book["asks"][0][0] if book["asks"] else None
        best_bid = book["bids"][0][0] if book["bids"] else None
        marketable = (side == "BUY" and best_ask is not None and price >= best_ask) or (
            side == "SELL" and best_bid is not None and price <= best_bid
        )
        if marketable:
            self._apply_fill(order)
        self.orders[order.order_id] = order
        return order.order_id

    def fill(self, order_id: str) -> None:
        """Force-fill a resting order (test/simulation hook)."""
        order = self.orders[order_id]
        if order.status == "OPEN":
            self._apply_fill(order)

    def _apply_fill(self, order: _Order) -> None:
        order.status = "FILLED"
        signed = order.qty if order.side == "BUY" else -order.qty
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + signed

    def get_order_status(self, order_id: str) -> OrderStatus:
        return self.orders[order_id].status

    def cancel_order(self, order_id: str) -> None:
        order = self.orders.get(order_id)
        if order is not None and order.status == "OPEN":
            order.status = "CANCELLED"

    def cancel_all_open_orders(self) -> None:
        for order in self.orders.values():
            if order.status == "OPEN":
                order.status = "CANCELLED"

    def flatten_all_positions(self) -> None:
        self.positions = {symbol: 0.0 for symbol in self.positions}

    def disable_trading_loop(self) -> None:
        self.trading_enabled = False
