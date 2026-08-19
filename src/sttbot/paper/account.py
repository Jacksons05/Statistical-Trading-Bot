"""A durable paper-trading account.

The design rule here is that **fills are the only state**. Cash, positions,
average cost and realised P&L are all recomputed from the fill ledger rather
than stored and mutated alongside it. That is slower and entirely worth it: a
long-running bot that maintains a running cash balance *and* a fill log will
eventually disagree with itself after a crash mid-write, and the disagreement
is invisible until the numbers are already wrong. With one source of truth
there is nothing to reconcile.

Every fill carries a ``ref``, a caller-supplied idempotency key. A runner that
dies after placing a trade but before recording it will retry on restart, and
the retry must not double-book. Recording an existing ref is a no-op that
reports itself rather than raising, because the retry path is the normal path.

Shorts are supported and signed consistently: a position's ``qty`` is negative
when short, ``avg_cost`` is always the positive average price paid or received,
and realised P&L accrues only when a position is reduced or closed.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import duckdb

BUY = "BUY"
SELL = "SELL"


class AccountBusy(RuntimeError):
    """Another process holds the account's write lock.

    DuckDB allows a single writer, so a scheduled tick and a human inspecting
    the book cannot both hold the file. That is a routine collision rather than
    a fault, and callers should skip the tick rather than crash -- a stack
    trace in a cron log looks identical to a real failure and buries the ones
    that matter.
    """


@dataclass(frozen=True)
class Fill:
    """One executed trade. The unit of truth."""

    ts: dt.datetime
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    strategy: str = ""
    # Idempotency key. Two fills with the same ref are the same event.
    ref: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be {BUY} or {SELL}, got {self.side!r}")
        if self.qty <= 0:
            raise ValueError("qty must be positive; direction comes from side")
        if self.price < 0:
            raise ValueError("price must be non-negative")
        if self.fee < 0:
            raise ValueError("fee must be non-negative")

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == BUY else -self.qty

    @property
    def cash_delta(self) -> float:
        """Effect on cash: buying spends, selling receives, fees always cost."""
        return -self.signed_qty * self.price - self.fee


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12

    def market_value(self, mark: float) -> float:
        return self.qty * mark

    def unrealized_pnl(self, mark: float) -> float:
        return self.qty * (mark - self.avg_cost)


@dataclass(frozen=True)
class Snapshot:
    """Account state at one moment, marked against supplied prices."""

    ts: dt.datetime
    cash: float
    positions: dict[str, Position]
    marks: dict[str, float]
    starting_cash: float

    @property
    def market_value(self) -> float:
        return sum(
            p.market_value(self.marks[s])
            for s, p in self.positions.items()
            if s in self.marks
        )

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            p.unrealized_pnl(self.marks[s])
            for s, p in self.positions.items()
            if s in self.marks
        )

    @property
    def total_return(self) -> float:
        return self.equity / self.starting_cash - 1.0 if self.starting_cash else 0.0

    @property
    def unmarked(self) -> tuple[str, ...]:
        """Open positions with no price supplied.

        These are silently excluded from equity, which is exactly the kind of
        omission that flatters a book -- a position nobody will quote is
        usually a position that has gone to nothing. Callers should surface
        this rather than ignore it.
        """
        return tuple(
            s for s, p in self.positions.items()
            if not p.is_flat and s not in self.marks
        )


class PaperAccount:
    """Durable paper account backed by DuckDB.

    Use ``":memory:"`` for tests; a filesystem path to survive restarts.
    """

    def __init__(
        self,
        path: str = ":memory:",
        starting_cash: float = 10_000.0,
        busy_timeout: float = 0.0,
    ):
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.path = path
        self._con = self._connect(path, busy_timeout)
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                ts TIMESTAMP,
                symbol VARCHAR,
                side VARCHAR,
                qty DOUBLE,
                price DOUBLE,
                fee DOUBLE,
                strategy VARCHAR,
                ref VARCHAR,
                note VARCHAR,
                seq BIGINT
            )
            """
        )
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS meta (key VARCHAR, value DOUBLE)"
        )
        existing = self._con.execute(
            "SELECT value FROM meta WHERE key = 'starting_cash'"
        ).fetchone()
        if existing is None:
            self._con.execute(
                "INSERT INTO meta VALUES ('starting_cash', ?)", [starting_cash]
            )
            self.starting_cash = starting_cash
        else:
            # An existing book keeps its original capital base. Silently
            # adopting a new one would rewrite the return series.
            self.starting_cash = float(existing[0])

    @staticmethod
    def _connect(path: str, busy_timeout: float):
        """Open the store, waiting out a transient writer if asked.

        Raises :class:`AccountBusy` rather than DuckDB's IOException so callers
        can distinguish "someone else has it" from a corrupt or missing file.
        """
        deadline = time.monotonic() + max(busy_timeout, 0.0)
        while True:
            try:
                return duckdb.connect(path)
            except duckdb.IOException as exc:
                # Match the contention message specifically. A looser test on
                # "lock" also catches "Could not set lock on file ...: No such
                # file or directory", so a genuinely broken path would be
                # reported as busy and every scheduled tick would skip in
                # silence rather than fail.
                if "conflicting lock" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise AccountBusy(
                        f"{path} is locked by another process"
                    ) from exc
                time.sleep(0.5)

    # --- writing ---------------------------------------------------------

    def record(self, fill: Fill) -> bool:
        """Append a fill. Returns ``False`` if the ref was already recorded."""
        if fill.ref and self._has_ref(fill.ref):
            return False
        seq = self._next_seq()
        self._con.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [fill.ts, fill.symbol, fill.side, fill.qty, fill.price, fill.fee,
             fill.strategy, fill.ref, fill.note, seq],
        )
        return True

    def record_all(self, fills: Iterable[Fill]) -> int:
        """Append many fills, skipping any whose ref is already present."""
        return sum(1 for f in fills if self.record(f))

    def _has_ref(self, ref: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM fills WHERE ref = ? LIMIT 1", [ref]
        ).fetchone()
        return row is not None

    def _next_seq(self) -> int:
        row = self._con.execute("SELECT COALESCE(MAX(seq), 0) FROM fills").fetchone()
        return int(row[0]) + 1

    # --- reading ---------------------------------------------------------

    def fills(self) -> list[Fill]:
        rows = self._con.execute(
            "SELECT ts, symbol, side, qty, price, fee, strategy, ref, note "
            "FROM fills ORDER BY seq"
        ).fetchall()
        return [Fill(*row) for row in rows]

    def positions(self) -> dict[str, Position]:
        """Replay the ledger into positions, average cost and realised P&L."""
        book: dict[str, Position] = {}
        for fill in self.fills():
            position = book.setdefault(fill.symbol, Position(fill.symbol))
            _apply(position, fill)
        return book

    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions().items() if not p.is_flat}

    @property
    def cash(self) -> float:
        return self.starting_cash + sum(f.cash_delta for f in self.fills())

    def snapshot(
        self, marks: dict[str, float] | None = None,
        *, ts: dt.datetime | None = None,
    ) -> Snapshot:
        return Snapshot(
            ts=ts or dt.datetime.now(dt.timezone.utc),
            cash=self.cash,
            positions=self.positions(),
            marks=dict(marks or {}),
            starting_cash=self.starting_cash,
        )

    def close(self) -> None:
        self._con.close()


def _apply(position: Position, fill: Fill) -> None:
    """Fold one fill into a position, accruing realised P&L on reductions.

    Handles the three cases that a naive average-cost update gets wrong:
    adding to a position, partially closing it, and reversing through zero in
    a single fill (where the closed portion realises and only the remainder
    opens at the new price).
    """
    incoming = fill.signed_qty
    existing = position.qty

    # Fees are a cost the moment they are paid, regardless of what the trade
    # does to the position.
    position.realized_pnl -= fill.fee

    if existing == 0 or (existing > 0) == (incoming > 0):
        # Opening or adding: weighted-average the cost basis.
        total = existing + incoming
        if total != 0:
            position.avg_cost = (
                abs(existing) * position.avg_cost + abs(incoming) * fill.price
            ) / (abs(existing) + abs(incoming))
        position.qty = total
        return

    # Reducing, closing, or reversing.
    closing = min(abs(incoming), abs(existing))
    direction = 1.0 if existing > 0 else -1.0
    position.realized_pnl += closing * (fill.price - position.avg_cost) * direction

    remaining = abs(incoming) - closing
    position.qty = existing + incoming
    if remaining > 1e-12:
        # Reversed through zero: the remainder is a new position at this price.
        position.avg_cost = fill.price
    elif abs(position.qty) < 1e-12:
        position.qty = 0.0
        position.avg_cost = 0.0
