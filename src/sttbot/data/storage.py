"""DuckDB columnar storage for tick / order-book data.

DuckDB is an embedded analytical database that reads and writes Parquet
natively, giving fast columnar queries over millions of rows with no server to
run. :class:`TickStore` wraps a connection with idempotent schema creation and
append/query helpers; pass ``":memory:"`` for ephemeral use in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import duckdb

_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    ts        TIMESTAMP,
    symbol    VARCHAR,
    bid       DOUBLE,
    ask       DOUBLE,
    volume    DOUBLE
)
"""

_COLUMNS = ("ts", "symbol", "bid", "ask", "volume")


@dataclass
class Tick:
    ts: Any  # datetime
    symbol: str
    bid: float
    ask: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class TickStore:
    def __init__(self, db_path: str = ":memory:", table: str = "market_ticks"):
        # Validate the table name up front: it is interpolated into DDL/DML and
        # must never carry untrusted input.
        if not table.isidentifier():
            raise ValueError(f"invalid table name {table!r}")
        self.table = table
        self.con = duckdb.connect(db_path)
        self.con.execute(_SCHEMA.format(table=table))

    def append(self, ticks: Sequence[Tick]) -> int:
        """Insert ticks; returns the number of rows written."""
        if not ticks:
            return 0
        rows = [(t.ts, t.symbol, t.bid, t.ask, t.volume) for t in ticks]
        placeholders = ", ".join(["(?, ?, ?, ?, ?)"] * len(rows))
        flat: list[Any] = [v for row in rows for v in row]
        self.con.execute(
            f"INSERT INTO {self.table} ({', '.join(_COLUMNS)}) VALUES {placeholders}",
            flat,
        )
        return len(rows)

    def count(self, symbol: str | None = None) -> int:
        if symbol is None:
            result = self.con.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()
        else:
            result = self.con.execute(
                f"SELECT COUNT(*) FROM {self.table} WHERE symbol = ?", [symbol]
            ).fetchone()
        return int(result[0])

    def latest(self, symbol: str) -> Tick | None:
        row = self.con.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM {self.table} "
            "WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            [symbol],
        ).fetchone()
        return Tick(*row) if row else None

    def to_parquet(self, path: str, symbol: str | None = None) -> None:
        """Export the store (optionally one symbol) to a Parquet file."""
        query = f"SELECT * FROM {self.table}"
        params: list[Any] = []
        if symbol is not None:
            query += " WHERE symbol = ?"
            params.append(symbol)
        self.con.execute(
            f"COPY ({query}) TO '{path}' (FORMAT PARQUET)", params
        )

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "TickStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
