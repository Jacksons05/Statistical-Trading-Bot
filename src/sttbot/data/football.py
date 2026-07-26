"""Loading and cleaning the open football dataset for Dixon-Coles work.

Turns the raw ``Matches.csv`` into tidy records with the fields the model and
the odds analytics actually need, using DuckDB to filter and project without
materialising all 230k rows in Python.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb

from .datasets import fetch


@dataclass(frozen=True)
class Match:
    date: dt.date
    division: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    # 1X2 decimal odds: a single book (Bet365) and the best-of-book maximum.
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    max_home: float | None = None
    max_draw: float | None = None
    max_away: float | None = None

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


_SELECT = """
SELECT
    MatchDate AS date, Division AS division,
    HomeTeam AS home, AwayTeam AS away,
    CAST(FTHome AS INTEGER) AS home_goals,
    CAST(FTAway AS INTEGER) AS away_goals,
    OddHome AS odds_home, OddDraw AS odds_draw, OddAway AS odds_away,
    MaxHome AS max_home, MaxDraw AS max_draw, MaxAway AS max_away
FROM read_csv_auto(?, header=true, sample_size=200000)
WHERE FTHome IS NOT NULL AND FTAway IS NOT NULL
"""


def load_matches(
    division: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    require_odds: bool = False,
    path: str | None = None,
) -> list[Match]:
    """Load matches, optionally filtered by division and date range.

    ``require_odds`` keeps only rows with a complete, valid (>1.0) 1X2 book,
    which is what the +EV and CLV analytics need.
    """
    csv_path = path or str(fetch("football_matches"))
    query = _SELECT
    params: list[object] = [csv_path]
    if division is not None:
        query += " AND Division = ?"
        params.append(division)
    if start is not None:
        query += " AND MatchDate >= ?"
        params.append(start)
    if end is not None:
        query += " AND MatchDate <= ?"
        params.append(end)
    if require_odds:
        query += " AND OddHome > 1 AND OddDraw > 1 AND OddAway > 1"
    query += " ORDER BY MatchDate"

    con = duckdb.connect()
    try:
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()
    return [Match(*row) for row in rows]


def overround(home: float, draw: float, away: float) -> float:
    """Sum of implied probabilities for a 1X2 book.

    Above 1.0 is the bookmaker's margin (the vig); below 1.0 across different
    books is a cross-book arbitrage.
    """
    for price in (home, draw, away):
        if price <= 1.0:
            raise ValueError("decimal odds must exceed 1.0")
    return 1.0 / home + 1.0 / draw + 1.0 / away


def devig(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """Strip the margin from a 1X2 book, returning probabilities summing to 1.

    Uses proportional (multiplicative) normalisation — the standard first-order
    approximation. It slightly overstates favourites relative to a
    power/Shin de-vig, which matters at long odds.
    """
    total = overround(home, draw, away)
    return (1.0 / home / total, 1.0 / draw / total, 1.0 / away / total)
