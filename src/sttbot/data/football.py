"""Loading and cleaning the open football dataset for Dixon-Coles work.

Turns the raw ``Matches.csv`` into tidy records with the fields the model and
the odds analytics actually need, using DuckDB to filter and project without
materialising all 230k rows in Python.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb

from .datasets import fetch, fetch_football_data_uk


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
    # Closing 1X2 odds, populated only by the football-data.co.uk loader. The
    # bulk football_matches dataset keeps a single odds snapshot per match, so
    # these stay None there and CLV cannot be computed from it.
    close_home: float | None = None
    close_draw: float | None = None
    close_away: float | None = None

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"

    @property
    def has_closing_odds(self) -> bool:
        return None not in (self.close_home, self.close_draw, self.close_away)


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


# --- football-data.co.uk (carries closing odds) -----------------------------
#
# Entry line = Bet365 pre-match (B365H/D/A), best-of-book pre-match = MaxH/D/A.
# Closing line prefers Pinnacle (PSCH/PSCD/PSCA) because it is the sharpest
# widely-published close; Bet365's close is the fallback when Pinnacle is
# absent. Column availability varies by season, so every odds column is read
# through COALESCE against a NULL literal via try-select below.

# Read Date as text: files before ~2018 use DD/MM/YY and later ones DD/MM/YYYY,
# and DuckDB's auto-detection silently misreads the two-digit form (14/08/10 in
# the 2010/11 file came back as 2001-01-11). Walk-forward ordering depends on
# these dates, so parse both formats explicitly instead.
_FDUK_READ = (
    "read_csv_auto(?, header=true, sample_size=-1, ignore_errors=true, "
    "types={'Date': 'VARCHAR'})"
)

# Pick the format by string length rather than trying one then the other:
# '%d/%m/%Y' happily parses "14/08/10" as the year 10 AD, so an ordered
# COALESCE silently produces dates two millennia off.
_FDUK_DATE = (
    'CAST(CASE WHEN length("Date") = 10 '
    "THEN try_strptime(\"Date\", '%d/%m/%Y') "
    "ELSE try_strptime(\"Date\", '%d/%m/%y') END AS DATE)"
)

# Logical field -> candidate source columns, in preference order. Seasons differ
# in which books they carry, so each field resolves against whatever the file
# actually has and falls back to NULL when none are present.
_FDUK_ODDS_COLUMNS: dict[str, tuple[str, ...]] = {
    "odds_home": ("B365H", "AvgH"),
    "odds_draw": ("B365D", "AvgD"),
    "odds_away": ("B365A", "AvgA"),
    "max_home": ("MaxH", "B365H"),
    "max_draw": ("MaxD", "B365D"),
    "max_away": ("MaxA", "B365A"),
    "close_home": ("PSCH", "B365CH", "AvgCH"),
    "close_draw": ("PSCD", "B365CD", "AvgCD"),
    "close_away": ("PSCA", "B365CA", "AvgCA"),
}


def _fduk_expression(field: str, present: set[str]) -> str:
    """COALESCE over whichever candidate columns this file actually has.

    Returns a bare expression with no alias, so it can be reused verbatim in a
    WHERE clause.
    """
    candidates = [f'"{c}"' for c in _FDUK_ODDS_COLUMNS[field] if c in present]
    if not candidates:
        return "CAST(NULL AS DOUBLE)"
    if len(candidates) == 1:
        return candidates[0]
    return f"COALESCE({', '.join(candidates)})"


def _fduk_query(con: duckdb.DuckDBPyConnection, path: str, require_closing: bool) -> str:
    present = {
        row[0]
        for row in con.execute(f"DESCRIBE SELECT * FROM {_FDUK_READ}", [path]).fetchall()
    }
    projections = ",\n    ".join(
        f"{_fduk_expression(field, present)} AS {field}"
        for field in _FDUK_ODDS_COLUMNS
    )
    conditions = [
        '"FTHG" IS NOT NULL',
        '"FTAG" IS NOT NULL',
        '"HomeTeam" IS NOT NULL',
        f"{_FDUK_DATE} IS NOT NULL",
    ]
    if require_closing:
        conditions += [
            f"{_fduk_expression(field, present)} > 1"
            for field in ("close_home", "close_draw", "close_away")
        ]
    return f"""
SELECT
    {_FDUK_DATE} AS date,
    "Div" AS division,
    "HomeTeam" AS home,
    "AwayTeam" AS away,
    CAST("FTHG" AS INTEGER) AS home_goals,
    CAST("FTAG" AS INTEGER) AS away_goals,
    {projections}
FROM {_FDUK_READ}
WHERE {' AND '.join(conditions)}
ORDER BY date
"""


def load_football_data_uk(
    seasons: list[str],
    division: str,
    *,
    require_closing: bool = False,
    paths: list[str] | None = None,
) -> list[Match]:
    """Load football-data.co.uk matches for ``seasons`` in one ``division``.

    Seasons are four-digit codes (``'2324'``). Unlike :func:`load_matches`,
    these rows carry a genuine closing line, which is what CLV needs; pass
    ``require_closing=True`` to keep only rows that have one. Closing columns
    only exist from season 2019/20 onward, so earlier seasons yield rows with
    ``close_*`` set to None.
    """
    if paths is None:
        paths = [str(fetch_football_data_uk(s, division)) for s in seasons]

    matches: list[Match] = []
    con = duckdb.connect()
    try:
        for path in paths:
            query = _fduk_query(con, path, require_closing)
            matches.extend(Match(*row) for row in con.execute(query, [path]).fetchall())
    finally:
        con.close()
    matches.sort(key=lambda m: m.date)
    return matches


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
