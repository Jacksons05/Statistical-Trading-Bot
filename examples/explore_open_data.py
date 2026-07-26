"""Profile the open football dataset and quantify the friction we must beat.

Downloads (once, into the cache) the openly-licensed match dataset and reports:
  1. coverage — rows, divisions, date span
  2. the empirical home advantage, i.e. the Dixon-Coles ``gamma``
  3. the low-score cell frequencies that motivate the ``tau`` correction
  4. bookmaker margin, single-book vs best-of-book, and how often the
     best-of-book 1X2 sum falls below 1.00 (a boundary arbitrage)

Run with:  python examples/explore_open_data.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import duckdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sttbot.data.datasets import describe, fetch  # noqa: E402


def main() -> None:
    print(describe("football_matches"), "\n")
    path = str(fetch("football_matches"))

    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW m AS SELECT * FROM "
        f"read_csv_auto('{path}', header=true, sample_size=200000)"
    )

    rows, lo, hi, divs, teams = con.execute(
        "SELECT COUNT(*), min(MatchDate), max(MatchDate), "
        "COUNT(DISTINCT Division), COUNT(DISTINCT HomeTeam) FROM m"
    ).fetchone()
    print(f"coverage: {rows:,} matches | {lo} -> {hi} | {divs} divisions | {teams} teams")

    home_goals, away_goals = con.execute(
        "SELECT AVG(FTHome), AVG(FTAway) FROM m "
        "WHERE FTHome IS NOT NULL AND FTAway IS NOT NULL"
    ).fetchone()
    print(
        f"\nhome advantage: {home_goals:.3f} vs {away_goals:.3f} goals "
        f"-> gamma ~ {math.log(home_goals / away_goals):.4f}"
    )

    print("\nlow-score cells (what the Dixon-Coles tau correction adjusts):")
    for h, a, n, pct in con.execute(
        "SELECT FTHome, FTAway, COUNT(*), "
        "ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2) "
        "FROM m WHERE FTHome<=1 AND FTAway<=1 GROUP BY 1,2 ORDER BY 1,2"
    ).fetchall():
        print(f"  {int(h)}-{int(a)}: {n:>6,}  ({pct}% of 0/1-goal games)")

    single, best, n = con.execute(
        "SELECT AVG(1/OddHome+1/OddDraw+1/OddAway), "
        "AVG(1/MaxHome+1/MaxDraw+1/MaxAway), COUNT(*) FROM m "
        "WHERE OddHome>1 AND OddDraw>1 AND OddAway>1 "
        "AND MaxHome>1 AND MaxDraw>1 AND MaxAway>1"
    ).fetchone()
    arb = con.execute(
        "SELECT COUNT(*) FROM m WHERE MaxHome>1 AND MaxDraw>1 AND MaxAway>1 "
        "AND (1/MaxHome+1/MaxDraw+1/MaxAway) < 1.0"
    ).fetchone()[0]

    print(f"\nfriction (n={n:,}):")
    print(f"  single-book (Bet365) margin : {100*(single-1):.2f}%")
    print(f"  best-of-book margin         : {100*(best-1):.2f}%")
    print(f"  recovered by line-shopping  : {100*(single-best):.2f} pp")
    print(f"  best-of-book sum < 1.00     : {arb:,} matches ({100*arb/n:.2f}%)")
    print(
        "\n  NOTE: those sub-1.00 sums are an upper bound, not realised profit -- the "
        "\n  maxima are peak prices across ~17 books that need simultaneous "
        "\n  availability, accounts everywhere, and survive stake limits."
    )
    con.close()


if __name__ == "__main__":
    main()
