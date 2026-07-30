"""Does the model's closing-line-value skill scale with market thinness?

An earlier pass over seven English and Scottish divisions found CLV skill only
in the Scottish lower leagues. That was a narrow read of a narrow sample: the
other five divisions were English tiers 1-3, which are among the most
efficiently priced football markets in the world. This script widens the test
to all 22 divisions football-data.co.uk publishes with both pre-match and
closing odds.

Pre-specified before looking at any out-of-sample result:

  H1      CLV skill increases with market thinness.
  Proxy   Mean single-book 1X2 overround. Computed from prices alone and
          entirely independent of the skill metric; a wider margin means a
          less competitively priced market.
  OOS set The 15 divisions not examined when the hypothesis was formed.
  Test    ONE Spearman correlation between overround and skill across those
          divisions -- not 15 separate significance tests.
  Metric  Excess CLV over the ODDS-MATCHED baseline, which controls for the
          model favouring a region of the price range.

Run with::

    python examples/clv_thin_markets.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy.stats import spearmanr

from sttbot.backtest.clv import clv_excess_per_bet
from sttbot.backtest.walk_forward import best_of_book_odds, walk_forward_backtest
from sttbot.data.football import load_football_data_uk, overround
from sttbot.economics.friction import FrictionModel

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425"]

# Divisions used to form the hypothesis. Reported for contrast, excluded from
# the test -- a hypothesis cannot be confirmed on the data that suggested it.
IN_SAMPLE = ["E0", "E1", "E2", "E3", "SC0", "SC2", "SC3"]

OUT_OF_SAMPLE = [
    "EC", "SC1", "D1", "D2", "I1", "I2", "SP1", "SP2",
    "F1", "F2", "N1", "B1", "P1", "T1", "G1",
]

BACKTEST_KWARGS = dict(
    train_window_days=730,
    min_train_matches=300,
    refit_every_days=7,
    friction=FrictionModel(taker_fee=0.02),
    min_net_edge=0.02,
    bankroll=1000.0,
    start=dt.date(2020, 8, 1),
)


def mean_overround(matches) -> float:
    """Thinness proxy. Prices only -- never touches CLV or results."""
    values = []
    for match in matches:
        if match.odds_home and match.odds_draw and match.odds_away:
            try:
                values.append(
                    overround(match.odds_home, match.odds_draw, match.odds_away)
                )
            except ValueError:
                pass
    return float(np.mean(values)) if values else float("nan")


def analyse(division: str) -> dict | None:
    matches = load_football_data_uk(SEASONS, division)
    result = walk_forward_backtest(
        matches, odds_selector=best_of_book_odds, **BACKTEST_KWARGS
    )
    bets = [b for b in result.bets if b.clv is not None]
    if len(bets) < 50:
        return None

    excess = clv_excess_per_bet(matches, bets, odds_selector=best_of_book_odds)
    if excess.size < 2:
        return None
    stderr = float(excess.std(ddof=1) / np.sqrt(excess.size))
    pnl = np.array([b.pnl for b in bets])

    return {
        "div": division,
        "bets": len(bets),
        "overround": mean_overround(matches),
        "roi": result.metrics.roi,
        "roi_t": float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(pnl.size))),
        "skill": float(excess.mean()),
        "skill_t": float(excess.mean() / stderr) if stderr else float("nan"),
    }


def report(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'div':<5}{'bets':>6}{'overround':>11}{'ROI':>9}{'ROI t':>7}"
          f"{'skill':>9}{'skill t':>9}")
    for row in sorted(rows, key=lambda r: -r["overround"]):
        print(
            f"{row['div']:<5}{row['bets']:>6}{row['overround']:>11.4f}"
            f"{row['roi']:>+9.2%}{row['roi_t']:>7.2f}"
            f"{row['skill']:>+9.4f}{row['skill_t']:>9.2f}"
        )


def main() -> None:
    in_sample = [r for r in (analyse(d) for d in IN_SAMPLE) if r]
    out_sample = [r for r in (analyse(d) for d in OUT_OF_SAMPLE) if r]

    report(in_sample, "IN-SAMPLE (used to form the hypothesis; not a test)")
    report(out_sample, "OUT-OF-SAMPLE (never examined before)")

    overrounds = [r["overround"] for r in out_sample]
    skills = [r["skill"] for r in out_sample]
    rho, p_value = spearmanr(overrounds, skills)

    print(
        f"\nH1 on {len(out_sample)} out-of-sample divisions: "
        f"Spearman rho = {rho:+.3f}, p = {p_value:.4f}"
    )
    print(
        f"Mean out-of-sample skill {np.mean(skills):+.5f} "
        f"(sd across divisions {np.std(skills, ddof=1):.5f}); "
        f"{sum(1 for r in out_sample if r['skill_t'] > 2)}/{len(out_sample)} "
        f"divisions individually significant at t > 2."
    )
    print(
        f"Profitable divisions: "
        f"{sum(1 for r in out_sample if r['roi'] > 0)}/{len(out_sample)}. "
        "Skill is not the same thing as profit -- the margin is larger."
    )


if __name__ == "__main__":
    main()
