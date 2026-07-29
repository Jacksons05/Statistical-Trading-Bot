"""Walk-forward Dixon-Coles backtest on real football odds, with a CLV control.

Downloads football-data.co.uk season CSVs (cached under ``STTBOT_DATA_DIR``),
fits the model on a trailing window, bets the next matchday, and reports P&L
after friction alongside closing-line value.

Crucially it also reports what CLV a *random* selection would have earned on
the same matches at the same entry prices. Without that control a positive CLV
number means nothing, because best-of-book entry prices beat a single book's
close most of the time on their own.

Run with::

    python examples/backtest_dixon_coles.py

Requires network access on the first run; afterwards it reads from cache.
"""

from __future__ import annotations

import datetime as dt

from sttbot.backtest.clv import clv_baseline, clv_skill
from sttbot.backtest.walk_forward import (
    best_of_book_odds,
    single_book_odds,
    walk_forward_backtest,
)
from sttbot.data.football import load_football_data_uk
from sttbot.economics.friction import FrictionModel

# Closing odds exist from 2019/20 onward; 2018/19 is included so the first
# predicted season already has a full training window behind it.
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425"]

# England tiers 1-4 and Scotland's top and bottom tiers. The premise worth
# testing is that thinner, less-watched markets are less efficient.
DIVISIONS = ["E0", "E1", "E2", "E3", "SC0", "SC2", "SC3"]

# 2% commission, in the region an exchange charges on net winnings.
FRICTION = FrictionModel(taker_fee=0.02)

BACKTEST_KWARGS = dict(
    train_window_days=730,
    min_train_matches=300,
    refit_every_days=7,
    friction=FRICTION,
    min_net_edge=0.02,
    bankroll=1000.0,
    start=dt.date(2020, 8, 1),
)

ENTRY_RULES = (
    ("single-book", single_book_odds),
    ("best-of-book", best_of_book_odds),
)


def main() -> None:
    header = (
        f"{'div':<5}{'entry':<14}{'bets':>6}{'ROI':>9}{'hit':>8}"
        f"{'Sharpe':>8}{'maxDD':>8}{'CLV':>10}{'random':>10}{'skill':>10}"
    )
    print(header)
    print("-" * len(header))

    for division in DIVISIONS:
        try:
            matches = load_football_data_uk(SEASONS, division)
        except OSError as exc:
            print(f"{division:<5}download failed: {exc}")
            continue

        for label, selector in ENTRY_RULES:
            result = walk_forward_backtest(
                matches, odds_selector=selector, **BACKTEST_KWARGS
            )
            metrics = result.metrics
            if not metrics.n_bets:
                print(f"{division:<5}{label:<14}{'no bets':>6}")
                continue

            # The control: same matches, same entry prices, random selections.
            baseline = clv_baseline(matches, odds_selector=selector, seed=0)
            skill = clv_skill(metrics.mean_clv, baseline)

            print(
                f"{division:<5}{label:<14}{metrics.n_bets:>6}"
                f"{metrics.roi:>+9.2%}{metrics.hit_rate:>8.1%}"
                f"{metrics.sharpe:>8.2f}{metrics.max_drawdown:>8.1%}"
                f"{metrics.mean_clv:>+10.4f}{baseline.mean_clv:>+10.4f}"
                f"{skill:>+10.4f}"
            )

    print(
        "\n'CLV' is the strategy's mean closing-line value, 'random' is what an\n"
        "unskilled random selection earned on the same matches at the same entry\n"
        "prices, and 'skill' is the difference. Only 'skill' reflects the model."
    )


if __name__ == "__main__":
    main()
