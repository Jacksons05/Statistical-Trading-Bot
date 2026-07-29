"""Market-making (scalping) on binary prediction markets.

Run with:  python examples/scalp_prediction_markets.py

Offline: every price path is simulated. Shows the three things that decide
whether prediction-market scalping works, in the order they bind:

  1. fees decide *where* in the book you can quote at all,
  2. inventory skew must be scaled to the spread or it unwinds you at a loss,
  3. adverse selection decides whether the captured spread is actually yours.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sttbot.backtest.mm_simulator import random_walk, simulate, trending_path
from sttbot.strategies.market_making import (
    MarketMaker,
    QuoteParams,
    breakeven_spread,
    untradeable_band,
)
from sttbot.venues.prediction import KALSHI_FEES, POLYMARKET_FEES, FeeModel


def where_can_you_quote() -> None:
    print("=" * 70)
    print("1. WHERE YOU CAN QUOTE AT ALL (Kalshi's quadratic fee)")
    print("=" * 70)
    print(f"\n{'price':>7} {'fee/side':>9} {'round trip':>11} {'2c mkt?':>9} {'4c mkt?':>9}")
    for p in (0.02, 0.05, 0.10, 0.25, 0.50):
        side = KALSHI_FEES.cost(p)
        rt = breakeven_spread(p, KALSHI_FEES)
        print(f"{p:>7.2f} {side*100:>8.2f}c {rt*100:>10.2f}c "
              f"{'yes' if rt < 0.02 else 'NO':>9} {'yes' if rt < 0.04 else 'NO':>9}")

    print()
    for spread in (0.01, 0.02, 0.04, 0.08):
        band = untradeable_band(KALSHI_FEES, spread)
        if band is None:
            text = "none - whole book workable"
        else:
            text = f"{band[0]:.2f}-{band[1]:.2f} blocked ({band[1]-band[0]:.0%} of the book)"
        print(f"  {spread*100:>2.0f}c market spread -> {text}")
    print("\n  Mid-book needs >3.5c of spread before any profit. That is the")
    print("  constraint: on Kalshi you scalp the tails, not the coin flips.")


def inventory_skew_scaling() -> None:
    print("\n" + "=" * 70)
    print("2. INVENTORY SKEW MUST BE SCALED TO THE SPREAD")
    print("=" * 70)
    path = [0.50 + (0.003 if i % 2 else -0.003) for i in range(120)]
    print("\n  Jitter of +/-0.3c around a stable 50c fair value, 0.5c half-spread.")
    print(f"\n  {'risk_aversion':>14} {'skew @ full inv':>16} {'net P&L':>10} {'fills':>7}")
    for gamma in (0.01, 0.05, 0.25, 1.00):
        mm = MarketMaker(
            QuoteParams(target_edge=0.01, risk_aversion=gamma, max_inventory=100.0),
            fees=FeeModel(),
        )
        result = simulate(mm, path, settlement=0.50)
        print(f"  {gamma:>14.2f} {gamma*0.25*100:>15.2f}c "
              f"{result.net_pnl:>+10.3f} {result.num_fills:>7}")
    print("\n  At gamma=1.0 the skew is 25c at full inventory against a 1c spread:")
    print("  the maker buys back its own position far above fair value. Keep")
    print("  gamma*0.25 within a few half-spreads.")


def adverse_selection() -> None:
    print("\n" + "=" * 70)
    print("3. ADVERSE SELECTION IS THE REAL COST")
    print("=" * 70)
    mm = MarketMaker(
        QuoteParams(target_edge=0.01, risk_aversion=0.05, max_inventory=100.0),
        fees=FeeModel(),
    )
    scenarios = [
        ("uninformed (flat)", random_walk(0.50, 400, 0.004, seed=21)),
        ("informed (trend)", trending_path(0.30, 400, 0.002, 0.004, seed=21)),
    ]
    print(f"\n  {'flow':>20} {'fills':>7} {'net P&L':>10} {'mean markout':>14} {'adverse %':>10}")
    for label, path in scenarios:
        r = simulate(mm, path, markout_steps=20)
        markout = "n/a" if r.mean_markout is None else f"{r.mean_markout:+.5f}"
        rate = "n/a" if r.adverse_fill_rate is None else f"{r.adverse_fill_rate:.0%}"
        print(f"  {label:>20} {r.num_fills:>7} {r.net_pnl:>+10.3f} "
              f"{markout:>14} {rate:>10}")
    print("\n  Mark-out is the diagnostic that P&L alone hides: it is the price")
    print("  move against you *after* each fill. Negative means you are being")
    print("  picked off, and it shows up before the P&L does.")
    print("\n  CAVEAT: fills here assume queue priority and no partial fills, so")
    print("  these numbers are an upper bound. Real quoting is worse.")


if __name__ == "__main__":
    where_can_you_quote()
    inventory_skew_scaling()
    adverse_selection()
