"""Scalping Polymarket, where makers pay nothing.

Run with:  python examples/scalp_polymarket.py

Polymarket's 2026 schedule charges *takers* a quadratic fee and pays part of it
back to makers. A market maker is passive on both legs of a round trip, so it
pays zero — and with a rebate, is paid to provide liquidity. That removes the
constraint that dominates Kalshi and changes where the strategy can operate.

What replaces it: the tick. Once fees are gone the 1c grid is the floor on
what a round trip can earn, and adverse selection is the only remaining cost.
Offline throughout — no venue calls.
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
from sttbot.venues.prediction import (
    KALSHI_FEES,
    POLYMARKET_FEES,
    POLYMARKET_US_FEES,
    polymarket_fees,
)


def venue_comparison() -> None:
    print("=" * 72)
    print("1. WHAT ZERO MAKER FEES BUYS YOU")
    print("=" * 72)
    print("\nRound-trip cost for a MAKER (passive on both legs), cents/contract:")
    print(f"\n{'price':>7} {'Kalshi':>10} {'Polymarket':>12} {'Poly + rebate':>15}")
    rebate = polymarket_fees("politics", maker_rebate=0.0125)
    for p in (0.05, 0.15, 0.30, 0.50):
        print(
            f"{p:>7.2f} {breakeven_spread(p, KALSHI_FEES)*100:>9.2f}c "
            f"{breakeven_spread(p, POLYMARKET_FEES)*100:>11.2f}c "
            f"{breakeven_spread(p, rebate)*100:>14.2f}c"
        )

    print("\nBlocked region of the book at a 2c market spread:")
    for name, fees in (
        ("Kalshi", KALSHI_FEES),
        ("Polymarket", POLYMARKET_FEES),
        ("Polymarket + rebate", rebate),
    ):
        band = untradeable_band(fees, 0.02)
        text = "none - whole book workable" if band is None else \
            f"{band[0]:.2f}-{band[1]:.2f} ({band[1]-band[0]:.0%} blocked)"
        print(f"  {name:>20}: {text}")

    print("\n  Kalshi confines you to the tails. On Polymarket the entire book")
    print("  is workable, and with a rebate the round-trip cost is negative -")
    print("  you are paid to quote, before adverse selection.")

    print("\nTaker fees still exist and vary by category (makers pay none):")
    for cat in ("geopolitical", "sports", "politics", "economics", "crypto"):
        f = polymarket_fees(cat)
        print(f"  {cat:>13}: taker {f.cost(0.50)*100:>5.2f}c/contract at 50c, "
              f"maker {f.cost(0.50, maker=True)*100:.2f}c")
    print("\n  US exchange schedule: taker "
          f"{POLYMARKET_US_FEES.cost(0.50)*100:.2f}c, maker "
          f"{POLYMARKET_US_FEES.cost(0.50, maker=True)*100:+.2f}c (a rebate).")


def tick_is_the_new_floor() -> None:
    print("\n" + "=" * 72)
    print("2. THE TICK REPLACES THE FEE AS THE BINDING CONSTRAINT")
    print("=" * 72)
    mm = MarketMaker(
        QuoteParams(target_edge=0.0, min_half_spread=0.0, tick_size=0.01),
        fees=POLYMARKET_FEES,
    )
    quote = mm.quote(0.50)
    print(f"\n  Zero fees, zero target edge, 1c tick -> bid {quote.bid} / "
          f"ask {quote.ask}, spread {quote.spread:.2f}")
    print("  You cannot quote tighter than the grid, so 1-2c is the floor on")
    print("  what a round trip can earn no matter how cheap the venue is.")

    print("\n  Consequence: sub-cent noise is unmonetisable.")
    subtick = [0.50 + (0.003 if i % 2 else -0.003) for i in range(80)]
    print(f"    +/-0.3c wobble  -> {simulate(mm, subtick).num_fills} fills")
    excursion = [0.50, 0.51, 0.50, 0.49] * 20
    r = simulate(mm, excursion, settlement=0.50)
    print(f"    +/-1c excursion -> {r.num_fills} fills, net {r.net_pnl:+.3f}")


def what_is_left_to_lose_to() -> None:
    print("\n" + "=" * 72)
    print("3. WITH FEES GONE, ADVERSE SELECTION IS THE WHOLE GAME")
    print("=" * 72)
    mm = MarketMaker(
        QuoteParams(target_edge=0.01, risk_aversion=0.05, max_inventory=100.0),
        fees=POLYMARKET_FEES,
    )
    scenarios = [
        ("uninformed (flat)", random_walk(0.50, 500, 0.006, seed=33)),
        ("informed (trend)", trending_path(0.30, 500, 0.002, 0.006, seed=33)),
    ]
    print(f"\n  {'flow':>20} {'fills':>7} {'net P&L':>10} {'mean markout':>14} {'adverse %':>10}")
    for label, path in scenarios:
        r = simulate(mm, path, markout_steps=20)
        mo = "n/a" if r.mean_markout is None else f"{r.mean_markout:+.5f}"
        rate = "n/a" if r.adverse_fill_rate is None else f"{r.adverse_fill_rate:.0%}"
        print(f"  {label:>20} {r.num_fills:>7} {r.net_pnl:>+10.3f} {mo:>14} {rate:>10}")

    print("\n  Zero fees do not make a maker profitable; they only remove one")
    print("  cost. Mark-out is what decides it, and no fee schedule helps there.")
    print("\n  CAVEAT: fills assume queue priority and no partial fills, so these")
    print("  are upper bounds. On a zero-fee venue queue position is contested")
    print("  precisely because quoting is cheap for everyone else too.")


if __name__ == "__main__":
    venue_comparison()
    tick_is_the_new_floor()
    what_is_left_to_lose_to()
