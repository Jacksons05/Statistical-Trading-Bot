"""Demonstrate the prediction-market, AMM, and token-screening modules.

Run with:  python examples/crypto_and_prediction.py

Everything is computed from inline book/pool fixtures — no venue, key, or
network call is involved. The point is to show the three places where naive
implementations lose money:

  1. flat-rate fee assumptions on a venue whose fee is quadratic,
  2. arbitrage sized to price equality rather than to maximum profit,
  3. position sizing that ignores whether the exit can actually fill.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sttbot.strategies.amm import Pool, impermanent_loss, no_arb_band, optimal_arbitrage
from sttbot.strategies.token_screen import TokenMetadata, position_limit, screen
from sttbot.venues.prediction import (
    KALSHI_FEES,
    POLYMARKET_FEES,
    VenueQuote,
    find_cross_venue_arb,
    kelly_fraction,
)


def prediction_markets() -> None:
    print("=" * 68)
    print("1. CROSS-VENUE PREDICTION MARKETS")
    print("=" * 68)

    print("\nKalshi's fee is quadratic -- 0.07 * P * (1-P) -- not a flat rate:")
    for price in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"  contract at {price:.2f}: fee {KALSHI_FEES.cost(price):.4f}/contract")

    print("\nSo an identical 1c gross edge flips sign depending on where it sits:")
    for label, poly, kalshi in (
        ("mid-book", (0.47, 0.49), (0.50, 0.52)),
        ("tail", (0.03, 0.04), (0.05, 0.06)),
    ):
        arb = find_cross_venue_arb(
            VenueQuote("polymarket", "EVT", *poly, POLYMARKET_FEES, 5_000, 5_000),
            VenueQuote("kalshi", "EVT", *kalshi, KALSHI_FEES, 5_000, 5_000),
            equivalent=True,
        )
        if arb is None:
            print(f"  {label:9}: 1c gross -> no viable trade (fee exceeds edge)")
        else:
            print(
                f"  {label:9}: 1c gross -> net {arb.net_edge:.4f}/contract, "
                f"{arb.contracts:,.0f} contracts, ${arb.net_profit:,.2f}"
            )

    print("\n  Legs are only paired when declared equivalent:")
    try:
        find_cross_venue_arb(
            VenueQuote("polymarket", "EVT", 0.40, 0.42),
            VenueQuote("kalshi", "EVT", 0.50, 0.52),
        )
    except ValueError as exc:
        print(f"  refused -> {str(exc)[:60]}...")

    edge = kelly_fraction(probability=0.60, price=0.50)
    print(f"\n  Kelly at p=0.60 vs price 0.50: {edge:.1%} of bankroll")
    print(f"  quarter-Kelly (estimates, not truth): {edge / 4:.1%}")


def amm_arbitrage() -> None:
    print("\n" + "=" * 68)
    print("2. AMM / CEX ARBITRAGE")
    print("=" * 68)

    pool = Pool(token_reserve=1_000.0, numeraire_reserve=100_000.0, fee=0.003)
    low, high = no_arb_band(pool)
    print(f"\npool spot {pool.spot_price:.2f}, 30bp fee")
    print(f"no-arb band: [{low:.2f}, {high:.2f}] -- divergence inside is not tradeable")

    external = 130.0
    trade = optimal_arbitrage(pool, external, gas_cost=25.0)
    assert trade is not None
    print(f"\nexternal price {external:.2f} -> {trade.direction}")
    print(f"  optimal size   : {trade.size_in:,.2f} numeraire")
    print(f"  tokens out     : {trade.size_out:,.4f}")
    print(f"  gross / net    : {trade.gross_profit:,.2f} / {trade.net_profit:,.2f} (gas 25)")
    print(f"  pool price     : {trade.pool_price_before:.2f} -> {trade.pool_price_after:.2f}")

    # The common mistake: trade until the pool price equals the external price.
    equalising = (pool.k * external) ** 0.5 - pool.numeraire_reserve
    naive = pool.buy_token(equalising) * external - equalising
    print(f"\n  sizing to price equality instead: {equalising:,.2f} in")
    print(f"  -> gross {naive:,.2f} vs optimal {trade.gross_profit:,.2f} "
          f"({naive - trade.gross_profit:+,.2f})")

    capped = optimal_arbitrage(pool, external, gas_cost=25.0, max_capital=500.0)
    assert capped is not None
    print(f"\n  with only 500 capital: net {capped.net_profit:,.2f} "
          f"(the usual binding constraint)")

    print(f"\n  LP impermanent loss on a 2x move: {impermanent_loss(2.0):.2%}"
          f"  (a 0.5x move costs the same: {impermanent_loss(0.5):.2%})")


def token_screening() -> None:
    print("\n" + "=" * 68)
    print("3. MEME-COIN SCREENING AND EXIT SIZING")
    print("=" * 68)

    candidates = [
        TokenMetadata(
            symbol="SOLID", liquidity_usd=250_000, lp_locked_fraction=1.0,
            top_holder_fraction=0.04, top10_holder_fraction=0.16,
            sniper_bundled_fraction=0.02, holder_count=8_000, age_hours=2_000,
            buy_tax=0.0, sell_tax=0.0, can_mint=False, can_freeze=False,
            ownership_renounced=True, sell_simulation_passed=True,
        ),
        TokenMetadata(
            symbol="RUGME", liquidity_usd=8_000, lp_locked_fraction=0.0,
            top_holder_fraction=0.72, top10_holder_fraction=0.91,
            sniper_bundled_fraction=0.60, holder_count=41, age_hours=3,
            buy_tax=0.0, sell_tax=0.0, can_mint=True, can_freeze=True,
            ownership_renounced=False, sell_simulation_passed=True,
        ),
        TokenMetadata(
            symbol="TRAP", liquidity_usd=120_000, lp_locked_fraction=1.0,
            top_holder_fraction=0.05, top10_holder_fraction=0.12,
            sniper_bundled_fraction=0.04, holder_count=3_000, age_hours=300,
            buy_tax=0.01, sell_tax=0.09, can_mint=False, can_freeze=False,
            ownership_renounced=True, sell_simulation_passed=False,
        ),
        TokenMetadata(
            symbol="CARTEL", liquidity_usd=180_000, lp_locked_fraction=1.0,
            top_holder_fraction=0.05, top10_holder_fraction=0.50,
            sniper_bundled_fraction=0.05, holder_count=2_200, age_hours=400,
            buy_tax=0.0, sell_tax=0.0, can_mint=False, can_freeze=False,
            ownership_renounced=True, sell_simulation_passed=True,
        ),
        TokenMetadata(symbol="NODATA", liquidity_usd=500_000),
    ]
    print()
    for token in candidates:
        print(f"  {screen(token).summary()}")

    print("\nExit sizing against real depth (thin pool, 100k numeraire):")
    thin = Pool(token_reserve=1_000_000.0, numeraire_reserve=100_000.0, fee=0.003)
    for equity in (5_000.0, 1_000_000.0):
        limit = position_limit(thin, equity, max_price_impact=0.02)
        binding = "risk budget" if limit == equity * 0.02 else "pool depth"
        print(f"  equity ${equity:>10,.0f} -> max position ${limit:>9,.2f}  ({binding} binds)")
    print("\n  A position inside your risk budget is still untradeable if the")
    print("  pool cannot absorb the exit. Take the smaller of the two.")


if __name__ == "__main__":
    prediction_markets()
    amm_arbitrage()
    token_screening()
