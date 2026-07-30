"""Build a breadth book over fresh Solana tokens, and price what it requires.

Runs the whole pipeline: discover, screen, size against real pool depth, and
report the orders that would be placed. It then states plainly what the world
has to look like for the book to make money.

Two tensions this surfaces rather than hides:

**The screen and the strategy disagree.** ``token_screen`` defaults to a 24-hour
minimum age, because ~69% of pump.fun launches never trade past their first
day. But the only cohort with real flow is *under* a day old -- established
tokens turn over 0.00x. So a breadth book on the liquid cohort means explicitly
lowering that floor and accepting the risk the screen exists to avoid, managed
by size instead of by selection. The checks that survive are the ones size
cannot protect you from: live mint authority, a freeze authority, and top-10
concentration.

**Execution is not implemented.** This prints intended orders. Placing them
needs a funded wallet, signing, slippage-bounded routing and MEV protection,
none of which is here, and all of which makes real fills worse than modelled.

Run with::

    python examples/breadth_portfolio.py
"""

from __future__ import annotations

import time

from sttbot.data.tokens import (
    goplus_security,
    latest_token_addresses,
    pairs_for_tokens,
    to_token_metadata,
)
from sttbot.strategies.breadth import (
    BreadthParams,
    Candidate,
    allocate,
    breakeven_multiple,
    names_for_confidence,
    probability_of_no_winner,
    round_trip_cost,
    simulate,
)
from sttbot.strategies.token_screen import ScreenThresholds, screen

BANKROLL = 10_000.0

# Sized for width, not conviction: 0.5% a name over 100 names puts half the
# bankroll to work and survives a long run of total losses.
PARAMS = BreadthParams(
    bankroll=BANKROLL,
    max_positions=100,
    risk_per_position=0.005,
    max_deployed_fraction=0.50,
    max_price_impact=0.02,
    min_position_usd=25.0,
    max_round_trip_cost=0.10,
)

# The age floor is the deliberate concession described above. Concentration and
# authority limits stay strict, because no position size protects you from a
# mint authority.
BREADTH_THRESHOLDS = ScreenThresholds(
    min_liquidity_usd=8_000.0,
    min_age_hours=1.0,
    min_holder_count=50,
    max_top_holder_fraction=0.15,
    max_top10_holder_fraction=0.30,
)

# Base rate for the assumed hit rate. Roughly 3% of these tokens survive in any
# meaningful sense; this is a literature figure, not something measured here.
ASSUMED_WIN_RATE = 0.03


def main() -> None:
    now_ms = time.time() * 1000.0

    print("discovering ...", flush=True)
    addresses = latest_token_addresses(chain="solana")
    pairs = pairs_for_tokens(addresses)
    print(f"  {len(pairs)} tokens with a tradeable pool", flush=True)
    if not pairs:
        print("nothing discovered; stopping")
        return

    print("fetching contract safety ...", flush=True)
    security = goplus_security(list(pairs))

    candidates, blocked = [], []
    for address, pair in pairs.items():
        meta = to_token_metadata(pair, security.get(address), now_ms=now_ms)
        result = screen(meta, BREADTH_THRESHOLDS)
        pool = pair.pool()
        if pool is None:
            blocked.append((meta.symbol, "no pool reserves"))
            continue
        # Unknowns are tolerated here and failures are not: at this size the
        # data gaps are unavoidable, but a check that actually ran and lost is
        # a verdict.
        if result.failures:
            blocked.append((meta.symbol, result.failures[0]))
            continue
        candidates.append(Candidate(meta.symbol, pool))

    print(f"  {len(candidates)} passed, {len(blocked)} blocked\n")

    book = allocate(candidates, PARAMS)
    print("=" * 72)
    print("BOOK")
    print("=" * 72)
    print("  " + book.summary())
    if book.allocations:
        print(f"\n  {'symbol':<14}{'size $':>9}{'round trip':>12}  capped by")
        for alloc in book.allocations[:20]:
            print(f"  {alloc.symbol[:13]:<14}{alloc.size_usd:>9,.0f}"
                  f"{alloc.round_trip_cost:>12.2%}  {alloc.capped_by}")
    for symbol, reason in book.rejected[:8]:
        print(f"  rejected {symbol[:12]:<14} {reason}")

    cost = book.mean_cost if book.allocations else 0.04
    print("\n" + "=" * 72)
    print("WHAT THIS REQUIRES  (arithmetic, not a forecast)")
    print("=" * 72)
    print(f"  assumed hit rate                {ASSUMED_WIN_RATE:.0%}")
    print(f"  mean round-trip cost            {cost:.2%}")
    print(f"  winners must return             "
          f"{breakeven_multiple(ASSUMED_WIN_RATE, cost):.1f}x just to break even")
    print(f"  names for 95% chance of one     "
          f"{names_for_confidence(ASSUMED_WIN_RATE):,}")
    held = len(book.allocations)
    if held:
        print(f"  this book holds {held}, so it misses entirely "
              f"{probability_of_no_winner(ASSUMED_WIN_RATE, held):.0%} of the time")

    print("\n" + "=" * 72)
    print("SIMULATION  (assumption-driven; NOT a backtest)")
    print("=" * 72)
    for alpha, label in ((0.8, "fat tail   (alpha 0.8)"),
                         (1.5, "medium     (alpha 1.5)"),
                         (3.0, "thin tail  (alpha 3.0)")):
        result = simulate(
            PARAMS, win_rate=ASSUMED_WIN_RATE, cost_fraction=cost,
            rounds=12, trials=1_500, pareto_alpha=alpha, seed=11,
        )
        print(f"  {label}  median ${result.median:>10,.0f}   "
              f"mean ${result.mean:>12,.0f}   "
              f"P(loss) {result.probability_of_loss:>5.0%}   "
              f"P(ruin) {result.probability_of_ruin:>5.0%}")

    print(
        "\n  The payoff tail is assumed, not measured, and it decides the answer.\n"
        "  Nothing in this repository establishes which alpha is real, so treat\n"
        "  these as the consequences of three guesses. Measuring the realised\n"
        "  distribution of token returns is the work that would make this a\n"
        "  strategy rather than a structure."
    )


if __name__ == "__main__":
    main()
