"""Is there an "established but thin" sweet spot in Solana tokens?

Fresh launches capped out at a ~$300 exit, which kills the strategy on capacity
before edge is ever the question. The obvious next hypothesis is that tokens
which survived their first month are still small enough to be inefficient but
big enough to trade. This tests that.

Capacity alone is the wrong measure, because liquidity with no flow is stranded
capital rather than an opportunity. So each token is scored on two axes:

* **exit capacity** -- the position unwindable inside a 2% price-impact budget,
  solved against the real constant-product pool.
* **turnover** -- 24h volume divided by pool liquidity, i.e. whether anyone is
  actually trading it.

An inefficiency worth scalping needs both. Run with::

    python examples/token_capacity_cohorts.py

Discovery combines GeckoTerminal's volume-ranked pools with a DexScreener term
sweep, because DexScreener's profile feed is newest-first and structurally
cannot show established tokens. Neither is exhaustive: this is a sample of
actively traded pools, not a census.
"""

from __future__ import annotations

import statistics as stats
import time

from sttbot.data.tokens import (
    geckoterminal_top_tokens,
    pairs_for_tokens,
    search_pairs,
)
from sttbot.strategies.token_screen import max_exit_size

IMPACT_BUDGET = 0.02
TERMS = [
    "SOL", "USDC", "BONK", "WIF", "pump", "dog", "cat", "moon", "inu", "pepe",
    "AI", "meme", "baby", "shib", "frog", "gold", "king", "chad", "gork",
]
COHORTS = [
    (0, 1, "< 1 day"),
    (1, 7, "1-7 days"),
    (7, 30, "7-30 days"),
    (30, 180, "1-6 months"),
    (180, 1e9, "> 6 months"),
]
# "Thin" in the sense the thesis means: too small for serious capital.
THIN_BAND = (25_000.0, 1_000_000.0)
# What a strategy would actually need to be worth running.
USEFUL_EXIT = 5_000.0
USEFUL_TURNOVER = 0.5


def exit_capacity(pair) -> float | None:
    """Unwindable position in dollars, against real pool depth."""
    pool = pair.pool()
    if pool is None:
        return None
    return max_exit_size(pool, max_price_impact=IMPACT_BUDGET) * pool.spot_price


def main() -> None:
    now_ms = time.time() * 1000.0

    print("discovering established tokens (paced for rate limits) ...", flush=True)
    addresses = set(geckoterminal_top_tokens(network="solana"))
    print(f"  {len(addresses)} from volume-ranked pools", flush=True)
    for term in TERMS:
        for pair in search_pairs(term):
            if pair.chain == "solana" and pair.token_address:
                addresses.add(pair.token_address)
        time.sleep(0.4)
    print(f"  {len(addresses)} after the term sweep", flush=True)

    pairs = pairs_for_tokens(sorted(addresses))
    rows = []
    for pair in pairs.values():
        age = pair.age_hours(now_ms)
        capacity = exit_capacity(pair)
        liquidity = pair.liquidity_usd or 0.0
        if age is None or capacity is None or liquidity <= 0:
            continue
        rows.append((pair, age / 24.0, capacity, (pair.volume_24h or 0.0) / liquidity))
    print(f"  {len(rows)} with a usable pool\n")

    print("=" * 74)
    print("CAPACITY BY AGE  (median exit inside a 2% impact budget)")
    print("=" * 74)
    print(f"{'cohort':<14}{'n':>5}{'median liq $':>15}{'median exit $':>15}"
          f"{'median turnover':>17}")
    for low, high, label in COHORTS:
        cohort = [r for r in rows if low <= r[1] < high]
        if not cohort:
            continue
        print(
            f"{label:<14}{len(cohort):>5}"
            f"{stats.median(r[0].liquidity_usd or 0 for r in cohort):>15,.0f}"
            f"{stats.median(r[2] for r in cohort):>15,.0f}"
            f"{stats.median(r[3] for r in cohort):>17.2f}"
        )

    established = [r for r in rows if r[1] >= 30]
    if not established:
        print("\nno established tokens in the sample")
        return

    print("\n" + "=" * 74)
    print("DOES CAPACITY COME WITH FLOW?  (established tokens only)")
    print("=" * 74)
    dead = [r for r in established if r[3] < 0.1]
    print(f"  established tokens              {len(established):>5}")
    print(f"  turnover < 0.1x/day (stranded)  {len(dead):>5}"
          f"  ({len(dead)/len(established):.0%})")

    thin = [r for r in established if THIN_BAND[0] <= (r[0].liquidity_usd or 0) <= THIN_BAND[1]]
    thin_dead = [r for r in thin if r[3] < 0.1]
    if thin:
        print(f"  in the thin band ${THIN_BAND[0]:,.0f}-${THIN_BAND[1]:,.0f}    {len(thin):>5}")
        print(f"    of those, stranded            {len(thin_dead):>5}"
              f"  ({len(thin_dead)/len(thin):.0%})")

    useful = [r for r in established if r[2] >= USEFUL_EXIT and r[3] >= USEFUL_TURNOVER]
    useful.sort(key=lambda r: -r[2])
    print(f"\n  exit >= ${USEFUL_EXIT:,.0f} AND turnover >= {USEFUL_TURNOVER}x: "
          f"{len(useful)} of {len(established)}")
    print(f"\n  {'symbol':<12}{'liq $':>13}{'24h vol $':>14}{'turn':>7}"
          f"{'age d':>7}{'exit $':>11}")
    for pair, age_days, capacity, turnover in useful[:12]:
        print(f"  {pair.symbol[:11]:<12}{pair.liquidity_usd or 0:>13,.0f}"
              f"{pair.volume_24h or 0:>14,.0f}{turnover:>7.1f}"
              f"{age_days:>7.0f}{capacity:>11,.0f}")

    print(
        "\nThe two properties are anticorrelated. Tokens are fresh and thin with\n"
        "flow but no capacity, or established and liquid with both but no longer\n"
        "thin, or established and thin with capacity and no flow at all. A thin\n"
        "market that is genuinely active does not stay thin -- it either attracts\n"
        "liquidity or the activity dies. There is no sweet spot to sit in."
    )


if __name__ == "__main__":
    main()
