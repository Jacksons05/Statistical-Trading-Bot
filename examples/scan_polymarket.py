"""Scan the entire open Polymarket universe for tradeable inefficiency.

Three passes, cheapest first:

1. **Inventory** -- how much of the venue is actually alive.
2. **Boundary arbitrage** on complete multi-outcome baskets, shortlisted on
   gamma's top of book then re-priced against real CLOB depth.
3. **Scalping viability** -- where a maker could quote with room to work.

The structure matters more than any single number. gamma prices carry no size,
so a candidate found there is a hypothesis; only the CLOB pass decides. Running
step 2 without step 3's depth check is how a scan reports edges it cannot fill.

Run with::

    python examples/scan_polymarket.py            # live, several minutes
    python examples/scan_polymarket.py cache.jsonl  # replay a saved pull
"""

from __future__ import annotations

import json
import sys
import time

from sttbot.strategies.market_making import breakeven_spread
from sttbot.venues.polymarket import (
    complete_baskets,
    executable_arbitrage,
    fetch_book,
    is_live,
    iter_events,
    market_fee_model,
    two_sided_quote,
    as_float,
)

# Only re-price candidates whose paper edge could plausibly survive depth.
PAPER_EDGE_FLOOR = 0.002
# A maker needs room to improve on both sides; one tick leaves none.
MIN_SCALP_SPREAD = 0.02
# Outside this band a market is effectively resolved.
SCALP_PRICE_BAND = (0.05, 0.95)
MIN_SCALP_VOLUME = 10_000.0


def load(path: str | None) -> list[dict]:
    if path:
        with open(path) as fh:
            return [json.loads(line) for line in fh]
    print("pulling the full open universe from gamma ...", flush=True)
    started = time.time()
    events = list(iter_events(closed=False))
    print(f"  {len(events):,} events in {time.time()-started:.0f}s", flush=True)
    return events


def main() -> None:
    events = load(sys.argv[1] if len(sys.argv) > 1 else None)

    markets = {}
    for event in events:
        for market in event.get("markets") or []:
            markets[market.get("id")] = market
    markets = list(markets.values())

    live = [m for m in markets if is_live(m)]
    quoted = [m for m in live if two_sided_quote(m)]
    volumes = [as_float(m.get("volume24hr"), 0.0) or 0.0 for m in markets]
    dead = sum(1 for v in volumes if v == 0)

    print("\n== inventory ==")
    print(f"  events                  {len(events):>10,}")
    print(f"  markets                 {len(markets):>10,}")
    print(f"  live                    {len(live):>10,}")
    print(f"  live and two-sided      {len(quoted):>10,}")
    print(f"  traded $0 in 24h        {dead:>10,}  ({dead/max(len(markets),1):.1%})")
    print(f"  total 24h volume        ${sum(volumes):>9,.0f}")

    # --- boundary arbitrage -------------------------------------------------
    baskets = complete_baskets(events)
    shortlist = [b for b in baskets if b.paper_edge() > PAPER_EDGE_FLOOR]
    shortlist.sort(key=lambda b: -b.paper_edge())

    print("\n== boundary arbitrage ==")
    print(f"  complete baskets        {len(baskets):>10,}")
    print(f"  positive on paper       {len(shortlist):>10,}")
    print("  (paper edge is top-of-book with no size behind it)")

    total_profit = total_capital = 0.0
    confirmed = 0
    print(f"\n  {'paper':>8}{'size':>8}{'capital':>10}{'profit':>9}{'ret':>7}  event")
    for basket in shortlist[:25]:
        token_ids = [t for t in basket.token_ids if t]
        if len(token_ids) != len(basket.token_ids):
            continue
        try:
            books = [fetch_book(t) for t in token_ids]
        except RuntimeError:
            continue
        arb = executable_arbitrage(books, basket.fees)
        if arb is None:
            print(f"  {basket.paper_edge():>+8.4f}{'none':>8}{'-':>10}{'-':>9}"
                  f"{'-':>7}  {basket.title[:38]}")
            continue
        confirmed += 1
        total_profit += arb.profit
        total_capital += arb.capital
        print(f"  {basket.paper_edge():>+8.4f}{arb.contracts:>8,.0f}"
              f"${arb.capital:>9,.0f}${arb.profit:>8,.2f}"
              f"{arb.return_on_capital:>7.1%}  {basket.title[:38]}")

    print(f"\n  executable: {confirmed} of {len(shortlist[:25])} checked, "
          f"${total_profit:,.2f} profit on ${total_capital:,.0f} capital")

    # --- scalping -----------------------------------------------------------
    print("\n== scalping viability ==")
    fee_viable = spread_ok = in_band = active = 0
    candidates = []
    for market in quoted:
        bid, ask = two_sided_quote(market)  # type: ignore[misc]
        spread, mid = ask - bid, (ask + bid) / 2.0
        if spread > breakeven_spread(mid, market_fee_model(market)):
            fee_viable += 1
        if not SCALP_PRICE_BAND[0] <= mid <= SCALP_PRICE_BAND[1]:
            continue
        in_band += 1
        if spread < MIN_SCALP_SPREAD:
            continue
        spread_ok += 1
        volume = as_float(market.get("volume24hr"), 0.0) or 0.0
        if volume < MIN_SCALP_VOLUME:
            continue
        active += 1
        candidates.append((volume, spread, mid, market.get("question")))

    print(f"  clears round-trip fee   {fee_viable:>10,}   <- vacuous: makers are rebated")
    print(f"  mid in {SCALP_PRICE_BAND}      {in_band:>10,}")
    print(f"  spread >= {MIN_SCALP_SPREAD:<4}           {spread_ok:>10,}")
    print(f"  and >= ${MIN_SCALP_VOLUME:,.0f} 24h vol  {active:>10,}")

    candidates.sort(reverse=True)
    print(f"\n  {'24h vol':>12}{'spread':>8}{'mid':>7}  question")
    for volume, spread, mid, question in candidates[:15]:
        print(f"  ${volume:>11,.0f}{spread:>8.3f}{mid:>7.2f}  {str(question)[:44]}")

    print(
        "\n  A fee screen every market passes is not a screen. Polymarket rebates\n"
        "  makers, so breakeven spread is negative and any positive spread clears.\n"
        "  The binding constraint is adverse selection, which backtest.mm_simulator\n"
        "  measures directly -- quote against informed flow and the rebate is noise."
    )


if __name__ == "__main__":
    main()
