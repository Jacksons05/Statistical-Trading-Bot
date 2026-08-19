"""Scan Kalshi with the same discipline the Polymarket study used.

Kalshi is the better venue on the two things that cap a book -- at matched
liquidity its spreads are 1c against Polymarket's 2c, and 768 of its markets
clear a $10k daily-volume bar against Polymarket's 57 -- despite a worse fee
(7% quadratic against 5%).

That does not mean the arbitrage is there. This applies the same three filters
that reduced 172 phantom Polymarket opportunities to 7 real ones:

1. enumerate via /events, because /markets is 99.7% auto-generated parlay combos
2. require the basket to look complete, because Kalshi's mutually_exclusive
   means "cannot co-occur", not "covers the probability space"
3. confirm against real order-book depth, because top-of-book carries no size

Run with::

    python examples/scan_kalshi.py
"""

from __future__ import annotations

import statistics as st
import time

from sttbot.venues.kalshi import (
    KALSHI_FEES,
    complete_baskets,
    executable_arbitrage,
    fetch_orderbook,
    is_live,
    iter_events,
    two_sided_quote,
)
from sttbot.venues.prediction import POLYMARKET_US_FEES

LIQUID_VOLUME = 10_000.0
MAX_CANDIDATES = 15


def main() -> None:
    print("enumerating Kalshi events ...", flush=True)
    events = list(iter_events(status="open"))
    markets = [m for e in events for m in (e.get("markets") or [])]
    quoted = [(m, q) for m in markets if is_live(m) and (q := two_sided_quote(m))]
    print(f"  {len(events):,} events / {len(markets):,} markets, "
          f"{len(quoted):,} live and two-sided")

    def vol(m):
        try:
            return float(m.get("volume_24h_fp") or 0)
        except (TypeError, ValueError):
            return 0.0

    traded = sum(1 for m in markets if vol(m) > 0)
    liquid = [(m, q) for m, q in quoted if vol(m) >= LIQUID_VOLUME]
    print(f"  ${sum(vol(m) for m in markets):,.0f} of 24h volume, "
          f"{traded:,} markets traded ({traded/max(len(markets),1):.1%})")
    print(f"  {len(liquid):,} markets with >= ${LIQUID_VOLUME:,.0f} of 24h volume")

    if liquid:
        spreads = sorted(a - b for _, (b, a) in liquid)
        print(f"  their spread: median {st.median(spreads)*100:.2f}c, "
              f"p75 {spreads[3*len(spreads)//4]*100:.2f}c")
        print("\n  cost to cross, liquid markets only (spread + fee, vs mid):")
        print(f"    {'band':<12}{'n':>6}{'spread':>9}{'K fee':>8}{'K total':>9}"
              f"{'PM total':>10}")
        for lo, hi in ((0.02, 0.15), (0.15, 0.35), (0.35, 0.65),
                       (0.65, 0.85), (0.85, 0.98)):
            band = [(b, a) for _, (b, a) in liquid if lo <= (a + b) / 2 < hi]
            if len(band) < 10:
                continue
            sp = st.median(a - b for b, a in band)
            mid = st.median((a + b) / 2 for b, a in band)
            kf, pf = KALSHI_FEES.cost(mid), POLYMARKET_US_FEES.cost(mid)
            print(f"    {lo:.2f}-{hi:.2f}  {len(band):>6,}{sp*100:>8.2f}c"
                  f"{kf*100:>7.2f}c{(sp+kf)/mid:>9.1%}{(0.02+pf)/mid:>10.1%}")

    baskets = complete_baskets(events)
    priced = [b for b in baskets if b.paper_edge() > 0.002]
    plausible = [b for b in priced if b.looks_complete]
    print(f"\n  {len(baskets):,} mutually-exclusive baskets fully quoted")
    print(f"  {len(priced)} show positive edge on paper")
    print(f"  {len(plausible)} survive the completeness rule "
          f"({len(priced) - len(plausible)} rejected as missing legs)")

    if not plausible:
        print("\n  nothing to confirm against depth.")
        return

    print("\n  confirming against real order-book depth ...", flush=True)
    total_profit = total_capital = 0.0
    confirmed = 0
    plausible.sort(key=lambda b: -b.paper_edge())
    print(f"    {'paper':>8}{'size':>8}{'capital':>10}{'profit':>9}  event")
    for basket in plausible[:MAX_CANDIDATES]:
        books = []
        for ticker in basket.tickers:
            book = fetch_orderbook(ticker)
            time.sleep(0.15)
            if book is None:
                books = None
                break
            books.append(book)
        if books is None:
            continue
        arb = executable_arbitrage(books, fees=KALSHI_FEES)
        if arb is None:
            print(f"    {basket.paper_edge():>+8.4f}{'none':>8}{'-':>10}{'-':>9}"
                  f"  {basket.title[:38]}")
            continue
        confirmed += 1
        total_profit += arb.profit
        total_capital += arb.capital
        print(f"    {basket.paper_edge():>+8.4f}{arb.contracts:>8,.0f}"
              f"${arb.capital:>9,.2f}${arb.profit:>8,.2f}  {basket.title[:38]}")

    print(f"\n  executable: {confirmed} of {len(plausible[:MAX_CANDIDATES])} checked, "
          f"${total_profit:,.2f} profit on ${total_capital:,.2f} capital")


if __name__ == "__main__":
    main()
