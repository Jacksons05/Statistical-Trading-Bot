"""Measure post-earnings drift on real earnings surprises and real prices.

Wires `strategies.pead` to actual data for the first time. Earnings come from
Alpha Vantage (an API key, or a cache written out-of-band); prices and the
benchmark come from Yahoo, which needs no key.

Everything that decides whether a PEAD number is honest lives in
`backtest.event_study` and `data.earnings`:

* entry is the first session a post-market report could actually be traded,
* SUE is standardised by *prior* surprises only,
* returns are net of a benchmark over the identical sessions,
* the announcement-day gap is reported separately and never counted.

Run with::

    python examples/pead_event_study.py --cache earnings_cache.jsonl
    python examples/pead_event_study.py --symbols IBM,CULP    # needs a key
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

from sttbot.backtest.event_study import bucket_by_sue, build_observations, long_short
from sttbot.data.earnings import (
    fetch_earnings,
    fetch_prices,
    load_earnings_cache,
)
from sttbot.strategies.pead import PostEarningsDriftStrategy

BENCHMARK = "SPY"
HORIZONS = (5, 10, 20, 40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", help="JSONL of raw Alpha Vantage EARNINGS payloads")
    parser.add_argument("--symbols", help="comma-separated; requires an API key")
    parser.add_argument("--horizon", type=int, default=20,
                        help="holding period in trading sessions")
    args = parser.parse_args()

    if args.cache:
        by_symbol = load_earnings_cache(args.cache)
    elif args.symbols:
        by_symbol = {s: fetch_earnings(s) for s in args.symbols.split(",")}
    else:
        parser.error("need --cache or --symbols")

    print(f"{len(by_symbol)} symbols, "
          f"{sum(len(v) for v in by_symbol.values())} quarterly reports")

    earliest = min(e.reported_date for v in by_symbol.values() for e in v)
    start = earliest - dt.timedelta(days=10)
    end = dt.date.today()

    print(f"fetching prices {start} .. {end} ...", flush=True)
    benchmark = fetch_prices(BENCHMARK, start, end)
    if benchmark is None:
        print(f"could not fetch {BENCHMARK}; cannot market-adjust")
        return
    print(f"  {BENCHMARK}: {len(benchmark.closes)} sessions")

    observations = []
    for symbol, events in sorted(by_symbol.items()):
        prices = fetch_prices(symbol, start, end)
        time.sleep(0.5)
        if prices is None:
            print(f"  {symbol}: no prices, skipped")
            continue
        obs = build_observations(events, prices, benchmark, horizon=args.horizon)
        observations.extend(obs)
        print(f"  {symbol}: {len(prices.closes)} sessions, {len(obs)} usable events")

    if not observations:
        print("\nno usable observations")
        return

    print(f"\n{'=' * 76}")
    print(f"POST-EARNINGS DRIFT  ({len(observations)} events, "
          f"{args.horizon}-session holding period, market-adjusted)")
    print("=" * 76)
    print(f"  {'bucket':<22}{'n':>6}{'mean excess':>14}{'t':>8}{'hit rate':>10}"
          f"{'entry gap':>12}")
    for b in bucket_by_sue(observations):
        print(f"  {b.label:<22}{b.n:>6}{b.mean_excess:>+14.2%}{b.t_stat:>8.2f}"
              f"{b.hit_rate:>10.1%}{b.mean_gap:>+12.2%}")

    print("\n  PEAD predicts this column rises monotonically: the biggest positive")
    print("  surprises drift up most, the biggest misses drift down most. One")
    print("  significant bucket without that shape is noise.")

    threshold = PostEarningsDriftStrategy().sue_threshold
    n_long, n_short, mean, t = long_short(observations, threshold=threshold)
    print(f"\n  long-short at the strategy's SUE threshold of {threshold}:")
    print(f"    {n_long} long, {n_short} short, mean excess {mean:+.2%}, t={t:.2f}")
    print("    (borrow cost on the short leg is not modelled, so this is a bound)")

    print(f"\n{'=' * 76}")
    print("DRIFT BY HORIZON  (long-short spread)")
    print("=" * 76)
    for h in HORIZONS:
        by_h = []
        for symbol, events in sorted(by_symbol.items()):
            prices = fetch_prices(symbol, start, end)
            time.sleep(0.3)
            if prices is None:
                continue
            by_h.extend(build_observations(events, prices, benchmark, horizon=h))
        if not by_h:
            continue
        nl, ns, m, tt = long_short(by_h, threshold=threshold)
        print(f"  {h:>3} sessions   n={nl + ns:>4}   mean {m:>+7.2%}   t={tt:>6.2f}")

    gaps = [o.entry_gap for o in observations]
    print(f"\n  For scale, the announcement gap this study excludes averages "
          f"{sum(gaps) / len(gaps):+.2%}.")
    print("  A study that entered at the announcement close would book that as drift.")


if __name__ == "__main__":
    main()
