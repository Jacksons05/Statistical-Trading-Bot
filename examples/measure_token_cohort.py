"""Measure the realised payoff distribution of a token birth cohort.

The breadth strategy's viability is decided by one number -- the fatness of the
payoff tail -- and until now that number was assumed. This measures it.

Two phases, because the measurement is inherently forward-looking:

    python examples/measure_token_cohort.py capture cohort.jsonl
    ... wait ...
    python examples/measure_token_cohort.py measure cohort.jsonl

Why it cannot be done in one pass: Solana creates roughly 1,600 pools an hour
and the discovery feed holds 200, so it spans about seven minutes. There is no
reaching back. And a cohort assembled later from whatever is still listed has
already had its failures removed, which is the survivorship bias that makes
this asset class look survivable. So the cohort is recorded at birth and
resolved afterwards from candle history, which remains available for pools that
have since died.

The measurement reports both what a perfect exit would have made (candle highs)
and what holding would have made (final close). The first is not achievable --
nobody sells the high -- but it bounds every possible take-profit rule, so if
the bound is below breakeven then no exit rule rescues the strategy.
"""

from __future__ import annotations

import json
import sys
import time

from sttbot.backtest.cohort import (
    detect_clone_groups,
    CohortMember,
    best_take_profit,
    expected_value_with_take_profit,
    measure_outcomes,
    summarise,
)
from sttbot.data.tokens import geckoterminal_new_pools, pool_ohlcv
from sttbot.strategies.breadth import breakeven_multiple, names_for_confidence  # noqa: F401

SWEEPS = 6
SWEEP_PAUSE = 140.0
ASSUMED_COST = 0.04  # round trip on a thin pool, from strategies.breadth
TAKE_PROFITS = (1.5, 2, 3, 5, 10, 25, 50, 100)


def capture(path: str) -> None:
    """Poll the new-pool feed and persist everything seen, with birth times."""
    seen: dict[str, dict] = {}
    for sweep in range(SWEEPS):
        for pool in geckoterminal_new_pools(network="solana"):
            seen.setdefault(pool.address, {
                "pool": pool.address,
                "name": pool.name,
                "base_token": pool.base_token,
                "created_at": pool.created_at,
                "price_at_capture": pool.price_usd,
                "reserve_at_capture": pool.reserve_usd,
            })
        print(f"  sweep {sweep+1}/{SWEEPS}: {len(seen)} unique pools", flush=True)
        if sweep < SWEEPS - 1:
            time.sleep(SWEEP_PAUSE)

    with open(path, "w") as fh:
        for record in seen.values():
            fh.write(json.dumps(record) + "\n")
    births = sorted(r["created_at"] for r in seen.values() if r["created_at"])
    print(f"captured {len(seen)} pools -> {path}")
    if births:
        print(f"  birth window {births[0]} .. {births[-1]}")
    print("\nLet these age, then run the measure phase.")


RULES = [
    ("A raw (candle-0 open, any peak)", {}),
    ("B skip the birth candle", {"skip_first_candle": True}),
    ("C peak needs $100 volume", {"min_peak_volume": 100.0}),
    ("D peak confirmed by a close", {"confirm_peak_fraction": 0.5}),
    ("E all three defences", {"skip_first_candle": True, "min_peak_volume": 100.0,
                              "confirm_peak_fraction": 0.5}),
]


def measure(path: str, limit: int | None = None) -> None:
    with open(path) as fh:
        members = [
            CohortMember(
                pool=r["pool"], base_token=r.get("base_token", ""),
                created_at=r.get("created_at", ""),
                price_at_capture=r.get("price_at_capture"),
                reserve_at_capture=r.get("reserve_at_capture"),
            )
            for r in (json.loads(line) for line in fh)
        ]
    if limit:
        members = members[:limit]
    print(f"resolving {len(members)} pools from candle history ...", flush=True)

    cache: dict[str, list | None] = {}

    def fetch_candles(pool: str):
        if pool not in cache:
            cache[pool] = pool_ohlcv(pool, timeframe="minute", aggregate=1, limit=200)
            time.sleep(2.2)  # GeckoTerminal free tier: ~30 calls/minute
        return cache[pool]

    baseline = measure_outcomes(members, fetch_candles)
    print(f"  {baseline.summary()}")
    if baseline.fetch_failed:
        print(f"  WARNING: {len(baseline.fetch_failed)} pools could not be fetched. "
              "These are NOT failures -- excluding them inflates every statistic "
              "below. Re-run to recover them before trusting the numbers.")
    if not baseline.outcomes:
        print("\nnothing resolvable yet -- let the cohort age and re-run")
        return

    print("\n" + "=" * 78)
    print("PEAK DISTRIBUTION UNDER EACH ARTIFACT DEFENCE")
    print("=" * 78)
    print(f"  {'rule':<34}{'n':>5}{'median':>8}{'p90':>8}{'p99':>10}"
          f"{'best':>12}{'alpha':>8}")
    reports = {}
    for label, kwargs in RULES:
        report = measure_outcomes(members, fetch_candles, **kwargs)
        reports[label] = report
        stats = summarise(report.outcomes)
        alpha = f"{stats.tail_alpha:.2f}" if stats.tail_alpha is not None else "n/a"
        print(f"  {label:<34}{stats.n:>5}{stats.median_max_multiple:>8.2f}"
              f"{stats.p90_max_multiple:>8.2f}{stats.p99_max_multiple:>10.1f}"
              f"{stats.best_max_multiple:>12,.1f}{alpha:>8}")
    print("\n  A spread this wide across defensible rules means the tail index is not"
          "\n  identified by this sample. Do not quote a single alpha, and do not quote"
          "\n  a mean at all -- one dust tick moved it by three orders of magnitude.")

    strict = reports["E all three defences"]
    outcomes = strict.outcomes
    peaks = [o.max_multiple for o in outcomes]

    print("\n" + "=" * 78)
    print("UNDER THE STRICTEST RULE")
    print("=" * 78)
    stats = summarise(outcomes)
    print(f"  n                               {stats.n:>8}")
    print(f"  median hold-to-end              {stats.median_final_multiple:>8.2f}x")
    print(f"  median peak                     {stats.median_max_multiple:>8.2f}x")
    print(f"  best peak                       {stats.best_max_multiple:>8.2f}x")
    print("\n  fraction whose peak reached:")
    for threshold in (2, 5, 10, 33, 100):
        hit = sum(1 for m in peaks if m >= threshold) / len(peaks)
        print(f"    {threshold:>4}x   {hit:>7.2%}")

    print("\n" + "=" * 78)
    print("HIT RATE NEEDED vs HIT RATE MEASURED")
    print("=" * 78)
    print(f"  {'target':>8}{'needed':>10}{'measured':>11}{'margin':>10}")
    for target in (2, 3, 5, 10, 25):
        needed = (1.0 + ASSUMED_COST) / target
        got = sum(1 for m in peaks if m >= target) / len(peaks)
        flag = "  clears" if got > needed else ""
        print(f"  {target:>7}x{needed:>10.2%}{got:>11.2%}{got - needed:>+10.2%}{flag}")
    print("\n  'needed' assumes losers go to zero, the easiest possible bar, and")
    print("  'measured' assumes you sell the exact high. Both favour the strategy.")

    print("\n" + "=" * 78)
    print("EXPECTED VALUE BY TAKE-PROFIT  (perfect trigger fills, so optimistic)")
    print("=" * 78)
    for tp in TAKE_PROFITS:
        clean = expected_value_with_take_profit(
            outcomes, tp, cost_fraction=ASSUMED_COST
        )
        slipped = expected_value_with_take_profit(
            outcomes, tp, cost_fraction=ASSUMED_COST, slippage_on_exit=0.10
        )
        print(f"  {tp:>6}x   EV {clean:>+9.4f}   with 10% exit slip {slipped:>+9.4f}")

    volumes = {o.pool: 0.0 for o in outcomes}
    clones = detect_clone_groups(
        members, volumes, name_of=lambda m: m.base_token
    )
    if clones:
        print(f"\n  clone families detected: {len(clones)} -- these are one actor "
              "running several mints,\n  so counting them separately inflates any "
              "hit rate and breaks independence.")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("capture", "measure"):
        print(__doc__)
        raise SystemExit(1)
    mode, path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    capture(path) if mode == "capture" else measure(path, limit)


if __name__ == "__main__":
    main()
