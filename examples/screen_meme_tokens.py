"""Screen live Solana meme tokens, and size the exit before the entry.

Pulls the DexScreener discovery feed, joins GoPlus contract safety, and runs
:func:`sttbot.strategies.token_screen.screen` over the result.

The number to watch is the rejection rate. This screen is built for
**avoidance, not selection** -- roughly 97% of these tokens die, so a screen
that passes most of what it sees is broken regardless of how good the passes
look. Rejections are also reported split by cause, because a *failure* (a check
ran and the token lost) and an *unknown* (the data was never there) call for
different responses: one is a verdict, the other is a hole in the pipeline.

Run with::

    python examples/screen_meme_tokens.py
"""

from __future__ import annotations

import time
from collections import Counter

from sttbot.data.tokens import (
    goplus_security,
    latest_token_addresses,
    pairs_for_tokens,
    to_token_metadata,
)
from sttbot.strategies.token_screen import max_exit_size, position_limit, screen


def main() -> None:
    now_ms = time.time() * 1000.0

    print("discovering recent Solana tokens ...", flush=True)
    addresses = latest_token_addresses(chain="solana")
    print(f"  {len(addresses)} candidates")
    if not addresses:
        print("discovery feed returned nothing; nothing to screen")
        return

    pairs = pairs_for_tokens(addresses)
    print(f"  {len(pairs)} have a tradeable pool")

    print("fetching contract safety ...", flush=True)
    security = goplus_security(list(pairs))

    rows = []
    for address, pair in pairs.items():
        meta = to_token_metadata(pair, security.get(address), now_ms=now_ms)
        rows.append((pair, meta, screen(meta)))

    passed = [r for r in rows if r[2].passed]
    failed = [r for r in rows if r[2].failures]
    only_unknown = [r for r in rows if not r[2].failures and not r[2].passed]

    print(f"\n{'='*72}\nSCREEN RESULTS  ({len(rows)} tokens)\n{'='*72}")
    print(f"  passed                        {len(passed):>5}"
          f"  ({len(passed)/len(rows):.1%})")
    print(f"  rejected on a failed check    {len(failed):>5}"
          f"  ({len(failed)/len(rows):.1%})")
    print(f"  blocked only by missing data  {len(only_unknown):>5}"
          f"  ({len(only_unknown)/len(rows):.1%})")

    causes = Counter(f.split("—")[0].strip() for _, _, r in rows for f in r.failures)
    print("\nmost common failure causes:")
    for cause, count in causes.most_common(10):
        print(f"  {count:>4}  {cause[:62]}")

    gaps = Counter(u for _, _, r in rows for u in r.unknowns)
    print("\nmost common data gaps (pipeline limits, not verdicts):")
    for gap, count in gaps.most_common(6):
        print(f"  {count:>4}  {gap}")

    print(f"\n{'='*72}\nEXIT SIZING  (liquidity bounds the position, not conviction)\n{'='*72}")
    print("Position cap assumes $10,000 of equity and a 2% price-impact budget.")
    print(f"\n{'symbol':<12}{'liq $':>11}{'24h vol $':>13}{'age h':>8}"
          f"{'exit $':>10}{'max pos $':>11}")
    equity = 10_000.0
    deepest = sorted(rows, key=lambda r: -(r[0].liquidity_usd or 0))[:12]
    for pair, meta, _ in deepest:
        pool = pair.pool()
        if pool is None:
            continue
        # max_exit_size is in tokens; value it at spot to compare with dollars.
        exit_usd = max_exit_size(pool, max_price_impact=0.02) * pool.spot_price
        limit = position_limit(pool, equity, max_price_impact=0.02)
        print(f"{meta.symbol[:11]:<12}{pair.liquidity_usd or 0:>11,.0f}"
              f"{pair.volume_24h or 0:>13,.0f}"
              f"{(pair.age_hours(now_ms) or 0):>8.1f}"
              f"{exit_usd:>10,.0f}{limit:>11,.0f}")

    print(
        "\nA high rejection rate is the screen working. Base rates for this asset\n"
        "class: ~97% of tokens die, ~69% of pump.fun launches never trade past\n"
        "their first day, and roughly 0.26% graduate. Passing this screen means\n"
        "'not obviously rigged', which is a long way from 'good trade'."
    )


if __name__ == "__main__":
    main()
