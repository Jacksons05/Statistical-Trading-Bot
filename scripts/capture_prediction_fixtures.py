"""Capture real Polymarket / Kalshi payloads as offline test fixtures.

The test suite must never touch the network, but adapters must be written
against *real* response shapes rather than guessed ones. This script is the
bridge: run it once from a machine with venue access, commit the JSON it
writes, and the adapters and their tests are then built against schemas that
actually exist.

    python scripts/capture_prediction_fixtures.py --out tests/fixtures

Standard library only, so it runs on a bare checkout. Read-only public
endpoints: no keys, no signing, no order placement. Nothing here is imported by
library code or by any test.

If a venue is unreachable the script reports it and continues, so a partial
capture still produces usable fixtures for the venue that did respond.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "sttbot/0.1 (research; contact via repository issues)"

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def get(url: str, params: dict | None = None, *, timeout: float = 20.0,
        retries: int = 3) -> object:
    """GET JSON with backoff. Raises the last error if every attempt fails."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "Accept": "application/json",
            })
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 4xx other than rate-limiting will not fix themselves.
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            last = exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2**attempt)
    assert last is not None
    raise last


def write(out_dir: pathlib.Path, name: str, payload: object) -> None:
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    size = path.stat().st_size
    print(f"  wrote {path} ({size:,} bytes)")


def describe(label: str, payload: object) -> None:
    """Print the payload's shape so schemas can be inspected without opening files."""
    if isinstance(payload, dict):
        print(f"  {label}: dict keys = {sorted(payload)[:14]}")
    elif isinstance(payload, list):
        print(f"  {label}: list[{len(payload)}]")
        if payload and isinstance(payload[0], dict):
            print(f"    item keys = {sorted(payload[0])[:14]}")
    else:
        print(f"  {label}: {type(payload).__name__}")


def capture_polymarket(out: pathlib.Path, limit: int) -> None:
    print("\nPolymarket")
    markets = get(f"{POLYMARKET_GAMMA}/markets",
                  {"limit": limit, "closed": "false", "order": "volume24hr",
                   "ascending": "false"})
    describe("markets", markets)
    write(out, "polymarket_markets.json", markets)

    rows = markets if isinstance(markets, list) else markets.get("data", [])
    # clobTokenIds is a JSON-encoded string of outcome token ids; the book is
    # per-token, so both outcomes must be fetched separately.
    for market in rows:
        raw = market.get("clobTokenIds")
        if not raw:
            continue
        try:
            token_ids = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            continue
        if not token_ids:
            continue
        try:
            book = get(f"{POLYMARKET_CLOB}/book", {"token_id": token_ids[0]})
        except Exception as exc:  # noqa: BLE001 - report and try the next market
            print(f"  book fetch failed ({exc}); trying next market")
            continue
        describe("book", book)
        write(out, "polymarket_book.json", book)
        write(out, "polymarket_market_single.json", market)
        return
    print("  no market exposed a usable clobTokenIds; book not captured")


def capture_kalshi(out: pathlib.Path, limit: int) -> None:
    print("\nKalshi")
    markets = get(f"{KALSHI}/markets", {"limit": limit, "status": "open"})
    describe("markets", markets)
    write(out, "kalshi_markets.json", markets)

    rows = markets.get("markets", []) if isinstance(markets, dict) else []
    if not rows:
        print("  no open markets returned; orderbook not captured")
        return
    ticker = rows[0].get("ticker")
    if not ticker:
        print("  first market had no ticker; orderbook not captured")
        return
    try:
        book = get(f"{KALSHI}/markets/{urllib.parse.quote(ticker)}/orderbook",
                   {"depth": 10})
    except Exception as exc:  # noqa: BLE001
        print(f"  orderbook fetch failed ({exc})")
        return
    describe("orderbook", book)
    write(out, "kalshi_orderbook.json", book)
    write(out, "kalshi_market_single.json", rows[0])
    print(f"  NOTE: Kalshi prices are integer cents (1-99). Sample: "
          f"yes_bid={rows[0].get('yes_bid')} yes_ask={rows[0].get('yes_ask')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/fixtures", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=25,
                        help="markets to request per venue")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    failures = []
    for name, fn in (("Polymarket", capture_polymarket), ("Kalshi", capture_kalshi)):
        try:
            fn(args.out, args.limit)
        except Exception as exc:  # noqa: BLE001 - one venue must not stop the other
            print(f"\n{name}: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"captured with failures: {', '.join(failures)} unreachable")
        print("A 403 on every host usually means an egress proxy, not the venue.")
    else:
        print("captured both venues")
    print(f"fixtures in {args.out}/ -- commit them, then build adapters against")
    print("these shapes rather than against guessed ones.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
