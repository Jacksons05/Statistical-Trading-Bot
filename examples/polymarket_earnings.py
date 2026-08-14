"""Price Polymarket's "Will X beat quarterly earnings?" markets from earnings history.

The measured conclusion of this repo's Polymarket research is that the venue
pays for a forecasting view and not for arbitrage. This is that view: a
company's own record of beating consensus, shrunk toward the base rate of the
universe Polymarket actually lists.

Which base rate is not a detail. Companies beat consensus at wildly different
rates by size -- measured here at 86% for large caps against 53% for a
micro-cap -- and Polymarket only lists large, well-covered names. Shrinking
toward a pooled rate that includes small caps would systematically say "the
market is too high, sell" on every contract, and be wrong every time.

Run with::

    python examples/polymarket_earnings.py --cache earnings_cache.jsonl
    python examples/polymarket_earnings.py --cache c.jsonl --paper book.duckdb
"""

from __future__ import annotations

import argparse

from sttbot.data.earnings import load_earnings_cache
from sttbot.paper.account import BUY, PaperAccount
from sttbot.paper.runner import Basket, Intent, PaperRunner, limit_fill_model
from sttbot.risk.circuit_breaker import RiskCircuitBreaker
from sttbot.strategies.earnings_market import (
    beat_probability,
    edge_against_market,
    extract_ticker,
    measure_base_rate,
)
from sttbot.venues.polymarket import (
    is_live,
    iter_events,
    market_fee_model,
    two_sided_quote,
    yes_token_id,
)

MIN_EDGE = 0.02          # per contract, after fees
MAX_CONTRACTS = 25       # these books are thin; see the capacity study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="earnings JSONL cache")
    parser.add_argument("--paper", help="paper account DuckDB path; omit to dry-run")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    args = parser.parse_args()

    histories = load_earnings_cache(args.cache)
    print(f"{len(histories)} companies in cache")

    print("\nscanning Polymarket for earnings markets ...", flush=True)
    candidates = []
    unmatched = []
    for event in iter_events(closed=False):
        ticker = extract_ticker(str(event.get("title") or ""))
        if not ticker:
            continue
        legs = [m for m in (event.get("markets") or []) if is_live(m) and two_sided_quote(m)]
        if not legs:
            continue
        if ticker not in histories:
            unmatched.append(ticker)
            continue
        candidates.append((ticker, event, legs[0]))

    print(f"  {len(candidates) + len(unmatched)} earnings markets live; "
          f"{len(candidates)} have earnings history cached")
    if unmatched:
        print(f"  no history for: {', '.join(sorted(set(unmatched))[:18])}"
              f"{' ...' if len(set(unmatched)) > 18 else ''}")

    if not candidates:
        print("\nnothing priceable -- pull earnings for the tickers listed above")
        return

    # The base rate MUST come from the universe Polymarket actually lists, not
    # from whatever happens to sit in the cache. Measured on this data, large
    # caps beat 86% of the time and a micro-cap beat 53%; letting one
    # unlisted micro-cap into the average drags the prior down ~11 points,
    # which on a live HD market manufactured a +3.5c "edge" that is -0.1c
    # under the correct prior. The contamination does not look like an error
    # -- it looks like a signal.
    listed = sorted({t for t, _, _ in candidates})
    base_rate = measure_base_rate([histories[t] for t in listed])
    excluded = sorted(set(histories) - set(listed))
    print(f"\n  base rate from the {len(listed)} listed name(s) only: {base_rate:.1%}")
    if excluded:
        pooled = measure_base_rate(list(histories.values()))
        print(f"  excluded from the prior: {', '.join(excluded)} "
              f"(pooling them would give {pooled:.1%})")

    print(f"\n{'=' * 78}")
    print("MODEL vs MARKET")
    print("=" * 78)
    print(f"  {'ticker':<8}{'model P':>9}{'bid':>7}{'ask':>7}"
          f"{'YES edge':>10}{'NO edge':>9}  action")

    baskets = []
    marks = {}
    for ticker, event, market in sorted(candidates):
        events = histories[ticker]
        forecast = beat_probability(events, len(events), base_rate=base_rate)
        if forecast is None:
            print(f"  {ticker:<8}{'--':>9}  not enough history")
            continue
        bid, ask = two_sided_quote(market)
        fees = market_fee_model(market)
        edge = edge_against_market(forecast.probability, bid, ask, fees)

        side = edge.best_side if edge.best_edge >= args.min_edge else None
        action = f"BUY {side}" if side else "no trade"
        print(f"  {ticker:<8}{forecast.probability:>9.3f}{bid:>7.2f}{ask:>7.2f}"
              f"{edge.yes_edge:>+10.4f}{edge.no_edge:>+9.4f}  {action}")

        if not side:
            continue
        token = yes_token_id(market)
        if token is None or side != "YES":
            # Only the YES token is addressable from this snapshot; taking the
            # NO side needs its own token id, which is deliberately not guessed.
            continue
        marks[token] = bid
        baskets.append(Basket(
            (Intent(symbol=token, side=BUY, qty=MAX_CONTRACTS, limit_price=ask,
                    strategy="polymarket_earnings",
                    ref=f"earn:{ticker}:{event.get('slug')}",
                    fee=fees.cost(ask, contracts=MAX_CONTRACTS),
                    note=f"{ticker} beat-earnings, model {forecast.probability:.3f}"),),
            rationale=f"{ticker} earnings beat @ model {forecast.probability:.3f}",
        ))

    print(f"\n  no-trade band exists because both sides cross the spread: with a")
    print(f"  5c spread and the quadratic fee, the model has to disagree with the")
    print(f"  market by roughly 4-6c before either side clears.")

    if not args.paper:
        print(f"\n{len(baskets)} tradeable signal(s); dry run, nothing recorded.")
        return

    account = PaperAccount(args.paper, starting_cash=1_000.0)
    runner = PaperRunner(account=account, breaker=RiskCircuitBreaker(),
                         fill_model=limit_fill_model)
    result = runner.run_once(baskets, marks)
    print(f"\n{result.summary()}")
    for fill in result.filled:
        print(f"  FILLED {fill.qty:>6,.0f} @ {fill.price:.3f}  {fill.note}")
    account.close()


if __name__ == "__main__":
    main()
