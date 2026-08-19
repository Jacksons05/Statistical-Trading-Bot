"""Run the Polymarket boundary-arbitrage scanner against a durable paper account.

This is the first strategy wired into ``sttbot.paper`` because it is the one
strategy in this repository with a measured, if tiny, genuine positive edge:
the README's Polymarket scan found $66 of executable arbitrage across the
*entire* open venue in one snapshot, confirmed against live order-book depth
rather than the top-of-book prices that manufacture false edges. Running it
repeatedly against a paper account is "fake money, real prices" end to end:
enumerate -> confirm against live depth -> size atomically -> record durably
-> guard with a circuit breaker.

Each run is a single tick: scan, size what clears, trade it, print the
account. Schedule it (cron, a systemd timer, ``watch``) to accumulate a real
history rather than running it once and calling that a track record.
Re-running does not pile into a basket the account already holds a leg of, so
repeated invocations are safe.

Symbols in the ledger are raw CLOB token ids rather than readable names,
because a token id is what lets a *later* run refetch that leg's book to mark
an open position -- a readable label would need a side-table to reverse, for
no benefit the fill's ``note`` field doesn't already provide.

Run with::

    python examples/paper_trade_polymarket.py                       # live
    python examples/paper_trade_polymarket.py --cache events.jsonl   # replay
"""

from __future__ import annotations

import argparse
import json

from sttbot.paper.account import BUY, AccountBusy, PaperAccount
from sttbot.paper.runner import Basket as PaperBasket
from sttbot.paper.runner import Intent, PaperRunner, limit_fill_model
from sttbot.paper.settlement import fetch_resolutions, settle_account
from sttbot.risk.circuit_breaker import RiskCircuitBreaker
from sttbot.venues.polymarket import (
    http_fetch,
    complete_baskets,
    executable_arbitrage,
    fetch_book,
    fetch_book_or_none,
    iter_events,
)

# Matches the shortlist filter in examples/scan_polymarket.py.
PAPER_EDGE_FLOOR = 0.002
# Cap on how many candidates get re-priced against live depth per run --
# each check is a handful of real HTTP calls.
MAX_CANDIDATES = 10
DEFAULT_DB = "paper_polymarket.duckdb"
DEFAULT_STARTING_CASH = 1_000.0
# Horizon cap, in days. DEFAULTS TO OFF, and that default is a measurement
# rather than caution.
#
# The obvious way to recycle capital faster is to skip long-dated markets --
# several baskets held on a live book do not settle until January 2027, and
# exiting early was priced at roughly -6% against a ~1% edge because a
# multi-leg basket crosses the spread on every leg twice.
#
# But the edge is not distributed evenly across horizons. Measured across
# 7,170 complete baskets in one venue-wide scan:
#
#     past due    70 baskets    0 with edge
#     <1d        417 baskets    0 with edge
#     1-7d     3,557 baskets    0 with edge
#     7-30d    2,866 baskets    0 with edge
#     30-180d    236 baskets    1 with edge   (0.42%)
#     180d+       22 baskets    1 with edge   (4.55%)
#
# Zero out of 4,044 inside a week. The mispricing lives precisely where nobody
# is watching, which is the same thing that makes it slow. Setting this to 7
# does not make the strategy faster, it stops it trading -- so the cap ships
# off and has to be asked for.
DEFAULT_MAX_DAYS = float("inf")


def load_events(cache_path: str | None) -> list[dict]:
    if cache_path:
        with open(cache_path) as fh:
            return [json.loads(line) for line in fh]
    return list(iter_events(closed=False))


def price_basket(basket) -> tuple[PaperBasket, dict[str, float]] | None:
    """Re-price a candidate against live order-book depth.

    Returns the paper-runner basket to submit plus the marks its legs need for
    equity accounting, or ``None`` if it does not clear real depth -- which is
    the common outcome, since the shortlist is built from top-of-book prices
    that carry no size.
    """
    token_ids = [t for t in basket.token_ids if t]
    if len(token_ids) != len(basket.token_ids):
        return None  # a leg has no CLOB token; the basket cannot be completed

    books = [fetch_book(t) for t in token_ids]
    arb = executable_arbitrage(books, basket.fees)
    if arb is None:
        return None

    intents = []
    marks: dict[str, float] = {}
    for token_id, book, fee_model, outcome in zip(
        token_ids, books, basket.fees, basket.outcomes
    ):
        cost = book.cost_to_buy(arb.contracts)
        if cost is None:
            # Should not happen: executable_arbitrage just confirmed this same
            # size against these same books. Treat a mismatch as reason to
            # skip the whole basket rather than trust a partial computation.
            return None
        avg_price = cost / arb.contracts
        marks[token_id] = book.best_bid if book.best_bid is not None else avg_price
        intents.append(
            Intent(
                symbol=token_id,
                side=BUY,
                qty=arb.contracts,
                limit_price=avg_price,
                strategy="polymarket_boundary_arb",
                fee=fee_model.cost(avg_price, contracts=arb.contracts),
                note=f"{basket.title} :: {outcome}",
            )
        )
    rationale = f"{basket.title} ({basket.event_slug})"
    return PaperBasket(tuple(intents), rationale=rationale), marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", default=None,
        help="replay a saved gamma events.jsonl instead of a live scan",
    )
    parser.add_argument("--db", default=DEFAULT_DB,
                        help="path to the paper account's DuckDB file")
    parser.add_argument("--starting-cash", type=float, default=DEFAULT_STARTING_CASH)
    parser.add_argument(
        "--max-days", type=float, default=DEFAULT_MAX_DAYS,
        help="skip baskets resolving further out than this (default: no cap). "
             "Measured: no basket inside 7 days had any edge, so a short cap "
             "stops trading rather than speeding it up.",
    )
    args = parser.parse_args()

    try:
        # Wait briefly for a concurrent tick or a human inspecting the book to
        # finish. DuckDB is single-writer, and a stack trace in a cron log
        # looks exactly like a real failure -- which is how 362 genuine crashes
        # went unnoticed for four days.
        account = PaperAccount(args.db, starting_cash=args.starting_cash,
                               busy_timeout=30.0)
    except AccountBusy as exc:
        print(f"account busy ({exc}); skipping this tick")
        return
    breaker = RiskCircuitBreaker()  # 5% high-water-mark and rolling drawdown
    runner = PaperRunner(account=account, breaker=breaker, fill_model=limit_fill_model)

    source = f"cache {args.cache}" if args.cache else "live scan"
    print(f"loading events ({source}) ...", flush=True)
    events = load_events(args.cache)
    print(f"  {len(events):,} events", flush=True)

    # Settle first: a resolved basket pays $1 a set, and nothing else in this
    # loop ever books that. Skipping it means cash only ever leaves the account
    # and the reported return drifts down regardless of whether the strategy
    # works.
    open_before = set(account.open_positions())
    if open_before:
        resolutions = fetch_resolutions(open_before, fetch=http_fetch)
        report = settle_account(account, resolutions)
        if report.settled:
            print(f"  {report.summary()}", flush=True)

    already_held = set(account.open_positions())
    priced = [
        b for b in complete_baskets(events) if b.paper_edge() > PAPER_EDGE_FLOOR
    ]
    print(f"  {len(priced)} complete baskets clear {PAPER_EDGE_FLOOR:.1%} "
          "on paper", flush=True)

    # Horizon filter. A basket with no stated end date is skipped rather than
    # assumed short: an unknown lock-up is the thing being avoided.
    candidates = []
    no_date = long_dated = 0
    for basket in priced:
        days = basket.days_to_resolution()
        if days is None:
            no_date += 1
        elif days > args.max_days:
            long_dated += 1
        else:
            candidates.append(basket)
    print(f"  {len(candidates)} resolve within {args.max_days:g}d "
          f"({long_dated} too far out, {no_date} undated)", flush=True)

    # Soonest first: same edge, less time with the capital locked.
    candidates.sort(key=lambda b: (b.days_to_resolution() or 0.0))

    baskets_to_submit: list[PaperBasket] = []
    marks: dict[str, float] = {}
    checked = 0
    for basket in candidates:
        if checked >= MAX_CANDIDATES:
            break
        if any(t in already_held for t in basket.token_ids if t):
            continue  # already positioned in this basket; do not pile in
        checked += 1
        priced = price_basket(basket)
        if priced is None:
            continue
        paper_basket, basket_marks = priced
        baskets_to_submit.append(paper_basket)
        marks.update(basket_marks)

    # Mark every already-open position too, not just this tick's candidates,
    # so equity and the circuit breaker see the whole book.
    unmarkable = 0
    for symbol in account.open_positions():
        if symbol in marks:
            continue
        # A resolved market's token is delisted and 404s permanently. That is a
        # terminal state, not a transient error, so it must not abort the tick:
        # letting it raise here killed every run for four days.
        book = fetch_book_or_none(symbol)
        if book is None:
            unmarkable += 1
            continue
        if book.best_bid is not None:
            marks[symbol] = book.best_bid
    if unmarkable:
        print(f"  {unmarkable} open position(s) have no live book "
              "(resolved or delisted) and are excluded from equity")

    print(f"  {len(baskets_to_submit)} baskets confirmed against live depth, "
          "submitting", flush=True)
    result = runner.run_once(baskets_to_submit, marks)

    print("\n" + "=" * 72)
    print(result.summary())
    print("=" * 72)
    for fill in result.filled:
        print(f"  FILLED {fill.side:<4} {fill.qty:>8,.0f} @ {fill.price:.4f}  "
              f"fee ${fill.fee:.4f}  {fill.note[:60]}")
    for rationale, reason in result.rejected:
        print(f"  rejected: {reason}  ({rationale[:60]})")

    snap = result.snapshot
    print(f"\n  cash            ${snap.cash:,.2f}")
    print(f"  market value    ${snap.market_value:,.2f}")
    print(f"  equity          ${snap.equity:,.2f}")
    print(f"  return          {snap.total_return:+.2%}")
    print(f"  realised P&L    ${snap.realized_pnl:,.2f}")
    print(f"  unrealised P&L  ${snap.unrealized_pnl:,.2f}")
    if snap.unmarked:
        print(f"  WARNING: {len(snap.unmarked)} open position(s) have no live "
              f"mark and are excluded from equity above: {snap.unmarked}")
    if result.halted:
        print("\n  CIRCUIT BREAKER TRIPPED -- trading is halted until reset().")

    account.close()


if __name__ == "__main__":
    main()
