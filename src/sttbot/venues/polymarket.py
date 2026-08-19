"""Polymarket market data: the full open universe, and what is tradeable in it.

Two feeds, and they answer different questions:

* **gamma** (``/events/keyset``) enumerates every market with a top-of-book
  price. It is the only way to see the whole venue, but its ``bestBid`` and
  ``bestAsk`` carry **no size**, so a price there is a rumour, not a trade.
* **CLOB** (``/book``) gives the real order book for one token. This is where a
  candidate is confirmed or killed.

The scan pattern that follows from that split is: enumerate cheaply on gamma,
then re-price the shortlist against CLOB depth before believing anything.

Two traps are encoded here because both silently produce wrong answers rather
than errors:

*Pagination.* The gamma response field is ``next_cursor`` but the request
parameter is ``after_cursor``. Passing the value back under the name it arrived
with is accepted and ignored, so you re-read page one forever and a "complete"
scan quietly contains one page repeated. Plain ``offset`` is not an escape --
it hard-caps around 2,400 while the real universe is far larger, so it
truncates silently too.

*Basket completeness.* A multi-outcome basket is only an arbitrage if it spans
**every** outcome. Scoring the subset of legs that happen to be quoted is the
single easiest way to manufacture a spectacular fake edge: drop half the legs
and the rest naturally sum below 1.00, but the missing outcomes can still
happen and you are simply short them, unhedged.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from .prediction import FeeModel

GAMMA_EVENTS_KEYSET = "https://gamma-api.polymarket.com/events/keyset"
CLOB_BOOK = "https://clob.polymarket.com/book"

# The response names this next_cursor; the request must name it after_cursor.
CURSOR_REQUEST_PARAM = "after_cursor"
CURSOR_RESPONSE_FIELD = "next_cursor"

Fetch = Callable[[str], dict]


def http_fetch(url: str, *, timeout: float = 60.0, attempts: int = 4) -> dict:
    """GET ``url`` as JSON with bounded exponential backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "sttbot/0.1 (research)"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:  # network, timeout, or malformed JSON
            last = exc
            if attempt < attempts - 1:
                time.sleep(2.0**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}")


# --- enumeration ------------------------------------------------------------


def iter_events(
    *,
    closed: bool = False,
    page: int = 100,
    fetch: Fetch = http_fetch,
    pause: float = 0.25,
    max_pages: int | None = None,
) -> Iterator[dict]:
    """Yield every event from gamma, following keyset pagination correctly.

    Stops on an empty page, a missing cursor, or a cursor that repeats -- the
    last of these is what a wrong parameter name looks like from the outside,
    so it is treated as termination rather than looping forever.
    """
    cursor: str | None = None
    seen: set[str] = set()
    pages = 0

    while True:
        url = f"{GAMMA_EVENTS_KEYSET}?limit={page}&closed={str(closed).lower()}"
        if cursor:
            url += f"&{CURSOR_REQUEST_PARAM}={cursor}"
        payload = fetch(url)

        events = payload.get("events") or []
        if not events:
            return
        yield from events

        pages += 1
        if max_pages is not None and pages >= max_pages:
            return

        cursor = payload.get(CURSOR_RESPONSE_FIELD)
        if not cursor or cursor in seen:
            return
        seen.add(cursor)
        if pause:
            time.sleep(pause)


# --- market predicates ------------------------------------------------------


def as_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def is_live(market: dict) -> bool:
    """Whether an order could actually rest on this market right now."""
    return bool(
        market.get("active")
        and market.get("acceptingOrders")
        and market.get("enableOrderBook")
        and not market.get("closed")
        and not market.get("archived")
    )


def two_sided_quote(market: dict) -> tuple[float, float] | None:
    """``(bid, ask)`` when both sides are genuinely quoted, else ``None``.

    Rejects crossed books and the 0/1 degenerate prices that a resolved or
    abandoned market leaves behind.
    """
    bid = as_float(market.get("bestBid"))
    ask = as_float(market.get("bestAsk"))
    if bid is None or ask is None:
        return None
    if not 0.0 < bid <= ask < 1.0:
        return None
    return bid, ask


def market_fee_model(market: dict) -> FeeModel:
    """Fee model from the market's own ``feeSchedule``.

    Polymarket charges takers and rebates makers, so the maker leg is income.
    Reading the schedule per market rather than assuming a venue-wide constant
    matters because the rebate rate is not uniform.
    """
    if not market.get("feesEnabled"):
        return FeeModel()
    schedule = market.get("feeSchedule") or {}
    rate = as_float(schedule.get("rate"), 0.0) or 0.0
    rebate = as_float(schedule.get("rebateRate"), 0.0) or 0.0
    return FeeModel(quadratic_taker=rate, quadratic_maker=-rate * rebate)


def yes_token_id(market: dict) -> str | None:
    """CLOB token id for the YES outcome."""
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return str(ids[0]) if ids else None


# --- order book -------------------------------------------------------------


@dataclass(frozen=True)
class OrderBook:
    """A CLOB book, both sides sorted best-price-first."""

    bids: tuple[tuple[float, float], ...]  # descending price
    asks: tuple[tuple[float, float], ...]  # ascending price

    @classmethod
    def from_payload(cls, payload: dict) -> OrderBook:
        # The API returns asks worst-price-first, which reads as a best price
        # of 0.98 when the real one is 0.45.
        def levels(key: str) -> list[tuple[float, float]]:
            out = []
            for entry in payload.get(key) or []:
                price = as_float(entry.get("price"))
                size = as_float(entry.get("size"))
                if price is not None and size is not None and size > 0:
                    out.append((price, size))
            return out

        return cls(
            bids=tuple(sorted(levels("bids"), reverse=True)),
            asks=tuple(sorted(levels("asks"))),
        )

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def ask_depth(self) -> float:
        return sum(size for _, size in self.asks)

    def cost_to_buy(self, contracts: float) -> float | None:
        """Cost of buying ``contracts`` by walking the asks.

        ``None`` when the book is too thin to fill the whole order -- a partial
        fill is not a completed basket, so it is not a fillable size.
        """
        if contracts <= 0:
            raise ValueError("contracts must be positive")
        remaining, cost = contracts, 0.0
        for price, size in self.asks:
            take = min(remaining, size)
            cost += take * price
            remaining -= take
            if remaining <= 1e-9:
                return cost
        return None


def fetch_book(token_id: str, *, fetch: Fetch = http_fetch) -> OrderBook:
    """The live book for a token. Raises if it cannot be fetched.

    Deliberately strict: a candidate you are about to trade must not be priced
    off a book you failed to read.
    """
    return OrderBook.from_payload(fetch(f"{CLOB_BOOK}?token_id={token_id}"))


def fetch_book_or_none(
    token_id: str, *, fetch: Fetch = http_fetch
) -> OrderBook | None:
    """The live book, or ``None`` when there isn't one.

    For *marking* an existing position rather than pricing a new trade. When a
    market resolves its token is delisted and the CLOB returns 404 forever,
    which is a real terminal state and not a transient failure. Letting that
    raise killed a paper-trading run on every tick for four days while the
    account sat frozen, because one held position had resolved.

    Callers should treat ``None`` as "no mark available" and surface it --
    :attr:`sttbot.paper.account.Snapshot.unmarked` exists for exactly this --
    rather than substituting a price nobody is quoting.
    """
    try:
        return fetch_book(token_id, fetch=fetch)
    except Exception:
        return None


# --- multi-outcome baskets --------------------------------------------------


@dataclass(frozen=True)
class Basket:
    """A complete set of mutually exclusive, exhaustive outcomes."""

    event_slug: str
    title: str
    outcomes: tuple[str, ...]
    token_ids: tuple[str | None, ...]
    bids: tuple[float, ...]
    asks: tuple[float, ...]
    fees: tuple[FeeModel, ...]
    liquidity: float
    volume_24h: float
    end_date: str = ""  # ISO8601 from gamma; "" when the event omits one

    def days_to_resolution(self, now: dt.datetime | None = None) -> float | None:
        """Days until the event's stated end, or ``None`` if it has none.

        Negative for a market already past its end date. Those are common and
        not an error -- the venue had 17,296 live markets past their endDate in
        one snapshot, mostly in-play sport awaiting settlement -- and they are
        the *fastest* capital to recycle, so they are worth including rather
        than filtering out.

        The horizon is what decides how long capital sits, and it has nothing
        to do with the size of the edge: a basket paying 1% in three days and
        one paying 1% in five months look identical to an edge filter.
        """
        if not self.end_date:
            return None
        text = self.end_date.replace("Z", "+00:00")
        try:
            end = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        current = now or dt.datetime.now(dt.timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt.timezone.utc)
        return (end - current).total_seconds() / 86400.0

    @property
    def ask_sum(self) -> float:
        return sum(self.asks)

    @property
    def bid_sum(self) -> float:
        return sum(self.bids)

    def paper_edge(self) -> float:
        """Best per-contract edge at top-of-book, after taker fees.

        Top-of-book only, so this is a shortlist filter and nothing more --
        confirm with :func:`executable_arbitrage` before believing it.
        """
        buy_fee = sum(f.cost(p) for f, p in zip(self.fees, self.asks))
        sell_fee = sum(f.cost(p) for f, p in zip(self.fees, self.bids))
        return max((1.0 - self.ask_sum) - buy_fee, (self.bid_sum - 1.0) - sell_fee)


def complete_baskets(events: Sequence[dict]) -> list[Basket]:
    """Baskets for negRisk events where *every* outcome is live and quoted.

    Events with any unquoted or inactive leg are dropped rather than scored on
    the remainder: an incomplete basket is an unhedged short on the outcomes
    you left out, not an arbitrage.
    """
    baskets: list[Basket] = []
    for event in events:
        if not event.get("negRisk"):
            continue
        legs = event.get("markets") or []
        if len(legs) < 2:
            continue
        if not all(is_live(m) for m in legs):
            continue
        quotes = [two_sided_quote(m) for m in legs]
        if any(q is None for q in quotes):
            continue

        baskets.append(
            Basket(
                event_slug=str(event.get("slug") or ""),
                title=str(event.get("title") or event.get("slug") or ""),
                outcomes=tuple(
                    str(m.get("groupItemTitle") or m.get("question") or "") for m in legs
                ),
                token_ids=tuple(yes_token_id(m) for m in legs),
                bids=tuple(q[0] for q in quotes),  # type: ignore[index]
                asks=tuple(q[1] for q in quotes),  # type: ignore[index]
                fees=tuple(market_fee_model(m) for m in legs),
                liquidity=sum(as_float(m.get("liquidityNum"), 0.0) or 0.0 for m in legs),
                volume_24h=sum(as_float(m.get("volume24hr"), 0.0) or 0.0 for m in legs),
                end_date=str(event.get("endDate") or ""),
            )
        )
    return baskets


@dataclass(frozen=True)
class ExecutableArb:
    contracts: float
    capital: float
    profit: float

    @property
    def edge_per_contract(self) -> float:
        return self.profit / self.contracts if self.contracts else 0.0

    @property
    def return_on_capital(self) -> float:
        return self.profit / self.capital if self.capital else 0.0


def executable_arbitrage(
    books: Sequence[OrderBook],
    fees: Sequence[FeeModel],
    *,
    step: float = 5.0,
    max_contracts: float = 20_000.0,
    min_profit: float = 1e-6,
) -> ExecutableArb | None:
    """Largest genuinely fillable buy-the-basket profit, walking real depth.

    Sizes in ``step`` increments because venues enforce a minimum order size,
    and stops at the first size the book cannot fill. Returns ``None`` when no
    size clears fees -- which is the common outcome for a candidate that looked
    profitable at top of book.

    ``min_profit`` exists because a basket priced exactly at par accumulates
    floating-point dust (three legs at 1/3 sum to 0.9999999999999998), and a
    1e-13 "profit" reported as an arbitrage is worse than no answer.
    """
    if len(books) != len(fees):
        raise ValueError("books and fees must correspond")
    if not books:
        return None

    best: ExecutableArb | None = None
    contracts = step
    while contracts <= max_contracts:
        spend = 0.0
        fee_total = 0.0
        fillable = True
        for book, fee in zip(books, fees):
            cost = book.cost_to_buy(contracts)
            if cost is None:
                fillable = False
                break
            spend += cost
            fee_total += fee.cost(cost / contracts, contracts=contracts)
        if not fillable:
            break

        profit = contracts - spend - fee_total
        if best is None or profit > best.profit:
            best = ExecutableArb(contracts, spend + fee_total, profit)
        contracts += step

    if best is None or best.profit < min_profit:
        return None
    return best
