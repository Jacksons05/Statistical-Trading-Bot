"""Kalshi market data, with the traps this venue sets written down.

Measured against Polymarket on the two things that cap a book, Kalshi is the
better venue: at matched liquidity (>=$10k of 24h volume) its markets quote a
**1 cent spread at every price band**, against Polymarket's ~2c, and **768 of
its markets clear that bar against Polymarket's 57**. Its fee is worse -- 7%
quadratic against 5%, so 3.50% of a 50c contract against 2.50% -- but spread
dominates fee at every band, so the total cost to cross is lower everywhere.

Four things here will silently produce wrong answers if taken at face value.

**Use /events, not /markets.** A full paginated walk of ``/markets`` returns
tens of thousands of rows that are 99.7% ``KXMVECROSSCATEGORY`` -- auto-generated
cross-category parlay combinations that nobody trades. One such scan reported
$15,079 of venue volume against a true figure near $72M, an error of ~4,800x,
while paginating perfectly and looking entirely healthy.

**The order book is two bid ladders, not bids and asks.** ``orderbook_fp``
carries ``yes_dollars`` and ``no_dollars``, and *both are bids*. There is no ask
ladder: the YES ask is the complement of the NO bid, so buying YES means
lifting NO bids at ``1 - price``. Reading ``yes_dollars`` as asks prices every
trade off the wrong side of the book.

**``mutually_exclusive`` does not mean exhaustive.** It means the outcomes
cannot co-occur. "LA-01 Republican nominee?" is flagged mutually exclusive and
lists two candidates whose asks sum to 0.108 -- the field says nothing about
whether the listed outcomes cover the probability space, which is what an
arbitrage basket needs. Polymarket's ``negRisk`` does guarantee that; this does
not.

**``liquidity_dollars`` is unusable.** It reads 0.0000 on markets doing $52k of
daily volume.

Prices are decimal dollars ("0.0460"), not cents.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from .prediction import FeeModel

API = "https://api.elections.kalshi.com/trade-api/v2"
EVENTS = API + "/events"
ORDERBOOK = API + "/markets/{}/orderbook"

# Kalshi's published fee: 0.07 * contracts * p * (1-p), charged on the trade.
KALSHI_FEE_RATE = 0.07
KALSHI_FEES = FeeModel(quadratic_taker=KALSHI_FEE_RATE, quadratic_maker=KALSHI_FEE_RATE)

# An implied edge larger than this is treated as evidence the basket is missing
# outcomes rather than as an opportunity. Measured across 4,468 quoted
# mutually-exclusive events, ask_sum runs p05 1.000 / median 1.090, and only
# 0.8% fall below 0.90 -- so a basket implying 80c of edge is not a find, it is
# a basket whose other legs were never listed. The cost of this guard is that
# genuine edge above the threshold is invisible; real arbitrage on a liquid
# venue is low single digits, so that is a trade worth making.
MAX_PLAUSIBLE_EDGE = 0.10

Fetch = Callable[[str], object]


def http_fetch(url: str, *, timeout: float = 45.0, attempts: int = 4) -> object:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "sttbot/0.1 (research)"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(2.0**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def as_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# --- enumeration ------------------------------------------------------------


def iter_events(
    *,
    status: str = "open",
    page: int = 200,
    fetch: Fetch = http_fetch,
    pause: float = 0.2,
    max_pages: int | None = None,
) -> Iterator[dict]:
    """Yield events with their markets nested.

    Deliberately not built on ``/markets``: that endpoint is swamped by
    auto-generated parlay combinations (see the module docstring). Terminates on
    an empty page, a missing cursor, or a cursor that repeats.
    """
    cursor: str | None = None
    seen: set[str] = set()
    pages = 0

    while True:
        url = f"{EVENTS}?limit={page}&status={status}&with_nested_markets=true"
        if cursor:
            url += f"&cursor={cursor}"
        payload = fetch(url)
        if not isinstance(payload, dict):
            return

        events = payload.get("events") or []
        if not events:
            return
        for event in events:
            ticker = str(event.get("event_ticker") or "")
            if ticker and ticker in seen:
                continue
            if ticker:
                seen.add(ticker)
            yield event

        pages += 1
        if max_pages is not None and pages >= max_pages:
            return
        nxt = payload.get("cursor")
        if not nxt or nxt == cursor:
            return
        cursor = nxt
        if pause:
            time.sleep(pause)


def is_live(market: dict) -> bool:
    return str(market.get("status") or "").lower() in {"active", "open"}


def two_sided_quote(market: dict) -> tuple[float, float] | None:
    """``(yes_bid, yes_ask)`` in dollars, or ``None``.

    Rejects crossed books and the 0/1 degenerate quotes a settled or untouched
    market leaves behind.
    """
    bid = as_float(market.get("yes_bid_dollars"))
    ask = as_float(market.get("yes_ask_dollars"))
    if bid is None or ask is None:
        return None
    if not 0.0 < bid <= ask < 1.0:
        return None
    return bid, ask


# --- order book -------------------------------------------------------------


@dataclass(frozen=True)
class OrderBook:
    """A Kalshi book, normalised to YES-side asks.

    ``yes_asks`` is derived from the NO bid ladder, because that is what a YES
    buyer actually lifts. Levels are ``(price, contracts)``, best (cheapest)
    first.
    """

    yes_asks: tuple[tuple[float, float], ...]
    yes_bids: tuple[tuple[float, float], ...]

    @classmethod
    def from_payload(cls, payload: object) -> OrderBook | None:
        if not isinstance(payload, dict):
            return None
        book = payload.get("orderbook_fp") or payload.get("orderbook")
        if not isinstance(book, dict):
            return None

        def levels(key: str) -> list[tuple[float, float]]:
            out = []
            for row in book.get(key) or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                price, size = as_float(row[0]), as_float(row[1])
                if price is None or size is None or size <= 0:
                    continue
                if not 0.0 < price < 1.0:
                    continue
                out.append((price, size))
            return out

        # Both ladders are BIDS. A YES ask is the complement of a NO bid, so the
        # best (highest) NO bid becomes the cheapest YES ask.
        yes_asks = sorted((1.0 - p, s) for p, s in levels("no_dollars"))
        yes_bids = sorted(levels("yes_dollars"), reverse=True)
        if not yes_asks and not yes_bids:
            return None
        return cls(tuple(yes_asks), tuple(yes_bids))

    @property
    def best_yes_ask(self) -> float | None:
        return self.yes_asks[0][0] if self.yes_asks else None

    @property
    def best_yes_bid(self) -> float | None:
        return self.yes_bids[0][0] if self.yes_bids else None

    def cost_to_buy_yes(self, contracts: float) -> float | None:
        """Cost of buying ``contracts`` of YES by walking the NO bids.

        ``None`` when the book is too thin to fill the whole order: a partial
        fill does not complete a basket, so it is not a fillable size.
        """
        if contracts <= 0:
            raise ValueError("contracts must be positive")
        remaining, cost = contracts, 0.0
        for price, size in self.yes_asks:
            take = min(remaining, size)
            cost += take * price
            remaining -= take
            if remaining <= 1e-9:
                return cost
        return None


def fetch_orderbook(
    ticker: str, *, depth: int | None = None, fetch: Fetch = http_fetch
) -> OrderBook | None:
    url = ORDERBOOK.format(ticker)
    if depth:
        url += f"?depth={depth}"
    try:
        return OrderBook.from_payload(fetch(url))
    except Exception:
        return None


# --- baskets ----------------------------------------------------------------


@dataclass(frozen=True)
class Basket:
    """A mutually-exclusive event, quoted on every leg."""

    event_ticker: str
    title: str
    tickers: tuple[str, ...]
    outcomes: tuple[str, ...]
    bids: tuple[float, ...]
    asks: tuple[float, ...]
    volume_24h: float

    @property
    def ask_sum(self) -> float:
        return sum(self.asks)

    def paper_edge(self, fees: FeeModel = KALSHI_FEES) -> float:
        """Top-of-book edge per contract-set after fees.

        Carries no size, so it is a shortlist filter only -- confirm against
        real depth before believing it.
        """
        return (1.0 - self.ask_sum) - sum(fees.cost(a) for a in self.asks)

    @property
    def looks_complete(self) -> bool:
        """Whether the listed outcomes plausibly cover the probability space.

        Kalshi does not tell you. A basket whose asks sum far below 1 is not a
        large arbitrage, it is a basket with legs missing -- and taking it is an
        unhedged short on the outcomes nobody quoted.
        """
        return self.ask_sum >= 1.0 - MAX_PLAUSIBLE_EDGE


def complete_baskets(events: Sequence[dict]) -> list[Basket]:
    """Baskets for mutually-exclusive events with every leg live and quoted.

    Events with any unquoted or inactive leg are dropped rather than scored on
    the remainder, and :attr:`Basket.looks_complete` must still be checked
    before treating one as an arbitrage.
    """
    out: list[Basket] = []
    for event in events:
        if not event.get("mutually_exclusive"):
            continue
        legs = event.get("markets") or []
        if len(legs) < 2:
            continue
        if not all(is_live(m) for m in legs):
            continue
        quotes = [two_sided_quote(m) for m in legs]
        if any(q is None for q in quotes):
            continue

        out.append(
            Basket(
                event_ticker=str(event.get("event_ticker") or ""),
                title=str(event.get("title") or ""),
                tickers=tuple(str(m.get("ticker") or "") for m in legs),
                outcomes=tuple(str(m.get("yes_sub_title") or "") for m in legs),
                bids=tuple(q[0] for q in quotes),  # type: ignore[index]
                asks=tuple(q[1] for q in quotes),  # type: ignore[index]
                volume_24h=sum(as_float(m.get("volume_24h_fp"), 0.0) or 0.0
                               for m in legs),
            )
        )
    return out


@dataclass(frozen=True)
class ExecutableArb:
    contracts: float
    capital: float
    profit: float

    @property
    def return_on_capital(self) -> float:
        return self.profit / self.capital if self.capital else 0.0


def executable_arbitrage(
    books: Sequence[OrderBook],
    *,
    fees: FeeModel = KALSHI_FEES,
    step: float = 1.0,
    max_contracts: float = 20_000.0,
    min_profit: float = 1e-6,
) -> ExecutableArb | None:
    """Largest genuinely fillable buy-the-basket profit, walking real depth.

    Stops at the first size the book cannot fill. Returns ``None`` when no size
    clears fees, which is the common outcome for a candidate that looked
    profitable at top of book.
    """
    if not books:
        return None
    best: ExecutableArb | None = None
    contracts = step
    while contracts <= max_contracts:
        spend = 0.0
        fee_total = 0.0
        fillable = True
        for book in books:
            cost = book.cost_to_buy_yes(contracts)
            if cost is None:
                fillable = False
                break
            spend += cost
            fee_total += fees.cost(cost / contracts, contracts=contracts)
        if not fillable:
            break
        profit = contracts - spend - fee_total
        if best is None or profit > best.profit:
            best = ExecutableArb(contracts, spend + fee_total, profit)
        contracts += step

    if best is None or best.profit < min_profit:
        return None
    return best
