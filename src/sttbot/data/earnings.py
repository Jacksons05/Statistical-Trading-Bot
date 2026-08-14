"""Earnings surprises and the prices to measure drift against.

Two sources, split by what needs a credential:

* **Earnings** come from Alpha Vantage, which needs an API key. When
  ``ALPHAVANTAGE_API_KEY`` is set this fetches directly; otherwise it reads a
  cache written out-of-band (an MCP client, a manual download). The parsing and
  the SUE maths are the same either way, so the strategy code never knows which
  path supplied the data.
* **Prices** come from Yahoo's chart endpoint, which needs no key. (Stooq was
  the obvious choice and is documented as reachable elsewhere in this repo, but
  it now sits behind a JavaScript proof-of-work challenge and cannot be fetched
  from a script.)

The one field here that decides whether a PEAD backtest is honest is
``report_time``. A company reporting *after* the close cannot be traded on that
day: the first price a real trader could act at is the next session's. Treating
the announcement day's close as the entry is the classic way a PEAD study
invents returns, and it is worth more than every other refinement combined.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

ALPHAVANTAGE = "https://www.alphavantage.co/query?function=EARNINGS&symbol={}&apikey={}"
YAHOO_CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{}"
    "?period1={}&period2={}&interval=1d"
)

Fetch = Callable[[str], object]

PRE_MARKET = "pre-market"
POST_MARKET = "post-market"


def http_json(url: str, *, timeout: float = 30.0, attempts: int = 3) -> object:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (sttbot research)"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(2.0**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _num(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # reject NaN


def _date(value: object) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# --- earnings ---------------------------------------------------------------


@dataclass(frozen=True)
class EarningsEvent:
    """One quarterly report, as announced."""

    symbol: str
    fiscal_end: dt.date
    reported_date: dt.date
    reported_eps: float
    estimated_eps: float
    report_time: str  # "pre-market" | "post-market" | ""

    @property
    def surprise(self) -> float:
        return self.reported_eps - self.estimated_eps

    def first_tradeable_date(self) -> dt.date:
        """Earliest calendar date a trader could act on this announcement.

        A post-market report is not actionable until the following day. This
        returns a calendar date; aligning it to an actual trading session is
        the price series' job, since holidays are not knowable from here.
        """
        if self.report_time == POST_MARKET:
            return self.reported_date + dt.timedelta(days=1)
        return self.reported_date


def parse_earnings(payload: object, symbol: str | None = None) -> list[EarningsEvent]:
    """Parse an Alpha Vantage EARNINGS payload, oldest first.

    Rows missing a reported date, a reported EPS or an estimate are dropped:
    without an estimate there is no surprise, and without a date there is no
    event to align prices to.
    """
    if not isinstance(payload, dict):
        return []
    sym = str(payload.get("symbol") or symbol or "")
    out: list[EarningsEvent] = []
    for row in payload.get("quarterlyEarnings") or []:
        if not isinstance(row, dict):
            continue
        reported = _date(row.get("reportedDate"))
        fiscal = _date(row.get("fiscalDateEnding"))
        actual = _num(row.get("reportedEPS"))
        estimate = _num(row.get("estimatedEPS"))
        if reported is None or fiscal is None or actual is None or estimate is None:
            continue
        out.append(
            EarningsEvent(
                symbol=sym,
                fiscal_end=fiscal,
                reported_date=reported,
                reported_eps=actual,
                estimated_eps=estimate,
                report_time=str(row.get("reportTime") or ""),
            )
        )
    out.sort(key=lambda e: e.reported_date)
    return out


def fetch_earnings(symbol: str, *, fetch: Fetch | None = None) -> list[EarningsEvent]:
    """Fetch from Alpha Vantage. Requires ``ALPHAVANTAGE_API_KEY``."""
    if fetch is None:
        key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "no ALPHAVANTAGE_API_KEY set; use load_earnings_cache() instead"
            )
        fetch = http_json
        return parse_earnings(fetch(ALPHAVANTAGE.format(symbol, key)), symbol)
    return parse_earnings(fetch(symbol), symbol)


def load_earnings_cache(path: str) -> dict[str, list[EarningsEvent]]:
    """Read a JSONL cache of raw Alpha Vantage payloads, one per line."""
    out: dict[str, list[EarningsEvent]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events = parse_earnings(json.loads(line))
            if events:
                out[events[0].symbol] = events
    return out


def trailing_sue(
    events: Sequence[EarningsEvent], index: int, *, window: int = 8, min_history: int = 4
) -> float | None:
    """SUE for ``events[index]`` using only surprises that preceded it.

    This is the whole ballgame for look-ahead. Standardising by the dispersion
    of *all* surprises, including ones that had not happened yet, quietly tells
    a 2015 trade how volatile 2024's results were. The denominator here is the
    standard deviation of the previous ``window`` surprises only.

    Returns ``None`` when there is not yet enough history, or when the trailing
    surprises are identical (a zero denominator is undefined, not infinite
    conviction).
    """
    if index < 0 or index >= len(events):
        raise IndexError("index outside events")
    prior = events[max(0, index - window):index]
    if len(prior) < min_history:
        return None
    surprises = [e.surprise for e in prior]
    mean = sum(surprises) / len(surprises)
    var = sum((s - mean) ** 2 for s in surprises) / (len(surprises) - 1)
    sd = var**0.5
    if sd <= 0:
        return None
    return events[index].surprise / sd


# --- prices -----------------------------------------------------------------


@dataclass(frozen=True)
class PriceSeries:
    """Adjusted daily closes, oldest first."""

    symbol: str
    dates: tuple[dt.date, ...]
    closes: tuple[float, ...]

    def index_on_or_after(self, when: dt.date) -> int | None:
        """First trading session on or after ``when``.

        Announcements land on weekends and holidays, and a post-market report
        on a Friday is not tradeable until Monday. Resolving that here keeps
        the calendar arithmetic out of the strategy.
        """
        for i, d in enumerate(self.dates):
            if d >= when:
                return i
        return None

    def forward_return(self, start: int, horizon: int) -> float | None:
        """Simple return from ``start`` to ``start + horizon`` sessions."""
        end = start + horizon
        if start < 0 or end >= len(self.closes):
            return None
        p0 = self.closes[start]
        if p0 <= 0:
            return None
        return self.closes[end] / p0 - 1.0


def parse_yahoo_chart(payload: object, symbol: str | None = None) -> PriceSeries | None:
    """Parse a Yahoo chart response into adjusted closes.

    Adjusted closes are used deliberately: a split or dividend inside the
    holding window would otherwise read as a return the trade never earned.
    """
    if not isinstance(payload, dict):
        return None
    results = ((payload.get("chart") or {}).get("result")) or []
    if not results:
        return None
    r = results[0]
    stamps = r.get("timestamp") or []
    indicators = r.get("indicators") or {}
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    if not adj:
        adj = (indicators.get("quote") or [{}])[0].get("close")
    if not stamps or not adj:
        return None

    pairs = []
    for ts, px in zip(stamps, adj):
        value = _num(px)
        if ts is None or value is None or value <= 0:
            continue
        pairs.append((dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).date(), value))
    if not pairs:
        return None
    pairs.sort()
    sym = str(((r.get("meta") or {}).get("symbol")) or symbol or "")
    return PriceSeries(sym, tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


def fetch_prices(
    symbol: str, start: dt.date, end: dt.date, *, fetch: Fetch = http_json
) -> PriceSeries | None:
    p1 = int(dt.datetime.combine(start, dt.time()).timestamp())
    p2 = int(dt.datetime.combine(end, dt.time()).timestamp())
    try:
        payload = fetch(YAHOO_CHART.format(symbol, p1, p2))
    except Exception:
        return None
    return parse_yahoo_chart(payload, symbol)
