"""Settle paper positions whose markets have resolved.

Without this a hold-to-resolution book is unmeasurable. It pays cash out to buy
a basket and never collects the $1 the basket was bought to receive, so equity
can only ever drift downward and the reported P&L says nothing about whether
the strategy works. On a real account, a 20-basket book showed -11.7% while
being worth +0.8% once its resolved legs were valued.

The whole difficulty is deciding what "resolved" means, and the venue does not
say it directly.

``closed == true`` is **not** sufficient. Surveyed across 60 closed Polymarket
markets, 36 reported ``outcomePrices`` of ``["0", "0"]`` and one reported
``["0.58", "0.42"]``. Neither is a settlement: the first is a market that has
stopped trading without a payout being written, the second is simply the last
price anyone traded at. Settling on either would be catastrophic in opposite
directions -- ``["0","0"]`` would zero out *every* leg of a basket including
the winner, turning a $1 payout into nothing, and a fractional price would
book a mark as if it were an outcome.

So a resolution is only accepted when the prices form an actual binary
payout: they sum to 1, one of them is ~1 and the rest are ~0. Anything else
leaves the position open and is reported, because an unsettled position is a
knowable state and a wrongly settled one is a silent, permanent error in the
ledger.

Settlement is recorded as an ordinary fill -- a sale of the whole position at
the settled price -- so cash, average cost and realised P&L all fall out of
the existing ledger with no special-casing. Redemption is not a trade, so no
fee is charged.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .account import BUY, SELL, Fill, PaperAccount

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"

# A settled binary pays exactly 1 on one outcome and 0 on the others.
_SUM_TOLERANCE = 1e-3
_WINNER_FLOOR = 0.99
_LOSER_CEILING = 0.01

Fetch = Callable[[str], object]


@dataclass(frozen=True)
class Resolution:
    """A token whose market has genuinely paid out."""

    token_id: str
    settled_price: float
    question: str
    market_id: str


def is_settled(prices: Sequence[float]) -> bool:
    """Whether these outcome prices represent a real payout.

    Rejects the two states that masquerade as one: all-zero (closed without a
    payout) and fractional (a last-traded mark, not an outcome).
    """
    if len(prices) < 2:
        return False
    if abs(sum(prices) - 1.0) > _SUM_TOLERANCE:
        return False
    if max(prices) < _WINNER_FLOOR:
        return False
    others = sorted(prices)[:-1]
    return all(p <= _LOSER_CEILING for p in others)


def parse_resolution(market: dict) -> list[Resolution]:
    """Resolutions for each token of a market, or [] if it has not settled."""
    if not market.get("closed"):
        return []
    try:
        tokens = json.loads(market.get("clobTokenIds") or "[]")
        prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if len(tokens) != len(prices) or not is_settled(prices):
        return []
    return [
        Resolution(
            token_id=str(t),
            settled_price=p,
            question=str(market.get("question") or ""),
            market_id=str(market.get("id") or ""),
        )
        for t, p in zip(tokens, prices)
    ]


def fetch_resolutions(
    token_ids: Iterable[str],
    *,
    fetch: Fetch,
    batch_size: int = 20,
    pause: float = 0.3,
) -> dict[str, Resolution]:
    """Look up settled prices for the given tokens.

    Gamma's ``clob_token_ids`` filter takes *repeated* parameters; a
    comma-separated list is rejected outright as "invalid clob token ids".
    ``closed=true`` is required or resolved markets are filtered out of the
    default view entirely and every lookup comes back empty.
    """
    import time

    wanted = list(dict.fromkeys(token_ids))
    out: dict[str, Resolution] = {}
    for start in range(0, len(wanted), batch_size):
        batch = wanted[start:start + batch_size]
        query = "&".join(f"clob_token_ids={t}" for t in batch) + "&closed=true"
        try:
            payload = fetch(f"{GAMMA_MARKETS}?{query}")
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for market in payload:
            if isinstance(market, dict):
                for resolution in parse_resolution(market):
                    if resolution.token_id in wanted:
                        out[resolution.token_id] = resolution
        if pause and start + batch_size < len(wanted):
            time.sleep(pause)
    return out


@dataclass(frozen=True)
class SettlementReport:
    settled: tuple[Fill, ...]
    proceeds: float
    still_open: int

    def summary(self) -> str:
        return (
            f"settled {len(self.settled)} position(s) for ${self.proceeds:,.2f}; "
            f"{self.still_open} still open"
        )


def settle_account(
    account: PaperAccount,
    resolutions: dict[str, Resolution],
    *,
    now: dt.datetime | None = None,
) -> SettlementReport:
    """Close every open position whose market has resolved.

    Recorded as a sale of the full position at the settled price, so a winner
    credits its face value and a loser closes at zero. Both realise the P&L
    that was sitting unrealised.

    Idempotent on the market id: re-running after a settled market is still
    listed will not pay out twice.
    """
    stamp = now or dt.datetime.now(dt.timezone.utc)
    fills: list[Fill] = []
    proceeds = 0.0

    for symbol, position in account.open_positions().items():
        resolution = resolutions.get(symbol)
        if resolution is None:
            continue
        fill = Fill(
            ts=stamp,
            symbol=symbol,
            # Closing a long is a sale; a short position buys back.
            side=SELL if position.qty > 0 else BUY,
            qty=abs(position.qty),
            price=resolution.settled_price,
            fee=0.0,  # redemption is not a trade
            strategy="settlement",
            ref=f"settle:{symbol}:{resolution.market_id}",
            note=f"resolved @ {resolution.settled_price:.0f} :: {resolution.question}"[:200],
        )
        if account.record(fill):
            fills.append(fill)
            proceeds += abs(position.qty) * resolution.settled_price

    return SettlementReport(
        settled=tuple(fills),
        proceeds=proceeds,
        still_open=len(account.open_positions()),
    )
