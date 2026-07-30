"""Feed real on-chain token data into :mod:`sttbot.strategies.token_screen`.

The screen is only as good as what reaches it, and no single free source
carries every field, so this joins two:

* **DexScreener** -- discovery plus market state: pool liquidity, 24h volume,
  transaction counts, and pair age.
* **GoPlus** -- contract safety: top-holder concentration, LP holders, mint and
  freeze authority, and transfer fees.

The screen treats ``None`` as a failure rather than a pass, so the mapping here
is deliberately conservative about what it claims to know. The one place that
needs care is the difference between *zero* and *unknown*: a token with no
transfer-fee extension genuinely has a 0% tax, while a token whose LP holders
the indexer could not enumerate has an unknown lock fraction. Collapsing those
two into the same value is how a zero-tax token gets scored as if it charged
100%, and how an unauditable token gets scored as safe.

Neither source covers sniper/bundler concentration or a real sell simulation,
so those stay ``None`` and the screen reports them as gaps in the pipeline
rather than silently passing them.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Sequence

from ..strategies.amm import Pool
from ..strategies.token_screen import TokenMetadata

DEXSCREENER_PROFILES = "https://api.dexscreener.com/token-profiles/latest/v1"
DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search"
DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens"
GOPLUS_SOLANA = "https://api.gopluslabs.io/api/v1/solana/token_security"

# DexScreener accepts at most 30 comma-separated addresses per request.
DEXSCREENER_BATCH = 30

Fetch = Callable[[str], object]


def http_fetch(url: str, *, timeout: float = 30.0, attempts: int = 3) -> object:
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


def _num(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# --- DexScreener ------------------------------------------------------------


@dataclass(frozen=True)
class TokenPair:
    """A traded pool, as DexScreener reports it."""

    chain: str
    token_address: str
    symbol: str
    name: str
    liquidity_usd: float | None
    liquidity_base: float | None  # token-side reserve, in tokens
    volume_24h: float | None
    buys_24h: int | None
    sells_24h: int | None
    price_usd: float | None
    fdv: float | None
    price_change_24h: float | None
    pair_created_ms: int | None

    def age_hours(self, now_ms: float) -> float | None:
        if self.pair_created_ms is None:
            return None
        return max((now_ms - self.pair_created_ms) / 3_600_000.0, 0.0)

    @property
    def trades_24h(self) -> int | None:
        if self.buys_24h is None or self.sells_24h is None:
            return None
        return self.buys_24h + self.sells_24h

    def pool(self, fee: float = 0.003) -> Pool | None:
        """Constant-product pool with the numeraire side valued in USD.

        DexScreener reports the quote reserve in the quote token (usually SOL),
        which would make price impact come out in SOL rather than dollars. The
        USD figure covers *both* sides of the pool, so half of it is the
        numeraire leg -- that keeps spot price in dollars per token and makes
        exit sizes directly comparable to a dollar position limit.
        """
        if not self.liquidity_base or not self.liquidity_usd:
            return None
        if self.liquidity_base <= 0 or self.liquidity_usd <= 0:
            return None
        return Pool(
            token_reserve=self.liquidity_base,
            numeraire_reserve=self.liquidity_usd / 2.0,
            fee=fee,
        )


def parse_pair(raw: dict) -> TokenPair:
    base = raw.get("baseToken") or {}
    txns = (raw.get("txns") or {}).get("h24") or {}
    created = raw.get("pairCreatedAt")
    return TokenPair(
        chain=str(raw.get("chainId") or ""),
        token_address=str(base.get("address") or ""),
        symbol=str(base.get("symbol") or ""),
        name=str(base.get("name") or ""),
        liquidity_usd=_num((raw.get("liquidity") or {}).get("usd")),
        liquidity_base=_num((raw.get("liquidity") or {}).get("base")),
        volume_24h=_num((raw.get("volume") or {}).get("h24")),
        buys_24h=int(txns["buys"]) if txns.get("buys") is not None else None,
        sells_24h=int(txns["sells"]) if txns.get("sells") is not None else None,
        price_usd=_num(raw.get("priceUsd")),
        fdv=_num(raw.get("fdv")),
        price_change_24h=_num((raw.get("priceChange") or {}).get("h24")),
        pair_created_ms=int(created) if created is not None else None,
    )


def latest_token_addresses(
    *, chain: str = "solana", fetch: Fetch = http_fetch
) -> list[str]:
    """Recently listed/promoted tokens -- the discovery feed for new launches."""
    payload = fetch(DEXSCREENER_PROFILES)
    if not isinstance(payload, list):
        return []
    return [
        str(entry.get("tokenAddress"))
        for entry in payload
        if entry.get("chainId") == chain and entry.get("tokenAddress")
    ]


def pairs_for_tokens(
    addresses: Sequence[str], *, fetch: Fetch = http_fetch, pause: float = 0.2
) -> dict[str, TokenPair]:
    """Deepest pool per token address, batched within DexScreener's limit.

    When a token trades in several pools the deepest one is the relevant
    market: it sets the price and bounds any realistic exit.
    """
    best: dict[str, TokenPair] = {}
    for start in range(0, len(addresses), DEXSCREENER_BATCH):
        batch = addresses[start : start + DEXSCREENER_BATCH]
        payload = fetch(f"{DEXSCREENER_TOKENS}/{','.join(batch)}")
        pairs = (payload or {}).get("pairs") if isinstance(payload, dict) else None
        for raw in pairs or []:
            pair = parse_pair(raw)
            if not pair.token_address:
                continue
            current = best.get(pair.token_address)
            if current is None or (pair.liquidity_usd or 0) > (current.liquidity_usd or 0):
                best[pair.token_address] = pair
        if pause and start + DEXSCREENER_BATCH < len(addresses):
            time.sleep(pause)
    return best


def search_pairs(query: str, *, fetch: Fetch = http_fetch) -> list[TokenPair]:
    payload = fetch(f"{DEXSCREENER_SEARCH}?q={urllib.parse.quote(query)}")
    if not isinstance(payload, dict):
        return []
    return [parse_pair(raw) for raw in payload.get("pairs") or []]


# --- GeckoTerminal (discovery of *established* tokens) ----------------------
#
# DexScreener's profile feed is newest-first by construction, so it can only
# ever show fresh launches. Answering "what about tokens that survived their
# first month?" needs a source ordered by activity rather than recency.

GECKOTERMINAL_POOLS = (
    "https://api.geckoterminal.com/api/v2/networks/{network}/pools?page={page}"
)

# The free tier allows roughly 30 calls a minute and caps out at page 10.
GECKOTERMINAL_MAX_PAGE = 10
GECKOTERMINAL_PAUSE = 4.0


def geckoterminal_top_tokens(
    *,
    network: str = "solana",
    pages: int = GECKOTERMINAL_MAX_PAGE,
    fetch: Fetch = http_fetch,
    pause: float = GECKOTERMINAL_PAUSE,
) -> list[str]:
    """Base-token addresses of the most-traded pools, ordered by 24h volume.

    Paced by default, because the free tier rate-limits hard and signals it two
    different ways: an HTTP 429, or a 200 whose *body* carries an
    ``error_code`` of 429. The second is the dangerous one -- it parses as a
    normal response, so an unguarded caller reads it as "no more pools" and
    treats a truncated walk as the whole venue.

    Both are treated as a stop signal. The result is then whatever was
    collected before the limit hit, which may be short of ``pages``; callers
    that need to distinguish a complete walk from a truncated one should
    compare the length against ``pages * 20``.
    """
    addresses: list[str] = []
    seen: set[str] = set()

    for page in range(1, min(pages, GECKOTERMINAL_MAX_PAGE) + 1):
        try:
            payload = fetch(GECKOTERMINAL_POOLS.format(network=network, page=page))
        except Exception:
            break  # rate limited or unreachable: keep what we have
        if not isinstance(payload, dict):
            break
        if (payload.get("status") or {}).get("error_code"):
            break

        rows = payload.get("data") or []
        if not rows:
            break
        for row in rows:
            base = ((row.get("relationships") or {}).get("base_token") or {}).get("data")
            ident = str((base or {}).get("id") or "")
            prefix = f"{network}_"
            if ident.startswith(prefix):
                address = ident[len(prefix) :]
                if address not in seen:
                    seen.add(address)
                    addresses.append(address)
        if pause and page < pages:
            time.sleep(pause)

    return addresses


# --- GoPlus -----------------------------------------------------------------


def goplus_security(
    addresses: Sequence[str], *, fetch: Fetch = http_fetch, pause: float = 0.5
) -> dict[str, dict]:
    """Contract-safety records keyed by token address (Solana)."""
    out: dict[str, dict] = {}
    for address in addresses:
        payload = fetch(f"{GOPLUS_SOLANA}?contract_addresses={address}")
        if isinstance(payload, dict):
            out.update(payload.get("result") or {})
        if pause:
            time.sleep(pause)
    return out


def _authority_absent(entry: object) -> bool | None:
    """GoPlus reports authorities as ``{'authority': [...], 'status': '0|1'}``."""
    if not isinstance(entry, dict):
        return None
    status = entry.get("status")
    if status in ("0", "1"):
        return status == "0"
    authority = entry.get("authority")
    if isinstance(authority, list):
        return len(authority) == 0
    return None


def _holder_concentration(security: dict) -> tuple[float | None, float | None]:
    """(top holder share, top-10 share) excluding pool and burn addresses.

    GoPlus returns only the top ten, so the top-10 figure is exact and anything
    deeper is unavailable. Pool and burn addresses are excluded where GoPlus
    tags them; untagged entries are counted, which makes this an *upper* bound
    on real concentration rather than a flattering one.
    """
    holders = security.get("holders")
    if not isinstance(holders, list) or not holders:
        return None, None

    shares = []
    for holder in holders:
        if not isinstance(holder, dict):
            continue
        tag = str(holder.get("tag") or "").lower()
        if any(word in tag for word in ("pool", "lp", "burn", "amm")):
            continue
        share = _num(holder.get("percent"))
        if share is not None:
            shares.append(share)
    if not shares:
        return None, None
    return max(shares), sum(shares)


def _lp_locked_fraction(security: dict) -> float | None:
    """Share of LP tokens locked or burned.

    Returns ``None`` when GoPlus enumerates no LP holders at all: that is an
    indexing gap, not evidence the LP is unlocked, and the two must not collapse
    to the same number.
    """
    lp_holders = security.get("lp_holders")
    if not isinstance(lp_holders, list) or not lp_holders:
        return None
    locked = 0.0
    for holder in lp_holders:
        if not isinstance(holder, dict):
            continue
        share = _num(holder.get("percent")) or 0.0
        tag = str(holder.get("tag") or "").lower()
        if str(holder.get("is_locked")) == "1" or "burn" in tag:
            locked += share
    return locked


def _transfer_taxes(security: dict) -> tuple[float | None, float | None]:
    """(buy tax, sell tax) from the token-2022 transfer-fee extension.

    An absent extension is a genuine 0%, not missing data -- a plain SPL token
    has no mechanism to charge one. Reporting that as unknown would fail a
    clean token on a gap that does not exist.
    """
    fee = security.get("transfer_fee")
    if fee is None:
        return None, None
    if isinstance(fee, dict) and not fee:
        return 0.0, 0.0
    if not isinstance(fee, dict):
        return None, None
    basis = _num(fee.get("transfer_fee_basis_points"))
    if basis is None:
        return None, None
    rate = basis / 10_000.0
    return rate, rate  # the extension charges symmetrically


def to_token_metadata(
    pair: TokenPair, security: dict | None, *, now_ms: float
) -> TokenMetadata:
    """Join a DexScreener pool and a GoPlus record into the screen's input."""
    security = security or {}
    top, top10 = _holder_concentration(security)
    buy_tax, sell_tax = _transfer_taxes(security)

    can_mint = _authority_absent(security.get("mintable"))
    can_freeze = _authority_absent(security.get("freezable"))
    mutable = _authority_absent(security.get("balance_mutable_authority"))
    renounced = (
        None
        if None in (can_mint, can_freeze, mutable)
        else bool(can_mint and can_freeze and mutable)
    )

    holder_count = security.get("holder_count")
    return TokenMetadata(
        symbol=pair.symbol or pair.token_address[:8],
        liquidity_usd=pair.liquidity_usd,
        lp_locked_fraction=_lp_locked_fraction(security),
        top_holder_fraction=top,
        top10_holder_fraction=top10,
        # Neither source exposes launch-block sniper/bundle share, or runs a
        # sell simulation. Left unknown rather than assumed benign.
        sniper_bundled_fraction=None,
        holder_count=int(holder_count) if _num(holder_count) is not None else None,
        age_hours=pair.age_hours(now_ms),
        buy_tax=buy_tax,
        sell_tax=sell_tax,
        can_mint=(not can_mint) if can_mint is not None else None,
        can_freeze=(not can_freeze) if can_freeze is not None else None,
        ownership_renounced=renounced,
        sell_simulation_passed=None,
    )
