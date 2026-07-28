"""Catalog of openly-licensed market datasets, with a caching fetcher.

Every entry here is public, openly licensed, and reachable over plain HTTPS
without an API key. Files are downloaded once into a local cache directory
(override with ``STTBOT_DATA_DIR``) and reused thereafter, so a backtest is
reproducible without re-hitting the network.

Note on network policy: reachability varies by environment. Hosts served from
``raw.githubusercontent.com`` are reachable almost everywhere; direct-egress
venues (exchange REST APIs, prediction-market APIs, SEC EDGAR) are blocked from
some sandboxes. ``LIVE_ENDPOINTS`` records the direct-egress hosts we depend on
along with their access requirements, and ``BLOCKED_SOURCES`` records the ones
that stay unreachable even with unrestricted egress. Run
``python -m sttbot.data.probe`` to re-measure both from the current machine.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CACHE = Path.home() / ".cache" / "sttbot"

# Some publishers gate on User-Agent. SEC EDGAR in particular rejects generic
# agents with 403 and requires a contact address in the header; set
# STTBOT_USER_AGENT to something like "myproject me@example.com" before
# fetching from data.sec.gov.
_DEFAULT_USER_AGENT = "sttbot/0.1 (research; contact via repository issues)"


def user_agent() -> str:
    """User-Agent sent with every download (``STTBOT_USER_AGENT`` overrides)."""
    return os.environ.get("STTBOT_USER_AGENT") or _DEFAULT_USER_AGENT


@dataclass(frozen=True)
class Dataset:
    """A downloadable, openly-licensed dataset."""

    name: str
    url: str
    filename: str
    license: str
    description: str
    approx_bytes: int
    source: str
    sha256: str | None = None  # pin once known, to detect upstream drift


DATASETS: dict[str, Dataset] = {
    "football_matches": Dataset(
        name="football_matches",
        url=(
            "https://raw.githubusercontent.com/xgabora/"
            "Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
        ),
        filename="football_matches.csv",
        license="MIT",
        description=(
            "230k club football matches (2000-2025, 38 divisions, 1214 teams) with "
            "full-time/half-time scores, shots, corners, cards, Elo ratings, and 1X2 / "
            "over-under / Asian-handicap odds from Bet365 plus best-of-book maxima. "
            "The reference dataset for fitting Dixon-Coles and measuring CLV."
        ),
        approx_bytes=43_632_197,
        source="https://github.com/xgabora/Club-Football-Match-Data-2000-2025",
    ),
    "btcusd_1min": Dataset(
        name="btcusd_1min",
        url=(
            "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/"
            "main/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
        ),
        filename="btcusd_bitstamp_1min.csv.gz",
        license="MIT",
        description=(
            "6.85M one-minute BTC/USD OHLCV candles from Bitstamp (2012-01 to 2025-01). "
            "Single-venue, so it supports volatility/momentum research but NOT the "
            "cross-venue arbitrage thesis, which needs synchronised multi-venue books."
        ),
        approx_bytes=94_796_215,
        source="https://github.com/ff137/bitstamp-btcusd-minute-data",
    ),
    "football_json_en1": Dataset(
        name="football_json_en1",
        url=(
            "https://raw.githubusercontent.com/openfootball/football.json/"
            "master/2023-24/en.1.json"
        ),
        filename="football_en1_2023_24.json",
        license="Public Domain (CC0)",
        description=(
            "English Premier League 2023-24 fixtures and results in JSON. Small, "
            "no odds; useful as a lightweight fixture/result cross-check."
        ),
        approx_bytes=60_000,
        source="https://github.com/openfootball/football.json",
    ),
}


@dataclass(frozen=True)
class LiveEndpoint:
    """A direct-egress host we read from, and what it takes to use it."""

    host: str
    probe_url: str
    purpose: str
    notes: str


# Direct-egress hosts confirmed reachable on 2026-07-28 from an unrestricted
# network. An earlier sandbox saw 403 on all of these because its proxy blocked
# every non-allowlisted host -- that was a network-policy artifact, not an
# upstream restriction, so treat a blanket 403 here as "re-probe" rather than
# "the venue is gone".
LIVE_ENDPOINTS: dict[str, LiveEndpoint] = {
    "polymarket": LiveEndpoint(
        host="gamma-api.polymarket.com",
        probe_url="https://gamma-api.polymarket.com/markets?limit=1",
        purpose="prediction-market prices for multi-outcome boundary arbitrage",
        notes="public read API, no key; returns JSON market list with prices",
    ),
    "kalshi": LiveEndpoint(
        host="api.elections.kalshi.com",
        probe_url="https://api.elections.kalshi.com/trade-api/v2/markets?limit=1",
        purpose="prediction-market prices for multi-outcome boundary arbitrage",
        notes="public read on /trade-api/v2/markets; trading needs a signed key",
    ),
    "sec_edgar": LiveEndpoint(
        host="data.sec.gov",
        probe_url="https://data.sec.gov/submissions/CIK0000320193.json",
        purpose="EDGAR filings/XBRL facts for PEAD earnings surprises",
        notes=(
            "403s unless User-Agent carries a contact address -- set "
            "STTBOT_USER_AGENT; SEC asks for <=10 requests/second"
        ),
    ),
    "football_data_uk": LiveEndpoint(
        host="www.football-data.co.uk",
        probe_url="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        purpose="per-season odds CSVs carrying BOTH pre-match and closing prices",
        notes=(
            "the only wired-up source with a genuine closing line, so the only "
            "one that supports real CLV measurement -- see fetch_football_data_uk"
        ),
    ),
    "stooq": LiveEndpoint(
        host="stooq.com",
        probe_url="https://stooq.com/q/d/l/?s=btcusd&i=d",
        purpose="free daily OHLCV for equities/indices/FX",
        notes="no key; aggressive rate limiting, cache and back off",
    ),
    "coinbase": LiveEndpoint(
        host="api.exchange.coinbase.com",
        probe_url="https://api.exchange.coinbase.com/products",
        purpose="crypto OHLCV and order books",
        notes="public read endpoints, no key",
    ),
}


# Venues that stay unreachable even with unrestricted egress. Documented so the
# reason for their absence is explicit, not forgotten.
BLOCKED_SOURCES: dict[str, str] = {
    "api.binance.com": (
        "crypto OHLCV/order books -- HTTP 451 from this jurisdiction "
        "(Binance geo-restriction, not a proxy). Use coinbase instead."
    ),
}


def cache_dir() -> Path:
    """Directory holding downloaded datasets (``STTBOT_DATA_DIR`` overrides)."""
    env = os.environ.get("STTBOT_DATA_DIR")
    path = Path(env) if env else _DEFAULT_CACHE
    path.mkdir(parents=True, exist_ok=True)
    return path


def available() -> list[str]:
    return sorted(DATASETS)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_to_cache(
    url: str,
    filename: str,
    *,
    force: bool = False,
    timeout: float = 300.0,
    sha256: str | None = None,
    label: str | None = None,
) -> Path:
    """Download ``url`` to ``filename`` in the cache, reusing an existing copy.

    Raises ``ValueError`` if ``sha256`` is given and does not match.
    """
    target = cache_dir() / filename
    if target.exists() and not force:
        return target

    # Download to a temporary sibling first so an interrupted transfer never
    # leaves a truncated file that later looks like a valid cache hit.
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tmp.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)

    if sha256:
        actual = sha256_of(tmp)
        if actual != sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"{label or filename}: checksum mismatch "
                f"(expected {sha256}, got {actual})"
            )

    tmp.replace(target)
    return target


def fetch(name: str, *, force: bool = False, timeout: float = 300.0) -> Path:
    """Download ``name`` into the cache if absent and return its local path.

    Raises ``KeyError`` for an unknown dataset and ``ValueError`` if a pinned
    ``sha256`` does not match what was downloaded.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(available())}")
    spec = DATASETS[name]
    return download_to_cache(
        spec.url,
        spec.filename,
        force=force,
        timeout=timeout,
        sha256=spec.sha256,
        label=name,
    )


# --- football-data.co.uk ----------------------------------------------------
#
# One CSV per (season, division). Unlike the bulk football_matches dataset,
# these carry closing odds alongside the pre-match line, which is what CLV
# measurement needs. Closing columns (the ``*C*`` family, e.g. B365CH, PSCH)
# exist from season 2019/20 onward; earlier files have pre-match odds only.

FOOTBALL_DATA_UK_BASE = "https://www.football-data.co.uk/mmz4281"

CLOSING_ODDS_FROM_SEASON = "1920"

# Division codes, by country. E0 is the Premier League, D1 the Bundesliga, etc.
FOOTBALL_DATA_UK_DIVISIONS: tuple[str, ...] = (
    "E0", "E1", "E2", "E3", "EC",           # England
    "SC0", "SC1", "SC2", "SC3",             # Scotland
    "D1", "D2",                             # Germany
    "I1", "I2",                             # Italy
    "SP1", "SP2",                           # Spain
    "F1", "F2",                             # France
    "N1", "B1", "P1", "T1", "G1",           # NL, BE, PT, TR, GR
)


def football_data_uk_season(start_year: int) -> str:
    """Season code for the campaign starting in ``start_year`` (2023 -> '2324')."""
    if not 1993 <= start_year <= 2100:
        raise ValueError(f"implausible season start year {start_year}")
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_football_data_uk(
    season: str, division: str, *, force: bool = False, timeout: float = 120.0
) -> Path:
    """Fetch one football-data.co.uk season/division CSV into the cache.

    ``season`` is the four-digit code (``'2324'``); build it from a start year
    with :func:`football_data_uk_season`.
    """
    if division not in FOOTBALL_DATA_UK_DIVISIONS:
        raise KeyError(
            f"unknown division {division!r}; "
            f"known: {', '.join(FOOTBALL_DATA_UK_DIVISIONS)}"
        )
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"season must be a four-digit code like '2324', got {season!r}")
    return download_to_cache(
        f"{FOOTBALL_DATA_UK_BASE}/{season}/{division}.csv",
        f"fduk_{season}_{division}.csv",
        force=force,
        timeout=timeout,
        label=f"football-data.co.uk {season}/{division}",
    )


def describe(name: str) -> str:
    spec = DATASETS[name]
    mb = spec.approx_bytes / 1e6
    return (
        f"{spec.name}  [{spec.license}]  ~{mb:.1f} MB\n"
        f"  {spec.description}\n"
        f"  source: {spec.source}"
    )
