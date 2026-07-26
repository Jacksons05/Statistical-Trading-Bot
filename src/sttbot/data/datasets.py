"""Catalog of openly-licensed market datasets, with a caching fetcher.

Every entry here is public, openly licensed, and reachable over plain HTTPS
without an API key. Files are downloaded once into a local cache directory
(override with ``STTBOT_DATA_DIR``) and reused thereafter, so a backtest is
reproducible without re-hitting the network.

Note on network policy: many market-data hosts (exchange REST APIs,
football-data.co.uk, prediction-market APIs) are blocked from restricted
environments such as CI sandboxes. The sources below are hosted on
``raw.githubusercontent.com``, which is far more commonly reachable. See
``BLOCKED_SOURCES`` for the venues that need direct egress instead.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CACHE = Path.home() / ".cache" / "sttbot"


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


# Venues that require direct egress and are commonly blocked by sandbox network
# policy. Documented so the reason for their absence is explicit, not forgotten.
BLOCKED_SOURCES: dict[str, str] = {
    "football-data.co.uk": "canonical odds CSVs; superseded here by football_matches",
    "api.binance.com": "crypto OHLCV/order books",
    "api.exchange.coinbase.com": "crypto OHLCV",
    "gamma-api.polymarket.com": "prediction-market prices (needed for live arb)",
    "api.elections.kalshi.com": "prediction-market prices (needed for live arb)",
    "data.sec.gov": "EDGAR XBRL filings (needed for PEAD earnings surprises)",
    "stooq.com": "free daily equity OHLCV",
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


def fetch(name: str, *, force: bool = False, timeout: float = 300.0) -> Path:
    """Download ``name`` into the cache if absent and return its local path.

    Raises ``KeyError`` for an unknown dataset and ``ValueError`` if a pinned
    ``sha256`` does not match what was downloaded.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(available())}")
    spec = DATASETS[name]
    target = cache_dir() / spec.filename

    if target.exists() and not force:
        return target

    # Download to a temporary sibling first so an interrupted transfer never
    # leaves a truncated file that later looks like a valid cache hit.
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "sttbot/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tmp.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)

    if spec.sha256:
        actual = sha256_of(tmp)
        if actual != spec.sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"{name}: checksum mismatch (expected {spec.sha256}, got {actual})"
            )

    tmp.replace(target)
    return target


def describe(name: str) -> str:
    spec = DATASETS[name]
    mb = spec.approx_bytes / 1e6
    return (
        f"{spec.name}  [{spec.license}]  ~{mb:.1f} MB\n"
        f"  {spec.description}\n"
        f"  source: {spec.source}"
    )
