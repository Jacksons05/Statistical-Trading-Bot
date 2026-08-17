"""Tests for the dataset catalog and football loader.

These never touch the network: `fetch` is exercised against a `file://` URL and
the loader against a small inline CSV fixture.
"""

import datetime as dt

import pytest

from sttbot.data import datasets
from sttbot.data.football import (
    Match,
    devig,
    load_football_data_uk,
    load_matches,
    overround,
)


def test_catalog_entries_well_formed():
    assert datasets.available()
    for name in datasets.available():
        spec = datasets.DATASETS[name]
        assert spec.name == name
        assert spec.url.startswith("https://")
        assert spec.license and spec.description and spec.source
        assert spec.approx_bytes > 0


def test_blocked_sources_documented():
    # The venues we cannot reach should stay explicit rather than silently
    # absent, and each needs a stated reason.
    assert "api.binance.com" in datasets.BLOCKED_SOURCES
    for reason in datasets.BLOCKED_SOURCES.values():
        assert reason.strip()


def test_reachable_hosts_are_not_also_listed_as_blocked():
    # These were blocked only by a sandbox proxy, not upstream; re-probed and
    # reachable on 2026-07-28. Keeping them in both places would be a lie.
    live_hosts = {e.host for e in datasets.LIVE_ENDPOINTS.values()}
    assert not (live_hosts & set(datasets.BLOCKED_SOURCES))
    assert "gamma-api.polymarket.com" in live_hosts
    assert "data.sec.gov" in live_hosts


def test_live_endpoints_well_formed():
    for name, endpoint in datasets.LIVE_ENDPOINTS.items():
        assert endpoint.probe_url.startswith("https://")
        assert endpoint.host in endpoint.probe_url
        assert endpoint.purpose and endpoint.notes, name


def test_user_agent_is_overridable(monkeypatch):
    # SEC EDGAR 403s without a contact address in the User-Agent.
    monkeypatch.setenv("STTBOT_USER_AGENT", "myproject me@example.com")
    assert datasets.user_agent() == "myproject me@example.com"
    monkeypatch.delenv("STTBOT_USER_AGENT")
    assert datasets.user_agent() == datasets._DEFAULT_USER_AGENT


def test_football_data_uk_season_code():
    assert datasets.football_data_uk_season(2023) == "2324"
    assert datasets.football_data_uk_season(1999) == "9900"
    assert datasets.football_data_uk_season(2009) == "0910"
    with pytest.raises(ValueError):
        datasets.football_data_uk_season(1800)


def test_fetch_football_data_uk_validates_arguments():
    with pytest.raises(KeyError):
        datasets.fetch_football_data_uk("2324", "NOT_A_DIVISION")
    with pytest.raises(ValueError):
        datasets.fetch_football_data_uk("23-24", "E0")


def test_cache_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STTBOT_DATA_DIR", str(tmp_path / "cache"))
    assert datasets.cache_dir() == tmp_path / "cache"
    assert datasets.cache_dir().is_dir()


def test_fetch_unknown_dataset():
    with pytest.raises(KeyError):
        datasets.fetch("does_not_exist")


def test_fetch_downloads_and_caches(tmp_path, monkeypatch):
    payload = b"col_a,col_b\n1,2\n"
    source = tmp_path / "source.csv"
    source.write_bytes(payload)
    monkeypatch.setenv("STTBOT_DATA_DIR", str(tmp_path / "cache"))

    spec = datasets.Dataset(
        name="fixture",
        url=source.as_uri(),
        filename="fixture.csv",
        license="MIT",
        description="test fixture",
        approx_bytes=len(payload),
        source="local",
        sha256=datasets.sha256_of(source),
    )
    monkeypatch.setitem(datasets.DATASETS, "fixture", spec)

    path = datasets.fetch("fixture")
    assert path.read_bytes() == payload

    # Second call is served from cache: deleting the origin must not break it.
    source.unlink()
    assert datasets.fetch("fixture").read_bytes() == payload
    # And no partial-download artifact is left behind.
    assert not list(path.parent.glob("*.part"))


def test_fetch_rejects_checksum_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "bad.csv"
    source.write_bytes(b"whatever")
    monkeypatch.setenv("STTBOT_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setitem(
        datasets.DATASETS,
        "bad",
        datasets.Dataset(
            name="bad",
            url=source.as_uri(),
            filename="bad.csv",
            license="MIT",
            description="mismatched checksum",
            approx_bytes=8,
            source="local",
            sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets.fetch("bad")


# --- football loader -------------------------------------------------------

_CSV = """Division,MatchDate,HomeTeam,AwayTeam,FTHome,FTAway,OddHome,OddDraw,OddAway,MaxHome,MaxDraw,MaxAway
E0,2023-08-12,Arsenal,Forest,2,1,1.50,4.20,6.50,1.55,4.40,7.00
E0,2023-08-13,Chelsea,Liverpool,1,1,2.70,3.40,2.60,2.80,3.55,2.75
D1,2023-08-19,Bayern,Werder,4,0,1.20,7.00,13.0,1.25,7.50,15.0
E0,2024-01-05,Forest,Arsenal,0,0,,,,,,
"""


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "matches.csv"
    path.write_text(_CSV)
    return str(path)


def test_load_matches_all(csv_path):
    matches = load_matches(path=csv_path)
    assert len(matches) == 4
    assert matches[0].home == "Arsenal"
    assert matches[0].date == dt.date(2023, 8, 12)


def test_load_matches_filters(csv_path):
    assert len(load_matches(division="E0", path=csv_path)) == 3
    assert len(load_matches(require_odds=True, path=csv_path)) == 3  # drops the blank row
    windowed = load_matches(
        start=dt.date(2023, 8, 13), end=dt.date(2023, 12, 31), path=csv_path
    )
    assert [m.home for m in windowed] == ["Chelsea", "Bayern"]


def test_results_derived():
    assert Match(dt.date(2024, 1, 1), "E0", "A", "B", 2, 1).result == "H"
    assert Match(dt.date(2024, 1, 1), "E0", "A", "B", 1, 2).result == "A"
    assert Match(dt.date(2024, 1, 1), "E0", "A", "B", 1, 1).result == "D"


def test_overround_is_positive_margin():
    # A typical book prices above 100%.
    assert overround(1.50, 4.20, 6.50) > 1.0
    with pytest.raises(ValueError):
        overround(1.0, 4.2, 6.5)


# --- football-data.co.uk loader --------------------------------------------

# Modern layout: four-digit years, Pinnacle + Bet365 closing odds present.
_FDUK_MODERN = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,"
    "MaxH,MaxD,MaxA,PSCH,PSCD,PSCA,B365CH,B365CD,B365CA\n"
    "E0,11/08/2023,Burnley,Man City,0,3,8.00,5.00,1.40,"
    "9.00,5.20,1.44,9.62,5.60,1.38,9.00,5.50,1.40\n"
    "E0,12/08/2023,Arsenal,Forest,2,1,1.30,5.50,10.0,"
    "1.35,5.75,11.0,1.28,6.00,11.5,1.30,5.80,11.0\n"
)

# Legacy layout: two-digit years and no closing-odds columns at all.
_FDUK_LEGACY = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
    "E0,14/08/10,Tottenham,Man City,0,0,2.40,3.30,3.10\n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_fduk_loader_reads_closing_odds(tmp_path):
    path = _write(tmp_path, "modern.csv", _FDUK_MODERN)
    matches = load_football_data_uk(["2324"], "E0", paths=[path])
    assert len(matches) == 2
    first = matches[0]
    assert first.date == dt.date(2023, 8, 11)
    assert first.home == "Burnley"
    # Entry line is Bet365 pre-match; the close prefers Pinnacle over Bet365.
    assert first.odds_home == pytest.approx(8.00)
    assert first.close_home == pytest.approx(9.62)
    assert first.has_closing_odds


def test_fduk_loader_parses_two_digit_years(tmp_path):
    # Regression: '%d/%m/%Y' parses "14/08/10" as the year 10 AD, which would
    # silently corrupt every walk-forward ordering.
    path = _write(tmp_path, "legacy.csv", _FDUK_LEGACY)
    (match,) = load_football_data_uk(["1011"], "E0", paths=[path])
    assert match.date == dt.date(2010, 8, 14)


def test_fduk_loader_tolerates_missing_closing_columns(tmp_path):
    path = _write(tmp_path, "legacy.csv", _FDUK_LEGACY)
    (match,) = load_football_data_uk(["1011"], "E0", paths=[path])
    assert match.close_home is None
    assert not match.has_closing_odds
    # Requiring a close drops the row rather than raising on absent columns.
    assert load_football_data_uk(["1011"], "E0", require_closing=True, paths=[path]) == []


def test_fduk_loader_tolerates_a_non_numeric_odds_cell(tmp_path):
    """Regression: one bad cell used to kill an entire division.

    A stray non-numeric value makes DuckDB type the whole column VARCHAR, and
    COALESCE then refuses to mix it with a DOUBLE sibling column. Several
    divisions (I2, SP2, T1) failed to load at all because of a handful of
    cells. The bad value should become NULL and fall through to the next book.
    """
    csv = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,"
        "MaxH,MaxD,MaxA,PSCH,PSCD,PSCA,B365CH,B365CD,B365CA\n"
        # PSCH is junk here; the Bet365 close should be used instead.
        "E0,11/08/2023,Burnley,Man City,0,3,8.00,5.00,1.40,"
        "9.00,5.20,1.44,n/a,5.60,1.38,9.10,5.50,1.40\n"
        "E0,12/08/2023,Arsenal,Forest,2,1,1.30,5.50,10.0,"
        "1.35,5.75,11.0,1.28,6.00,11.5,1.30,5.80,11.0\n"
    )
    path = _write(tmp_path, "dirty.csv", csv)
    matches = load_football_data_uk(["2324"], "E0", paths=[path])

    assert len(matches) == 2
    assert matches[0].close_home == pytest.approx(9.10)  # fell through to B365
    assert matches[1].close_home == pytest.approx(1.28)  # Pinnacle still wins


def test_fduk_loader_merges_seasons_in_date_order(tmp_path):
    modern = _write(tmp_path, "modern.csv", _FDUK_MODERN)
    legacy = _write(tmp_path, "legacy.csv", _FDUK_LEGACY)
    matches = load_football_data_uk(["2324", "1011"], "E0", paths=[modern, legacy])
    assert [m.date for m in matches] == sorted(m.date for m in matches)
    assert matches[0].date == dt.date(2010, 8, 14)


def test_devig_normalises_to_one():
    probs = devig(1.50, 4.20, 6.50)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]  # favourite ranks highest
