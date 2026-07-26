"""Tests for the dataset catalog and football loader.

These never touch the network: `fetch` is exercised against a `file://` URL and
the loader against a small inline CSV fixture.
"""

import datetime as dt

import pytest

from sttbot.data import datasets
from sttbot.data.football import Match, devig, load_matches, overround


def test_catalog_entries_well_formed():
    assert datasets.available()
    for name in datasets.available():
        spec = datasets.DATASETS[name]
        assert spec.name == name
        assert spec.url.startswith("https://")
        assert spec.license and spec.description and spec.source
        assert spec.approx_bytes > 0


def test_blocked_sources_documented():
    # The venues we cannot reach should stay explicit rather than silently absent.
    assert "gamma-api.polymarket.com" in datasets.BLOCKED_SOURCES
    assert "data.sec.gov" in datasets.BLOCKED_SOURCES


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


def test_devig_normalises_to_one():
    probs = devig(1.50, 4.20, 6.50)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]  # favourite ranks highest
