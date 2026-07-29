import datetime as dt

import pytest

from sttbot.data.storage import Tick, TickStore


def _tick(minute, symbol="BTC", bid=100.0, ask=100.2, volume=1.0):
    return Tick(
        ts=dt.datetime(2026, 1, 1, 0, minute, 0),
        symbol=symbol,
        bid=bid,
        ask=ask,
        volume=volume,
    )


def test_append_and_count():
    with TickStore(":memory:") as store:
        assert store.append([_tick(0), _tick(1)]) == 2
        assert store.count() == 2
        assert store.count("BTC") == 2
        assert store.count("ETH") == 0


def test_append_empty_is_noop():
    with TickStore(":memory:") as store:
        assert store.append([]) == 0
        assert store.count() == 0


def test_latest_returns_most_recent():
    with TickStore(":memory:") as store:
        store.append([_tick(0, bid=100.0), _tick(5, bid=105.0), _tick(2, bid=102.0)])
        latest = store.latest("BTC")
        assert latest is not None
        assert latest.bid == 105.0
        assert latest.mid == pytest.approx((105.0 + 100.2) / 2)


def test_latest_missing_symbol():
    with TickStore(":memory:") as store:
        store.append([_tick(0)])
        assert store.latest("DOGE") is None


def test_invalid_table_name_rejected():
    with pytest.raises(ValueError):
        TickStore(":memory:", table="bad-name; DROP TABLE x")


def test_parquet_export(tmp_path):
    path = tmp_path / "ticks.parquet"
    with TickStore(":memory:") as store:
        store.append([_tick(0), _tick(1, symbol="ETH")])
        store.to_parquet(str(path))
        assert path.exists()
        # Round-trip: DuckDB can read the file back and see both rows.
        rows = store.con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')"
        ).fetchone()
        assert rows[0] == 2
