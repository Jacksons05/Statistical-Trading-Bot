"""Tests for the durable paper account.

The dangerous bugs here are silent: an average cost that drifts, a realised
P&L that double-counts, or a restart that re-books a trade. All three produce
a plausible-looking number that is simply wrong, so they get explicit tests.
"""

import datetime as dt

import pytest

from sttbot.paper.account import BUY, SELL, Fill, PaperAccount, Position, _apply

TS = dt.datetime(2026, 8, 2, 12, 0, 0)


def fill(side, qty, price, **over):
    base = dict(ts=TS, symbol="X", side=side, qty=qty, price=price)
    base.update(over)
    return Fill(**base)


def account(cash=1_000.0):
    return PaperAccount(":memory:", starting_cash=cash)


# --- fill validation --------------------------------------------------------


def test_fill_rejects_nonsense():
    with pytest.raises(ValueError):
        fill("LONG", 1, 1.0)          # not a side
    with pytest.raises(ValueError):
        fill(BUY, 0, 1.0)             # direction comes from side, not sign
    with pytest.raises(ValueError):
        fill(BUY, -1, 1.0)
    with pytest.raises(ValueError):
        fill(BUY, 1, -1.0)
    with pytest.raises(ValueError):
        fill(BUY, 1, 1.0, fee=-0.5)


def test_cash_delta_signs():
    assert fill(BUY, 10, 2.0).cash_delta == pytest.approx(-20.0)
    assert fill(SELL, 10, 2.0).cash_delta == pytest.approx(20.0)
    # Fees always cost, on both sides.
    assert fill(BUY, 10, 2.0, fee=1.0).cash_delta == pytest.approx(-21.0)
    assert fill(SELL, 10, 2.0, fee=1.0).cash_delta == pytest.approx(19.0)


# --- position arithmetic ----------------------------------------------------


def test_adding_to_a_long_weights_the_cost_basis():
    p = Position("X")
    _apply(p, fill(BUY, 10, 1.00))
    _apply(p, fill(BUY, 30, 2.00))
    assert p.qty == pytest.approx(40)
    assert p.avg_cost == pytest.approx(1.75)  # (10*1 + 30*2)/40
    assert p.realized_pnl == pytest.approx(0.0)


def test_partial_close_realises_only_the_closed_portion():
    p = Position("X")
    _apply(p, fill(BUY, 100, 1.00))
    _apply(p, fill(SELL, 40, 1.50))
    assert p.qty == pytest.approx(60)
    assert p.avg_cost == pytest.approx(1.00)  # basis unchanged by a reduction
    assert p.realized_pnl == pytest.approx(40 * 0.50)


def test_full_close_flattens_and_clears_the_basis():
    p = Position("X")
    _apply(p, fill(BUY, 100, 1.00))
    _apply(p, fill(SELL, 100, 0.80))
    assert p.is_flat
    assert p.avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(-20.0)


def test_reversing_through_zero_splits_realise_from_reopen():
    """The case a naive average-cost update gets wrong.

    Selling 150 against a 100 long closes 100 and opens a 50 short; only the
    closed portion realises, and the new short is based at the fill price.
    """
    p = Position("X")
    _apply(p, fill(BUY, 100, 1.00))
    _apply(p, fill(SELL, 150, 2.00))
    assert p.qty == pytest.approx(-50)
    assert p.avg_cost == pytest.approx(2.00)
    assert p.realized_pnl == pytest.approx(100 * 1.00)


def test_shorts_realise_with_the_right_sign():
    p = Position("X")
    _apply(p, fill(SELL, 50, 2.00))
    assert p.qty == pytest.approx(-50)
    _apply(p, fill(BUY, 50, 1.50))   # covering lower is a profit
    assert p.is_flat
    assert p.realized_pnl == pytest.approx(50 * 0.50)


def test_fees_hit_realised_pnl_immediately():
    p = Position("X")
    _apply(p, fill(BUY, 10, 1.00, fee=2.0))
    assert p.realized_pnl == pytest.approx(-2.0)  # cost even while still open
    assert p.avg_cost == pytest.approx(1.00)      # and not folded into basis


def test_unrealized_pnl_follows_the_mark():
    p = Position("X")
    _apply(p, fill(BUY, 10, 1.00))
    assert p.unrealized_pnl(1.50) == pytest.approx(5.0)
    assert p.unrealized_pnl(0.50) == pytest.approx(-5.0)
    short = Position("Y")
    _apply(short, fill(SELL, 10, 1.00))
    assert short.unrealized_pnl(0.50) == pytest.approx(5.0)  # short gains as it falls


# --- ledger and cash --------------------------------------------------------


def test_cash_and_equity_reconcile():
    acc = account(1_000.0)
    acc.record(fill(BUY, 100, 2.00, fee=1.0))
    assert acc.cash == pytest.approx(1_000 - 200 - 1)

    snap = acc.snapshot({"X": 2.50})
    assert snap.market_value == pytest.approx(250.0)
    assert snap.equity == pytest.approx(799.0 + 250.0)
    assert snap.unrealized_pnl == pytest.approx(50.0)
    assert snap.realized_pnl == pytest.approx(-1.0)  # the fee
    assert snap.total_return == pytest.approx((1049.0 / 1000.0) - 1.0)


def test_a_round_trip_returns_cash_and_leaves_only_pnl():
    acc = account(1_000.0)
    acc.record(fill(BUY, 100, 1.00, ref="a"))
    acc.record(fill(SELL, 100, 1.20, ref="b"))
    assert acc.cash == pytest.approx(1_020.0)
    assert acc.open_positions() == {}
    assert acc.snapshot().equity == pytest.approx(1_020.0)


def test_positions_are_derived_not_stored():
    """Cash and positions are projections of the ledger, so they cannot drift."""
    acc = account()
    acc.record(fill(BUY, 10, 1.0, ref="1"))
    acc.record(fill(BUY, 10, 3.0, ref="2"))
    assert acc.positions()["X"].avg_cost == pytest.approx(2.0)
    assert len(acc.fills()) == 2


# --- idempotency and durability ---------------------------------------------


def test_replaying_a_ref_is_a_no_op():
    """A runner that dies after trading and before recording will retry."""
    acc = account()
    assert acc.record(fill(BUY, 10, 1.0, ref="order-1")) is True
    assert acc.record(fill(BUY, 10, 1.0, ref="order-1")) is False
    assert acc.positions()["X"].qty == pytest.approx(10)


def test_record_all_reports_how_many_were_new():
    acc = account()
    batch = [fill(BUY, 1, 1.0, ref="a"), fill(BUY, 1, 1.0, ref="b")]
    assert acc.record_all(batch) == 2
    assert acc.record_all(batch) == 0


def test_fills_without_a_ref_are_not_deduplicated():
    """An empty ref means "no idempotency claim", not "same as every other"."""
    acc = account()
    acc.record(fill(BUY, 1, 1.0))
    acc.record(fill(BUY, 1, 1.0))
    assert acc.positions()["X"].qty == pytest.approx(2)


def test_the_book_survives_a_restart(tmp_path):
    path = str(tmp_path / "book.duckdb")
    acc = PaperAccount(path, starting_cash=500.0)
    acc.record(fill(BUY, 20, 1.5, ref="x"))
    acc.close()

    reopened = PaperAccount(path, starting_cash=999_999.0)
    # The original capital base is kept; adopting the new one would rewrite
    # the return series.
    assert reopened.starting_cash == pytest.approx(500.0)
    assert reopened.positions()["X"].qty == pytest.approx(20)
    assert reopened.cash == pytest.approx(500.0 - 30.0)
    assert reopened.record(fill(BUY, 20, 1.5, ref="x")) is False
    reopened.close()


def test_starting_cash_must_be_positive():
    with pytest.raises(ValueError):
        PaperAccount(":memory:", starting_cash=0)


# --- the flattering-omission guard ------------------------------------------


def test_unmarked_positions_are_reported_not_hidden():
    """A position nobody will quote is usually one that went to nothing.

    Excluding it from equity silently is how a paper book reports a flat P&L
    on a portfolio of dead tokens.
    """
    acc = account()
    acc.record(fill(BUY, 10, 1.0, symbol="ALIVE", ref="1"))
    acc.record(fill(BUY, 10, 1.0, symbol="DEAD", ref="2"))

    snap = acc.snapshot({"ALIVE": 1.0})
    assert snap.unmarked == ("DEAD",)
    # DEAD contributes nothing to equity, which is exactly why it must surface.
    assert snap.market_value == pytest.approx(10.0)


def test_no_unmarked_when_everything_is_priced():
    acc = account()
    acc.record(fill(BUY, 10, 1.0, ref="1"))
    assert acc.snapshot({"X": 1.0}).unmarked == ()


def test_flat_positions_are_not_reported_as_unmarked():
    acc = account()
    acc.record(fill(BUY, 10, 1.0, ref="1"))
    acc.record(fill(SELL, 10, 1.0, ref="2"))
    assert acc.snapshot({}).unmarked == ()
