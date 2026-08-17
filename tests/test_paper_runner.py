"""Tests for the paper runner.

The two properties that matter most for this runner: a basket is genuinely
atomic (no partial-fill leakage), and the circuit breaker's trip actually
closes the book rather than merely flagging it.
"""

import datetime as dt

import pytest

from sttbot.paper.account import BUY, SELL, Fill, PaperAccount
from sttbot.paper.runner import Basket, Intent, PaperRunner
from sttbot.risk.circuit_breaker import RiskCircuitBreaker

TS = dt.datetime(2026, 8, 2, 12, 0, 0, tzinfo=dt.timezone.utc)


def runner(cash=1_000.0, breaker=None, fill_model=None, **kw):
    acc = PaperAccount(":memory:", starting_cash=cash)
    kwargs = dict(account=acc, breaker=breaker, clock=lambda: TS)
    if fill_model is not None:
        kwargs["fill_model"] = fill_model
    kwargs.update(kw)
    return PaperRunner(**kwargs)


def intent(symbol, side, qty, limit, **over):
    base = dict(symbol=symbol, side=side, qty=qty, limit_price=limit)
    base.update(over)
    return Intent(**base)


# --- basic execution ---------------------------------------------------------


def test_a_simple_buy_fills_and_updates_the_book():
    r = runner(1_000.0)
    basket = Basket((intent("X", BUY, 10, 2.0),), rationale="test")
    result = r.run_once([basket], marks={"X": 2.0})
    assert len(result.filled) == 1
    assert r.account.positions()["X"].qty == pytest.approx(10)
    assert result.snapshot.cash == pytest.approx(980.0)


def test_intent_validates_side_and_quantity():
    with pytest.raises(ValueError):
        intent("X", "HOLD", 1, 1.0)
    with pytest.raises(ValueError):
        intent("X", BUY, 0, 1.0)


def test_basket_requires_at_least_one_intent():
    with pytest.raises(ValueError):
        Basket(())


# --- atomicity: the whole point of Basket -----------------------------------


def test_a_leg_that_cannot_fill_kills_the_whole_basket():
    """The reason baskets exist: three good legs plus one bad leg must trade
    zero legs, not three."""
    def flaky(i):
        return None if i.symbol == "C" else i.limit_price

    r = runner(10_000.0, fill_model=flaky)
    basket = Basket(
        (intent("A", BUY, 10, 1.0), intent("B", BUY, 10, 1.0),
         intent("C", BUY, 10, 1.0)),
        rationale="arb",
    )
    result = r.run_once([basket], marks={"A": 1.0, "B": 1.0, "C": 1.0})
    assert result.filled == ()
    assert r.account.open_positions() == {}
    assert "no fill available for C" in result.rejected[0][1]


def test_a_leg_that_fills_through_its_limit_kills_the_whole_basket():
    def worse_than_quoted(i):
        return i.limit_price * 1.5  # buy fills higher than the limit allows

    r = runner(10_000.0, fill_model=worse_than_quoted)
    basket = Basket((intent("A", BUY, 10, 1.0), intent("B", BUY, 10, 1.0)))
    result = r.run_once([basket], marks={"A": 1.0, "B": 1.0})
    assert result.filled == ()
    assert r.account.open_positions() == {}


def test_insufficient_cash_kills_the_whole_basket_not_a_prefix():
    r = runner(50.0)  # can afford one leg but not both
    basket = Basket((intent("A", BUY, 10, 3.0), intent("B", BUY, 10, 3.0)))
    result = r.run_once([basket], marks={"A": 3.0, "B": 3.0})
    assert result.filled == ()
    assert r.account.open_positions() == {}
    assert "needs $" in result.rejected[0][1]


def test_a_selling_leg_frees_cash_within_the_same_basket():
    """Net cash need across the basket, not per leg."""
    r = runner(50.0)
    acc = r.account
    acc.record(Fill(ts=TS, symbol="Y", side=BUY, qty=10, price=1.0, ref="seed"))
    basket = Basket((intent("X", BUY, 20, 2.0), intent("Y", SELL, 10, 4.0)))
    result = r.run_once([basket], marks={"X": 2.0, "Y": 4.0})
    assert len(result.filled) == 2


def test_one_bad_basket_does_not_block_a_good_one_in_the_same_run():
    r = runner(1_000.0)

    def flaky(i):
        return None if i.symbol == "BAD" else i.limit_price

    r.fill_model = flaky
    bad = Basket((intent("BAD", BUY, 1, 1.0),))
    good = Basket((intent("GOOD", BUY, 1, 1.0),))
    result = r.run_once([bad, good], marks={"GOOD": 1.0})
    assert len(result.filled) == 1
    assert result.filled[0].symbol == "GOOD"
    assert len(result.rejected) == 1


# --- idempotency across a run -------------------------------------------------


def test_intents_carry_refs_that_deduplicate_on_replay():
    r = runner(1_000.0)
    basket = Basket((intent("X", BUY, 10, 1.0, ref="order-7"),))
    r.run_once([basket], marks={"X": 1.0})
    result = r.run_once([basket], marks={"X": 1.0})
    assert result.filled == ()  # already recorded; not a second fill
    assert r.account.positions()["X"].qty == pytest.approx(10)


# --- circuit breaker integration ---------------------------------------------


def test_a_tripped_breaker_blocks_every_basket():
    breaker = RiskCircuitBreaker(max_drawdown=0.05, clock=lambda: 0.0)
    r = runner(1_000.0, breaker=breaker)
    r.account.record(Fill(ts=TS, symbol="X", side=BUY, qty=100, price=1.0, ref="seed"))

    # Establish the high-water mark at full value before the price drops --
    # the breaker's peak is whatever the first evaluate() sees, so a mark-down
    # on the very first call has nothing to be a drawdown *from*.
    r.run_once([], marks={"X": 1.0})

    basket = Basket((intent("Y", BUY, 1, 1.0),))
    result = r.run_once([basket], marks={"X": 0.5})  # equity down >5% from peak
    assert result.halted is True
    assert result.filled == ()
    assert result.rejected[0][1] == "circuit breaker tripped"


def test_tripping_the_breaker_actually_flattens_open_positions():
    """A kill switch that leaves positions open is not a kill switch."""
    breaker = RiskCircuitBreaker(max_drawdown=0.05, clock=lambda: 0.0)
    r = runner(1_000.0, breaker=breaker)
    r.account.record(Fill(ts=TS, symbol="X", side=BUY, qty=100, price=1.0, ref="s"))
    r.run_once([], marks={"X": 1.0})  # establish the peak

    r.run_once([Basket((intent("Y", BUY, 1, 1.0),))], marks={"X": 0.5})
    assert r.account.open_positions() == {}
    flatten_fills = [f for f in r.account.fills() if f.strategy == "circuit-breaker"]
    assert len(flatten_fills) == 1
    assert flatten_fills[0].price == pytest.approx(0.5)


def test_flatten_skips_positions_with_no_mark_rather_than_guessing():
    breaker = RiskCircuitBreaker(max_drawdown=0.05, clock=lambda: 0.0)
    r = runner(1_000.0, breaker=breaker)
    r.account.record(Fill(ts=TS, symbol="X", side=BUY, qty=100, price=1.0, ref="s1"))
    r.account.record(Fill(ts=TS, symbol="UNMARKED", side=BUY, qty=10, price=1.0,
                          ref="s2"))
    r.run_once([], marks={"X": 1.0, "UNMARKED": 1.0})  # establish the peak

    r.run_once([Basket((intent("Y", BUY, 1, 1.0),))], marks={"X": 0.5})
    remaining = r.account.open_positions()
    assert "UNMARKED" in remaining
    assert "X" not in remaining


def test_reset_alone_does_not_undo_a_crystallised_loss():
    """Flattening realises the loss at the tripping mark, so cash-only equity
    is frozen at exactly the trip level. reset() clears the latch but the
    very next evaluate() sees the same equity and correctly retrips -- a real
    loss is not undone by acknowledging it."""
    breaker = RiskCircuitBreaker(max_drawdown=0.05, clock=lambda: 0.0)
    r = runner(1_000.0, breaker=breaker)
    r.account.record(Fill(ts=TS, symbol="X", side=BUY, qty=100, price=1.0, ref="s"))
    r.run_once([], marks={"X": 1.0})  # establish the peak
    r.run_once([Basket((intent("Y", BUY, 1, 1.0),))], marks={"X": 0.5})

    breaker.reset()
    result = r.run_once([Basket((intent("Y", BUY, 1, 1.0),))], marks={})
    assert result.halted is True  # equity is still 5% below the old peak


def test_rolling_drawdown_recovers_as_the_old_peak_ages_out():
    """The rolling half of the breaker is the one built to recover with time
    (see RiskCircuitBreaker's docstring). The high-water mark is disabled here
    so only that behaviour is under test."""
    clock_time = {"t": 0.0}
    breaker = RiskCircuitBreaker(
        max_drawdown=1.0,  # effectively disabled: equity can't go negative
        rolling_window_seconds=100.0,
        max_rolling_drawdown=0.05,
        clock=lambda: clock_time["t"],
    )
    r = runner(1_000.0, breaker=breaker)
    r.account.record(Fill(ts=TS, symbol="X", side=BUY, qty=100, price=1.0, ref="s"))
    r.run_once([], marks={"X": 1.0})  # rolling peak 1000 @ t=0

    clock_time["t"] = 10.0
    result = r.run_once([Basket((intent("Y", BUY, 1, 1.0),))], marks={"X": 0.5})
    assert result.halted is True  # 950 vs a 1000 rolling peak, 5% down

    breaker.reset()
    # Past the window: the t=0 and t=10 entries age out, so the frozen
    # post-flatten equity becomes the new rolling peak with nothing to be a
    # drawdown from.
    clock_time["t"] = 10.0 + 100.0 + 1.0
    result = r.run_once([Basket((intent("Y", BUY, 1, 1.0),))], marks={})
    assert result.halted is False
    assert len(result.filled) == 1


# --- cash floor ---------------------------------------------------------------


def test_negative_cash_is_refused_by_default():
    r = runner(10.0)
    basket = Basket((intent("X", BUY, 100, 1.0),))
    result = r.run_once([basket], marks={"X": 1.0})
    assert result.filled == ()
    assert r.account.cash == pytest.approx(10.0)


def test_allow_negative_cash_opts_out_explicitly():
    r = runner(10.0, allow_negative_cash=True)
    basket = Basket((intent("X", BUY, 100, 1.0),))
    result = r.run_once([basket], marks={"X": 1.0})
    assert len(result.filled) == 1
    assert r.account.cash == pytest.approx(-90.0)


# --- per-intent fee override -------------------------------------------------


def test_intent_fee_overrides_the_basket_flat_rate():
    """Polymarket's fee is quadratic in price, not a flat proportional rate,
    so the caller must be able to hand the runner a precomputed fee."""
    r = runner(1_000.0)
    basket = Basket(
        (intent("X", BUY, 10, 2.0, fee=3.33),), fee_rate=0.10,
    )
    result = r.run_once([basket], marks={"X": 2.0})
    assert len(result.filled) == 1
    assert result.filled[0].fee == pytest.approx(3.33)  # not 0.10*10*2.0=2.0


def test_intent_without_a_fee_falls_back_to_the_basket_rate():
    r = runner(1_000.0)
    basket = Basket((intent("X", BUY, 10, 2.0),), fee_rate=0.10)
    result = r.run_once([basket], marks={"X": 2.0})
    assert result.filled[0].fee == pytest.approx(2.0)


def test_intent_rejects_a_negative_fee():
    with pytest.raises(ValueError):
        intent("X", BUY, 1, 1.0, fee=-0.01)
