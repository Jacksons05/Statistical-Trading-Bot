"""Tests for settling resolved paper positions.

The dangerous failure is not "we missed a settlement" -- that is visible and
recoverable. It is settling a market that has not actually resolved, which
writes a permanent wrong number into the ledger. Both real states that look
like resolutions get explicit tests.
"""

import datetime as dt

import pytest

from sttbot.paper.account import BUY, SELL, Fill, PaperAccount
from sttbot.paper.settlement import (
    Resolution,
    fetch_resolutions,
    is_settled,
    parse_resolution,
    settle_account,
)

TS = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)


def market(prices, tokens=("A", "B"), closed=True, **over):
    import json
    base = {
        "id": "m1",
        "question": "Will it?",
        "closed": closed,
        "clobTokenIds": json.dumps(list(tokens)),
        "outcomePrices": json.dumps([str(p) for p in prices]),
    }
    base.update(over)
    return base


def account_with(*positions, cash=1_000.0):
    acc = PaperAccount(":memory:", starting_cash=cash)
    for i, (sym, qty, price) in enumerate(positions):
        acc.record(Fill(ts=TS, symbol=sym, side=BUY, qty=qty, price=price,
                        ref=f"seed-{i}"))
    return acc


# --- what counts as resolved -------------------------------------------------


def test_a_clean_binary_payout_is_settled():
    assert is_settled([0.0, 1.0])
    assert is_settled([1.0, 0.0])
    # Near-exact resolutions really occur on this venue.
    assert is_settled([0.000001, 0.999999])


def test_all_zero_prices_are_not_a_resolution():
    """36 of 60 surveyed closed markets looked like this.

    Settling here would zero out every leg of a basket including the winner,
    turning a $1 payout into nothing.
    """
    assert not is_settled([0.0, 0.0])
    assert not is_settled([0.0, 0.0, 0.0])


def test_fractional_prices_are_a_mark_not_an_outcome():
    """A closed market can still report the last price anyone traded at."""
    assert not is_settled([0.5843, 0.4157])
    assert not is_settled([0.9, 0.1])      # sums to 1 but nobody won
    assert not is_settled([0.98, 0.02])    # still short of a payout


def test_multi_outcome_resolution_needs_exactly_one_winner():
    assert is_settled([1.0, 0.0, 0.0, 0.0])
    assert not is_settled([0.5, 0.5, 0.0, 0.0])
    assert not is_settled([1.0, 1.0, 0.0])   # sums to 2


def test_degenerate_input_is_rejected():
    assert not is_settled([])
    assert not is_settled([1.0])


# --- parsing -----------------------------------------------------------------


def test_parse_maps_each_token_to_its_settled_price():
    res = parse_resolution(market([0, 1], tokens=("TOK_A", "TOK_B")))
    assert {r.token_id: r.settled_price for r in res} == {"TOK_A": 0.0, "TOK_B": 1.0}


def test_an_open_market_never_parses_as_resolved():
    assert parse_resolution(market([0, 1], closed=False)) == []


def test_unresolved_closed_markets_parse_to_nothing():
    assert parse_resolution(market([0, 0])) == []
    assert parse_resolution(market([0.58, 0.42])) == []


def test_malformed_payloads_are_survivable():
    assert parse_resolution({"closed": True}) == []
    assert parse_resolution(market([0, 1], tokens=("A",))) == []  # length mismatch
    assert parse_resolution({"closed": True, "clobTokenIds": "not json",
                             "outcomePrices": "[]"}) == []


# --- fetching ----------------------------------------------------------------


def test_fetch_uses_repeated_params_and_requires_closed():
    """Comma-separated ids are rejected by the API as invalid, and without
    closed=true resolved markets are filtered out and every lookup is empty."""
    seen = []

    def fake(url):
        seen.append(url)
        return [market([0, 1], tokens=("A", "B"))]

    got = fetch_resolutions(["A", "B"], fetch=fake, pause=0)
    assert set(got) == {"A", "B"}
    assert seen[0].count("clob_token_ids=") == 2
    assert "closed=true" in seen[0]
    assert "clob_token_ids=A,B" not in seen[0]


def test_fetch_batches_and_survives_a_failure():
    calls = {"n": 0}

    def flaky(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return [market([0, 1], tokens=("C", "D"))]

    got = fetch_resolutions([f"T{i}" for i in range(25)] + ["C", "D"],
                            fetch=flaky, batch_size=20, pause=0)
    assert calls["n"] >= 2          # batched
    assert set(got) == {"C", "D"}   # the surviving batch still lands


def test_fetch_ignores_tokens_it_did_not_ask_about():
    def fake(url):
        return [market([0, 1], tokens=("WANTED", "STRANGER"))]

    got = fetch_resolutions(["WANTED"], fetch=fake, pause=0)
    assert set(got) == {"WANTED"}


# --- settling ----------------------------------------------------------------


def res(token, price, market_id="m1"):
    return Resolution(token, price, "Q?", market_id)


def test_a_winner_credits_face_value_and_closes():
    acc = account_with(("W", 100, 0.30))
    report = settle_account(acc, {"W": res("W", 1.0)}, now=TS)
    assert len(report.settled) == 1
    assert acc.open_positions() == {}
    assert acc.cash == pytest.approx(1_000 - 30 + 100)
    assert report.proceeds == pytest.approx(100.0)


def test_a_loser_closes_at_zero_and_realises_the_loss():
    acc = account_with(("L", 100, 0.30))
    settle_account(acc, {"L": res("L", 0.0)}, now=TS)
    assert acc.open_positions() == {}
    assert acc.cash == pytest.approx(970.0)
    assert acc.positions()["L"].realized_pnl == pytest.approx(-30.0)


def test_a_whole_basket_settles_to_its_face_value():
    """The point of the exercise: a 4-leg arbitrage bought for under $1 a set
    pays exactly $1 a set, and only settlement ever books that."""
    acc = account_with(("A", 50, 0.20), ("B", 50, 0.30),
                       ("C", 50, 0.25), ("D", 50, 0.20))
    cost = 50 * (0.20 + 0.30 + 0.25 + 0.20)          # $47.50 for $50 of payout
    assert acc.cash == pytest.approx(1_000 - cost)

    resolutions = {"A": res("A", 0.0), "B": res("B", 1.0),
                   "C": res("C", 0.0), "D": res("D", 0.0)}
    report = settle_account(acc, resolutions, now=TS)
    assert len(report.settled) == 4
    assert acc.open_positions() == {}
    assert acc.cash == pytest.approx(1_000 - cost + 50)
    assert acc.snapshot().equity == pytest.approx(1_002.50)


def test_settlement_is_idempotent():
    acc = account_with(("W", 100, 0.30))
    first = settle_account(acc, {"W": res("W", 1.0)}, now=TS)
    second = settle_account(acc, {"W": res("W", 1.0)}, now=TS)
    assert len(first.settled) == 1
    assert second.settled == ()
    assert acc.cash == pytest.approx(1_070.0)


def test_unresolved_positions_are_left_alone_and_counted():
    acc = account_with(("DONE", 10, 0.5), ("PENDING", 10, 0.5))
    report = settle_account(acc, {"DONE": res("DONE", 1.0)}, now=TS)
    assert len(report.settled) == 1
    assert report.still_open == 1
    assert "PENDING" in acc.open_positions()
    assert "settled 1" in report.summary()


def test_settlement_charges_no_fee():
    """Redemption is not a trade."""
    acc = account_with(("W", 100, 0.30))
    report = settle_account(acc, {"W": res("W", 1.0)}, now=TS)
    assert report.settled[0].fee == 0.0


def test_a_short_position_settles_by_buying_back():
    acc = PaperAccount(":memory:", starting_cash=1_000.0)
    acc.record(Fill(ts=TS, symbol="S", side=SELL, qty=100, price=0.40, ref="s"))
    settle_account(acc, {"S": res("S", 1.0)}, now=TS)
    assert acc.open_positions() == {}
    # Sold at 0.40, bought back at 1.00: a 60c loss per contract.
    assert acc.cash == pytest.approx(1_000 + 40 - 100)


def test_settling_an_empty_book_is_harmless():
    acc = PaperAccount(":memory:", starting_cash=1_000.0)
    report = settle_account(acc, {"X": res("X", 1.0)}, now=TS)
    assert report.settled == ()
    assert report.proceeds == 0.0
