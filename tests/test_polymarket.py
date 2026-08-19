"""Tests for the Polymarket client and scans.

Every test injects a fake fetcher, so the suite never touches the network. The
two failure modes with dedicated tests are the ones that produce confidently
wrong answers rather than errors: pagination that silently repeats page one,
and baskets scored on an incomplete set of outcomes.
"""

import datetime as dt

import pytest

from sttbot.venues.polymarket import (
    CURSOR_REQUEST_PARAM,
    Basket,
    OrderBook,
    complete_baskets,
    executable_arbitrage,
    fetch_book,
    fetch_book_or_none,
    is_live,
    iter_events,
    market_fee_model,
    two_sided_quote,
    yes_token_id,
)
from sttbot.venues.prediction import FeeModel


def market(**over):
    base = dict(
        id="1",
        active=True,
        acceptingOrders=True,
        enableOrderBook=True,
        closed=False,
        archived=False,
        bestBid=0.30,
        bestAsk=0.32,
        groupItemTitle="Alpha",
        clobTokenIds='["tok-yes", "tok-no"]',
        liquidityNum=1000.0,
        volume24hr=500.0,
        feesEnabled=True,
        feeSchedule={"rate": 0.05, "rebateRate": 0.15, "takerOnly": True},
    )
    base.update(over)
    return base


def event(markets, **over):
    base = dict(slug="ev", title="An event", negRisk=True, markets=markets)
    base.update(over)
    return base


# --- pagination -------------------------------------------------------------


def test_pagination_uses_after_cursor_not_next_cursor():
    """The response field and the request parameter have different names.

    Sending it back as next_cursor is accepted and ignored by the real API, so
    this asserts on the URL rather than only on the collected output.
    """
    urls = []
    pages = {
        None: {"events": [{"id": "a"}], "next_cursor": "C1"},
        "C1": {"events": [{"id": "b"}], "next_cursor": "C2"},
        "C2": {"events": [], "next_cursor": None},
    }

    def fake(url):
        urls.append(url)
        cursor = None
        if f"{CURSOR_REQUEST_PARAM}=" in url:
            cursor = url.split(f"{CURSOR_REQUEST_PARAM}=")[1].split("&")[0]
        return pages[cursor]

    got = list(iter_events(fetch=fake, pause=0))
    assert [e["id"] for e in got] == ["a", "b"]
    assert f"{CURSOR_REQUEST_PARAM}=C1" in urls[1]
    assert "next_cursor=C1" not in urls[1]


def test_pagination_terminates_on_a_repeating_cursor():
    """A stuck cursor is what a wrong parameter name looks like from outside.

    Without this guard the scan loops forever collecting duplicates and calls
    itself complete.
    """
    calls = {"n": 0}

    def fake(url):
        calls["n"] += 1
        return {"events": [{"id": "same"}], "next_cursor": "STUCK"}

    got = list(iter_events(fetch=fake, pause=0))
    assert len(got) == 2  # first page, then the repeat is detected
    assert calls["n"] == 2


def test_pagination_stops_on_empty_page_and_respects_max_pages():
    def fake(url):
        return {"events": [{"id": "x"}], "next_cursor": "c" + str(len(url))}

    assert len(list(iter_events(fetch=fake, pause=0, max_pages=3))) == 3

    def empty(url):
        return {"events": []}

    assert list(iter_events(fetch=empty, pause=0)) == []


# --- market predicates ------------------------------------------------------


def test_is_live_requires_every_condition():
    assert is_live(market())
    for field, value in [
        ("active", False),
        ("acceptingOrders", False),
        ("enableOrderBook", False),
        ("closed", True),
        ("archived", True),
    ]:
        assert not is_live(market(**{field: value})), field


def test_two_sided_quote_rejects_degenerate_and_crossed_books():
    assert two_sided_quote(market()) == (0.30, 0.32)
    assert two_sided_quote(market(bestBid=0.5, bestAsk=0.4)) is None  # crossed
    assert two_sided_quote(market(bestBid=0.0, bestAsk=0.4)) is None  # resolved
    assert two_sided_quote(market(bestAsk=1.0)) is None
    assert two_sided_quote(market(bestBid=None)) is None


def test_fee_model_reads_the_per_market_schedule():
    fees = market_fee_model(market())
    assert fees.quadratic_taker == pytest.approx(0.05)
    # Makers are rebated, so their leg is income.
    assert fees.quadratic_maker == pytest.approx(-0.05 * 0.15)
    assert fees.cost(0.5, maker=True) < 0
    # Fees off means no cost at all, not the venue default.
    assert market_fee_model(market(feesEnabled=False)).cost(0.5) == 0.0


def test_yes_token_id_parses_the_json_string():
    assert yes_token_id(market()) == "tok-yes"
    assert yes_token_id(market(clobTokenIds=None)) is None
    assert yes_token_id(market(clobTokenIds="not json")) is None


# --- order book -------------------------------------------------------------


def test_order_book_sorts_asks_best_first():
    """The API returns asks worst-price-first, which inverts best_ask."""
    payload = {
        "asks": [{"price": "0.98", "size": "25"}, {"price": "0.45", "size": "5"}],
        "bids": [{"price": "0.10", "size": "50"}, {"price": "0.42", "size": "80"}],
    }
    book = OrderBook.from_payload(payload)
    assert book.best_ask == pytest.approx(0.45)
    assert book.best_bid == pytest.approx(0.42)


def test_order_book_drops_zero_size_and_malformed_levels():
    payload = {
        "asks": [
            {"price": "0.40", "size": "0"},
            {"price": "bad", "size": "5"},
            {"price": "0.50", "size": "10"},
        ],
        "bids": [],
    }
    book = OrderBook.from_payload(payload)
    assert book.asks == ((0.50, 10.0),)
    assert book.best_bid is None


def test_cost_to_buy_walks_levels_and_reports_insufficient_depth():
    book = OrderBook.from_payload(
        {"asks": [{"price": "0.40", "size": "10"}, {"price": "0.50", "size": "10"}],
         "bids": []}
    )
    assert book.cost_to_buy(10) == pytest.approx(4.0)
    assert book.cost_to_buy(15) == pytest.approx(4.0 + 2.5)
    # 20 is the whole book; 21 is not fillable, and a partial fill is not a fill.
    assert book.cost_to_buy(20) == pytest.approx(9.0)
    assert book.cost_to_buy(21) is None
    with pytest.raises(ValueError):
        book.cost_to_buy(0)


# --- basket completeness ----------------------------------------------------


def test_incomplete_baskets_are_dropped_not_scored():
    """The bug that manufactured +0.85 "edges".

    Three outcomes priced at 0.30 each sum to 0.90, but that is only an
    arbitrage if those three are all the outcomes there are. With one leg
    unquoted, the remaining two sum to 0.60 and look spectacular while leaving
    you short whatever the third outcome was.
    """
    quoted = [market(id="1", bestBid=0.28, bestAsk=0.30),
              market(id="2", bestBid=0.28, bestAsk=0.30)]
    missing = market(id="3", bestBid=None, bestAsk=None)

    assert complete_baskets([event(quoted + [missing])]) == []
    # And with the third leg present it is scored, correctly, as near par.
    full = quoted + [market(id="3", bestBid=0.28, bestAsk=0.30)]
    baskets = complete_baskets([event(full)])
    assert len(baskets) == 1
    assert baskets[0].ask_sum == pytest.approx(0.90)


def test_inactive_leg_also_makes_a_basket_incomplete():
    legs = [market(id="1"), market(id="2"), market(id="3", acceptingOrders=False)]
    assert complete_baskets([event(legs)]) == []


def test_non_negrisk_events_are_not_treated_as_baskets():
    """Outcomes must be mutually exclusive before a sum below 1.00 means anything."""
    legs = [market(id="1"), market(id="2")]
    assert complete_baskets([event(legs, negRisk=False)]) == []


def test_basket_paper_edge_accounts_for_fees():
    legs = [market(id=str(i), bestBid=0.28, bestAsk=0.30) for i in range(3)]
    (basket,) = complete_baskets([event(legs)])
    gross = 1.0 - basket.ask_sum
    assert basket.paper_edge() < gross  # taker fees eat into it
    assert basket.paper_edge() > 0


def test_basket_at_par_shows_no_edge():
    legs = [market(id=str(i), bestBid=0.32, bestAsk=1 / 3) for i in range(3)]
    (basket,) = complete_baskets([event(legs)])
    assert basket.paper_edge() < 0


# --- executable sizing ------------------------------------------------------


def _book(levels):
    return OrderBook.from_payload(
        {"asks": [{"price": str(p), "size": str(s)} for p, s in levels], "bids": []}
    )


def test_executable_arbitrage_is_capped_by_the_thinnest_leg():
    """Top of book says 0.90; depth says you can only have a little of it."""
    deep = _book([(0.30, 10_000)])
    thin = _book([(0.30, 50), (0.45, 10_000)])
    books = [deep, deep, thin]
    fees = [FeeModel()] * 3

    arb = executable_arbitrage(books, fees, step=5, max_contracts=1000)
    assert arb is not None
    # Beyond 50 contracts the thin leg costs 0.45 and the basket is at par.
    assert arb.contracts == pytest.approx(50)
    assert arb.profit == pytest.approx(50 * (1.0 - 0.90))
    assert arb.return_on_capital > 0


def test_executable_arbitrage_returns_none_when_fees_swallow_the_edge():
    books = [_book([(0.33, 1000)])] * 3
    heavy = [FeeModel(quadratic_taker=0.5)] * 3
    assert executable_arbitrage(books, heavy, step=5, max_contracts=100) is None


def test_executable_arbitrage_returns_none_on_a_par_basket():
    books = [_book([(1 / 3, 1000)])] * 3
    assert executable_arbitrage(books, [FeeModel()] * 3, step=5) is None


def test_executable_arbitrage_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        executable_arbitrage([_book([(0.3, 10)])], [FeeModel(), FeeModel()])
    assert executable_arbitrage([], []) is None


def test_executable_arbitrage_stops_at_the_end_of_the_book():
    """No depth beyond 100 contracts, so that is the maximum size."""
    books = [_book([(0.30, 100)])] * 3
    arb = executable_arbitrage(books, [FeeModel()] * 3, step=5, max_contracts=5000)
    assert arb is not None
    assert arb.contracts == pytest.approx(100)


# --- marking a position whose market has resolved ---------------------------


def test_fetch_book_raises_so_a_trade_is_never_priced_off_a_failed_read():
    def boom(url):
        raise RuntimeError("HTTP Error 404: Not Found")

    with pytest.raises(RuntimeError):
        fetch_book("tok", fetch=boom)


def test_fetch_book_or_none_survives_a_delisted_market():
    """Regression: this crashed a live paper run on every tick for four days.

    A resolved market's token is delisted and its book 404s permanently. That
    is terminal, not transient, so marking an existing position must tolerate
    it -- the position simply has no mark, which Snapshot.unmarked reports.
    """
    def boom(url):
        raise RuntimeError("HTTP Error 404: Not Found")

    assert fetch_book_or_none("tok", fetch=boom) is None


def test_fetch_book_or_none_returns_the_book_when_there_is_one():
    payload = {"asks": [{"price": "0.40", "size": "10"}],
               "bids": [{"price": "0.30", "size": "10"}]}
    book = fetch_book_or_none("tok", fetch=lambda url: payload)
    assert book is not None
    assert book.best_bid == pytest.approx(0.30)


# --- resolution horizon -----------------------------------------------------

def _basket(end_date=""):
    legs = [market(id=str(i), bestBid=0.28, bestAsk=0.30) for i in range(3)]
    (b,) = complete_baskets([event(legs, endDate=end_date)])
    return b


def test_days_to_resolution_reads_the_event_end_date():
    now = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)
    assert _basket("2026-08-26T12:00:00Z").days_to_resolution(now) == pytest.approx(7.0)
    assert _basket("2027-01-12T12:00:00Z").days_to_resolution(now) > 140


def test_a_past_due_market_reports_negative_days_not_an_error():
    """The venue had 17,296 live markets past their endDate in one snapshot,
    mostly in-play sport. They resolve soonest, so they must stay usable."""
    now = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)
    assert _basket("2026-08-17T12:00:00Z").days_to_resolution(now) == pytest.approx(-2.0)


def test_a_missing_or_unparseable_end_date_is_none_not_zero():
    """None means "unknown horizon" and gets skipped; zero would read as
    "resolves today" and be entered eagerly."""
    now = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)
    assert _basket("").days_to_resolution(now) is None
    assert _basket("not a date").days_to_resolution(now) is None


def test_naive_timestamps_are_treated_as_utc():
    now = dt.datetime(2026, 8, 19, 12, 0)          # naive
    assert _basket("2026-08-20T12:00:00Z").days_to_resolution(now) == pytest.approx(1.0)


def test_the_horizon_is_independent_of_the_edge():
    """The whole point: a long-dated basket can look identical on edge."""
    soon = _basket("2026-08-20T12:00:00Z")
    far = _basket("2027-01-12T12:00:00Z")
    assert soon.paper_edge() == pytest.approx(far.paper_edge())
    now = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.timezone.utc)
    assert far.days_to_resolution(now) > 100 * soon.days_to_resolution(now)
