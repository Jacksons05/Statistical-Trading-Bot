"""Tests for the Kalshi client.

Every test injects a fake fetcher, so the suite never touches the network. The
tests that matter are the ones pinning this venue's specific traps -- each
produces a confidently wrong answer rather than an error, and two of them
caught me during the build.
"""

import pytest

from sttbot.venues.kalshi import (
    KALSHI_FEES,
    MAX_PLAUSIBLE_EDGE,
    Basket,
    OrderBook,
    complete_baskets,
    executable_arbitrage,
    fetch_orderbook,
    is_live,
    iter_events,
    two_sided_quote,
)


def market(**over):
    base = dict(
        ticker="T-A",
        status="active",
        yes_bid_dollars="0.3000",
        yes_ask_dollars="0.3100",
        yes_sub_title="Alpha",
        volume_24h_fp="1000.0",
    )
    base.update(over)
    return base


def event(markets, **over):
    base = dict(event_ticker="EV", title="An event",
                mutually_exclusive=True, markets=markets)
    base.update(over)
    return base


# --- prices are decimal dollars ---------------------------------------------


def test_quotes_are_decimal_dollars_not_cents():
    """A cents reading would make every price 100x too small."""
    assert two_sided_quote(market()) == (0.30, 0.31)


def test_quote_rejects_crossed_and_degenerate_books():
    assert two_sided_quote(market(yes_bid_dollars="0.5", yes_ask_dollars="0.4")) is None
    assert two_sided_quote(market(yes_bid_dollars="0")) is None
    assert two_sided_quote(market(yes_ask_dollars="1.0")) is None
    assert two_sided_quote(market(yes_bid_dollars=None)) is None


def test_is_live_reads_status():
    assert is_live(market())
    assert is_live(market(status="open"))
    assert not is_live(market(status="settled"))
    assert not is_live(market(status="closed"))


# --- the order book is two BID ladders --------------------------------------


def test_yes_asks_are_derived_from_the_no_bid_ladder():
    """The trap: `yes_dollars` and `no_dollars` are BOTH bids.

    Reading `yes_dollars` as asks prices every trade off the wrong side. This
    reproduces a real book: no bid 0.9540 implies a yes ask of 0.0460, with the
    size carried across.
    """
    payload = {"orderbook_fp": {
        "yes_dollars": [["0.0440", "1136.78"], ["0.0450", "10.38"]],
        "no_dollars": [["0.9500", "23284.96"], ["0.9540", "30030.52"]],
    }}
    book = OrderBook.from_payload(payload)
    assert book.best_yes_ask == pytest.approx(0.0460)   # 1 - 0.9540
    assert book.best_yes_bid == pytest.approx(0.0450)
    assert book.yes_asks[0][1] == pytest.approx(30030.52)  # size carries over
    # And the ask is genuinely above the bid, as a sane book requires.
    assert book.best_yes_ask > book.best_yes_bid


def test_walking_the_book_takes_the_cheapest_yes_first():
    payload = {"orderbook_fp": {
        "yes_dollars": [],
        # no bids 0.90 and 0.80 -> yes asks 0.10 and 0.20
        "no_dollars": [["0.9000", "10"], ["0.8000", "10"]],
    }}
    book = OrderBook.from_payload(payload)
    assert book.cost_to_buy_yes(10) == pytest.approx(1.0)        # 10 @ 0.10
    assert book.cost_to_buy_yes(15) == pytest.approx(1.0 + 1.0)  # + 5 @ 0.20
    assert book.cost_to_buy_yes(20) == pytest.approx(3.0)
    assert book.cost_to_buy_yes(21) is None      # partial fill is not a fill
    with pytest.raises(ValueError):
        book.cost_to_buy_yes(0)


def test_orderbook_survives_junk_and_the_legacy_key():
    assert OrderBook.from_payload(None) is None
    assert OrderBook.from_payload({}) is None
    assert OrderBook.from_payload({"orderbook_fp": {}}) is None
    legacy = {"orderbook": {"no_dollars": [["0.9", "5"]], "yes_dollars": []}}
    assert OrderBook.from_payload(legacy) is not None
    bad = {"orderbook_fp": {"no_dollars": [["x", "5"], ["0.9", "0"], ["0.8", "3"]],
                            "yes_dollars": []}}
    assert OrderBook.from_payload(bad).yes_asks == ((pytest.approx(0.2), 3.0),)


def test_fetch_orderbook_returns_none_on_failure():
    def boom(url):
        raise RuntimeError("404")
    assert fetch_orderbook("T", fetch=boom) is None


# --- mutually_exclusive is NOT exhaustive -----------------------------------


def test_mutually_exclusive_does_not_mean_complete():
    """The live case: "LA-01 Republican nominee?" is flagged mutually
    exclusive, lists two candidates, and their asks sum to 0.108. That is not
    an 89c arbitrage, it is a basket whose other legs were never listed."""
    legs = [market(ticker="A", yes_bid_dollars="0.034", yes_ask_dollars="0.070"),
            market(ticker="B", yes_bid_dollars="0.004", yes_ask_dollars="0.038")]
    (basket,) = complete_baskets([event(legs)])
    assert basket.ask_sum == pytest.approx(0.108)
    assert basket.paper_edge() > 0.80          # looks spectacular
    assert basket.looks_complete is False      # and is rejected


def test_a_near_par_basket_is_treated_as_complete():
    legs = [market(ticker=t, yes_bid_dollars="0.32", yes_ask_dollars="0.33")
            for t in ("A", "B", "C")]
    (basket,) = complete_baskets([event(legs)])
    assert basket.ask_sum == pytest.approx(0.99)
    assert basket.looks_complete is True


def test_the_completeness_threshold_is_where_it_claims():
    floor = 1.0 - MAX_PLAUSIBLE_EDGE
    at = [market(ticker="A", yes_bid_dollars=f"{floor-0.01:.4f}",
                 yes_ask_dollars=f"{floor:.4f}"),
          market(ticker="B", yes_bid_dollars="0.0010", yes_ask_dollars="0.0020")]
    (basket,) = complete_baskets([event(at)])
    assert basket.ask_sum >= floor
    assert basket.looks_complete is True

    below = [market(ticker="A", yes_bid_dollars=f"{floor-0.06:.4f}",
                    yes_ask_dollars=f"{floor-0.05:.4f}"),
             market(ticker="B", yes_bid_dollars="0.0010", yes_ask_dollars="0.0020")]
    (basket,) = complete_baskets([event(below)])
    assert basket.looks_complete is False


def test_non_exclusive_events_are_not_baskets():
    legs = [market(ticker="A"), market(ticker="B")]
    assert complete_baskets([event(legs, mutually_exclusive=False)]) == []


def test_an_unquoted_or_dead_leg_drops_the_basket():
    quoted = [market(ticker="A"), market(ticker="B")]
    assert complete_baskets([event(quoted + [market(ticker="C",
                                                    yes_bid_dollars=None)])]) == []
    assert complete_baskets([event(quoted + [market(ticker="C",
                                                    status="settled")])]) == []


# --- enumeration -------------------------------------------------------------


def test_iter_events_paginates_by_cursor_and_stops_on_repeat():
    pages = {
        None: {"events": [{"event_ticker": "a"}], "cursor": "C1"},
        "C1": {"events": [{"event_ticker": "b"}], "cursor": "C1"},   # stuck
    }

    def fake(url):
        cur = url.split("cursor=")[1].split("&")[0] if "cursor=" in url else None
        return pages[cur]

    got = list(iter_events(fetch=fake, pause=0))
    assert [e["event_ticker"] for e in got] == ["a", "b"]


def test_iter_events_uses_the_nested_markets_endpoint():
    """/markets is 99.7% auto-generated parlay combos; /events is the real one."""
    seen = []

    def fake(url):
        seen.append(url)
        return {"events": []}

    list(iter_events(fetch=fake, pause=0))
    assert "/events?" in seen[0]
    assert "with_nested_markets=true" in seen[0]


def test_iter_events_deduplicates_and_respects_max_pages():
    def fake(url):
        return {"events": [{"event_ticker": "same"}], "cursor": "x" + str(len(url))}

    assert len(list(iter_events(fetch=fake, pause=0, max_pages=3))) == 1


# --- executable sizing -------------------------------------------------------


def _book(no_levels):
    return OrderBook.from_payload(
        {"orderbook_fp": {"yes_dollars": [],
                          "no_dollars": [[str(p), str(s)] for p, s in no_levels]}}
    )


def test_executable_arbitrage_is_capped_by_the_thinnest_leg():
    deep = _book([("0.70", 10_000)])      # yes ask 0.30
    thin = _book([("0.70", 40), ("0.55", 10_000)])  # 0.30 then 0.45
    arb = executable_arbitrage([deep, deep, thin], fees=KALSHI_FEES,
                               step=10, max_contracts=500)
    assert arb is not None
    assert arb.contracts == pytest.approx(40)
    assert arb.return_on_capital > 0


def test_executable_arbitrage_returns_none_at_par():
    books = [_book([("0.6667", 1000)])] * 3   # yes asks ~1/3 each
    assert executable_arbitrage(books, step=10) is None
    assert executable_arbitrage([]) is None


def test_kalshis_fee_swallows_a_four_cent_edge_on_three_legs():
    """Worth stating as a number, because it sets the bar for the venue.

    Three legs at 0.32 sum to 0.96 -- a 4c gross edge. Kalshi charges
    0.07*p*(1-p) per leg, which is 1.52c each and 4.57c across the basket. The
    trade is underwater before it is placed.
    """
    books = [_book([("0.68", 1000)])] * 3          # yes asks 0.32
    assert executable_arbitrage(books, fees=KALSHI_FEES, step=10,
                                max_contracts=100) is None

    # At 0.31 a leg the gross edge is 7c against a 4.5c fee, and it clears.
    better = [_book([("0.69", 1000)])] * 3
    arb = executable_arbitrage(better, fees=KALSHI_FEES, step=10,
                               max_contracts=100)
    assert arb is not None and arb.profit > 0
