"""Tests for the token data pipeline feeding token_screen.

The mapping's whole job is to be honest about what it knows, so most of these
test the boundary between *zero* and *unknown*. Getting that wrong is not a
rounding error: it either fails a clean token on a gap that does not exist, or
passes an unauditable one as safe.
"""

import pytest

from sttbot.data.tokens import (
    TokenPair,
    goplus_security,
    latest_token_addresses,
    pairs_for_tokens,
    parse_pair,
    search_pairs,
    to_token_metadata,
)
from sttbot.strategies.token_screen import screen

NOW_MS = 1_785_400_000_000


def raw_pair(**over):
    base = {
        "chainId": "solana",
        "baseToken": {"address": "AAA", "symbol": "WIF", "name": "dogwifhat"},
        "liquidity": {"usd": 32022.6},
        "volume": {"h24": 2343137.06},
        "txns": {"h24": {"buys": 500, "sells": 400}},
        "priceUsd": "0.0001624",
        "fdv": 154708,
        "priceChange": {"h24": 445},
        "pairCreatedAt": NOW_MS - 48 * 3_600_000,
    }
    base.update(over)
    return base


def security(**over):
    base = {
        "holder_count": 1266,
        "mintable": {"authority": [], "status": "0"},
        "freezable": {"authority": [], "status": "0"},
        "balance_mutable_authority": {"authority": [], "status": "0"},
        "transfer_fee": {},
        "holders": [{"percent": "0.10", "tag": ""}, {"percent": "0.04", "tag": ""}],
        "lp_holders": [{"percent": "1.0", "is_locked": 1, "tag": "burn"}],
    }
    base.update(over)
    return base


# --- DexScreener parsing ----------------------------------------------------


def test_parse_pair_extracts_market_state():
    pair = parse_pair(raw_pair())
    assert pair.symbol == "WIF"
    assert pair.liquidity_usd == pytest.approx(32022.6)
    assert pair.trades_24h == 900
    assert pair.age_hours(NOW_MS) == pytest.approx(48.0)


def test_age_is_unknown_without_a_creation_timestamp():
    assert parse_pair(raw_pair(pairCreatedAt=None)).age_hours(NOW_MS) is None


def test_age_never_goes_negative_on_clock_skew():
    future = parse_pair(raw_pair(pairCreatedAt=NOW_MS + 10_000_000))
    assert future.age_hours(NOW_MS) == 0.0


def test_pairs_for_tokens_keeps_the_deepest_pool():
    """A token trades in several pools; the deep one is the real market."""
    shallow = raw_pair(liquidity={"usd": 500.0}, pairAddress="p1")
    deep = raw_pair(liquidity={"usd": 90_000.0}, pairAddress="p2")

    def fake(url):
        return {"pairs": [shallow, deep]}

    got = pairs_for_tokens(["AAA"], fetch=fake, pause=0)
    assert got["AAA"].liquidity_usd == pytest.approx(90_000.0)


def test_pairs_for_tokens_batches_within_the_api_limit():
    seen = []

    def fake(url):
        seen.append(url.rsplit("/", 1)[1].split(","))
        return {"pairs": []}

    pairs_for_tokens([f"T{i}" for i in range(70)], fetch=fake, pause=0)
    assert [len(b) for b in seen] == [30, 30, 10]


def test_discovery_filters_by_chain():
    def fake(url):
        return [
            {"chainId": "solana", "tokenAddress": "S1"},
            {"chainId": "ton", "tokenAddress": "T1"},
            {"chainId": "solana"},  # malformed, no address
        ]

    assert latest_token_addresses(fetch=fake) == ["S1"]


def test_clients_tolerate_unexpected_payloads():
    assert latest_token_addresses(fetch=lambda u: {"error": "x"}) == []
    assert search_pairs("q", fetch=lambda u: []) == []
    assert pairs_for_tokens(["A"], fetch=lambda u: None, pause=0) == {}


def test_goplus_merges_results_per_address():
    def fake(url):
        addr = url.split("=")[1]
        return {"result": {addr: {"holder_count": 5}}}

    got = goplus_security(["A", "B"], fetch=fake, pause=0)
    assert set(got) == {"A", "B"}


# --- zero versus unknown ----------------------------------------------------


def test_absent_transfer_fee_extension_is_a_real_zero_tax():
    """A plain SPL token cannot charge a tax, so 0% is a fact, not a gap."""
    meta = to_token_metadata(parse_pair(raw_pair()), security(), now_ms=NOW_MS)
    assert meta.buy_tax == 0.0
    assert meta.sell_tax == 0.0
    assert "buy_tax" not in screen(meta).unknowns


def test_missing_transfer_fee_key_is_unknown_not_zero():
    sec = security()
    del sec["transfer_fee"]
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.buy_tax is None


def test_transfer_fee_basis_points_are_converted():
    sec = security(transfer_fee={"transfer_fee_basis_points": 250})
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.sell_tax == pytest.approx(0.025)


def test_unenumerated_lp_holders_are_unknown_not_unlocked():
    """An indexing gap is not evidence the LP is free to be pulled."""
    meta = to_token_metadata(
        parse_pair(raw_pair()), security(lp_holders=[]), now_ms=NOW_MS
    )
    assert meta.lp_locked_fraction is None
    assert "lp_locked_fraction" in screen(meta).unknowns


def test_burned_lp_counts_as_locked():
    meta = to_token_metadata(parse_pair(raw_pair()), security(), now_ms=NOW_MS)
    assert meta.lp_locked_fraction == pytest.approx(1.0)


def test_unlocked_lp_is_zero_not_unknown():
    sec = security(lp_holders=[{"percent": "1.0", "is_locked": 0, "tag": ""}])
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.lp_locked_fraction == 0.0


# --- concentration ----------------------------------------------------------


def test_top10_concentration_is_summed_and_top1_is_the_max():
    sec = security(holders=[{"percent": "0.20"}, {"percent": "0.05"},
                            {"percent": "0.03"}])
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.top_holder_fraction == pytest.approx(0.20)
    assert meta.top10_holder_fraction == pytest.approx(0.28)


def test_pool_and_burn_addresses_are_excluded_from_concentration():
    """The pool holding most of the supply is not a whale."""
    sec = security(holders=[{"percent": "0.80", "tag": "Raydium Pool"},
                            {"percent": "0.06", "tag": ""}])
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.top_holder_fraction == pytest.approx(0.06)


def test_concentration_unknown_when_holders_absent():
    meta = to_token_metadata(parse_pair(raw_pair()), security(holders=[]),
                             now_ms=NOW_MS)
    assert meta.top_holder_fraction is None
    assert meta.top10_holder_fraction is None


def test_distributed_whales_are_caught_by_the_top10_check():
    """Ten wallets at 5% pass a top-1 check and must still fail the screen."""
    sec = security(holders=[{"percent": "0.05"} for _ in range(10)])
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.top_holder_fraction == pytest.approx(0.05)  # looks fine alone
    assert meta.top10_holder_fraction == pytest.approx(0.50)
    assert any("top 10" in f for f in screen(meta).failures)


# --- authorities ------------------------------------------------------------


def test_authority_status_maps_to_capability():
    clean = to_token_metadata(parse_pair(raw_pair()), security(), now_ms=NOW_MS)
    assert clean.can_mint is False
    assert clean.can_freeze is False
    assert clean.ownership_renounced is True

    risky = to_token_metadata(
        parse_pair(raw_pair()),
        security(mintable={"authority": ["someone"], "status": "1"}),
        now_ms=NOW_MS,
    )
    assert risky.can_mint is True
    assert risky.ownership_renounced is False
    assert any("mint authority" in f for f in screen(risky).failures)


def test_missing_authority_record_is_unknown():
    sec = security()
    del sec["mintable"]
    meta = to_token_metadata(parse_pair(raw_pair()), sec, now_ms=NOW_MS)
    assert meta.can_mint is None
    assert meta.ownership_renounced is None


def test_fields_no_source_covers_stay_unknown():
    """Better a reported gap than a silent assumption of safety."""
    meta = to_token_metadata(parse_pair(raw_pair()), security(), now_ms=NOW_MS)
    assert meta.sniper_bundled_fraction is None
    assert meta.sell_simulation_passed is None
    result = screen(meta)
    assert "sniper_bundled_fraction" in result.unknowns
    assert "sell_simulation_passed" in result.unknowns
    assert not result.passed


def test_a_token_with_no_security_record_is_all_unknowns():
    meta = to_token_metadata(parse_pair(raw_pair()), None, now_ms=NOW_MS)
    assert meta.liquidity_usd is not None  # market data still present
    assert meta.top10_holder_fraction is None
    assert not screen(meta).passed
