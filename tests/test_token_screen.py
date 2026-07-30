import pytest

from sttbot.strategies.amm import Pool
from sttbot.strategies.token_screen import (
    ScreenThresholds,
    TokenMetadata,
    max_exit_size,
    position_limit,
    screen,
)


def clean(**overrides) -> TokenMetadata:
    """A token that passes every check, so tests can spoil one field at a time."""
    base = dict(
        symbol="GOOD",
        liquidity_usd=250_000.0,
        lp_locked_fraction=1.0,
        top_holder_fraction=0.05,
        top10_holder_fraction=0.18,
        sniper_bundled_fraction=0.03,
        holder_count=5_000,
        age_hours=720.0,
        buy_tax=0.0,
        sell_tax=0.0,
        can_mint=False,
        can_freeze=False,
        ownership_renounced=True,
        sell_simulation_passed=True,
    )
    base.update(overrides)
    return TokenMetadata(**base)


def test_clean_token_passes():
    result = screen(clean())
    assert result.passed
    assert result.summary() == "GOOD: pass"


@pytest.mark.parametrize(
    "field,value",
    [
        ("liquidity_usd", 1_000.0),
        ("lp_locked_fraction", 0.10),
        ("top_holder_fraction", 0.60),
        ("top10_holder_fraction", 0.55),
        ("sniper_bundled_fraction", 0.40),
        ("holder_count", 12),
        ("age_hours", 0.5),
        ("sell_tax", 0.35),
        ("can_mint", True),
        ("can_freeze", True),
        ("ownership_renounced", False),
        ("sell_simulation_passed", False),
    ],
)
def test_each_risk_factor_blocks(field, value):
    result = screen(clean(**{field: value}))
    assert not result.passed
    assert result.failures


def test_honeypot_tax_asymmetry_detected():
    # Cheap to buy, punitive to sell — the classic trap. Each tax alone is
    # under the absolute cap, so only the asymmetry check catches this.
    result = screen(clean(buy_tax=0.01, sell_tax=0.09))
    assert not result.passed
    assert any("honeypot" in f for f in result.failures)


def test_symmetric_tax_within_cap_is_allowed():
    assert screen(clean(buy_tax=0.05, sell_tax=0.05)).passed


@pytest.mark.parametrize(
    "field,value",
    [("sell_tax", 0.0), ("buy_tax", 0.0), ("top_holder_fraction", 0.0)],
)
def test_zero_is_a_value_not_a_missing_field(field, value):
    """Regression: `value or default` treated a legitimate 0.0 as absent,
    scoring a zero-tax token as if it charged 100%."""
    result = screen(clean(**{field: value}))
    assert result.passed, result.summary()
    assert field not in result.unknowns


def test_distributed_whale_supply_caught_by_top10_not_top1():
    """Ten coordinated wallets at 5% each pass a top-1 check comfortably.

    This is the gap the single-holder measure leaves: 50% of supply positioned
    to exit together, with no individual wallet looking unusual.
    """
    token = clean(top_holder_fraction=0.05, top10_holder_fraction=0.50)
    result = screen(token)
    assert not result.passed
    assert any("top 10 holders" in f for f in result.failures)
    # The single-holder check alone sees nothing wrong.
    assert not any("top holder owns" in f for f in result.failures)


def test_sniped_launch_blocks_despite_clean_distribution():
    token = clean(top_holder_fraction=0.02, top10_holder_fraction=0.10,
                  sniper_bundled_fraction=0.45)
    result = screen(token)
    assert not result.passed
    assert any("snipers/bundled" in f for f in result.failures)


def test_unknown_data_blocks_but_is_reported_separately():
    result = screen(clean(lp_locked_fraction=None, holder_count=None))
    assert not result.passed
    assert not result.failures            # nothing was *judged* and lost
    assert set(result.unknowns) == {"lp_locked_fraction", "holder_count"}
    assert "unknown" in result.summary()


def test_all_unknown_token_never_passes():
    # Absence of evidence must not read as evidence of safety.
    assert not screen(TokenMetadata(symbol="MYSTERY")).passed


def test_failure_messages_are_actionable():
    result = screen(clean(lp_locked_fraction=0.0))
    assert any("liquidity can be pulled" in f for f in result.failures)


def test_thresholds_are_tunable():
    token = clean(holder_count=50)
    assert not screen(token).passed
    relaxed = ScreenThresholds(min_holder_count=10)
    assert screen(token, relaxed).passed


# --- exit sizing -----------------------------------------------------------


@pytest.fixture
def pool():
    return Pool(token_reserve=1_000_000.0, numeraire_reserve=100_000.0, fee=0.003)


def test_max_exit_size_hits_the_impact_budget(pool):
    size = max_exit_size(pool, max_price_impact=0.02)
    assert size > 0
    realised = 1.0 - (pool.sell_token(size) / size) / pool.spot_price
    assert realised == pytest.approx(0.02, abs=1e-9)


def test_exit_size_scales_with_depth(pool):
    deep = Pool(10_000_000.0, 1_000_000.0, fee=0.003)
    assert max_exit_size(deep, 0.02) == pytest.approx(
        10 * max_exit_size(pool, 0.02)
    )


def test_impact_budget_below_fee_is_unreachable(pool):
    # 30bp fee, 10bp budget: no trade size can satisfy it.
    assert max_exit_size(pool, max_price_impact=0.001) == 0.0


def test_bigger_budget_allows_bigger_exit(pool):
    assert max_exit_size(pool, 0.05) > max_exit_size(pool, 0.02)


def test_max_exit_size_validates_budget(pool):
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            max_exit_size(pool, bad)


def test_position_limit_takes_the_binding_constraint(pool):
    # Rich trader, thin pool -> depth binds.
    depth_bound = position_limit(pool, equity=10_000_000.0)
    assert depth_bound == pytest.approx(
        max_exit_size(pool, 0.02) * pool.spot_price
    )
    # Small trader, same pool -> the risk budget binds instead.
    risk_bound = position_limit(pool, equity=10_000.0)
    assert risk_bound == pytest.approx(200.0)
    assert risk_bound < depth_bound


def test_position_limit_validates_inputs(pool):
    with pytest.raises(ValueError):
        position_limit(pool, equity=0)
    with pytest.raises(ValueError):
        position_limit(pool, equity=1_000, max_equity_fraction=1.5)
