"""Tests for the breadth portfolio engine.

The arithmetic here decides whether the strategy is viable at all, so the
tests check the economics against hand-computed values rather than only
asserting that numbers come out.
"""

import math

import pytest

from sttbot.strategies.amm import Pool
from sttbot.strategies.breadth import (
    Allocation,
    BreadthParams,
    Candidate,
    allocate,
    breakeven_multiple,
    names_for_confidence,
    probability_of_no_winner,
    round_trip_cost,
    simulate,
)


def pool(liquidity_usd: float, fee: float = 0.003) -> Pool:
    """Constant-product pool with ``liquidity_usd`` on the numeraire side."""
    return Pool(token_reserve=liquidity_usd * 100.0, numeraire_reserve=liquidity_usd,
                fee=fee)


# --- round-trip cost --------------------------------------------------------


def test_round_trip_cost_is_at_least_both_swap_fees():
    """An infinitesimal trade still pays the fee twice."""
    deep = pool(10_000_000.0)
    cost = round_trip_cost(deep, 100.0, gas_usd=0.0)
    assert cost == pytest.approx(2 * 0.003, abs=5e-4)


def test_round_trip_cost_rises_with_size_against_a_thin_pool():
    """Regression: pricing the exit against your own moved pool inverts this.

    If the sell is simulated against the reserves the buy just shifted, the
    entry impact reverses and cost *falls* with size (0.598% at $50 down to
    0.480% at $5,000 in the thin pool below) -- which would tell a
    capacity-constrained strategy to trade bigger.
    """
    thin = pool(20_000.0)
    small = round_trip_cost(thin, 50.0, gas_usd=0.0)
    large = round_trip_cost(thin, 5_000.0, gas_usd=0.0)
    assert large > small > 0
    # Self-inflicted impact dominates once the trade is material to the pool.
    assert large > 2 * 0.003 * 2


def test_round_trip_cost_includes_gas_and_it_bites_at_small_size():
    thin = pool(50_000.0)
    tiny = round_trip_cost(thin, 25.0, gas_usd=0.02)
    big = round_trip_cost(thin, 2_500.0, gas_usd=0.02)
    # A fixed cost is a percentage cost, and the percentage is worse when small.
    assert tiny - round_trip_cost(thin, 25.0, gas_usd=0.0) == pytest.approx(
        0.04 / 25.0
    )
    assert big - round_trip_cost(thin, 2_500.0, gas_usd=0.0) < 1e-4


def test_round_trip_cost_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        round_trip_cost(pool(1000.0), 0.0)


# --- the economics ----------------------------------------------------------


def test_breakeven_multiple_matches_hand_calculation():
    """At a 3% hit rate and 4% costs the winners must return 33x."""
    assert breakeven_multiple(0.03, 0.04) == pytest.approx(1.04 + 0.97 / 0.03)
    assert breakeven_multiple(0.03, 0.04) == pytest.approx(33.37, abs=0.01)


def test_breakeven_multiple_is_brutal_at_low_hit_rates():
    assert breakeven_multiple(0.50, 0.0) == pytest.approx(2.0)
    assert breakeven_multiple(0.10, 0.0) == pytest.approx(10.0)
    assert breakeven_multiple(0.01, 0.0) == pytest.approx(100.0)
    # Halving the hit rate roughly doubles what the winners must deliver.
    assert breakeven_multiple(0.02, 0.04) > 1.9 * breakeven_multiple(0.04, 0.04)


def test_partial_recovery_on_failures_lowers_the_bar_a_lot():
    """Being able to exit a dying position at all changes the arithmetic."""
    total_loss = breakeven_multiple(0.03, 0.04, loss_given_failure=1.0)
    half_loss = breakeven_multiple(0.03, 0.04, loss_given_failure=0.5)
    assert half_loss < total_loss / 1.9


def test_breakeven_multiple_validates_inputs():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            breakeven_multiple(bad, 0.01)
    with pytest.raises(ValueError):
        breakeven_multiple(0.1, -0.01)


def test_names_for_confidence_matches_the_closed_form():
    """At a 3% hit rate you need ~99 concurrent names to be 95% sure of one."""
    assert names_for_confidence(0.03, 0.95) == 99
    assert names_for_confidence(0.03, 0.95) == math.ceil(
        math.log(0.05) / math.log(0.97)
    )
    assert names_for_confidence(0.5, 0.95) == 5


def test_a_small_book_misses_entirely_most_of_the_time():
    """Why breadth is not optional at these hit rates."""
    assert probability_of_no_winner(0.03, 20) == pytest.approx(0.5438, abs=1e-3)
    assert probability_of_no_winner(0.03, 100) < 0.05
    assert probability_of_no_winner(0.03, 0) == 1.0


def test_names_for_confidence_validates_inputs():
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            names_for_confidence(bad)
    with pytest.raises(ValueError):
        names_for_confidence(0.03, confidence=1.0)


# --- allocation -------------------------------------------------------------


def params(**over) -> BreadthParams:
    base = dict(bankroll=10_000.0, max_positions=10, risk_per_position=0.01,
                max_deployed_fraction=0.5, min_position_usd=25.0, gas_usd=0.0)
    base.update(over)
    return BreadthParams(**base)


def test_deep_pools_are_sized_by_the_risk_budget():
    cands = [Candidate(f"T{i}", pool(5_000_000.0)) for i in range(5)]
    book = allocate(cands, params())
    assert len(book.allocations) == 5
    assert all(a.size_usd == pytest.approx(100.0) for a in book.allocations)
    assert all(a.capped_by == "risk_budget" for a in book.allocations)


def test_thin_pools_are_sized_by_depth_not_conviction():
    """The market decides the position, and it decides it is small."""
    cands = [Candidate("THIN", pool(20_000.0))]
    book = allocate(cands, params(risk_per_position=0.5))  # wants $5,000
    (alloc,) = book.allocations
    assert alloc.capped_by == "pool_depth"
    assert alloc.size_usd < 500.0


def test_positions_too_small_to_bother_with_are_rejected_with_a_reason():
    cands = [Candidate("DUST", pool(50.0))]
    book = allocate(cands, params())
    assert book.allocations == ()
    assert "depth caps size" in book.rejected[0][1]


def test_ruinous_round_trip_costs_are_rejected():
    """A pool you cannot get out of cheaply is not an opportunity.

    The impact budget is loosened here so the name clears the depth check and
    is rejected on cost specifically, rather than on depth first.
    """
    cands = [Candidate("GRABBY", pool(1_000_000.0, fee=0.06))]
    book = allocate(cands, params(max_price_impact=0.20, max_round_trip_cost=0.10))
    assert book.allocations == ()
    assert "round trip costs" in book.rejected[0][1]


def test_portfolio_cap_binds_before_the_position_count():
    cands = [Candidate(f"T{i}", pool(5_000_000.0)) for i in range(50)]
    # 50 names x $500 wanted = $25,000, but only $5,000 may be deployed.
    book = allocate(cands, params(max_positions=50, risk_per_position=0.05))
    assert book.deployed <= 5_000.0 + 1e-9
    assert any("portfolio cap" in reason for _, reason in book.rejected)


def test_max_positions_is_respected():
    cands = [Candidate(f"T{i}", pool(5_000_000.0)) for i in range(30)]
    book = allocate(cands, params(max_positions=7))
    assert len(book.allocations) == 7
    assert any("max_positions" in reason for _, reason in book.rejected)


def test_book_reports_deployed_and_mean_cost():
    cands = [Candidate(f"T{i}", pool(1_000_000.0)) for i in range(4)]
    book = allocate(cands, params())
    assert book.deployed == pytest.approx(400.0)
    assert 0 < book.mean_cost < 0.02
    assert "4 positions" in book.summary()


def test_empty_candidate_list_is_an_empty_book():
    book = allocate([], params())
    assert book.allocations == () and book.rejected == ()
    assert book.deployed == 0.0
    assert book.mean_cost == 0.0


def test_params_validate():
    for bad in (dict(bankroll=0), dict(max_positions=0),
                dict(risk_per_position=0), dict(risk_per_position=1.5),
                dict(max_deployed_fraction=0)):
        with pytest.raises(ValueError):
            params(**bad)


# --- simulation -------------------------------------------------------------


def test_simulation_is_deterministic_for_a_seed():
    kwargs = dict(win_rate=0.03, cost_fraction=0.04, rounds=5, trials=200, seed=3)
    a = simulate(params(max_positions=100), **kwargs)
    b = simulate(params(max_positions=100), **kwargs)
    assert a.median == b.median and a.mean == b.mean


def test_below_breakeven_the_book_loses_most_of_the_time():
    """A thin tail cannot fund a 97% failure rate."""
    result = simulate(
        params(max_positions=100),
        win_rate=0.03,
        cost_fraction=0.04,
        rounds=10,
        trials=400,
        pareto_alpha=6.0,  # thin tail: winners rarely exceed ~3x
        seed=1,
    )
    assert result.probability_of_loss > 0.9
    assert result.median < result.starting_bankroll


def test_a_fat_tail_produces_the_characteristic_breadth_shape():
    """Mean far above median: most books lose, a few pay for everything."""
    result = simulate(
        params(max_positions=100),
        win_rate=0.03,
        cost_fraction=0.04,
        rounds=10,
        trials=600,
        pareto_alpha=0.8,
        seed=2,
    )
    assert result.mean > result.median
    assert result.p95 > result.p05


def test_more_names_reduce_the_spread_of_outcomes():
    """The actual argument for breadth: same edge, less variance."""
    common = dict(win_rate=0.05, cost_fraction=0.04, rounds=6, trials=500,
                  pareto_alpha=1.2, seed=7)
    narrow = simulate(params(max_positions=10, risk_per_position=0.05), **common)
    wide = simulate(params(max_positions=100, risk_per_position=0.005), **common)
    narrow_spread = (narrow.p95 - narrow.p05) / narrow.starting_bankroll
    wide_spread = (wide.p95 - wide.p05) / wide.starting_bankroll
    assert wide_spread < narrow_spread


def test_simulation_reports_ruin_and_never_goes_negative():
    result = simulate(
        params(max_positions=20, risk_per_position=0.05),
        win_rate=0.01,
        cost_fraction=0.05,
        rounds=20,
        trials=300,
        pareto_alpha=5.0,
        seed=4,
    )
    assert 0.0 <= result.probability_of_ruin <= 1.0
    assert result.probability_of_ruin > 0.5  # hopeless parameters, honestly so
    assert result.p05 >= 0.0


def test_simulation_validates_inputs():
    for bad in (dict(rounds=0), dict(trials=0), dict(pareto_alpha=0)):
        kwargs = dict(win_rate=0.03, cost_fraction=0.04, rounds=5, trials=10)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            simulate(params(), **kwargs)
