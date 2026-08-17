"""Breadth: many tokens, small size, automated.

The single-name capacity ceiling is the binding constraint on low-cap tokens.
The only cohort with real flow is under a day old, and it supports a few
hundred dollars of exit each. You cannot beat that with depth, so the only
remaining shape is width: many simultaneous positions, each small enough to
leave, sized by the pool rather than by conviction.

That changes what has to be true. A breadth book in this asset class is a
power-law bet -- almost every name goes to zero and the return comes from a
handful of survivors -- so the question is not "is this token good?" but
"given the hit rate and the round-trip cost, how large must the winners be,
and do I hold enough names to catch one?"

This module answers both, and deliberately does not pretend to answer a third.

* :func:`round_trip_cost` is **exact**, simulated through the constant-product
  pool: buy, move the reserves, sell back. It includes the fee on both legs and
  the impact you cause yourself, which is the cost small-size traders most
  often forget because it does not appear on a fee schedule.
* :func:`breakeven_multiple` and :func:`names_for_confidence` are **arithmetic**
  on an assumed hit rate. They tell you what the world must look like for the
  strategy to work.
* :func:`simulate` is a **Monte Carlo over an assumed payoff distribution**. It
  is not a backtest. Nothing here has been fitted to realised token returns,
  because that data was never collected -- the numbers it produces are
  conditional on the distribution you hand it, and are worth exactly as much as
  that assumption.

The honest summary of the arithmetic: at a 3% survival rate and a 4%
round-trip, winners must return **33x** to break even, and you need about
**100 concurrent names** for a 95% chance of holding a single one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .amm import Pool

# Solana base fee plus a competitive priority fee, per transaction. Small in
# absolute terms, but a fixed cost is a percentage cost at $50 a clip, which is
# exactly the regime a breadth book lives in.
DEFAULT_GAS_USD = 0.02


def round_trip_cost(
    pool: Pool, size_usd: float, *, gas_usd: float = DEFAULT_GAS_USD
) -> float:
    """Fraction of ``size_usd`` lost buying in and selling out at one price.

    Both legs are priced against the *same* pool state, which is the modelling
    decision that matters. Simulating the sell against the pool your own buy
    just moved lets your entry impact reverse itself, and the round trip then
    collapses to twice the swap fee no matter how large the trade -- it even
    falls slightly as size grows, which is obvious nonsense for a strategy that
    is capacity-constrained. That reversal is only available if you exit in the
    same instant, which a position held for hours or days does not.

    Pricing both legs at the unmoved pool charges the fee twice and the impact
    twice, holding the underlying price fixed so this measures friction alone
    rather than a view. Returns a positive fraction: 0.04 means 4% of the stake
    is gone before the price has done anything.
    """
    if size_usd <= 0:
        raise ValueError("size_usd must be positive")

    tokens = pool.buy_token(size_usd)
    if tokens <= 0:
        return 1.0
    returned = pool.sell_token(tokens)
    lost = size_usd - returned + 2.0 * gas_usd
    return max(lost / size_usd, 0.0)


def breakeven_multiple(
    win_rate: float, cost_fraction: float, *, loss_given_failure: float = 1.0
) -> float:
    """Price multiple a winner must reach for the book to break even.

    Solves ``p * (M - 1 - c) = (1 - p) * L`` for ``M``: the winners must fund
    their own costs and every loser beside them.

    ``loss_given_failure`` is the fraction of stake lost on a failure, 1.0
    meaning the position cannot be exited at all -- the realistic case when
    liquidity is pulled.
    """
    if not 0.0 < win_rate <= 1.0:
        raise ValueError("win_rate must be in (0, 1]")
    if cost_fraction < 0:
        raise ValueError("cost_fraction must be non-negative")
    if loss_given_failure < 0:
        raise ValueError("loss_given_failure must be non-negative")
    return 1.0 + cost_fraction + (1.0 - win_rate) * loss_given_failure / win_rate


def names_for_confidence(win_rate: float, confidence: float = 0.95) -> int:
    """Concurrent positions needed for ``confidence`` of holding >=1 winner.

    The reason breadth is not optional. At a 3% hit rate a 20-name book misses
    entirely 54% of the time, and a strategy that is right on average but
    bankrupt half the time is not a strategy.
    """
    if not 0.0 < win_rate < 1.0:
        raise ValueError("win_rate must be in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - win_rate)))


def probability_of_no_winner(win_rate: float, positions: int) -> float:
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    if positions < 0:
        raise ValueError("positions must be non-negative")
    return (1.0 - win_rate) ** positions


# --- allocation -------------------------------------------------------------


@dataclass(frozen=True)
class BreadthParams:
    """Position-sizing policy for a breadth book."""

    bankroll: float
    max_positions: int = 100
    # Fraction of bankroll risked per name. With ~97% of names going to zero,
    # this is the number that decides survival, not the entry signal.
    risk_per_position: float = 0.005
    # Ceiling on total capital at risk simultaneously.
    max_deployed_fraction: float = 0.50
    max_price_impact: float = 0.02
    # Below this a position is not worth the gas and the operational load.
    min_position_usd: float = 25.0
    # Reject a name whose round trip eats more than this share of the stake.
    max_round_trip_cost: float = 0.10
    gas_usd: float = DEFAULT_GAS_USD

    def __post_init__(self) -> None:
        if self.bankroll <= 0:
            raise ValueError("bankroll must be positive")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not 0.0 < self.risk_per_position <= 1.0:
            raise ValueError("risk_per_position must be in (0, 1]")
        if not 0.0 < self.max_deployed_fraction <= 1.0:
            raise ValueError("max_deployed_fraction must be in (0, 1]")


@dataclass(frozen=True)
class Candidate:
    symbol: str
    pool: Pool


@dataclass(frozen=True)
class Allocation:
    symbol: str
    size_usd: float
    round_trip_cost: float
    # Which constraint set the size: useful for knowing whether the book is
    # limited by risk policy or by the market itself.
    capped_by: str


@dataclass(frozen=True)
class BreadthBook:
    allocations: tuple[Allocation, ...]
    rejected: tuple[tuple[str, str], ...]  # (symbol, reason)

    @property
    def deployed(self) -> float:
        return sum(a.size_usd for a in self.allocations)

    @property
    def mean_cost(self) -> float:
        if not self.allocations:
            return 0.0
        weighted = sum(a.size_usd * a.round_trip_cost for a in self.allocations)
        return weighted / self.deployed if self.deployed else 0.0

    def summary(self) -> str:
        return (
            f"{len(self.allocations)} positions, ${self.deployed:,.0f} deployed, "
            f"mean round-trip cost {self.mean_cost:.2%}, "
            f"{len(self.rejected)} rejected"
        )


def allocate(candidates: Sequence[Candidate], params: BreadthParams) -> BreadthBook:
    """Size a book across candidates, letting the pool cap the position.

    Each name is sized at the tightest of three limits: the risk budget, what
    the pool can absorb inside the impact budget, and what remains of the
    portfolio cap. Names too illiquid to leave, or whose round trip is
    ruinous, are rejected with a reason rather than silently downsized to dust.
    """
    per_name_budget = params.bankroll * params.risk_per_position
    remaining = params.bankroll * params.max_deployed_fraction

    allocations: list[Allocation] = []
    rejected: list[tuple[str, str]] = []

    for candidate in candidates:
        if len(allocations) >= params.max_positions:
            rejected.append((candidate.symbol, "max_positions reached"))
            continue
        if remaining < params.min_position_usd:
            rejected.append((candidate.symbol, "portfolio cap reached"))
            continue

        depth_cap = _depth_cap(candidate.pool, params.max_price_impact)
        size = min(per_name_budget, depth_cap, remaining)
        if size < params.min_position_usd:
            rejected.append((candidate.symbol, f"depth caps size at ${size:,.2f}"))
            continue

        cost = round_trip_cost(candidate.pool, size, gas_usd=params.gas_usd)
        if cost > params.max_round_trip_cost:
            rejected.append((candidate.symbol, f"round trip costs {cost:.1%}"))
            continue

        limits = {
            "risk_budget": per_name_budget,
            "pool_depth": depth_cap,
            "portfolio_cap": remaining,
        }
        capped_by = min(limits, key=lambda name: limits[name])
        allocations.append(Allocation(candidate.symbol, size, cost, capped_by))
        remaining -= size

    return BreadthBook(tuple(allocations), tuple(rejected))


def _depth_cap(pool: Pool, max_price_impact: float) -> float:
    """Largest entry, in numeraire, whose own impact stays inside the budget.

    Bisection rather than an inversion: the impact function is monotonic in
    size, and solving it numerically keeps this correct if the pool's fee or
    curve changes.
    """
    if pool.price_impact_buy(pool.numeraire_reserve) <= max_price_impact:
        return pool.numeraire_reserve
    low, high = 0.0, pool.numeraire_reserve
    for _ in range(60):
        mid = (low + high) / 2.0
        if pool.price_impact_buy(mid) <= max_price_impact:
            low = mid
        else:
            high = mid
    return low


# --- Monte Carlo (assumption-driven, NOT a backtest) ------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Distribution of terminal bankroll over simulated rounds."""

    median: float
    mean: float
    p05: float
    p95: float
    probability_of_loss: float
    probability_of_ruin: float
    starting_bankroll: float

    @property
    def median_return(self) -> float:
        return self.median / self.starting_bankroll - 1.0


def simulate(
    params: BreadthParams,
    *,
    win_rate: float,
    cost_fraction: float,
    rounds: int,
    trials: int = 2_000,
    pareto_alpha: float = 1.2,
    min_winner_multiple: float = 2.0,
    loss_given_failure: float = 1.0,
    ruin_fraction: float = 0.2,
    seed: int = 0,
) -> SimulationResult:
    """Monte Carlo a breadth book over ``rounds`` of positions.

    **This is not a backtest.** Winner payoffs are drawn from a Pareto tail
    because that is the shape this asset class is usually claimed to have, not
    because it was measured here. ``pareto_alpha`` controls how fat that tail
    is and dominates the result: at 1.2 the mean is enormous and the median is
    poor, which is the characteristic breadth outcome and the reason median and
    mean are both reported. Treat the output as the consequence of your
    assumptions, made explicit.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if pareto_alpha <= 0:
        raise ValueError("pareto_alpha must be positive")

    rng = np.random.default_rng(seed)
    positions = params.max_positions
    stake = params.bankroll * params.risk_per_position

    terminal = np.empty(trials)
    for trial in range(trials):
        bankroll = params.bankroll
        for _ in range(rounds):
            deployable = min(
                bankroll * params.max_deployed_fraction, stake * positions
            )
            n = int(deployable // stake) if stake > 0 else 0
            if n <= 0:
                break
            wins = rng.random(n) < win_rate
            multiples = min_winner_multiple * (1.0 + rng.pareto(pareto_alpha, n))
            payoff = np.where(
                wins,
                stake * (multiples - 1.0 - cost_fraction),
                -stake * loss_given_failure,
            )
            bankroll += float(payoff.sum())
            if bankroll <= 0:
                bankroll = 0.0
                break
        terminal[trial] = bankroll

    return SimulationResult(
        median=float(np.median(terminal)),
        mean=float(terminal.mean()),
        p05=float(np.quantile(terminal, 0.05)),
        p95=float(np.quantile(terminal, 0.95)),
        probability_of_loss=float((terminal < params.bankroll).mean()),
        probability_of_ruin=float(
            (terminal < params.bankroll * ruin_fraction).mean()
        ),
        starting_bankroll=params.bankroll,
    )
