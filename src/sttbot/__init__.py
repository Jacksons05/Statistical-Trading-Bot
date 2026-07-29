"""sttbot — a modular systematic-trading ecosystem for the solo operator.

The package is organised around the pipeline described in the feasibility
study: data ingestion -> strategy/analysis -> orchestration/execution ->
operations/risk. Each layer is a small, independently testable module.
"""

from __future__ import annotations

from .backtest.clv import clv_baseline, clv_skill
from .backtest.metrics import BacktestMetrics, max_drawdown, sharpe_ratio
from .backtest.walk_forward import Bet, BacktestResult, walk_forward_backtest
from .economics.friction import FrictionModel, should_use_maker
from .execution.oms import OMS, PaperBroker
from .execution.order_manager import DynamicOrderManager, ExecutionResult
from .risk.circuit_breaker import RiskCircuitBreaker
from .strategies.amm import (
    ArbTrade,
    Pool,
    impermanent_loss,
    no_arb_band,
    optimal_arbitrage,
)
from .strategies.base import Param, Signal, Strategy
from .strategies.dixon_coles import DixonColesModel, TeamRating, expected_value
from .strategies.dixon_coles_fit import FitResult, fit_dixon_coles, time_decay_weights
from .strategies.pead import PostEarningsDriftStrategy
from .strategies.prob_arbitrage import Outcome, find_boundary_arbitrage
from .strategies.token_screen import (
    ScreenResult,
    ScreenThresholds,
    TokenMetadata,
    max_exit_size,
    position_limit,
    screen,
)
from .venues.prediction import (
    KALSHI_FEES,
    POLYMARKET_FEES,
    CrossVenueArb,
    FeeModel,
    VenueQuote,
    find_cross_venue_arb,
    implied_from_book,
    kelly_fraction,
)

__version__ = "0.1.0"

__all__ = [
    "FrictionModel",
    "should_use_maker",
    "OMS",
    "PaperBroker",
    "DynamicOrderManager",
    "ExecutionResult",
    "RiskCircuitBreaker",
    "Param",
    "Signal",
    "Strategy",
    "DixonColesModel",
    "TeamRating",
    "expected_value",
    "FitResult",
    "fit_dixon_coles",
    "time_decay_weights",
    "walk_forward_backtest",
    "BacktestResult",
    "BacktestMetrics",
    "Bet",
    "sharpe_ratio",
    "max_drawdown",
    "clv_baseline",
    "clv_skill",
    "PostEarningsDriftStrategy",
    "Outcome",
    "find_boundary_arbitrage",
    "Pool",
    "ArbTrade",
    "optimal_arbitrage",
    "no_arb_band",
    "impermanent_loss",
    "TokenMetadata",
    "ScreenThresholds",
    "ScreenResult",
    "screen",
    "max_exit_size",
    "position_limit",
    "VenueQuote",
    "FeeModel",
    "CrossVenueArb",
    "KALSHI_FEES",
    "POLYMARKET_FEES",
    "find_cross_venue_arb",
    "kelly_fraction",
    "implied_from_book",
    "__version__",
]
