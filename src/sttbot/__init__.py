"""sttbot — a modular systematic-trading ecosystem for the solo operator.

The package is organised around the pipeline described in the feasibility
study: data ingestion -> strategy/analysis -> orchestration/execution ->
operations/risk. Each layer is a small, independently testable module.
"""

from __future__ import annotations

from .economics.friction import FrictionModel, should_use_maker
from .execution.oms import OMS, PaperBroker
from .execution.order_manager import DynamicOrderManager, ExecutionResult
from .risk.circuit_breaker import RiskCircuitBreaker
from .strategies.base import Param, Signal, Strategy
from .strategies.dixon_coles import DixonColesModel, TeamRating, expected_value
from .strategies.pead import PostEarningsDriftStrategy
from .strategies.prob_arbitrage import Outcome, find_boundary_arbitrage

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
    "PostEarningsDriftStrategy",
    "Outcome",
    "find_boundary_arbitrage",
    "__version__",
]
