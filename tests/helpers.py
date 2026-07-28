"""Shared fixtures for the backtest tests.

Lives in its own module so test files can share it without importing each
other: pytest puts this directory on ``sys.path``, but the repository root is
not guaranteed to be there, so ``from tests.x import ...`` is fragile.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class FakeMatch:
    """Minimal record satisfying the ``BacktestMatch`` protocol."""

    date: dt.date
    home: str
    away: str
    home_goals: int
    away_goals: int
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    max_home: float | None = None
    max_draw: float | None = None
    max_away: float | None = None
    close_home: float | None = None
    close_draw: float | None = None
    close_away: float | None = None
