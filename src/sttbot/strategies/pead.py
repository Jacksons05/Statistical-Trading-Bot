"""Micro-cap Post-Earnings Announcement Drift (PEAD).

Standardised Unexpected Earnings (SUE) measures how large an earnings surprise
is relative to its own historical noise. In micro-caps, institutional
non-coverage lets the drift persist for weeks, so a simple SUE threshold is a
durable (if low-capacity) signal.
"""

from __future__ import annotations

from .base import Param, Signal, Strategy


def standardized_unexpected_earnings(
    actual: float, expected: float, surprise_std: float
) -> float:
    """SUE = (actual - expected) / stdev(historical surprises).

    ``surprise_std`` must be positive; a zero/negative dispersion is undefined.
    """
    if surprise_std <= 0:
        raise ValueError("surprise_std must be positive")
    return (actual - expected) / surprise_std


class PostEarningsDriftStrategy(Strategy):
    """Long strong positive surprises, short strong negative ones."""

    sue_threshold = Param(1.5, low=0.5, high=3.0, kind="float")
    holding_period_days = Param(10, low=5, high=20, kind="int")
    max_position_size_pct = Param(0.05, low=0.01, high=0.10, kind="float")

    def generate_signal(self, current_sue: float) -> Signal:
        if current_sue >= self.sue_threshold:
            return Signal.LONG
        if current_sue <= -self.sue_threshold:
            return Signal.SHORT
        return Signal.FLAT

    def target_weight(self, current_sue: float) -> float:
        """Signed target portfolio weight for the position."""
        return int(self.generate_signal(current_sue)) * self.max_position_size_pct
