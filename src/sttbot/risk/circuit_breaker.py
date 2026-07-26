"""Portfolio kill-switch on drawdown.

Two drawdown measures are tracked:

* **High-water mark** drawdown — decline from the all-time peak equity.
* **Rolling-window** drawdown — decline from the highest equity seen inside a
  trailing window (default 24h), which recovers as old peaks age out.

Breaching either threshold flattens the book and disables trading via the OMS.
The clock is injectable so the rolling window is testable without real time.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from ..execution.oms import OMS


@dataclass
class RiskCircuitBreaker:
    max_drawdown: float = 0.05  # from the all-time high-water mark
    rolling_window_seconds: float = 24 * 3600
    max_rolling_drawdown: float = 0.05
    clock: Callable[[], float] = time.monotonic

    peak_equity: float | None = None
    tripped: bool = False
    _history: deque = field(default_factory=deque)

    def evaluate(self, current_equity: float, oms: OMS) -> bool:
        """Record equity and return ``True`` while trading may continue.

        Once tripped, the breaker stays latched and keeps returning ``False``
        until :meth:`reset` is called — a recovering equity curve must not
        silently re-enable trading.
        """
        if current_equity < 0:
            raise ValueError("equity cannot be negative")
        if self.tripped:
            return False

        now = self.clock()
        self._history.append((now, current_equity))
        self._evict_old(now)

        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity

        hwm_dd = self._drawdown(self.peak_equity, current_equity)
        rolling_peak = max(equity for _, equity in self._history)
        rolling_dd = self._drawdown(rolling_peak, current_equity)

        if hwm_dd >= self.max_drawdown or rolling_dd >= self.max_rolling_drawdown:
            self._trip(oms, max(hwm_dd, rolling_dd))
            return False
        return True

    @staticmethod
    def _drawdown(peak: float, current: float) -> float:
        if peak is None or peak <= 0:
            return 0.0
        return max(0.0, (peak - current) / peak)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.rolling_window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _trip(self, oms: OMS, drawdown: float) -> None:
        self.tripped = True
        oms.cancel_all_open_orders()
        oms.flatten_all_positions()
        oms.disable_trading_loop()

    def reset(self) -> None:
        """Clear the latch after manual review; keeps the equity history."""
        self.tripped = False
