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
from ..monitoring.alerts import Notifier


@dataclass
class RiskCircuitBreaker:
    max_drawdown: float = 0.05  # from the all-time high-water mark
    rolling_window_seconds: float = 24 * 3600
    max_rolling_drawdown: float = 0.05
    clock: Callable[[], float] = time.monotonic
    # Paged on every trip (drawdown or manual). None keeps the breaker silent,
    # same as before this was added, so existing callers are unaffected.
    notifier: Notifier | None = None

    peak_equity: float | None = None
    tripped: bool = False
    trip_reason: str = ""
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
            dd = max(hwm_dd, rolling_dd)
            self._trip(oms, f"drawdown {dd:.1%} (hwm={hwm_dd:.1%}, rolling={rolling_dd:.1%})")
            return False
        return True

    def trip_manually(self, oms: OMS, reason: str) -> None:
        """Operator/external kill switch, independent of the drawdown checks.

        For a signal handler, a sentinel-file check, or a data-quality/
        reconciliation failure elsewhere in the system that should halt
        trading even though equity itself looks fine.
        """
        if self.tripped:
            return
        self._trip(oms, f"manual: {reason}")

    @staticmethod
    def _drawdown(peak: float, current: float) -> float:
        if peak is None or peak <= 0:
            return 0.0
        return max(0.0, (peak - current) / peak)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.rolling_window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _trip(self, oms: OMS, reason: str) -> None:
        self.tripped = True
        self.trip_reason = reason
        oms.cancel_all_open_orders()
        oms.flatten_all_positions()
        oms.disable_trading_loop()
        if self.notifier is not None:
            self.notifier.critical(f"circuit breaker tripped: {reason}")

    def reset(self) -> None:
        """Clear the latch after manual review; keeps the equity history."""
        self.tripped = False
        self.trip_reason = ""
