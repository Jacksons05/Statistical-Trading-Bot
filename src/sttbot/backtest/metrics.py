"""Performance statistics for a sequence of realised bets.

Deliberately conservative about what each number means:

* ROI is profit over *total staked*, not over bankroll, so it does not flatter
  a strategy that happened to bet rarely.
* Sharpe is computed on per-period returns against a fixed bankroll and
  annualised from the observed period length. Betting returns are lumpy and
  fat-tailed, so a betting Sharpe is a much weaker signal than an equity one --
  it is reported because it is comparable across strategies, not because it is
  distributionally well behaved.
* Max drawdown is peak-to-trough on the realised equity curve.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

TRADING_DAYS_PER_YEAR = 365.0


def sharpe_ratio(
    returns: Sequence[float], periods_per_year: float = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised Sharpe of ``returns`` (excess of a zero risk-free rate).

    Returns 0.0 for fewer than two observations or zero variance, rather than
    an infinity that would look like a spectacular result.
    """
    values = np.asarray(list(returns), dtype=float)
    if values.size < 2:
        return 0.0
    sd = float(values.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(values.mean() / sd * math.sqrt(periods_per_year))


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough fall, as a positive fraction of the peak.

    0.0 means the curve never fell below a previous high.
    """
    values = np.asarray(list(equity), dtype=float)
    if values.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    # Guard a non-positive peak: a ruined bankroll has no meaningful percentage.
    safe_peaks = np.where(peaks > 0, peaks, np.nan)
    drawdowns = (peaks - values) / safe_peaks
    worst = np.nanmax(drawdowns) if np.any(np.isfinite(drawdowns)) else 0.0
    return float(max(worst, 0.0))


@dataclass(frozen=True)
class BacktestMetrics:
    """Headline numbers for a completed backtest. All P&L is net of friction."""

    n_bets: int
    total_staked: float
    total_pnl: float
    roi: float
    hit_rate: float
    sharpe: float
    max_drawdown: float
    final_bankroll: float
    # CLV is only defined for bets whose match carried a closing line.
    n_bets_with_clv: int
    mean_clv: float
    clv_positive_rate: float

    def summary(self) -> str:
        lines = [
            f"bets              {self.n_bets}",
            f"staked            {self.total_staked:.2f}",
            f"P&L (net)         {self.total_pnl:+.2f}",
            f"ROI               {self.roi:+.2%}",
            f"hit rate          {self.hit_rate:.2%}",
            f"Sharpe (ann.)     {self.sharpe:.2f}",
            f"max drawdown      {self.max_drawdown:.2%}",
            f"final bankroll    {self.final_bankroll:.2f}",
        ]
        if self.n_bets_with_clv:
            lines += [
                f"CLV (mean)        {self.mean_clv:+.4f} "
                f"over {self.n_bets_with_clv} bets",
                f"CLV positive      {self.clv_positive_rate:.2%}",
            ]
        else:
            lines.append("CLV               n/a (no closing odds in this sample)")
        return "\n".join(lines)


def summarise(
    dates: Sequence[dt.date],
    pnls: Sequence[float],
    stakes: Sequence[float],
    wins: Sequence[bool],
    clvs: Sequence[float | None],
    *,
    bankroll: float,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> BacktestMetrics:
    """Roll per-bet records up into headline metrics.

    Returns are aggregated per calendar day before the Sharpe calculation:
    several bets struck on the same matchday are one observation, not many,
    because they resolve together and are not independent draws.
    """
    n = len(pnls)
    if not (len(dates) == len(stakes) == len(wins) == len(clvs) == n):
        raise ValueError("per-bet sequences must all have the same length")
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")

    total_staked = float(sum(stakes))
    total_pnl = float(sum(pnls))

    daily: dict[dt.date, float] = {}
    for date, pnl in zip(dates, pnls):
        daily[date] = daily.get(date, 0.0) + pnl
    ordered_days = sorted(daily)
    daily_pnl = [daily[d] for d in ordered_days]

    equity = [bankroll]
    for pnl in daily_pnl:
        equity.append(equity[-1] + pnl)

    returns = [pnl / bankroll for pnl in daily_pnl]

    measured = [c for c in clvs if c is not None]
    return BacktestMetrics(
        n_bets=n,
        total_staked=total_staked,
        total_pnl=total_pnl,
        roi=total_pnl / total_staked if total_staked > 0 else 0.0,
        hit_rate=(sum(1 for w in wins if w) / n) if n else 0.0,
        sharpe=sharpe_ratio(returns, periods_per_year),
        max_drawdown=max_drawdown(equity),
        final_bankroll=equity[-1],
        n_bets_with_clv=len(measured),
        mean_clv=float(np.mean(measured)) if measured else 0.0,
        clv_positive_rate=(
            sum(1 for c in measured if c > 0) / len(measured) if measured else 0.0
        ),
    )
