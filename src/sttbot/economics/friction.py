"""Transaction-cost economics: turning gross edge into net edge.

For a small-capital operator, gross mispricing is easily eaten by fees,
slippage, gas, and the bid-ask spread (the "vig"). Every signal must be
evaluated *after* friction, and execution should prefer maker orders whenever
the spread would consume too much of the edge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class FrictionModel:
    """Per-trade cost assumptions, all expressed in the same units as edge
    (e.g. fraction of notional)."""

    taker_fee: float = 0.0
    maker_fee: float = 0.0  # can be negative for a rebate
    slippage: float = 0.0
    gas: float = 0.0

    def net_edge(self, gross_mispricing: float, spread: float, *, maker: bool) -> float:
        """Net edge = gross - fees - (slippage + gas) - vig.

        A maker order captures (does not pay) the spread and earns the maker
        fee/rebate; a taker order crosses the full spread and pays taker fees.
        """
        fee = self.maker_fee if maker else self.taker_fee
        vig = 0.0 if maker else spread
        slip = 0.0 if maker else self.slippage
        return gross_mispricing - fee - slip - self.gas - vig


def should_use_maker(
    gross_edge: float, spread: float, max_spread_fraction: float = 0.30
) -> bool:
    """Suppress market (taker) orders when the spread would eat too much edge.

    Returns ``True`` (use a passive maker order) when the spread exceeds
    ``max_spread_fraction`` of the signal's gross edge. A non-positive edge can
    never justify crossing the spread.
    """
    if gross_edge <= 0:
        return True
    return spread > max_spread_fraction * gross_edge


def stealth_size(
    target_size: float, jitter: float = 0.03, rng: random.Random | None = None
) -> float:
    """Perturb an order size with Gaussian noise to avoid round-number
    fingerprints (e.g. 243 instead of 250).

    ``jitter`` is the relative standard deviation. The result is clamped to a
    small positive minimum so noise can never flip the sign or zero the order.
    """
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    rng = rng or random.Random()
    noisy = target_size * (1.0 + rng.gauss(0.0, jitter))
    return max(noisy, target_size * 0.5)


def stealth_delay(
    base_seconds: float = 0.0,
    max_extra_seconds: float = 1.5,
    rng: random.Random | None = None,
) -> float:
    """A randomised, non-negative execution delay to break periodic timing
    fingerprints."""
    if max_extra_seconds < 0:
        raise ValueError("max_extra_seconds must be non-negative")
    rng = rng or random.Random()
    return base_seconds + rng.uniform(0.0, max_extra_seconds)
