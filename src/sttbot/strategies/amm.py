"""Constant-product AMM mechanics and CEX-DEX arbitrage sizing.

A Uniswap-v2-style pool holds reserves ``(x, y)`` of a token and a numeraire and
enforces ``x * y = k`` on every swap. Its quoted price only moves when someone
trades against it, so when an external venue re-prices, the pool is stale until
an arbitrageur corrects it. That correction is the trade this module sizes.

The important part is *how much* to trade. Trading until the pool price equals
the external price overshoots: the marginal unit earns nothing while still
paying fees and slippage. The profit-maximising size stops where the pool's
marginal price reaches the external price, which for a fee factor
``gamma = 1 - fee`` is

    numeraire in:  dy* = (sqrt(p_ext * x * y * gamma) - y) / gamma
    token in:      dx* = (sqrt(x * y * gamma / p_ext) - x) / gamma

Both are derived by maximising profit directly (set ``dP/d(size) = 0``), not by
solving for price equality — see the tests, which check the analytic optimum
against a brute-force search.

Conventions: ``x`` is the token reserve, ``y`` the numeraire reserve, and every
price is numeraire per token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pool:
    """A constant-product pool.

    ``fee`` is the proportional swap fee taken on the input amount (0.003 for
    the common 30 bp tier).
    """

    token_reserve: float  # x
    numeraire_reserve: float  # y
    fee: float = 0.003

    def __post_init__(self) -> None:
        if self.token_reserve <= 0 or self.numeraire_reserve <= 0:
            raise ValueError("pool reserves must be positive")
        if not 0.0 <= self.fee < 1.0:
            raise ValueError("fee must be in [0, 1)")

    @property
    def gamma(self) -> float:
        """Fraction of an input that reaches the invariant after fees."""
        return 1.0 - self.fee

    @property
    def k(self) -> float:
        return self.token_reserve * self.numeraire_reserve

    @property
    def spot_price(self) -> float:
        """Marginal price ignoring fees and depth — the pool's quoted price."""
        return self.numeraire_reserve / self.token_reserve

    def buy_token(self, numeraire_in: float) -> float:
        """Tokens received for spending ``numeraire_in``."""
        if numeraire_in < 0:
            raise ValueError("numeraire_in must be non-negative")
        effective = self.gamma * numeraire_in
        return self.token_reserve * effective / (self.numeraire_reserve + effective)

    def sell_token(self, token_in: float) -> float:
        """Numeraire received for selling ``token_in``."""
        if token_in < 0:
            raise ValueError("token_in must be non-negative")
        effective = self.gamma * token_in
        return self.numeraire_reserve * effective / (self.token_reserve + effective)

    def execution_price_buy(self, numeraire_in: float) -> float:
        """Average price paid per token when buying — always worse than spot."""
        out = self.buy_token(numeraire_in)
        if out <= 0:
            raise ValueError("trade too small to return any token")
        return numeraire_in / out

    def price_impact_buy(self, numeraire_in: float) -> float:
        """Fractional gap between the average fill and the pre-trade spot."""
        return self.execution_price_buy(numeraire_in) / self.spot_price - 1.0

    def with_swap(self, numeraire_in: float = 0.0, token_in: float = 0.0) -> "Pool":
        """The pool after a swap, for simulating a sequence of trades."""
        if numeraire_in and token_in:
            raise ValueError("swap one direction at a time")
        if numeraire_in:
            return Pool(
                self.token_reserve - self.buy_token(numeraire_in),
                self.numeraire_reserve + numeraire_in,
                self.fee,
            )
        if token_in:
            return Pool(
                self.token_reserve + token_in,
                self.numeraire_reserve - self.sell_token(token_in),
                self.fee,
            )
        return self


@dataclass(frozen=True)
class ArbTrade:
    """A sized CEX-DEX arbitrage, already net of fees and gas."""

    direction: str  # "buy_dex" (pool cheap) | "sell_dex" (pool expensive)
    size_in: float  # numeraire in for buy_dex, tokens in for sell_dex
    size_out: float
    gross_profit: float  # numeraire, before gas
    net_profit: float  # numeraire, after gas
    pool_price_before: float
    pool_price_after: float

    @property
    def profitable(self) -> bool:
        return self.net_profit > 0


def optimal_arbitrage(
    pool: Pool,
    external_price: float,
    *,
    gas_cost: float = 0.0,
    max_capital: float | None = None,
) -> ArbTrade | None:
    """Size the profit-maximising arbitrage between ``pool`` and an external venue.

    ``external_price`` is numeraire per token on the reference venue (assumed
    deep enough to fill without moving). ``gas_cost`` is a fixed numeraire cost
    charged once. ``max_capital`` caps the input leg, which binds often for a
    small operator and makes the trade smaller — never larger — than optimal.

    Returns ``None`` when no direction has a positive-size optimum. A trade that
    is optimally sized but loses to gas is still returned, with
    ``profitable == False``, so the caller can see how close it came.
    """
    if external_price <= 0:
        raise ValueError("external_price must be positive")
    if gas_cost < 0:
        raise ValueError("gas_cost must be non-negative")
    if max_capital is not None and max_capital <= 0:
        raise ValueError("max_capital must be positive")

    x, y, gamma = pool.token_reserve, pool.numeraire_reserve, pool.gamma

    # Pool cheap relative to the external venue: buy tokens here, sell there.
    if external_price * gamma > pool.spot_price:
        size_in = (math.sqrt(external_price * x * y * gamma) - y) / gamma
        if max_capital is not None:
            size_in = min(size_in, max_capital)
        if size_in <= 0:
            return None
        tokens_out = pool.buy_token(size_in)
        gross = tokens_out * external_price - size_in
        after = pool.with_swap(numeraire_in=size_in)
        return ArbTrade(
            "buy_dex", size_in, tokens_out, gross, gross - gas_cost,
            pool.spot_price, after.spot_price,
        )

    # Pool expensive: buy tokens externally, sell them into the pool.
    if pool.spot_price * gamma > external_price:
        size_in = (math.sqrt(x * y * gamma / external_price) - x) / gamma
        if max_capital is not None:
            # max_capital is numeraire, and the token leg is bought externally.
            size_in = min(size_in, max_capital / external_price)
        if size_in <= 0:
            return None
        numeraire_out = pool.sell_token(size_in)
        gross = numeraire_out - size_in * external_price
        after = pool.with_swap(token_in=size_in)
        return ArbTrade(
            "sell_dex", size_in, numeraire_out, gross, gross - gas_cost,
            pool.spot_price, after.spot_price,
        )

    # Divergence is inside the fee band — no trade can profit at any size.
    return None


def no_arb_band(pool: Pool) -> tuple[float, float]:
    """External prices within which no arbitrage exists at any size.

    Fees make the pool's effective quote two-sided: it buys at
    ``spot * gamma`` and sells at ``spot / gamma``. Divergence inside this band
    is not an opportunity, however large it looks against mid.
    """
    return (pool.spot_price * pool.gamma, pool.spot_price / pool.gamma)


def impermanent_loss(price_ratio: float) -> float:
    """LP loss vs. holding, as a negative fraction, for a price move ``r``.

    ``2*sqrt(r)/(1+r) - 1``. Symmetric in ``r`` and ``1/r``: a halving and a
    doubling cost an LP the same 5.72%. Excludes fee income, which is what
    makes LPing viable in practice.
    """
    if price_ratio <= 0:
        raise ValueError("price_ratio must be positive")
    return 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0
