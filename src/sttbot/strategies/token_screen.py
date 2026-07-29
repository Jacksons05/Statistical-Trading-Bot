"""Pre-trade risk screening for low-cap / meme tokens.

For thin tokens the dominant loss mode is not a bad entry — it is being unable
to exit. Liquidity gets pulled, the deployer dumps a majority supply, or the
contract simply refuses to let you sell. Those losses are total and no stop-loss
protects against them, so the screen runs *before* any signal is considered.

Two distinct questions are answered here:

* **Can I get out at all?** — rug and honeypot heuristics (:func:`screen`).
* **How much can I get out with?** — exit sizing against real pool depth
  (:func:`max_exit_size`), which is usually the binding constraint and is
  routinely the number people skip.

These are heuristics over self-reported on-chain metadata. They filter obvious
traps; they are not a safety guarantee. A token can pass every check here and
still go to zero — indeed most will. Nothing in this module estimates whether a
token will go *up*.

Base rates (published figures, 2025-2026) — why the defaults are strict:

* ~97% of meme coins die or lose meaningful volume, average lifespan ~1 year
  (Binance Research).
* ~69% of pump.fun tokens record their last trade on the *same day* they
  launch, so a 24h minimum age filters most of the distribution at almost no
  opportunity cost.
* Graduation to a DEX listing ran below 2% through 2025, falling to ~0.26% by
  mid-2026.
* Rug pulls averaged ~$510k each, >$2.8bn total across 2025 (Chainalysis).

The strategic consequence: against a base rate that skewed, nearly all the
achievable edge is in *avoidance* rather than selection. A screen that rejects
almost everything is working as intended — the cost of a false reject (a missed
winner) is far smaller than a false accept (a total loss). Loosening thresholds
to surface more candidates is usually a mistake.

Threshold provenance: the top-10 concentration limit reflects the ~30% level
the rug-pull detection literature converges on; the LP-lock, single-holder and
sniper/bundled-wallet checks follow the same body of work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .amm import Pool


@dataclass(frozen=True)
class TokenMetadata:
    """On-chain facts about a token, as reported by an indexer or RPC node.

    Every field is optional because indexers disagree on coverage. ``None``
    means *unknown*, which the screen treats as a failure rather than a pass —
    absence of evidence is not evidence of safety for an asset like this.
    """

    symbol: str
    liquidity_usd: float | None = None
    # Fraction of LP tokens burned or held by a time-locked contract.
    lp_locked_fraction: float | None = None
    # Largest non-pool, non-burn holder's share of supply.
    top_holder_fraction: float | None = None
    # Combined share of the top 10 holders. Distinct from the single-holder
    # measure and the one the rug-pull literature actually keys on: supply
    # split across ten coordinated wallets is the same exit risk as one whale,
    # and passes a top-1 check comfortably.
    top10_holder_fraction: float | None = None
    # Share acquired by snipers/bundled wallets in the launch block(s). High
    # values mean the float is held by actors positioned to exit into the first
    # real bid, regardless of how the supply looks spread afterwards.
    sniper_bundled_fraction: float | None = None
    holder_count: int | None = None
    age_hours: float | None = None
    # Transfer taxes, as fractions. A large asymmetry is the classic honeypot.
    buy_tax: float | None = None
    sell_tax: float | None = None
    # Contract permissions that let an owner change the rules after you buy.
    can_mint: bool | None = None
    can_freeze: bool | None = None
    ownership_renounced: bool | None = None
    # Whether a simulated sell actually succeeded against a forked node.
    sell_simulation_passed: bool | None = None


@dataclass(frozen=True)
class ScreenThresholds:
    """Tunable limits. Defaults are deliberately strict."""

    min_liquidity_usd: float = 25_000.0
    min_lp_locked_fraction: float = 0.90
    max_top_holder_fraction: float = 0.15
    # 30% across the top 10 is the threshold the rug-pull literature converges
    # on as the point where coordinated exit becomes the dominant risk.
    max_top10_holder_fraction: float = 0.30
    max_sniper_bundled_fraction: float = 0.15
    min_holder_count: int = 200
    # ~69% of pump.fun tokens never trade again after their launch day, so a
    # minimum age filters the bulk of the distribution at essentially no cost.
    min_age_hours: float = 24.0
    max_sell_tax: float = 0.10
    max_tax_asymmetry: float = 0.05  # sell_tax - buy_tax


@dataclass
class ScreenResult:
    symbol: str
    failures: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures and not self.unknowns

    def summary(self) -> str:
        if self.passed:
            return f"{self.symbol}: pass"
        parts = []
        if self.failures:
            parts.append("fails " + "; ".join(self.failures))
        if self.unknowns:
            parts.append("unknown " + ", ".join(self.unknowns))
        return f"{self.symbol}: " + " | ".join(parts)


def screen(
    token: TokenMetadata, thresholds: ScreenThresholds | None = None
) -> ScreenResult:
    """Evaluate rug and honeypot heuristics.

    Failures and unknowns are reported separately: a failure means a check ran
    and the token lost, an unknown means the data was missing. Both block a
    pass, but they call for different responses — one is a verdict, the other a
    gap in your data pipeline.
    """
    limits = thresholds or ScreenThresholds()
    result = ScreenResult(symbol=token.symbol)

    def check(name: str, value, ok, detail) -> None:
        """Record a check. ``ok`` and ``detail`` are callables so neither is
        evaluated when the value is missing — and so a legitimate 0.0 is never
        confused with absent, which a ``value or default`` idiom would do."""
        if value is None:
            result.unknowns.append(name)
        elif not ok(value):
            result.failures.append(detail(value))

    check(
        "liquidity_usd",
        token.liquidity_usd,
        lambda v: v >= limits.min_liquidity_usd,
        lambda v: f"liquidity ${v:,.0f} < ${limits.min_liquidity_usd:,.0f}",
    )
    check(
        "lp_locked_fraction",
        token.lp_locked_fraction,
        lambda v: v >= limits.min_lp_locked_fraction,
        lambda v: f"only {v:.0%} of LP locked/burned "
        f"(need {limits.min_lp_locked_fraction:.0%}) — liquidity can be pulled",
    )
    check(
        "top_holder_fraction",
        token.top_holder_fraction,
        lambda v: v <= limits.max_top_holder_fraction,
        lambda v: f"top holder owns {v:.0%} "
        f"(max {limits.max_top_holder_fraction:.0%}) — single-wallet dump risk",
    )
    check(
        "top10_holder_fraction",
        token.top10_holder_fraction,
        lambda v: v <= limits.max_top10_holder_fraction,
        lambda v: f"top 10 holders own {v:.0%} "
        f"(max {limits.max_top10_holder_fraction:.0%}) — coordinated exit risk",
    )
    check(
        "sniper_bundled_fraction",
        token.sniper_bundled_fraction,
        lambda v: v <= limits.max_sniper_bundled_fraction,
        lambda v: f"snipers/bundled wallets hold {v:.0%} "
        f"(max {limits.max_sniper_bundled_fraction:.0%}) — float is pre-positioned to exit",
    )
    check(
        "holder_count",
        token.holder_count,
        lambda v: v >= limits.min_holder_count,
        lambda v: f"only {v} holders (need {limits.min_holder_count})",
    )
    check(
        "age_hours",
        token.age_hours,
        lambda v: v >= limits.min_age_hours,
        lambda v: f"only {v}h old (need {limits.min_age_hours}h)",
    )
    check(
        "sell_tax",
        token.sell_tax,
        lambda v: v <= limits.max_sell_tax,
        lambda v: f"sell tax {v:.1%} > {limits.max_sell_tax:.0%}",
    )
    if token.sell_tax is not None and token.buy_tax is not None:
        asymmetry = token.sell_tax - token.buy_tax
        if asymmetry > limits.max_tax_asymmetry:
            result.failures.append(
                f"sell tax exceeds buy tax by {asymmetry:.1%} — honeypot signature"
            )
    elif token.buy_tax is None:
        result.unknowns.append("buy_tax")

    check("can_mint", token.can_mint, lambda v: v is False,
          lambda v: "mint authority live — supply can be inflated after you buy")
    check("can_freeze", token.can_freeze, lambda v: v is False,
          lambda v: "freeze authority live — your balance can be frozen")
    check("ownership_renounced", token.ownership_renounced,
          lambda v: v is True,
          lambda v: "ownership not renounced — rules can change after you buy")
    check("sell_simulation_passed", token.sell_simulation_passed,
          lambda v: v is True,
          lambda v: "simulated sell FAILED — cannot exit this position")

    return result


def max_exit_size(pool: Pool, max_price_impact: float = 0.02) -> float:
    """Largest token position exitable within a price-impact budget.

    Solves the constant-product sell for the size whose *average* fill sits
    ``max_price_impact`` below spot:

        dx = x * (gamma - (1 - impact)) / (gamma * (1 - impact))

    A position larger than this cannot be unwound at the assumed price no matter
    how patient you are — the depth simply is not there. Returns 0.0 when the
    fee alone already exceeds the impact budget.
    """
    if not 0.0 < max_price_impact < 1.0:
        raise ValueError("max_price_impact must be in (0, 1)")
    gamma, remaining = pool.gamma, 1.0 - max_price_impact
    numerator = gamma - remaining
    if numerator <= 0:
        # Every trade costs at least the fee; the budget cannot be met.
        return 0.0
    return pool.token_reserve * numerator / (gamma * remaining)


def position_limit(
    pool: Pool,
    equity: float,
    *,
    max_price_impact: float = 0.02,
    max_equity_fraction: float = 0.02,
) -> float:
    """Position cap in numeraire: the tighter of exit depth and risk budget.

    Thin-token sizing has two independent ceilings, and taking the smaller is
    the whole point — a position inside your risk budget is still untradeable
    if the pool cannot absorb the exit.
    """
    if equity <= 0:
        raise ValueError("equity must be positive")
    if not 0.0 < max_equity_fraction <= 1.0:
        raise ValueError("max_equity_fraction must be in (0, 1]")
    depth_cap = max_exit_size(pool, max_price_impact) * pool.spot_price
    return min(depth_cap, equity * max_equity_fraction)
