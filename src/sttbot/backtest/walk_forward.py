"""Walk-forward backtest for the Dixon-Coles betting strategy.

The whole point of this module is to make look-ahead bias structurally
impossible rather than merely intended. Every prediction for matchday ``d`` is
made from a model fitted only on matches that finished strictly before ``d``,
and the training window is re-derived from ``d`` each time. Nothing in the
loop can see a result it is about to bet on.

The three ways a football backtest usually lies to you, and what is done here:

1. *Look-ahead through the fit.* Handled by the strict ``date < d`` cut above.
2. *Betting a price nobody could have taken.* We bet the pre-match line and
   settle at that same price. Best-of-book maxima are available but are peak
   prices across ~17 books; using them assumes simultaneous availability and
   non-binding limits, so single-book prices are the default.
3. *Ignoring friction.* Every edge is charged the same per-unit cost that is
   later subtracted from realised P&L, so the entry rule and the accounting can
   never disagree. See :func:`_cost_per_unit`.

Promoted teams have no rating on their first appearance in a window; those
matches are skipped and counted in ``n_skipped_unrated`` rather than silently
dropped.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from ..economics.friction import FrictionModel
from ..strategies.dixon_coles import (
    MatchProbabilities,
    closing_line_value,
    expected_value,
)
from ..strategies.dixon_coles_fit import DEFAULT_XI, fit_dixon_coles
from .metrics import BacktestMetrics, summarise

SELECTIONS = ("H", "D", "A")


class BacktestMatch(Protocol):
    """What the backtester needs from a match record."""

    date: dt.date
    home: str
    away: str
    home_goals: int
    away_goals: int
    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    max_home: float | None
    max_draw: float | None
    max_away: float | None
    close_home: float | None
    close_draw: float | None
    close_away: float | None


OddsSelector = Callable[[BacktestMatch], tuple[float | None, float | None, float | None]]


def single_book_odds(match: BacktestMatch) -> tuple[float | None, ...]:
    """Bet the single-book pre-match line. The conservative, realistic default."""
    return (match.odds_home, match.odds_draw, match.odds_away)


def best_of_book_odds(match: BacktestMatch) -> tuple[float | None, ...]:
    """Bet the best price across books.

    Optimistic: these maxima assume an account at whichever book was best,
    simultaneous availability, and stake limits that do not bind. Treat any
    result produced with this selector as an upper bound.
    """
    return (match.max_home, match.max_draw, match.max_away)


@dataclass(frozen=True)
class Bet:
    """One placed bet and its realised outcome, net of friction."""

    date: dt.date
    home: str
    away: str
    selection: str
    odds: float
    model_prob: float
    gross_edge: float
    net_edge: float
    stake: float
    pnl: float
    won: bool
    clv: float | None

    @property
    def fixture(self) -> str:
        return f"{self.home} v {self.away}"


@dataclass(frozen=True)
class BacktestResult:
    bets: list[Bet]
    metrics: BacktestMetrics
    n_matches_considered: int
    n_skipped_unrated: int
    n_fits: int
    train_sizes: list[int]

    def summary(self) -> str:
        head = (
            f"matches considered {self.n_matches_considered}  "
            f"skipped (unrated team) {self.n_skipped_unrated}  "
            f"refits {self.n_fits}"
        )
        return head + "\n" + self.metrics.summary()


def _cost_per_unit(friction: FrictionModel) -> float:
    """Friction charged per unit staked, in the same units as edge.

    Computed from :class:`FrictionModel` so the entry threshold and the P&L
    accounting cannot drift apart: ``net_edge`` is exactly the expectation of
    the realised P&L per unit staked.

    ``spread`` is deliberately passed as zero. The bookmaker's margin is
    already inside the quoted decimal odds, so an edge computed from those odds
    has paid the vig once; charging it again here would double-count it.
    """
    return -friction.net_edge(0.0, spread=0.0, maker=False)


def _probability_for(probs: MatchProbabilities, selection: str) -> float:
    return {
        "H": probs.home_win,
        "D": probs.draw,
        "A": probs.away_win,
    }[selection]


def _closing_odds_for(match: BacktestMatch, selection: str) -> float | None:
    return {
        "H": match.close_home,
        "D": match.close_draw,
        "A": match.close_away,
    }[selection]


def _result_of(match: BacktestMatch) -> str:
    if match.home_goals > match.away_goals:
        return "H"
    if match.home_goals < match.away_goals:
        return "A"
    return "D"


def _group_by_date(
    matches: Iterable[BacktestMatch],
) -> list[tuple[dt.date, list[BacktestMatch]]]:
    grouped: dict[dt.date, list[BacktestMatch]] = {}
    for match in matches:
        grouped.setdefault(match.date, []).append(match)
    return sorted(grouped.items())


def walk_forward_backtest(
    matches: Sequence[BacktestMatch],
    *,
    train_window_days: int = 730,
    min_train_matches: int = 200,
    refit_every_days: int = 7,
    xi: float = DEFAULT_XI,
    friction: FrictionModel | None = None,
    min_net_edge: float = 0.02,
    stake: float = 1.0,
    bankroll: float = 100.0,
    odds_selector: OddsSelector = single_book_odds,
    max_goals: int = 10,
    start: dt.date | None = None,
) -> BacktestResult:
    """Fit on a trailing window, bet the next matchday, never peek forward.

    ``min_net_edge`` is the after-friction edge required to place a bet, so a
    signal that only clears the threshold before costs is not taken. Staking is
    flat by default; flat staking keeps the P&L series interpretable and does
    not let a Kelly sizing rule quietly manufacture most of the return.

    ``start`` skips the burn-in period, which is useful when the sample begins
    part-way through a season.
    """
    friction = friction or FrictionModel()
    cost = _cost_per_unit(friction)
    if stake <= 0:
        raise ValueError("stake must be positive")
    if train_window_days <= 0:
        raise ValueError("train_window_days must be positive")

    ordered = sorted(matches, key=lambda m: m.date)
    batches = _group_by_date(ordered)

    bets: list[Bet] = []
    considered = skipped = n_fits = 0
    train_sizes: list[int] = []

    model = None
    last_fit_date: dt.date | None = None

    for predict_date, batch in batches:
        if start is not None and predict_date < start:
            continue

        # Strictly-before cut: this is what makes the backtest honest.
        window_start = predict_date - dt.timedelta(days=train_window_days)
        train = [m for m in ordered if window_start <= m.date < predict_date]
        if len(train) < min_train_matches:
            continue

        stale = (
            last_fit_date is None
            or (predict_date - last_fit_date).days >= refit_every_days
        )
        if stale or model is None:
            fit = fit_dixon_coles(train, reference_date=predict_date, xi=xi)
            model = fit.to_model(max_goals=max_goals)
            last_fit_date = predict_date
            n_fits += 1
            train_sizes.append(len(train))

        for match in batch:
            considered += 1
            if match.home not in model.ratings or match.away not in model.ratings:
                skipped += 1  # promoted/new team, cannot be priced
                continue

            prices = odds_selector(match)
            probs = model.match_probabilities(match.home, match.away)
            outcome = _result_of(match)

            for selection, price in zip(SELECTIONS, prices):
                if price is None or price <= 1.0:
                    continue
                prob = _probability_for(probs, selection)
                gross = expected_value(prob, price)
                net = friction.net_edge(gross, spread=0.0, maker=False)
                if net < min_net_edge:
                    continue

                won = selection == outcome
                pnl = stake * ((price - 1.0) if won else -1.0) - stake * cost
                closing = _closing_odds_for(match, selection)
                bets.append(
                    Bet(
                        date=match.date,
                        home=match.home,
                        away=match.away,
                        selection=selection,
                        odds=price,
                        model_prob=prob,
                        gross_edge=gross,
                        net_edge=net,
                        stake=stake,
                        pnl=pnl,
                        won=won,
                        clv=(
                            closing_line_value(price, closing)
                            if closing is not None and closing > 1.0
                            else None
                        ),
                    )
                )

    metrics = summarise(
        [b.date for b in bets],
        [b.pnl for b in bets],
        [b.stake for b in bets],
        [b.won for b in bets],
        [b.clv for b in bets],
        bankroll=bankroll,
    )
    return BacktestResult(
        bets=bets,
        metrics=metrics,
        n_matches_considered=considered,
        n_skipped_unrated=skipped,
        n_fits=n_fits,
        train_sizes=train_sizes,
    )
