"""Dixon-Coles bivariate-Poisson model for low-scoring team sports.

The independent double-Poisson model misprices low-scoring outcomes (0-0, 1-0,
0-1, 1-1) because goals early in a match are not independent. Dixon & Coles
(1997) correct the joint mass function with a dependence parameter ``rho``:

    P(X=x, Y=y) = tau(x, y; lam, mu, rho) * Poisson(x; lam) * Poisson(y; mu)

where the correction ``tau`` is 1 everywhere except the four lowest score
cells. Team scoring rates come from attack/defence strengths plus a home
advantage:

    lam = exp(attack_home + defence_away + gamma)   # home expected goals
    mu  = exp(attack_away + defence_home)           # away expected goals
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score dependence correction from Dixon & Coles (1997)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: np.ndarray, rate: float) -> np.ndarray:
    """Poisson PMF over an array of non-negative integer counts."""
    return np.exp(-rate) * rate**k / np.array([math.factorial(int(i)) for i in k])


def score_matrix(
    lam: float, mu: float, rho: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Joint probability matrix ``M[x, y] = P(home=x, away=y)``.

    The matrix is truncated at ``max_goals`` and renormalised so it sums to 1,
    absorbing the small tail beyond the truncation point.
    """
    if lam <= 0 or mu <= 0:
        raise ValueError("expected goals lam and mu must be positive")
    goals = np.arange(max_goals + 1)
    home = _poisson_pmf(goals, lam)
    away = _poisson_pmf(goals, mu)
    matrix = np.outer(home, away)
    # Apply the tau correction to the four low-score cells only.
    for x in (0, 1):
        for y in (0, 1):
            matrix[x, y] *= dixon_coles_tau(x, y, lam, mu, rho)
    total = matrix.sum()
    if total <= 0:  # pragma: no cover - defensive
        raise ValueError("degenerate score matrix")
    return matrix / total


@dataclass(frozen=True)
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.home_win, self.draw, self.away_win)


def outcome_probabilities(matrix: np.ndarray) -> MatchProbabilities:
    """Collapse a score matrix into 1X2 (home / draw / away) probabilities."""
    home_win = float(np.tril(matrix, -1).sum())  # home goals > away goals
    away_win = float(np.triu(matrix, 1).sum())  # away goals > home goals
    draw = float(np.trace(matrix))
    return MatchProbabilities(home_win, draw, away_win)


@dataclass
class TeamRating:
    attack: float
    defence: float


@dataclass
class DixonColesModel:
    """Forward model mapping team ratings to match probabilities.

    Ratings are typically fit by weighted maximum likelihood with an
    exponential time decay ``xi`` on past matches; this class holds the fitted
    ratings and evaluates the resulting distribution.
    """

    ratings: dict[str, TeamRating] = field(default_factory=dict)
    home_advantage: float = 0.25  # gamma
    rho: float = -0.05
    max_goals: int = 10

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if home not in self.ratings:
            raise KeyError(f"no rating for home team {home!r}")
        if away not in self.ratings:
            raise KeyError(f"no rating for away team {away!r}")
        h, a = self.ratings[home], self.ratings[away]
        lam = math.exp(h.attack + a.defence + self.home_advantage)
        mu = math.exp(a.attack + h.defence)
        return lam, mu

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        return score_matrix(lam, mu, self.rho, self.max_goals)

    def match_probabilities(self, home: str, away: str) -> MatchProbabilities:
        return outcome_probabilities(self.score_matrix(home, away))


def implied_probability(decimal_odds: float) -> float:
    """Bookmaker-implied probability (including the vig) from decimal odds."""
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    return 1.0 / decimal_odds


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """Expected profit per 1 unit staked at ``decimal_odds``.

    Positive means the model thinks the bet is +EV. A win returns
    ``decimal_odds - 1`` profit; a loss costs the stake.
    """
    if not 0.0 <= model_prob <= 1.0:
        raise ValueError("model_prob must be in [0, 1]")
    return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)


def closing_line_value(entry_odds: float, closing_odds: float) -> float:
    """CLV as the edge captured versus the (sharper) closing line.

    Positive CLV means we bet at better odds than the market closed at, the
    strongest available proxy for durable +EV.
    """
    return implied_probability(closing_odds) - implied_probability(entry_odds)
