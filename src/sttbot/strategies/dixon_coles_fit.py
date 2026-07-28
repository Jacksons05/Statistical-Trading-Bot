"""Weighted maximum-likelihood fitting for the Dixon-Coles model.

The forward model in :mod:`sttbot.strategies.dixon_coles` maps team ratings to
match probabilities but has no way to *learn* those ratings, which makes it
unusable on real data. This module estimates them.

We maximise the time-decay-weighted log-likelihood of Dixon & Coles (1997):

    ll = sum_k w_k * log P(X=x_k, Y=y_k ; lam_k, mu_k, rho)

with the same parametrisation the forward model already uses,

    lam = exp(attack_home + defence_away + gamma)
    mu  = exp(attack_away + defence_home)

so a fitted result drops straight into :class:`DixonColesModel`. Note the sign
convention: ``defence`` enters positively, so a *higher* defence value means a
team concedes more, i.e. defends worse.

Two details matter for this to work at all:

*Identifiability.* Adding a constant to every attack and subtracting it from
every defence leaves both ``lam`` and ``mu`` unchanged, so the likelihood has a
flat direction and no unique maximum. We pin it with ``mean(attack) = 0``,
carried structurally by fitting only ``n-1`` attack parameters and setting the
last to minus their sum. With attacks centred, the overall scoring level is
absorbed by the defence terms.

*Time decay.* Team strength drifts, so old matches should count for less.
Weights are ``w = exp(-xi * days_before_reference)``; the default ``xi``
corresponds to a half-life of about three months, in the range Dixon and Coles
found optimal for predicting the next match.

The optimiser is L-BFGS-B with an analytic gradient. The gradient is worth the
extra code: a numeric one needs ``2n+1`` extra likelihood evaluations per step,
which makes the repeated refitting inside a walk-forward backtest painfully
slow.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .dixon_coles import DixonColesModel, TeamRating

# Default decay rate, per day. exp(-0.0065 * 107) ~ 0.5, so a match from about
# three and a half months ago carries half the weight of today's.
DEFAULT_XI = 0.0065

# tau can be driven non-positive by an extreme rho, which would make log(tau)
# undefined. Flooring it keeps the objective finite and produces a very large
# penalty plus a steep gradient, so the line search backs out of the region
# instead of the optimiser dying on a NaN.
_TAU_FLOOR = 1e-10

# Keeps exp() well inside floating-point range during the search. Real fitted
# strengths sit near zero, so these never bind on sane data.
_STRENGTH_BOUND = 5.0
_GAMMA_BOUND = 3.0


class MatchLike(Protocol):
    """The subset of a match the fitter reads.

    :class:`sttbot.data.football.Match` satisfies this, but so does any simple
    record, which keeps the strategy layer independent of the data layer.
    """

    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class FitResult:
    """Fitted Dixon-Coles parameters plus optimiser diagnostics."""

    ratings: dict[str, TeamRating]
    home_advantage: float
    rho: float
    log_likelihood: float
    n_matches: int
    effective_sample_size: float
    converged: bool
    message: str
    n_iterations: int

    @property
    def teams(self) -> list[str]:
        return sorted(self.ratings)

    def to_model(self, max_goals: int = 10) -> DixonColesModel:
        """Package the fit as a forward model ready to price matches."""
        return DixonColesModel(
            ratings=dict(self.ratings),
            home_advantage=self.home_advantage,
            rho=self.rho,
            max_goals=max_goals,
        )


def time_decay_weights(
    dates: Sequence[dt.date], reference: dt.date, xi: float = DEFAULT_XI
) -> np.ndarray:
    """Exponential decay weights ``exp(-xi * days_before_reference)``.

    Matches on or after ``reference`` get weight 1.0 rather than an
    extrapolated weight above 1 -- in a walk-forward fit nothing should be
    dated after the reference anyway, and silently up-weighting a leaked future
    match would be worse than treating it as current.
    """
    if xi < 0:
        raise ValueError("xi must be non-negative")
    age = np.array([(reference - d).days for d in dates], dtype=float)
    return np.exp(-xi * np.maximum(age, 0.0))


def half_life_to_xi(days: float) -> float:
    """Decay rate whose weight halves after ``days``."""
    if days <= 0:
        raise ValueError("half-life must be positive")
    return math.log(2.0) / days


def _tau_terms(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``tau`` and its partials w.r.t. ``lam``, ``mu`` and ``rho``.

    ``tau`` is 1 with zero derivatives everywhere except the four low-score
    cells, so we only touch those.
    """
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    d_lam[m00] = -mu[m00] * rho
    d_mu[m00] = -lam[m00] * rho
    d_rho[m00] = -lam[m00] * mu[m00]

    tau[m01] = 1.0 + lam[m01] * rho
    d_lam[m01] = rho
    d_rho[m01] = lam[m01]

    tau[m10] = 1.0 + mu[m10] * rho
    d_mu[m10] = rho
    d_rho[m10] = mu[m10]

    tau[m11] = 1.0 - rho
    d_rho[m11] = -1.0

    return tau, d_lam, d_mu, d_rho


def _unpack(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Free parameter vector -> (attack, defence, gamma, rho).

    The last attack is not a free parameter: it is minus the sum of the others,
    which is what enforces ``mean(attack) = 0``.
    """
    attack = np.empty(n)
    attack[: n - 1] = theta[: n - 1]
    attack[n - 1] = -theta[: n - 1].sum()
    defence = theta[n - 1 : 2 * n - 1]
    return attack, defence, float(theta[2 * n - 1]), float(theta[2 * n])


def _objective(
    theta: np.ndarray,
    n: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood and its analytic gradient.

    The ``log(x!)`` terms are dropped here because they do not depend on the
    parameters; :func:`fit_dixon_coles` adds them back once so the reported
    log-likelihood is a true one.
    """
    attack, defence, gamma, rho = _unpack(theta, n)

    lam = np.exp(attack[home_idx] + defence[away_idx] + gamma)
    mu = np.exp(attack[away_idx] + defence[home_idx])

    tau, dtau_dlam, dtau_dmu, dtau_drho = _tau_terms(x, y, lam, mu, rho)
    tau_safe = np.maximum(tau, _TAU_FLOOR)

    ll = float(
        np.sum(w * (np.log(tau_safe) - lam + x * np.log(lam) - mu + y * np.log(mu)))
    )

    # d(ll)/d(lam) * lam, i.e. the derivative w.r.t. any log-scale parameter
    # that feeds lam (attack_home, defence_away, gamma). Same for mu.
    g_lam = w * ((x - lam) + lam * dtau_dlam / tau_safe)
    g_mu = w * ((y - mu) + mu * dtau_dmu / tau_safe)

    d_attack = np.bincount(home_idx, g_lam, n) + np.bincount(away_idx, g_mu, n)
    d_defence = np.bincount(away_idx, g_lam, n) + np.bincount(home_idx, g_mu, n)
    d_gamma = float(g_lam.sum())
    d_rho = float(np.sum(w * dtau_drho / tau_safe))

    # Chain rule through attack[n-1] = -sum(free attacks).
    grad = np.empty_like(theta)
    grad[: n - 1] = d_attack[: n - 1] - d_attack[n - 1]
    grad[n - 1 : 2 * n - 1] = d_defence
    grad[2 * n - 1] = d_gamma
    grad[2 * n] = d_rho

    return -ll, -grad


def _initial_theta(
    n: int, x: np.ndarray, y: np.ndarray, w: np.ndarray, rho: float
) -> np.ndarray:
    """Start from the weighted pooled scoring rates, attacks flat at zero.

    With every attack at 0, ``lam = exp(defence + gamma)`` and
    ``mu = exp(defence)``, so setting defence to log(mean away goals) and gamma
    to log(mean home / mean away) reproduces the sample averages exactly.
    """
    total_w = w.sum()
    mean_home = max(float((w * x).sum() / total_w), 1e-3)
    mean_away = max(float((w * y).sum() / total_w), 1e-3)

    theta = np.zeros(2 * n + 1)
    theta[n - 1 : 2 * n - 1] = math.log(mean_away)
    theta[2 * n - 1] = math.log(mean_home / mean_away)
    theta[2 * n] = rho
    return theta


def fit_dixon_coles(
    matches: Sequence[MatchLike],
    *,
    dates: Sequence[dt.date] | None = None,
    reference_date: dt.date | None = None,
    xi: float = DEFAULT_XI,
    weights: Sequence[float] | np.ndarray | None = None,
    rho_bounds: tuple[float, float] = (-0.9, 0.9),
    initial_rho: float = -0.05,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> FitResult:
    """Fit attack/defence strengths, home advantage and rho by weighted MLE.

    Weights come from ``weights`` if given, else from exponential time decay
    when ``reference_date`` is set (reading ``dates``, or each match's own
    ``date`` attribute), else all matches count equally.

    Raises ``ValueError`` for an empty sample or fewer than two teams, since
    neither identifies the parameters.
    """
    if not matches:
        raise ValueError("cannot fit on an empty match sample")

    teams = sorted({m.home for m in matches} | {m.away for m in matches})
    n = len(teams)
    if n < 2:
        raise ValueError("need at least two distinct teams to fit")

    index = {team: i for i, team in enumerate(teams)}
    home_idx = np.array([index[m.home] for m in matches], dtype=np.intp)
    away_idx = np.array([index[m.away] for m in matches], dtype=np.intp)
    x = np.array([m.home_goals for m in matches], dtype=float)
    y = np.array([m.away_goals for m in matches], dtype=float)

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("goal counts must be non-negative")

    w = _resolve_weights(matches, dates, reference_date, xi, weights)

    theta0 = _initial_theta(n, x, y, w, initial_rho)
    bounds = (
        [(-_STRENGTH_BOUND, _STRENGTH_BOUND)] * (n - 1)
        + [(-_STRENGTH_BOUND, _STRENGTH_BOUND)] * n
        + [(-_GAMMA_BOUND, _GAMMA_BOUND), rho_bounds]
    )

    result = minimize(
        _objective,
        theta0,
        args=(n, home_idx, away_idx, x, y, w),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    attack, defence, gamma, rho = _unpack(result.x, n)
    # Add back the parameter-free log(x!) terms so this is a real log-likelihood.
    constant = float(np.sum(w * (gammaln(x + 1.0) + gammaln(y + 1.0))))

    return FitResult(
        ratings={
            team: TeamRating(attack=float(attack[i]), defence=float(defence[i]))
            for team, i in index.items()
        },
        home_advantage=gamma,
        rho=rho,
        log_likelihood=-float(result.fun) - constant,
        n_matches=len(matches),
        effective_sample_size=float(w.sum()),
        converged=bool(result.success),
        message=str(result.message),
        n_iterations=int(result.nit),
    )


def _resolve_weights(
    matches: Sequence[MatchLike],
    dates: Sequence[dt.date] | None,
    reference_date: dt.date | None,
    xi: float,
    weights: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(matches),):
            raise ValueError(
                f"weights has length {w.shape}, expected {len(matches)}"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("weights must not sum to zero")
        return w

    if reference_date is None:
        return np.ones(len(matches))

    if dates is None:
        try:
            dates = [m.date for m in matches]  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "reference_date given but matches carry no 'date'; "
                "pass dates= explicitly"
            ) from exc
    if len(dates) != len(matches):
        raise ValueError("dates and matches must have the same length")
    return time_decay_weights(dates, reference_date, xi)
