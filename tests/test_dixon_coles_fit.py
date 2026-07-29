"""Validation of the Dixon-Coles MLE fitter.

The fitter is only trustworthy if it can recover parameters it did not see, so
the central tests simulate matches from known attack/defence/gamma/rho, refit,
and check the estimates come back. Everything here is network-free and seeded.
"""

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import spearmanr

from sttbot.strategies.dixon_coles import score_matrix
from sttbot.strategies.dixon_coles_fit import (
    DEFAULT_XI,
    fit_dixon_coles,
    half_life_to_xi,
    time_decay_weights,
)
from sttbot.strategies.dixon_coles_fit import _objective, _unpack


@dataclass(frozen=True)
class SimMatch:
    home: str
    away: str
    home_goals: int
    away_goals: int
    date: dt.date = dt.date(2024, 1, 1)


# Ground truth: attacks are centred because the fitter pins mean(attack) = 0,
# so only a centred truth is directly comparable to what it returns.
TRUE_ATTACK = {
    "Ajax": 0.60,
    "PSV": 0.35,
    "Feyenoord": 0.20,
    "AZ": 0.10,
    "Twente": 0.00,
    "Utrecht": -0.08,
    "Vitesse": -0.16,
    "Groningen": -0.25,
    "Heracles": -0.32,
    "Willem": -0.44,
}
TRUE_DEFENCE = {
    "Ajax": -0.05,
    "PSV": 0.02,
    "Feyenoord": 0.05,
    "Twente": 0.14,
    "Utrecht": 0.20,
    "AZ": 0.10,
    "Vitesse": 0.26,
    "Groningen": 0.30,
    "Heracles": 0.34,
    "Willem": 0.42,
}
TRUE_GAMMA = 0.26
TRUE_RHO = -0.08


def _centre(values: dict[str, float]) -> dict[str, float]:
    mean = sum(values.values()) / len(values)
    return {k: v - mean for k, v in values.items()}


def simulate(n_rounds: int, seed: int = 7, rho: float = TRUE_RHO) -> list[SimMatch]:
    """Sample matches from the true Dixon-Coles joint distribution.

    Draws from the exact tau-corrected score matrix rather than two independent
    Poissons, so a fitter that ignored the dependence correction would show up
    as a biased rho. The score distribution depends only on the fixture, so it
    is computed once per ordered pair and sampled ``n_rounds`` times.
    """
    rng = np.random.default_rng(seed)
    attack = _centre(TRUE_ATTACK)
    teams = sorted(attack)
    max_goals = 12
    width = max_goals + 1
    flat_index = np.arange(width * width)

    matches: list[SimMatch] = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            lam = math.exp(attack[home] + TRUE_DEFENCE[away] + TRUE_GAMMA)
            mu = math.exp(attack[away] + TRUE_DEFENCE[home])
            probs = score_matrix(lam, mu, rho, max_goals).ravel()
            cells = rng.choice(flat_index, size=n_rounds, p=probs / probs.sum())
            matches.extend(
                SimMatch(home, away, int(c // width), int(c % width)) for c in cells
            )
    return matches


@pytest.fixture(scope="module")
def big_sample() -> list[SimMatch]:
    # 10 teams x 90 ordered pairs x 80 rounds = 7200 matches. Sized from the
    # measured spread of the estimator rather than guessed: at 1080 matches the
    # worst-team standard error is about 0.08, so tolerances tight enough to be
    # meaningful failed on roughly one seed in ten. The tolerances below are set
    # from the observed worst case over 20 seeds at this size, with headroom.
    return simulate(n_rounds=80)


@pytest.fixture(scope="module")
def big_fit(big_sample):
    return fit_dixon_coles(big_sample)


# --- gradient correctness ---------------------------------------------------


def test_analytic_gradient_matches_numeric():
    """The analytic gradient is the main source of subtle fitter bugs."""
    matches = simulate(n_rounds=2, seed=3)
    teams = sorted({m.home for m in matches})
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    home_idx = np.array([index[m.home] for m in matches], dtype=np.intp)
    away_idx = np.array([index[m.away] for m in matches], dtype=np.intp)
    x = np.array([m.home_goals for m in matches], dtype=float)
    y = np.array([m.away_goals for m in matches], dtype=float)
    w = np.linspace(0.3, 1.0, len(matches))  # non-uniform, to catch weight bugs

    rng = np.random.default_rng(11)
    theta = np.concatenate(
        [rng.normal(0, 0.2, n - 1), rng.normal(0.1, 0.2, n), [0.25], [-0.07]]
    )

    _, analytic = _objective(theta, n, home_idx, away_idx, x, y, w)

    step = 1e-6
    numeric = np.empty_like(theta)
    for i in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (
            _objective(up, n, home_idx, away_idx, x, y, w)[0]
            - _objective(down, n, home_idx, away_idx, x, y, w)[0]
        ) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)


# --- parameter recovery -----------------------------------------------------


def test_recovers_attack_and_defence(big_fit):
    true_attack = _centre(TRUE_ATTACK)
    for team, rating in big_fit.ratings.items():
        assert rating.attack == pytest.approx(true_attack[team], abs=0.10), team
        assert rating.defence == pytest.approx(TRUE_DEFENCE[team], abs=0.10), team


def test_recovers_home_advantage_and_rho(big_fit):
    assert big_fit.home_advantage == pytest.approx(TRUE_GAMMA, abs=0.05)
    # rho is by far the noisiest parameter: it is identified only by the four
    # low-score cells, so it sees a fraction of the sample the strengths do.
    assert big_fit.rho == pytest.approx(TRUE_RHO, abs=0.08)


def test_recovers_team_ranking(big_fit):
    """Ordering should track the truth without requiring an exact permutation.

    Adjacent teams differ by as little as 0.08 in true attack, which is inside
    sampling error, so recovering the exact order is not a property the fitter
    can have at any realistic sample size. Rank correlation is the claim that
    survives; the strongest attack is separated widely enough to assert on its
    own.
    """
    true_attack = _centre(TRUE_ATTACK)
    teams = sorted(true_attack)
    rho, _ = spearmanr(
        [big_fit.ratings[t].attack for t in teams], [true_attack[t] for t in teams]
    )
    assert rho > 0.9
    assert max(big_fit.ratings, key=lambda t: big_fit.ratings[t].attack) == "Ajax"


def test_estimator_is_unbiased_across_seeds():
    """Errors should average out, which noise on a single sample cannot show."""
    errors = {team: [] for team in TRUE_ATTACK}
    gammas, rhos = [], []
    for seed in range(6):
        fit = fit_dixon_coles(simulate(n_rounds=30, seed=seed))
        for team, rating in fit.ratings.items():
            errors[team].append(rating.attack - TRUE_ATTACK[team])
        gammas.append(fit.home_advantage)
        rhos.append(fit.rho)

    for team, errs in errors.items():
        assert abs(float(np.mean(errs))) < 0.06, team
    assert float(np.mean(gammas)) == pytest.approx(TRUE_GAMMA, abs=0.04)
    assert float(np.mean(rhos)) == pytest.approx(TRUE_RHO, abs=0.04)


def test_fit_converges_and_reports_diagnostics(big_fit, big_sample):
    assert big_fit.converged, big_fit.message
    assert big_fit.n_iterations > 0
    assert big_fit.n_matches == len(big_sample)
    assert big_fit.effective_sample_size == pytest.approx(len(big_sample))
    assert big_fit.log_likelihood < 0


def test_attack_is_centred_for_identifiability(big_fit):
    mean_attack = sum(r.attack for r in big_fit.ratings.values()) / len(big_fit.ratings)
    assert mean_attack == pytest.approx(0.0, abs=1e-9)


def test_zero_rho_sample_recovers_zero_rho():
    """Independent Poisson data should not manufacture a dependence."""
    fit = fit_dixon_coles(simulate(n_rounds=80, seed=5, rho=0.0))
    assert fit.rho == pytest.approx(0.0, abs=0.05)


def test_fitted_model_prices_the_true_favourite_correctly(big_fit):
    model = big_fit.to_model()
    probs = model.match_probabilities("Ajax", "Willem")
    assert sum(probs.as_tuple()) == pytest.approx(1.0)
    # Best attack at home against the weakest side: a heavy favourite.
    assert probs.home_win > 0.6
    assert probs.home_win > probs.away_win


def test_more_data_reduces_error():
    """A consistent estimator should tighten as the sample grows."""
    true_attack = _centre(TRUE_ATTACK)

    def error(n_rounds):
        fit = fit_dixon_coles(simulate(n_rounds=n_rounds, seed=21))
        return max(
            abs(r.attack - true_attack[t]) for t, r in fit.ratings.items()
        )

    assert error(12) < error(1)


# --- time decay -------------------------------------------------------------


def test_time_decay_weights_halve_at_half_life():
    reference = dt.date(2024, 6, 1)
    xi = half_life_to_xi(100)
    weights = time_decay_weights(
        [reference, reference - dt.timedelta(days=100)], reference, xi
    )
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)


def test_future_matches_are_not_up_weighted():
    reference = dt.date(2024, 6, 1)
    weights = time_decay_weights([reference + dt.timedelta(days=30)], reference)
    assert weights[0] == pytest.approx(1.0)


def test_default_xi_has_a_sane_half_life():
    half_life = math.log(2) / DEFAULT_XI
    assert 90 < half_life < 130  # about a season's worth of recency


def test_decay_downweights_stale_form():
    """A team that was strong long ago and weak lately should rate weak."""
    old = dt.date(2023, 1, 1)
    recent = dt.date(2024, 6, 1)
    matches = (
        [SimMatch("A", "B", 5, 0, old) for _ in range(30)]
        + [SimMatch("B", "A", 0, 5, old) for _ in range(30)]
        + [SimMatch("A", "B", 0, 3, recent) for _ in range(30)]
        + [SimMatch("B", "A", 3, 0, recent) for _ in range(30)]
    )
    undecayed = fit_dixon_coles(matches)
    decayed = fit_dixon_coles(
        matches, reference_date=recent, xi=half_life_to_xi(60)
    )
    # Recency weighting should move A's attack down relative to the flat fit.
    assert decayed.ratings["A"].attack < undecayed.ratings["A"].attack


def test_explicit_weights_override_decay():
    matches = simulate(n_rounds=2, seed=9)
    weights = np.ones(len(matches))
    weights[: len(matches) // 2] = 0.0  # ignore the first half entirely
    fit = fit_dixon_coles(matches, weights=weights)
    assert fit.effective_sample_size == pytest.approx(weights.sum())


def test_half_life_conversion_roundtrip():
    assert half_life_to_xi(math.log(2) / DEFAULT_XI) == pytest.approx(DEFAULT_XI)
    with pytest.raises(ValueError):
        half_life_to_xi(0)


# --- input validation -------------------------------------------------------


def test_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        fit_dixon_coles([])


def test_rejects_single_team():
    with pytest.raises(ValueError, match="two distinct teams"):
        fit_dixon_coles([SimMatch("A", "A", 1, 1)])


def test_rejects_negative_goals():
    with pytest.raises(ValueError, match="non-negative"):
        fit_dixon_coles([SimMatch("A", "B", -1, 0), SimMatch("B", "A", 1, 0)])


def test_rejects_mismatched_weights():
    matches = [SimMatch("A", "B", 1, 0), SimMatch("B", "A", 2, 1)]
    with pytest.raises(ValueError, match="expected 2"):
        fit_dixon_coles(matches, weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        fit_dixon_coles(matches, weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        fit_dixon_coles(matches, weights=[0.0, 0.0])


def test_rejects_negative_xi():
    with pytest.raises(ValueError, match="non-negative"):
        time_decay_weights([dt.date(2024, 1, 1)], dt.date(2024, 1, 1), xi=-0.1)


def test_unpack_enforces_zero_mean_attack():
    n = 4
    theta = np.array([0.3, -0.1, 0.5, 0.0, 0.1, 0.2, 0.3, 0.25, -0.05])
    attack, defence, gamma, rho = _unpack(theta, n)
    assert attack.sum() == pytest.approx(0.0)
    assert attack[-1] == pytest.approx(-0.7)
    assert defence.size == n
    assert gamma == pytest.approx(0.25)
    assert rho == pytest.approx(-0.05)
