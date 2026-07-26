import numpy as np
import pytest

from sttbot.strategies.dixon_coles import (
    DixonColesModel,
    TeamRating,
    closing_line_value,
    dixon_coles_tau,
    expected_value,
    implied_probability,
    outcome_probabilities,
    score_matrix,
)


def test_tau_reduces_to_one_off_low_scores():
    assert dixon_coles_tau(3, 2, 1.4, 1.1, -0.05) == 1.0
    assert dixon_coles_tau(0, 0, 1.4, 1.1, -0.05) != 1.0


def test_score_matrix_normalised():
    m = score_matrix(1.4, 1.1, rho=-0.05, max_goals=12)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_rho_zero_is_independent_poisson():
    m = score_matrix(1.5, 1.2, rho=0.0, max_goals=15)
    # With rho=0 the joint mass factorises into the marginals.
    row = m.sum(axis=1)
    col = m.sum(axis=0)
    assert m[2, 3] == pytest.approx(row[2] * col[3], rel=1e-6)


def test_outcome_probabilities_sum_to_one():
    m = score_matrix(1.6, 1.0, rho=-0.03)
    probs = outcome_probabilities(m)
    assert sum(probs.as_tuple()) == pytest.approx(1.0)
    # Stronger home attack -> home win most likely.
    assert probs.home_win > probs.away_win


def test_negative_rho_raises_draw_probability():
    lam, mu = 1.1, 1.1
    base = outcome_probabilities(score_matrix(lam, mu, rho=0.0))
    corrected = outcome_probabilities(score_matrix(lam, mu, rho=-0.1))
    assert corrected.draw > base.draw


def test_model_expected_goals_and_home_edge():
    model = DixonColesModel(
        ratings={
            "A": TeamRating(attack=0.3, defence=-0.2),
            "B": TeamRating(attack=0.1, defence=-0.1),
        },
        home_advantage=0.3,
        rho=-0.05,
    )
    lam, mu = model.expected_goals("A", "B")
    assert lam > mu  # home attack + home advantage dominates
    probs = model.match_probabilities("A", "B")
    assert sum(probs.as_tuple()) == pytest.approx(1.0)


def test_unknown_team_raises():
    model = DixonColesModel(ratings={"A": TeamRating(0.0, 0.0)})
    with pytest.raises(KeyError):
        model.expected_goals("A", "Z")


def test_expected_value_and_implied_probability():
    # A fair coin priced at 2.0 decimal odds is exactly break-even.
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert implied_probability(2.0) == pytest.approx(0.5)
    # Model edge over the line -> positive EV.
    assert expected_value(0.6, 2.0) == pytest.approx(0.2)


def test_closing_line_value_positive_when_entry_beats_close():
    # Bet at 2.10, line closes at 1.90 -> we got better than the sharp close.
    assert closing_line_value(entry_odds=2.10, closing_odds=1.90) > 0


def test_score_matrix_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        score_matrix(0.0, 1.0)
