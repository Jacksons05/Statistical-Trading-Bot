import pytest

from sttbot.strategies.base import Signal
from sttbot.strategies.pead import (
    PostEarningsDriftStrategy,
    standardized_unexpected_earnings,
)


def test_sue_computation():
    assert standardized_unexpected_earnings(1.2, 1.0, 0.1) == pytest.approx(2.0)


def test_sue_rejects_nonpositive_std():
    with pytest.raises(ValueError):
        standardized_unexpected_earnings(1.0, 1.0, 0.0)


def test_signals_around_threshold():
    s = PostEarningsDriftStrategy(sue_threshold=1.5)
    assert s.generate_signal(2.0) is Signal.LONG
    assert s.generate_signal(-2.0) is Signal.SHORT
    assert s.generate_signal(0.5) is Signal.FLAT
    # Exactly at the threshold counts as a signal.
    assert s.generate_signal(1.5) is Signal.LONG


def test_target_weight_sign_and_size():
    s = PostEarningsDriftStrategy(sue_threshold=1.0, max_position_size_pct=0.05)
    assert s.target_weight(2.0) == pytest.approx(0.05)
    assert s.target_weight(-2.0) == pytest.approx(-0.05)
    assert s.target_weight(0.0) == 0.0
