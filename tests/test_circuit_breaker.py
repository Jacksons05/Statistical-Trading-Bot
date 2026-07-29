import pytest

from sttbot.execution.oms import PaperBroker
from sttbot.risk.circuit_breaker import RiskCircuitBreaker


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def broker():
    b = PaperBroker()
    b.set_orderbook("BTC", bids=[[99.0, 1.0]], asks=[[101.0, 1.0]])
    b.place_limit_order("BTC", "BUY", 101.0, 1.0)  # opens a position
    b.place_limit_order("BTC", "BUY", 90.0, 1.0)  # resting open order
    return b


def test_normal_equity_keeps_trading(broker):
    cb = RiskCircuitBreaker(max_drawdown=0.05)
    assert cb.evaluate(10_000, broker) is True
    assert cb.evaluate(10_100, broker) is True
    assert broker.trading_enabled is True


def test_hwm_drawdown_trips_kill_switch(broker):
    cb = RiskCircuitBreaker(max_drawdown=0.05)
    assert cb.evaluate(10_000, broker) is True
    assert cb.evaluate(9_400, broker) is False  # 6% drawdown from peak
    assert cb.tripped is True
    assert broker.trading_enabled is False
    assert all(p == 0.0 for p in broker.positions.values())  # flattened
    assert all(o.status != "OPEN" for o in broker.orders.values())  # cancelled


def test_breaker_stays_latched(broker):
    cb = RiskCircuitBreaker(max_drawdown=0.05)
    cb.evaluate(10_000, broker)
    cb.evaluate(9_000, broker)  # trips
    # Even a full recovery must not silently re-enable trading.
    assert cb.evaluate(10_500, broker) is False
    cb.reset()
    assert cb.evaluate(10_500, broker) is True


def test_rolling_window_peak_ages_out(broker):
    clock = FakeClock()
    cb = RiskCircuitBreaker(
        max_drawdown=0.50,  # effectively disable the HWM check
        rolling_window_seconds=100,
        max_rolling_drawdown=0.05,
        clock=clock.now,
    )
    assert cb.evaluate(10_000, broker) is True
    clock.advance(200)  # the 10_000 peak is now outside the window
    # 9_600 is a 4% drop from the all-time peak but the rolling peak has aged
    # out, so within-window drawdown is 0 and trading continues.
    assert cb.evaluate(9_600, broker) is True


def test_negative_equity_rejected(broker):
    cb = RiskCircuitBreaker()
    with pytest.raises(ValueError):
        cb.evaluate(-1.0, broker)
