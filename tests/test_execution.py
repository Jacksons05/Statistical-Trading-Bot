import pytest

from sttbot.execution.oms import PaperBroker
from sttbot.execution.order_manager import DynamicOrderManager, slippage_bps


@pytest.fixture
def broker():
    b = PaperBroker()
    b.set_orderbook("BTC", bids=[[99.0, 1.0]], asks=[[101.0, 1.0]])
    return b


def test_slippage_bps_sign():
    assert slippage_bps(101.0, 100.0) == pytest.approx(100.0)  # 1% = 100 bps


def test_marketable_order_fills_immediately(broker):
    oid = broker.place_limit_order("BTC", "BUY", 101.0, 0.5)
    assert broker.get_order_status(oid) == "FILLED"
    assert broker.positions["BTC"] == pytest.approx(0.5)


def test_resting_order_stays_open(broker):
    oid = broker.place_limit_order("BTC", "BUY", 100.0, 0.5)
    assert broker.get_order_status(oid) == "OPEN"


class FakeClock:
    """A monotonic clock that only advances when the manager sleeps."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_ttl_expires_and_cancels(broker):
    clock = FakeClock()
    mgr = DynamicOrderManager(
        max_allowed_slippage_bps=1_000,
        ttl_seconds=0.2,
        poll_interval=0.05,
        clock=clock.now,
        sleep=clock.sleep,
    )
    # Non-marketable buy rests, never fills, and is cancelled at TTL.
    result = mgr.execute_with_ttl(broker, "BTC", "BUY", 100.0, 0.5)
    assert result.filled is False
    assert result.reason == "ttl_expired"
    assert broker.get_order_status(result.order_id) == "CANCELLED"


def test_execution_fills_within_ttl(broker):
    clock = FakeClock()
    mgr = DynamicOrderManager(
        max_allowed_slippage_bps=1_000,
        ttl_seconds=1.0,
        clock=clock.now,
        sleep=clock.sleep,
    )
    result = mgr.execute_with_ttl(broker, "BTC", "BUY", 101.0, 0.5)
    assert result.filled is True
    assert result.reason == "filled"


def test_slippage_cap_rejects_before_placing(broker):
    mgr = DynamicOrderManager(max_allowed_slippage_bps=10.0)
    # Best ask 101 vs target 100 is ~99 bps of slippage, over the 10 bps cap.
    result = mgr.execute_with_ttl(broker, "BTC", "BUY", 100.0, 0.5)
    assert result.filled is False
    assert "slippage" in result.reason
    assert result.order_id is None
    assert broker.orders == {}  # nothing was placed


def test_disabled_trading_blocks_orders(broker):
    broker.disable_trading_loop()
    with pytest.raises(RuntimeError):
        broker.place_limit_order("BTC", "BUY", 101.0, 0.5)
