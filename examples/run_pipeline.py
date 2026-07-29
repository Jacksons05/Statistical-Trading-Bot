"""End-to-end paper-trading demo wiring the ecosystem's layers together.

Run with:  python examples/run_pipeline.py

It exercises: DuckDB ingestion -> a Dixon-Coles +EV signal and a prediction-
market boundary-arbitrage scan -> friction-aware maker/taker routing ->
TTL execution against a paper broker -> a drawdown circuit breaker -> alerts.
No live venues or credentials are touched.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys

# Allow running directly from a checkout without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sttbot.data.storage import Tick, TickStore
from sttbot.economics.friction import FrictionModel, should_use_maker
from sttbot.execution.oms import PaperBroker
from sttbot.execution.order_manager import DynamicOrderManager
from sttbot.monitoring.alerts import Notifier
from sttbot.risk.circuit_breaker import RiskCircuitBreaker
from sttbot.strategies.dixon_coles import DixonColesModel, TeamRating, expected_value
from sttbot.strategies.prob_arbitrage import Outcome, find_boundary_arbitrage


def main() -> None:
    notify = Notifier()  # no webhook -> logs only

    # 1. Ingest a couple of ticks into the columnar store.
    store = TickStore(":memory:")
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    store.append(
        [
            Tick(now, "BTC", bid=99.0, ask=101.0, volume=3.2),
            Tick(now, "BTC", bid=99.5, ask=100.5, volume=1.1),
        ]
    )
    print(f"ingested {store.count('BTC')} BTC ticks; latest mid={store.latest('BTC').mid:.2f}")

    # 2a. Sports model: is the home line +EV?
    model = DixonColesModel(
        ratings={"Home": TeamRating(0.35, -0.15), "Away": TeamRating(0.05, -0.05)},
        home_advantage=0.30,
        rho=-0.05,
    )
    probs = model.match_probabilities("Home", "Away")
    home_odds = 1.95
    ev = expected_value(probs.home_win, home_odds)
    print(f"P(home win)={probs.home_win:.3f}  home EV @ {home_odds}: {ev:+.3f}")
    if ev > 0:
        notify.info(f"+EV home bet: {ev:+.3f} per unit")

    # 2b. Prediction market: scan a categorical book for a boundary arb.
    outcomes = [
        Outcome("Cand-A", best_bid=0.40, best_ask=0.42),
        Outcome("Cand-B", best_bid=0.33, best_ask=0.35),
        Outcome("Cand-C", best_bid=0.19, best_ask=0.21),
    ]
    opp = find_boundary_arbitrage(outcomes, fee_rate=0.005)
    if opp:
        print(f"arb: {opp.side} net_edge={opp.net_edge:+.4f} legs={opp.legs}")
        notify.info(f"boundary arb {opp.side} net {opp.net_edge:+.4f}")

    # 3. Friction-aware routing + TTL execution against a paper broker.
    broker = PaperBroker()
    broker.set_orderbook("BTC", bids=[[99.5, 5]], asks=[[100.5, 5]])
    gross_edge, spread = 0.01, 0.002
    use_maker = should_use_maker(gross_edge, spread)
    friction = FrictionModel(taker_fee=0.001, maker_fee=-0.0002, slippage=0.0005)
    net = friction.net_edge(gross_edge, spread, maker=use_maker)
    print(f"routing: {'MAKER' if use_maker else 'TAKER'}  net_edge={net:+.4f}")

    mgr = DynamicOrderManager(max_allowed_slippage_bps=200, ttl_seconds=1.0)
    # Cross the spread so the demo fills deterministically.
    result = mgr.execute_with_ttl(broker, "BTC", "BUY", 100.5, qty=0.25)
    print(f"execution: filled={result.filled} ({result.reason})")

    # 4. Risk gate on marked-to-market equity.
    cb = RiskCircuitBreaker(max_drawdown=0.05)
    for equity in (10_000, 10_200, 9_600):  # last point is a >5% drawdown
        ok = cb.evaluate(equity, broker)
        print(f"equity={equity} trading_ok={ok}")
        if not ok:
            notify.critical("circuit breaker tripped; positions flattened")
            break


if __name__ == "__main__":
    main()
