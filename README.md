# Statistical-Trading-Bot (`sttbot`)

A modular, testable **systematic-trading ecosystem for the solo operator**. The
design premise is deliberately narrow: instead of competing with institutional
funds on liquid majors, target *fragmented, low-capacity, and behavioral*
niches where AUM capacity floors keep large capital out — micro-cap
post-earnings drift, prediction-market probability boundaries, cross-venue
crypto divergence, and quantitative sports pricing.

This repository is the software backbone for that thesis: a set of small,
independently testable modules mirroring the pipeline **data ingestion →
strategy/analysis → orchestration/execution → operations/risk**.

## Architecture

```
Data ingestion      →  sttbot.data.storage      DuckDB columnar tick store
Strategy / analysis →  sttbot.strategies         declarative params + signals
Execution           →  sttbot.execution          OMS interface, paper broker,
                                                  TTL + slippage order manager
Risk / ops          →  sttbot.risk               drawdown circuit breaker
                       sttbot.economics           net-edge / friction model
                       sttbot.monitoring          webhook alerting
```

## Modules

| Module | What it provides |
| --- | --- |
| `data.storage` | `TickStore` — DuckDB-backed tick/order-book storage with Parquet export; `:memory:` for tests. |
| `strategies.base` | `Strategy` + `Param` — declarative hyperparameters with grid-search introspection (the `# @param` convention, made programmatic). |
| `strategies.pead` | Micro-cap Post-Earnings Announcement Drift via Standardised Unexpected Earnings (SUE). |
| `strategies.dixon_coles` | Dixon-Coles bivariate-Poisson model for low-scoring sports, plus +EV and Closing Line Value helpers. |
| `strategies.prob_arbitrage` | Multi-outcome probability-boundary arbitrage for categorical prediction markets. |
| `economics.friction` | Net-edge formula, maker/taker routing rule, order-size/timing stealthing. |
| `execution.oms` | `OMS` protocol + in-memory `PaperBroker` for deterministic paper trading. |
| `execution.order_manager` | `DynamicOrderManager` — pre-trade slippage cap + Time-To-Live cancellation. |
| `risk.circuit_breaker` | High-water-mark **and** rolling-window drawdown kill switch. |
| `monitoring.alerts` | Fail-safe Discord/Telegram webhook notifier (stdlib only). |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the end-to-end paper-trading demo
python examples/run_pipeline.py

# run the tests
pip install -e .
pytest
```

## Example: a +EV sports signal

```python
from sttbot.strategies.dixon_coles import DixonColesModel, TeamRating, expected_value

model = DixonColesModel(
    ratings={"Home": TeamRating(0.35, -0.15), "Away": TeamRating(0.05, -0.05)},
    home_advantage=0.30,
    rho=-0.05,
)
probs = model.match_probabilities("Home", "Away")
print(expected_value(probs.home_win, decimal_odds=1.95))  # +EV if positive
```

## Example: prediction-market boundary arbitrage

```python
from sttbot.strategies.prob_arbitrage import Outcome, find_boundary_arbitrage

book = [
    Outcome("Cand-A", best_bid=0.40, best_ask=0.42),
    Outcome("Cand-B", best_bid=0.33, best_ask=0.35),
    Outcome("Cand-C", best_bid=0.19, best_ask=0.21),
]
opp = find_boundary_arbitrage(book, fee_rate=0.005)
print(opp)  # buy_basket when the asks sum below 1.00, net of fees
```

## Design notes

- **Testability first.** Clocks and sleep are injected into the order manager
  and circuit breaker, so time-dependent logic is deterministic under test.
  The `PaperBroker` implements the same `OMS` surface as a live adapter.
- **Friction is not optional.** Every signal is meant to be evaluated *after*
  fees, slippage, gas, and spread; `should_use_maker` suppresses taker orders
  when the spread would eat too much of the edge.
- **Fail-safe risk.** The circuit breaker latches once tripped — a recovering
  equity curve cannot silently re-enable trading; a human must `reset()`.

## Scope & disclaimer

This is research/engineering scaffolding, not financial advice and not a
turnkey money printer. Live venue adapters (Interactive Brokers, CCXT,
Kalshi/Polymarket) are intentionally left as thin integrations behind the
`OMS` protocol. Trade at your own risk and comply with the terms of every
venue you connect to.

## Requirements

Python 3.11+, `numpy`, `duckdb`, `pandas`; `pytest` for the test suite.
