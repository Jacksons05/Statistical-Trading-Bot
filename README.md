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
| `data.datasets` | Catalog of openly-licensed datasets + caching downloader (`STTBOT_DATA_DIR`), with checksum pinning; `LIVE_ENDPOINTS` records the direct-egress APIs and their access requirements. |
| `data.probe` | `python -m sttbot.data.probe` — re-measures which data hosts this machine can actually reach. |
| `data.football` | Loaders for the bulk football dataset **and** football-data.co.uk season CSVs (the only wired-up source carrying a genuine closing line); overround and de-vig helpers. |
| `strategies.base` | `Strategy` + `Param` — declarative hyperparameters with grid-search introspection (the `# @param` convention, made programmatic). |
| `strategies.pead` | Micro-cap Post-Earnings Announcement Drift via Standardised Unexpected Earnings (SUE). |
| `strategies.dixon_coles` | Dixon-Coles bivariate-Poisson model for low-scoring sports, plus +EV and Closing Line Value helpers. |
| `strategies.dixon_coles_fit` | Weighted-MLE fitter for the above: per-team attack/defence, home advantage γ and dependence ρ, with exponential time decay, `mean(attack)=0` identifiability, and an analytic gradient. |
| `backtest.walk_forward` | Walk-forward engine — fit on a trailing window, bet the next matchday, settle after friction. Look-ahead is structurally impossible, not merely intended. |
| `backtest.metrics` | Sharpe, max drawdown, hit rate, ROI on staked capital. |
| `backtest.clv` | Closing-line-value **control**: what CLV a random selection earns on the same matches, so mechanical CLV is not mistaken for skill. |
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

## Open data

`sttbot.data.datasets` catalogs openly-licensed, key-free datasets and caches
them locally on first use (override the location with `STTBOT_DATA_DIR`):

```python
from sttbot.data.datasets import available, describe, fetch

print(available())            # ['btcusd_1min', 'football_json_en1', 'football_matches']
path = fetch("football_matches")   # downloads once, then served from cache
```

| Dataset | License | Size | Contents |
| --- | --- | --- | --- |
| `football_matches` | MIT | 44 MB | 230,557 club matches (2000–2025, 38 divisions, 1,214 teams) with scores, match stats, Elo, and 1X2 / O-U / Asian-handicap odds from Bet365 plus best-of-book maxima |
| `btcusd_1min` | MIT | 95 MB | 6.85M one-minute BTC/USD OHLCV candles from Bitstamp (2012–2025) |
| `football_json_en1` | CC0 | <1 MB | EPL 2023-24 fixtures/results in JSON |

`python examples/explore_open_data.py` profiles the football data. Measured on
the full 230k rows:

- **Home advantage** — 1.488 vs 1.151 goals, i.e. γ ≈ **0.257**, which
  independently supports the model's 0.25 default.
- **Low-score cells** — 0-0, 0-1, 1-0 and 1-1 are 20.6 / 20.1 / 27.4 / 31.9% of
  all 0-or-1-goal games, the concentration the Dixon-Coles `tau` correction exists to fix.
- **Friction** — a single book's 1X2 margin averages **6.73%**, but the
  best-of-book margin is only **0.93%**; line-shopping recovers **5.8pp**, which
  is larger than most model edges. In **23.6%** of matches the best-of-book sum
  falls below 1.00 — a boundary arbitrage of the kind `find_boundary_arbitrage`
  detects. Treat that as an *upper bound*: those maxima are peak prices across
  ~17 books and assume simultaneous availability, accounts everywhere, and
  stake limits that don't bind.

### Direct-egress sources

Re-probed 2026-07-28 from an unrestricted network. An earlier sandbox returned
403 for all of these, which was its allowlisting proxy rather than any upstream
restriction — so a blanket 403 means "re-probe", not "the venue is gone". Run
`python -m sttbot.data.probe` to re-measure from your own machine.

| Host | Status | Notes |
| --- | --- | --- |
| `www.football-data.co.uk` | reachable | Per-season odds CSVs carrying **both** pre-match and closing prices — the only source here that supports real CLV |
| `gamma-api.polymarket.com` | reachable | Public read API, no key |
| `api.elections.kalshi.com` | reachable | Public read on `/trade-api/v2/markets`; trading needs a signed key |
| `data.sec.gov` | reachable | 403s unless the User-Agent carries a contact address — set `STTBOT_USER_AGENT` |
| `stooq.com` | reachable | Free daily OHLCV; rate-limited |
| `api.exchange.coinbase.com` | reachable | Public read endpoints |
| `api.binance.com` | **blocked** | HTTP 451 geo-restriction, not a proxy. Use Coinbase instead |

## Walk-forward results on real odds

`python examples/backtest_dixon_coles.py` fits the model on a trailing two-year
window, bets the next matchday, and settles after 2% commission. Seven seasons
(2018/19–2024/25), English tiers 1–4 plus Scotland, ~2,600 matches per division.

**The model does not beat these markets.** Every division loses money, and
against a single book its closing-line value is *negative* — it is
systematically taking worse prices than the market closes at, which is a
stronger indictment than the P&L, because CLV carries no settlement variance.

| Division | Entry | Bets | ROI | Mean CLV | Random CLV | Skill |
| --- | --- | --- | --- | --- | --- | --- |
| E0 (Premier League) | single-book | 1,675 | −9.43% | −0.0081 | −0.0079 | −0.0002 |
| E0 | best-of-book | 2,208 | −4.54% | +0.0078 | +0.0068 | +0.0010 |
| E2 (League One) | best-of-book | 2,738 | −6.54% | +0.0067 | +0.0057 | +0.0010 |
| SC2 (Scottish L1) | best-of-book | 556 | −2.65% | +0.0146 | +0.0067 | **+0.0078** |
| SC3 (Scottish L2) | best-of-book | 500 | +3.12% | +0.0135 | +0.0067 | **+0.0068** |

The `Random CLV` column is the reason this table is trustworthy. Best-of-book
entry prices beat a single book's close roughly 60% of the time **with no model
at all** — you are comparing the best of ~17 prices against one. A strategy
reporting "+0.007 mean CLV, 64% positive" on that basis has demonstrated
nothing. `backtest.clv.clv_skill` subtracts that free lunch.

What survives the control: in the two thinnest markets tested, the model shows
genuine selection skill (SC2 t = 5.1, SC3 t = 4.1 — significant even after
correcting for the 14 division/entry-rule combinations tried). That is the
project's "thin markets are less efficient" thesis showing up in data. But the
skill is worth ~0.7 percentage points of implied probability, and the margin
plus commission is worth more, so **it does not convert into profit**: SC3's
+3.12% ROI is 0.5 standard errors from zero (SE 6.7%) and is noise, not a
finding. Do not read that number as an edge.

Honest summary: the fitter works, the backtest is sound, and the strategy is
not yet profitable. The measurable signal is in lower-tier markets and in line
shopping — not in the Premier League.

## Design notes

- **Testability first.** Clocks and sleep are injected into the order manager
  and circuit breaker, so time-dependent logic is deterministic under test.
  The `PaperBroker` implements the same `OMS` surface as a live adapter.
- **Friction is not optional.** Every signal is meant to be evaluated *after*
  fees, slippage, gas, and spread; `should_use_maker` suppresses taker orders
  when the spread would eat too much of the edge.
- **Fail-safe risk.** The circuit breaker latches once tripped — a recovering
  equity curve cannot silently re-enable trading; a human must `reset()`.
- **No look-ahead by construction.** The walk-forward loop derives its training
  window from the prediction date and cuts strictly at `date < d`, so a fit
  cannot see a result it is about to bet on. A test tampers with a future
  matchday's scores and asserts every earlier bet is unchanged.
- **Measure against the right null.** A metric that a random strategy also
  earns is not evidence. `backtest.clv` makes that baseline a first-class
  object rather than something a reader is trusted to remember.

## Scope & disclaimer

This is research/engineering scaffolding, not financial advice and not a
turnkey money printer. Live venue adapters (Interactive Brokers, CCXT,
Kalshi/Polymarket) are intentionally left as thin integrations behind the
`OMS` protocol. Trade at your own risk and comply with the terms of every
venue you connect to.

## Requirements

Python 3.11+, `numpy`, `duckdb`, `pandas`, `scipy`; `pytest` for the test suite.
