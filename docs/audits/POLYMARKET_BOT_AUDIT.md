# Polymarket / sttbot Audit — Phase 1

Date: 2026-08-20
Scope: full repository read — `src/sttbot/**`, `tests/**`, `examples/**`,
`scripts/**`, `README.md`, `pyproject.toml`, `requirements.txt`,
`.github/workflows/ci.yml`. All 484 tests were executed locally (`pip install
-e .[dev] && pytest`) and pass. Every claim below cites `file:line`; anything
without a citation is a synthesis of cited evidence, marked as such.

---

## 0. Headline verification of the framing claims

| Claim in the assignment brief | Verified? | Evidence |
|---|---|---|
| Paper-trading only, no live order-submission path | **Confirmed** | No `py_clob_client`/`py-clob` dependency anywhere (`grep -ri` across the repo, zero hits, checked against `pyproject.toml`, `requirements.txt`, all `*.py`). `venues/polymarket.py` only issues `urllib.request` GET calls to `gamma-api.polymarket.com` and `clob.polymarket.com/book` (`src/sttbot/venues/polymarket.py:41-65`, `:233-234`). No signing library (`eth_account`, `web3`), no wallet/private-key code, no order-placement endpoint is ever called. The only components that implement the `OMS` protocol (`src/sttbot/execution/oms.py:20-29`) are `PaperBroker` (in-memory, `oms.py:43-117`) and `_AccountOms` (paper-account adapter, `src/sttbot/paper/runner.py:100-153`) — there is no live-broker adapter in the codebase at all. |
| DuckDB storage | **Confirmed** | `src/sttbot/data/storage.py` (`TickStore`), `src/sttbot/paper/account.py` (`PaperAccount`), both backed by `duckdb.connect()`. |
| Walk-forward backtesting | **Confirmed** | `src/sttbot/backtest/walk_forward.py:167-288`. |
| CLV controls | **Confirmed** | `src/sttbot/backtest/clv.py` implements a *baseline-relative* CLV metric, not just raw CLV (see §7). |
| Drawdown circuit breaker | **Confirmed** | `src/sttbot/risk/circuit_breaker.py:24-81`. |
| No live order-submission path found | **Confirmed** | See row 1. No WebSocket usage anywhere (`grep -rniE "websocket|wss://"` → zero hits). |

---

## 1. Architecture map and data flow

```
Ingestion                    Strategy / pricing              Backtest / Paper Execution        Risk / Monitoring
─────────                    ──────────────────              ───────────────────────────       ──────────────────
venues/polymarket.py   ───►  strategies/prob_arbitrage.py ─┐
 (gamma + CLOB REST,         strategies/dixon_coles(.py|   │
  read-only GET)              _fit.py)                     │
venues/prediction.py         strategies/pead.py            ├─► backtest/walk_forward.py
 (fee models, cross-venue    strategies/earnings_market.py │   backtest/mm_simulator.py
  arb math — no venue I/O)   strategies/market_making.py   │   backtest/clv.py
data/datasets.py             strategies/amm.py              │   backtest/cohort.py
 (cached, licensed CSVs)     strategies/breadth.py          │   backtest/event_study.py
data/football.py             strategies/token_screen.py     │   backtest/metrics.py
data/earnings.py             economics/friction.py          │
data/tokens.py               strategies/base.py (Strategy   │
 (DexScreener/GoPlus)         framework, Param grid search)  │
data/storage.py (TickStore,                                 └─► paper/runner.py (PaperRunner)
 DuckDB tick store)                                               │  ├─ execution/oms.py (OMS protocol,
data/probe.py (network                                            │  │   PaperBroker)
 reachability tool, not                                           │  ├─ execution/order_manager.py
 imported by lib code)                                             │  │   (DynamicOrderManager: TTL + slippage cap)
                                                                    │  ├─ paper/account.py (PaperAccount,
                                                                    │  │   DuckDB fill ledger, replay-based)
                                                                    │  └─ risk/circuit_breaker.py
                                                                    │      (RiskCircuitBreaker: HWM + rolling DD)
                                                                    └─► monitoring/alerts.py (Notifier: webhook,
                                                                        fail-open/fail-safe logging)
```

Entry points wiring this together live in `examples/` (e.g.
`examples/paper_trade_polymarket.py`, `examples/scan_polymarket.py`) and
`scripts/run_paper_trade_polymarket.sh` (a cron/flock wrapper). There is no
`__main__`/daemon inside `src/sttbot` itself — the package is a library, and
`examples/` are the runnable programs (`src/sttbot/__init__.py` was not
separately inspected for a CLI entry point beyond package metadata; `pyproject.toml:1-29`
declares no console-script entry points).

**Data flow for the one strategy actually wired end-to-end (Polymarket
boundary arbitrage):**
1. `iter_events()` paginates the gamma API for the full open market universe (`venues/polymarket.py:71-109`).
2. `complete_baskets()` filters to negRisk (mutually-exclusive/exhaustive) events where every leg is live and two-sided quoted (`venues/polymarket.py:273-308`).
3. Candidates above a paper-edge floor are re-priced against real CLOB depth via `fetch_book()` + `executable_arbitrage()`, which walks the actual ask ladder (`venues/polymarket.py:233-234`, `:326-373`).
4. Confirmed baskets become `Intent`/`Basket` objects (`paper/runner.py:28-68`) and are run through `PaperRunner.run_once()`, which checks the circuit breaker, prices each leg through an injectable fill model, and enforces cash and all-or-nothing basket semantics (`paper/runner.py:169-234`).
5. Fills are appended to `PaperAccount`'s DuckDB-backed, idempotent (`ref`-keyed) ledger (`paper/account.py:184-198`).
6. `monitoring/alerts.py` is available for push notifications but is not invoked by `paper/runner.py` or the example scripts (grep found no import of `Notifier` outside `monitoring/alerts.py` itself and `tests/test_alerts.py`) — **alerting is not wired into the paper-trading loop.**

---

## 2. Strategy modules (what each one does)

| Module | Purpose | Notably rigorous properties (evidence) |
|---|---|---|
| `strategies/dixon_coles.py` + `dixon_coles_fit.py` | Bivariate-Poisson (Dixon–Coles 1997) model for football 1X2 pricing; MLE fit with analytic gradient, time-decay weighting, identifiability constraint (`mean(attack)=0`) | `dixon_coles_fit.py:20-38` explains the identifiability fix; `_objective` (`:183-227`) has a hand-derived analytic gradient rather than finite differences, explicitly for backtest refit speed. |
| `strategies/pead.py` | Post-Earnings-Announcement-Drift: SUE-threshold long/short signal, declared via the `Strategy`/`Param` framework | Minimal; correctness delegated to `data/earnings.py` (`trailing_sue`) and `backtest/event_study.py` for honest entry timing. |
| `strategies/earnings_market.py` | Prices Polymarket "will X beat earnings" binaries via shrinkage toward a cross-sectional base rate (empirical-Bayes-style), with a `skill_vs_base_rate` Brier-skill control | `earnings_market.py:61-99` (shrinkage), `:133-147` (skill vs. base-rate control, explicitly modeled on the CLV-baseline discipline). |
| `strategies/prob_arbitrage.py` | Multi-outcome probability-boundary arbitrage math (sum of asks < 1 / sum of bids > 1), fee-aware | Simple, correctly handles buy-basket vs sell-basket selection (`:45-82`). |
| `strategies/market_making.py` | Avellaneda-Stoikov-style inventory-skewed quoting for binary contracts, fee-aware breakeven-spread and tick-snapping | `breakeven_spread()` prices both legs as **maker** fills, explicitly called out as the correct convention vs. taker (`:57-68`); quotes snap away from mid never inward (`:196-207`). |
| `strategies/amm.py` | Constant-product (Uniswap v2 style) pool mechanics; analytically-derived profit-maximizing CEX-DEX arb size (not solved by price-equality, by `dP/dsize=0`) | `optimal_arbitrage()` derivation documented and cross-checked against brute force in tests (`amm.py:8-23`, confirmed by `tests/test_amm.py`). |
| `strategies/breadth.py` | Portfolio-construction math for many-small-position "breadth" books in meme-coin markets; round-trip cost via full constant-product simulation, Kelly-style breakeven-multiple arithmetic, and an explicitly-labeled Monte Carlo (not a backtest) | Module docstring is unusually candid about which of its own functions are exact math vs. assumption-driven simulation (`breadth.py:15-32`). |
| `strategies/token_screen.py` | Pre-trade rug/honeypot heuristic screen for low-cap tokens; treats missing data as failure, not pass | `screen()` at `:126-220`; explicit design choice that `None` fails rather than defaults (`:56` docstring, `:139-146` `check()` helper). |
| `strategies/base.py` | Declarative `Strategy`/`Param` framework for hyperparameter search (grid enumeration), used by `pead.py` | `base.py:27-119`. |

All strategy math is unit-tested (see §11); the arbitrage and market-making
modules additionally carry adverse-selection/markout accounting in
`backtest/mm_simulator.py`, which is unusual rigor for a market-making
backtest (most naive simulators overstate profitability by filling every
quote against an unreactive path — this one explicitly measures and reports
that bias, `backtest/mm_simulator.py:1-20`).

---

## 3. `venues/polymarket.py` — what it actually calls

- **Gamma API** (`GAMMA_EVENTS_KEYSET = "https://gamma-api.polymarket.com/events/keyset"`, `venues/polymarket.py:41`): used only for enumeration (`iter_events`, `:71-109`). Top-of-book prices carry no size (explicitly noted in the module docstring, `:6-7`).
- **CLOB REST** (`CLOB_BOOK = "https://clob.polymarket.com/book"`, `venues/polymarket.py:42`): used only for `fetch_book()` (`:233-234`), a `GET` of the public order book. **Read-only.**
- **HTTP transport**: `http_fetch()` uses `urllib.request.Request`/`urlopen` from the standard library only — no `py_clob_client`, no `requests` session with auth headers, no API key or wallet header of any kind (`:51-65`).
- **WebSocket usage**: none anywhere in the repository (`grep -rniE "websocket|wss://|socket.io"` returns zero hits).
- **Signed/authenticated order submission**: none anywhere. No `eth_account`, `web3`, private-key handling, or CLOB-signing code exists in the repository (`grep -rniE "eth_account|web3|sign_order|clob_client|private_key|wallet"` finds only comments about *screening* on-chain wallet concentration in `strategies/token_screen.py` and `data/tokens.py` — unrelated to order execution — plus a comment in `examples/breadth_portfolio.py:19` noting that a *live* execution path "needs a funded wallet, signing, slippage-bounded routing and MEV protection" as something **not implemented**).

**Conclusion: `venues/polymarket.py` is exclusively a read-only market-data client.** There is no code path anywhere in the repository capable of submitting a real order to Polymarket, Kalshi, or any other venue.

---

## 4. Fee assumptions, tick sizes, order minimums

- **Fees are parameterized, not hardcoded**, via `FeeModel` (`venues/prediction.py:29-74`), which supports quadratic (Kalshi/Polymarket-shape) and proportional fee components plus a fee cap. Published venue schedules are named constants (`KALSHI_FEES`, `POLYMARKET_FEES`, `POLYMARKET_US_FEES`, `POLYMARKET_CATEGORY_RATES`, `:76-108`) rather than being inlined into strategy logic, and `market_fee_model()` reads the fee schedule **per market** from the live gamma payload rather than assuming a venue-wide constant (`venues/polymarket.py:148-160`), because "the rebate rate is not uniform" (comment at `:150-154`).
- **Tick size is parameterized**: `QuoteParams.tick_size: float = 0.01` (`strategies/market_making.py:124`) is a configurable dataclass field, not a magic number baked into logic; `_to_tick()` (`:196-207`) reads it generically.
- **Order minimums**: `executable_arbitrage()` sizes in a configurable `step` (default `5.0`) with a configurable `max_contracts` (default `20,000.0`) (`venues/polymarket.py:326-333`) — parameterized defaults, not hardcoded inline constants, though the *defaults themselves* (5, 20 000) are not sourced from a published Polymarket minimum-order-size figure anywhere in the repo (**unverified**: no citation to an actual venue minimum was found; this appears to be an engineering choice for step-search granularity, not a modeled venue constraint).
- **Minor hardcoded constant worth flagging**: `MIN_PRICE = 0.01` / `MAX_PRICE = 0.99` in `strategies/market_making.py:41-42` is a magic-number pair, though it is explained in a comment as "venues quote in whole cents ... 0.00/1.00 are settlement values rather than tradeable prices" — a reasonable domain constant, not an unexplained one.

---

## 5. Secrets / credentials scan

`grep -rniE "api[_-]?key|secret[_-]?key|private[_-]?key|BEGIN (RSA|EC|OPENSSH|PGP)|password\s*=|aws_access|0x[a-fA-F0-9]{40}"` across all `*.py`, `*.md`, `*.toml`, `*.txt`, `*.sh`, `*.yml` files found **no embedded secrets, keys, or credentials**. The only matches are:
- `src/sttbot/data/earnings.py:6,32,140,142,145` — references to the **environment variable name** `ALPHAVANTAGE_API_KEY`, read via `os.environ.get(...)` (`:142`), never a literal key value.
- `README.md:900,966` — prose referencing the same environment variable by name.

No `.env` file, no committed credentials file, and `.gitignore` was not inspected for secret-adjacent patterns as part of this scan but no secret material was found in tracked files regardless. **Verdict: clean.**

---

## 6. Paper/live mode gate

There is **no explicit runtime flag** (`PAPER_MODE`, `LIVE_TRADING=1`, etc.) anywhere in the codebase — `grep -rniE "\blive\b|paper_mode|PAPER_TRADING|LIVE_TRADING|is_live_mode"` across `src/` found no such gate.

**This is not a bug — it reflects a stronger property than a runtime flag would provide: there is no live-order-submission implementation to gate.** The `OMS` protocol (`execution/oms.py:20-29`) defines the execution surface, but the only two implementations in the entire repository are:
1. `PaperBroker` (`execution/oms.py:43-117`) — pure in-memory simulation.
2. `_AccountOms` (`paper/runner.py:100-153`) — adapts `PaperAccount` to the same surface, and its `place_limit_order` explicitly raises `NotImplementedError("the paper runner records fills directly")` (`paper/runner.py:119-120`).

Enabling "live" trading would require writing an entirely new adapter (e.g. a signed CLOB REST client) from scratch — there is no dormant or half-built live path, no commented-out live branch, and no config toggle that would activate one. `RiskCircuitBreaker` additionally enforces a hard kill switch on top of this (see §8). The paper account itself refuses to overdraw (`paper/runner.py:163-165`, `:230-233`).

**Confidence: verified.** Absence-of-capability is a stronger safety property than a flag defaulting to "paper," and is exactly what was found.

---

## 7. Backtesting realism

### `backtest/walk_forward.py` (Dixon-Coles / football)
- **Look-ahead bias**: structurally prevented — training window is `[predict_date - train_window_days, predict_date)`, a strict `<` cut (`walk_forward.py:213-215`), re-derived every date, not just at backtest start.
- **Fees/friction modeled**: yes, via `FrictionModel` (`economics/friction.py:15-34`), applied identically to the entry threshold and the P&L accounting so they "can never disagree" (`walk_forward.py:120-131`).
- **Slippage**: modeled implicitly as part of `FrictionModel.slippage` for taker orders (`economics/friction.py:20,25-34`), though for the football backtest specifically the code bets "the pre-match line and settle[s] at that same price" (`walk_forward.py:13-15`) — i.e., no price-impact model for size, appropriate for the fixed-odds-book context it targets but not a depth/partial-fill model.
- **Partial fills / depth**: not modeled in `walk_forward.py` (odds-book betting, not an order book) — this is domain-appropriate, not a defect, given fixed-price bookmaker odds.
- **Survivorship bias**: the module explicitly skips (and counts, `n_skipped_unrated`) matches involving newly-promoted teams with no rating rather than silently dropping them (`walk_forward.py:20-23`, `:230-234`).
- **Chronological split**: yes — `ordered = sorted(matches, key=lambda m: m.date)` (`walk_forward.py:199`) and batches are grouped/iterated by date.
- **Purging**: the strict `date < predict_date` cut (`:213-215`) is a purge in the sense that no training row can share or postdate the prediction date; there is no explicit embargo period between train and predict beyond the day boundary, which is standard for this walk-forward design (not flagged as a gap, since matches on `predict_date` itself are exactly what's being bet on, not leaking future information).

### `backtest/mm_simulator.py` (market-making)
- **Fees**: modeled per-fill via injectable `FeeModel` (`mm_simulator.py:112,129,137`).
- **Slippage / adverse selection**: explicitly and separately measured via post-fill markout (`mean_markout`, `adverse_fill_rate`, `:64-77`) — the module's stated purpose is to prevent the classic MM-backtest failure mode where "a simulator that fills your quote whenever the price touches it, against a path that does not react to your presence ... will report a profit almost regardless of the strategy" (`:1-8`).
- **Partial fills**: **not modeled** — fills are all-or-nothing at `quote.size`, and the module's own docstring flags this as an optimistic assumption ("assumes queue priority and no partial fills ... treat the output as an upper bound," `:18-21`). This is an honest, disclosed limitation rather than a silent one.
- **Depth**: the simulator operates against a single mid-price path, not a full order book — it does not model resting depth from other participants. This is again disclosed as a simplification, not hidden.
- **Look-ahead**: none found — markout is computed strictly forward from each fill's step (`:145-157`).

### CLV baseline discipline (`backtest/clv.py`)
This is a standout piece of statistical hygiene: the module does not report raw CLV as evidence of skill. It computes a **random-selection baseline** (`clv_baseline`, `:56-85`) and an **odds-matched baseline** (`clv_odds_matched_baseline`, `:88-121`, bucketed by implied probability) specifically because "a strategy reporting '+0.007 mean CLV, 64% positive' has therefore demonstrated nothing at all" when best-of-book entry is compared against single-book close (`clv.py:1-18`, citing a measured baseline of +0.0044 to +0.0067 mean CLV / 54–63% positive rate from random selections on the same data). `clv_skill()` (`:195-201`) reports excess-over-baseline as the only number that reflects actual selection skill. **This is a genuine strength** and should be highlighted, not just noted.

### `backtest/cohort.py` (token cohort survivorship handling)
Also a standout: it explicitly separates "no candles" (genuinely never traded), "fetch failed" (pipeline gap), and "bad entry" (unusable price) into different buckets rather than merging them, because "merging them silently deleted a third of one cohort and made the survivors look far better than the population" (`cohort.py:96-105`). It also documents and defends against a specific measured artifact — 70.6% of pools having their maximum high inside the birth candle, a single $0.00074-volume tick producing a 152,363x "peak" — with `skip_first_candle`/`min_peak_volume`/`confirm_peak_fraction` guards, all off by default so raw measurement stays visible (`:143-165`). It also detects wash-trading/clone pools that would otherwise triple a measured hit rate (`detect_clone_groups`, `:242-276`).

---

## 8. Risk controls (`risk/circuit_breaker.py`)

**Implemented:**
- Dual drawdown triggers: high-water-mark drawdown and rolling-window (default 24h) drawdown, either of which trips the breaker (`circuit_breaker.py:24-60`).
- On trip: `cancel_all_open_orders()`, `flatten_all_positions()`, `disable_trading_loop()` are all called (`:73-77`) — and in the paper-trading integration, `flatten_all_positions()` is a *real* implementation that records actual closing fills at supplied marks, not just a flag (`paper/runner.py:100-153`, explicitly commented: "A kill switch that leaves positions open is not a kill switch," `:105-106`).
- **Latching**: once tripped, `evaluate()` unconditionally returns `False` until `reset()` is called manually — a recovering equity curve cannot silently re-enable trading (`circuit_breaker.py:34-44`, `:79-81`).
- Negative-equity guard: `evaluate()` raises on negative equity input (`:41-42`).
- Injectable clock for deterministic testing of the rolling window (`:28`, `:46-70`).

**What's missing / not implemented (repo-wide, not just this module):**
- **No standalone "kill switch" independent of the drawdown breaker** — the only halt mechanism is drawdown-triggered. There is no manual/operator-triggered emergency stop, no external signal (e.g., SIGTERM handler, file-based flag) wired to `disable_trading_loop()`, and no monitoring-driven halt (e.g., data staleness, error-rate, or reconciliation failure triggering a stop). **Severity: medium**, since the only executable strategy today (`paper_trade_polymarket.py`) is a single-tick cron job with no persistent loop to kill mid-flight — the risk surface this would protect against does not yet exist in the wired system, but would be needed before any continuously-running (e.g., market-making) strategy went live.
- **No "cancel-all-on-shutdown" hook** — nothing in `paper/runner.py`, `execution/oms.py`, or the example scripts calls `cancel_all_open_orders()` on process exit/signal. Again, low present-day impact because there is no live order-placement code and the paper runner records fills synchronously rather than leaving resting orders, but this would need to be added before a live or continuously-quoting (market-making) execution path is built.
- **No exposure/position limits at the portfolio level** independent of the circuit breaker — `strategies/breadth.py` has its own per-name and portfolio-cap sizing (`BreadthParams`, `:126-153`) and `strategies/token_screen.py` has `position_limit()` (`token_screen.py:245-263`), but these are strategy-local, not a cross-strategy portfolio risk limit enforced centrally (e.g., in `PaperRunner` or `RiskCircuitBreaker`). `PaperRunner` does enforce a global cash-solvency check (`paper/runner.py:230-233`), which is a real, if narrow, exposure control.
- **No reconciliation logic** — nothing compares the paper account's recorded state against an external source of truth (there is none to reconcile against, since there's no live broker), and no drift-detection exists between, e.g., DuckDB state and a hypothetical exchange fill report. Not applicable today given the read-only architecture, but flagged as a gap that would need filling before any live integration.
- **`monitoring/alerts.py` is not wired into `risk/circuit_breaker.py` or `paper/runner.py`** — a trip event is silent except for the `RunResult.halted` flag returned to the caller (`paper/runner.py:184-190`, `execution/order_manager.py` printed by the example script, `examples/paper_trade_polymarket.py:182-183`). No `Notifier.critical()` call fires on a circuit-breaker trip anywhere in the repo (confirmed by grep: `Notifier`/`webhook` is imported only in its own module and its test file).

---

## 9. Data architecture (`data/storage.py`, `paper/account.py`)

- **`TickStore` (data/storage.py)**: idempotent **schema creation** (`CREATE TABLE IF NOT EXISTS`, `:16-24`), but **no row-level idempotency** — `append()` is a plain multi-row `INSERT` with no dedup key, unique constraint, or upsert logic (`:52-63`). Re-running an ingestion job over the same time range would duplicate rows. **No gap detection** — nothing in `TickStore` checks for missing timestamps in a sequence; `latest()` only returns the most recent row per symbol (`:74-80`). **Timestamp handling**: the schema stores `TIMESTAMP` with no explicit timezone annotation (`_SCHEMA`, `:16-24`), and `Tick.ts` is typed as `Any` with a docstring comment `# datetime` (`:31`) — there is no enforcement that callers pass UTC-aware datetimes, and no test exercises timezone handling for `TickStore` specifically. This is a real gap for a "tick store" whose stated purpose is durable historical data.
- **`PaperAccount` (paper/account.py)**: by contrast, this is well-built. It is **fully idempotent by design** — every `Fill` carries a caller-supplied `ref` key, and `record()` is a no-op (returns `False`, doesn't raise) on a duplicate ref specifically so "a runner that dies after placing a trade but before recording it will retry on restart... and the retry must not double-book" (`paper/account.py:11-18`, `:184-194`). State is **fully derived by replay** from the append-only fill ledger rather than mutated in place — cash, positions, and P&L are all recomputed from `fills()` (`:219-232`), explicitly to avoid the "running balance disagrees with the ledger after a crash" failure mode (`:1-9`). Timestamps use `dt.datetime.now(dt.timezone.utc)` as the default clock (`account.py:239`, `:166`) — **UTC-aware**, unlike `TickStore`.
- **Table-name injection guard**: `TickStore.__init__` validates `table.isidentifier()` before interpolating it into DDL/DML, specifically to prevent untrusted input from reaching raw SQL (`storage.py:46-47`) — good practice, though the table name isn't attacker-controlled in any current call site.

---

## 10. Observability (`monitoring/alerts.py`)

- **What it alerts on**: nothing automatically — `Notifier` is a generic three-level (`INFO`/`WARNING`/`CRITICAL`) webhook+log sender (`alerts.py:22-65`) that must be called explicitly by application code. As noted in §8, **no call site in the strategy, execution, risk, or paper-trading code actually invokes it** — it exists as infrastructure but is not wired into the risk or execution loop.
- **Fail-safe design**: webhook failures are caught and logged, never raised, so "a monitoring outage can never crash the trading loop" (`alerts.py:1-8`, `:52-56`) — a good defensive property for the code that *is* there.
- **Structured logging**: uses Python's standard `logging` module with a named logger (`sttbot.alerts`, `alerts.py:19`) and level mapping (`:68-72`), but every other module in the repo (`paper/runner.py`, `execution/order_manager.py`, `risk/circuit_breaker.py`) uses `print()` in the example scripts rather than the `logging` module for operational output (e.g., `examples/paper_trade_polymarket.py:122-185`) — there is no structured (JSON) logging anywhere, and no correlation IDs tying a log line to a specific run/basket/fill beyond the human-readable `note`/`rationale` fields on `Fill`/`Basket`.
- **Sequence-gap detection**: none found. `PaperAccount` does maintain a monotonic `seq` column (`account.py:162`, `:206-208`) for fill ordering, but nothing checks for or alerts on gaps in a data feed (e.g., missed gamma pagination pages, stale order-book fetches). `venues/polymarket.py`'s `iter_events()` does defend against one specific gap-adjacent failure mode — a repeating pagination cursor is treated as a termination signal rather than looping forever (`:104-107`) — but this is pagination-loop protection, not feed-gap alerting.

---

## 11. Test coverage

All 484 tests pass (`pytest`, run locally in this audit). Coverage mapping (module → dedicated test file):

**Covered**, with the test asserting real behavior via injected fakes (no real network calls anywhere in `tests/` — confirmed via grep for `urlopen`/`requests` in test code, zero hits outside fixture definitions):
`backtest/clv.py`→`test_clv_baseline.py`; `backtest/cohort.py`→`test_cohort.py`; `backtest/mm_simulator.py`→`test_mm_simulator.py`; `backtest/walk_forward.py`→`test_walk_forward.py`; `backtest/event_study.py`→covered by `test_earnings.py` (imports `sttbot.backtest.event_study`, `tests/test_earnings.py:13`); `data/datasets.py`→`test_datasets.py`; `data/earnings.py`→`test_earnings.py`; `data/storage.py`→`test_storage.py`; `data/tokens.py`→`test_tokens.py`; `economics/friction.py`→`test_friction.py`; `execution/oms.py` & `execution/order_manager.py`→`test_execution.py` (8 tests covering slippage sign, marketable fills, TTL expiry, disabled-trading rejection); `monitoring/alerts.py`→`test_alerts.py`; `paper/account.py`→`test_paper_account.py`; `paper/runner.py`→`test_paper_runner.py`; `risk/circuit_breaker.py`→`test_circuit_breaker.py`; every `strategies/*.py`→one test file each; `venues/polymarket.py`→`test_polymarket.py` (uses injected `fetch=fake` throughout, e.g. `tests/test_polymarket.py:76,94,103,108`); `venues/prediction.py`→`test_prediction_venues.py`.

**Not covered by a dedicated test file:**
- `backtest/metrics.py` — no `test_metrics.py`, but its public functions (`summarise`, `sharpe_ratio`, `max_drawdown`) are exercised indirectly through `test_walk_forward.py` and `test_circuit_breaker.py`/`test_paper_runner.py`/`test_cohort.py` call sites — **not a coverage gap in practice**, but there is no unit test isolating edge cases of `sharpe_ratio`/`max_drawdown` themselves (e.g., zero-variance returns, single-observation series) independent of a full backtest.
- `data/football.py` — **correction (post-audit verification): this claim was wrong.** `data/football.py` has no dedicated `test_football.py`, but it is thoroughly covered *indirectly* inside `tests/test_datasets.py` (`_FDUK_MODERN`/`_FDUK_LEGACY` fixtures, `tests/test_datasets.py:185-270`), including exactly the two failure modes flagged above as untested: `test_fduk_loader_parses_two_digit_years` regression-tests the "14/08/10" → year-10-AD bug directly, and `test_fduk_loader_tolerates_a_non_numeric_odds_cell` regression-tests the COALESCE/TRY_CAST division-killing bug. `load_matches` (the bulk-dataset loader) is separately covered at `tests/test_datasets.py:156-176`. **Not a coverage gap.** (Original finding retracted; left here rather than deleted so the correction is auditable.)
- `data/probe.py` — explicitly and deliberately excluded from the test suite by design ("Nothing here is imported by library code, and no test invokes it — the test suite stays hermetic," `probe.py:9-10`). Not a gap; a documented, reasonable exclusion for a manual network-reachability tool.

**Tests that could submit real orders**: none. Verified — no test file imports or exercises a live-order-submission path (none exists to import, per §3/§6), and `test_polymarket.py`/`test_prediction_venues.py` inject fake `fetch` callables rather than hitting real endpoints (`tests/test_polymarket.py:76` et al.).

---

## 12. CI (`.github/workflows/ci.yml`)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .[dev]
      - run: pytest
```
(`.github/workflows/ci.yml:9-21`)

**What it runs**: `pytest` only, on a single Python version (3.11), on `push` to `main` and on every `pull_request` (`:3-7`).

**What it does not run**: no linter (no `ruff`/`flake8`/`pylint` step), no type checker (no `mypy`/`pyright` step despite the codebase using type hints extensively throughout), no coverage reporting/threshold, no dependency-vulnerability scan, and no matrix testing across Python versions. **Severity: medium** — for a codebase this statistically careful, the absence of `mypy` in CI is a missed low-cost win given how much of the correctness argument rests on type-annotated `Protocol`s (`execution/oms.py:20-29`, `backtest/walk_forward.py:43-62`) that are never actually type-checked.

---

## 13. Obsolete / stale dependencies (checked against PyPI's current index, Aug 2026)

`pyproject.toml:12-17` and `requirements.txt` both pin **floors only**, no upper bounds, no lockfile:

| Package | Repo floor | Latest on PyPI (as probed) | Note |
|---|---|---|---|
| numpy | `>=1.26` | 2.4.6 | Floor predates numpy 2.0's breaking C-API/ABI changes; unpinned upper bound means CI silently picks up numpy 2.x today, which is fine only because nothing in the codebase appears to hit removed numpy 1.x APIs (tests currently pass) — but the floor itself gives no signal that this was verified deliberately. |
| duckdb | `>=0.10` | 1.5.5 | Floor is nearly two major-version lines behind current; DuckDB's 1.0 release included breaking changes to some APIs. Since `requirements.txt`/`pyproject.toml` set no ceiling, CI resolves to the latest (1.5.x) automatically, so this is a **stale floor**, not a stale *running* version — but it means a fresh `pip install duckdb==0.10` (i.e., pinning to the stated floor) would very likely fail or behave differently than what CI actually tests. |
| pandas | `>=2.0` | 3.0.5 | Same pattern — pandas 3.0 is a major release with removals; floor doesn't reflect it. |
| scipy | `>=1.11` | 1.17.1 | Least concerning of the four; no major-version jump since the floor. |
| pytest | `>=8.0` (dev) | 9.1.1 | Minor concern only. |

**No lockfile** (no `requirements-lock.txt`, no `poetry.lock`, no `uv.lock`) and **no upper-bound pins** exist anywhere in the repo. Combined with a CI job that always installs "whatever is latest today," this means: (a) the stated floors in `pyproject.toml` are not an accurate description of what is actually tested (CI has been running against much newer majors of numpy/duckdb/pandas than the floors state), and (b) there is no reproducibility guarantee — a build today and a build run against the same commit in a year could resolve to materially different dependency versions with no way to reproduce the original test environment. **Severity: medium.** This is a supply-chain/reproducibility issue, not a functional obsolescence issue — the tests do pass against current dependency versions (verified in this audit), so nothing here is actually broken today.

---

## 14. Verified strengths (explicit, not manufactured)

This codebase shows genuine, unusual statistical discipline for its size. Concretely, and each independently verified above:

1. **CLV-baseline discipline** (`backtest/clv.py`) — refuses to report raw CLV as evidence of skill; computes and requires beating a measured random-selection baseline before calling anything "edge." This is a level of rigor most quant codebases of this size do not have.
2. **Cohort survivorship handling** (`backtest/cohort.py`) — separates "never traded," "fetch failed," and "bad entry" into distinct buckets specifically because merging them previously and measurably distorted a real cohort's survival rate; defends against documented, specific artifacts (birth-candle wick inflation, wash-trading clone pools) with off-by-default guards that keep raw measurement visible.
3. **Walk-forward with structural no-look-ahead** (`backtest/walk_forward.py`) — the training cut is a strict date inequality re-derived every predict-date, not a single train/test split, and unrated (promoted) teams are counted rather than silently dropped.
4. **Adverse-selection-aware market-making backtest** (`backtest/mm_simulator.py`) — explicitly measures post-fill markout rather than only reporting gross spread capture, specifically to catch the standard failure mode where an MM backtest looks profitable purely because it fills against a non-reactive price path.
5. **Idempotent, replay-derived paper ledger** (`paper/account.py`) — `ref`-keyed fill dedup plus recompute-don't-mutate account state design, correctly reasoned as the way to avoid a running-balance/ledger disagreement after a crash.
6. **Real kill-switch semantics** (`risk/circuit_breaker.py` + `paper/runner.py`'s `_AccountOms`) — tripping the breaker actually closes positions at real marks rather than merely flipping a flag, and latches until manual reset.
7. **Read-only-by-construction venue client** (`venues/polymarket.py`) — the absence of any live-order path is not a flag defaulting to safe; it is the simple fact that no such code was ever written, which is a stronger safety property than a runtime gate.
8. **Honest scope-labeling throughout** — multiple modules (`strategies/breadth.py`, `backtest/cohort.py`, `backtest/mm_simulator.py`) explicitly distinguish "this is exact math," "this is a Monte Carlo over an assumption," and "this is an achievable vs. an unachievable upper bound" in their own docstrings, rather than presenting simulation output as backtested results.

---

## 15. Prioritized remediation table

| Issue | File:line | Severity | Confidence | Remedy | Required validation |
|---|---|---|---|---|---|
| ~~No test coverage for `data/football.py`~~ — **retracted**: already covered by `tests/test_datasets.py:185-270` (two-digit-year and bad-cell regressions both present) | `src/sttbot/data/football.py:119-126`, `:144-164` | N/A | Retracted post-verification | None needed | Confirmed by reading `tests/test_datasets.py` directly |
| `TickStore.append()` has no row-level idempotency (no dedup/unique key) | `src/sttbot/data/storage.py:52-63` | Medium | Verified | Add a caller-supplied idempotency key (mirroring `PaperAccount.Fill.ref`) or a `(ts, symbol)` unique constraint with `INSERT ... ON CONFLICT DO NOTHING` | Test that re-ingesting the same batch of ticks twice does not duplicate rows |
| `TickStore` has no explicit UTC enforcement or gap detection | `src/sttbot/data/storage.py:16-24, 31` | Low | Verified | Type `Tick.ts` as `dt.datetime` with a runtime check for `tzinfo is not None`/UTC; add a `gaps(symbol, expected_interval)` query helper | Test naive-datetime rejection; test gap detection against a synthetic series with a missing interval |
| ~~Circuit-breaker trip / halted run produces no alert~~ — **fixed in this pass**: `RiskCircuitBreaker` now takes an optional `notifier: Notifier`, calls `notifier.critical(...)` from `_trip()`, and `examples/paper_trade_polymarket.py` wires one from `ALERT_WEBHOOK_URL` | `src/sttbot/risk/circuit_breaker.py`, `examples/paper_trade_polymarket.py` | Medium | Fixed | Done | `tests/test_circuit_breaker.py::test_drawdown_trip_pages_notifier`, `::test_no_notifier_is_silent_by_default` |
| CI runs `pytest` only — no linter, no type checker, despite pervasive type hints and `Protocol` usage | `.github/workflows/ci.yml:16-21` | Medium | Verified | Add `ruff check` and `mypy src/` steps to CI | CI green on a clean run; fix any surfaced type errors |
| No dependency lockfile / upper bounds; floors (`numpy>=1.26`, `duckdb>=0.10`, `pandas>=2.0`) are 1-2 major versions behind what CI actually installs | `pyproject.toml:12-17`, `requirements.txt:1-5` | Medium | Verified | Add a lockfile (`pip-compile`/`uv lock`) pinned to tested versions, or at minimum bump floors to reflect versions actually exercised by CI and add sane upper bounds | Re-run full test suite pinned to the new lock; confirm CI reproduces the same resolution across runs |
| ~~No standalone kill switch independent of the drawdown breaker~~ — **partially fixed in this pass**: added `RiskCircuitBreaker.trip_manually(oms, reason)`, which flattens/cancels/disables and pages the same way a drawdown trip does. Still open: nothing in `examples/` calls it yet — no signal handler, sentinel-file check, or data-staleness/reconciliation hook invokes it, so it exists as a capability but is not wired into any run loop | `src/sttbot/risk/circuit_breaker.py` | Low (was Medium) | Fixed (capability) / Strongly inferred (wiring still open) | Wire a SIGTERM/SIGINT handler or sentinel-file check in a persistent runner, once one exists, to call `trip_manually` | `tests/test_circuit_breaker.py::test_manual_trip_flattens_and_pages` |
| No cancel-all-on-shutdown hook | `src/sttbot/execution/oms.py`, `src/sttbot/paper/runner.py` (module-wide absence) | Low (today), would become Medium/High before live/continuous execution | Strongly inferred | Register a signal handler (SIGTERM/SIGINT) calling `cancel_all_open_orders()` before any persistent execution loop is introduced | N/A until a persistent (non-single-tick) execution loop exists; add integration test once it does |
| No portfolio-level exposure limit enforced centrally across strategies (only per-strategy sizing exists) | `src/sttbot/strategies/breadth.py:126-153`, `src/sttbot/strategies/token_screen.py:245-263` (local only); no cross-strategy equivalent in `paper/runner.py` or `risk/circuit_breaker.py` | Low | Strongly inferred | Add an aggregate exposure/position-count cap to `PaperRunner` or `RiskCircuitBreaker`, evaluated across all open positions regardless of originating strategy | Test that total exposure across two simultaneously-run strategies is capped |
| Executable-arbitrage sizing defaults (`step=5.0`, `max_contracts=20_000.0`) are not sourced from a documented Polymarket minimum-order-size figure | `src/sttbot/venues/polymarket.py:326-333` | Low | Unverified | Confirm actual Polymarket minimum order size/increment from current API docs and either cite it in a comment or parameterize per-market from the market payload if the venue exposes it | Cross-check against Polymarket's published CLOB minimum order size at time of any live integration |
| Executable-arbitrage sizing / examples default `MAX_CANDIDATES=10`, `PAPER_EDGE_FLOOR=0.002` in `examples/paper_trade_polymarket.py` are un-parameterized magic numbers at the script level (not the library level) | `examples/paper_trade_polymarket.py:46-51` | Low | Verified | Low priority — these are example-script constants, not library defaults; document provenance or move to CLI flags if the script becomes a long-running operational tool | N/A |

---

## Appendix: files read in full for this audit

`pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`, `README.md` (headings scanned; key sections spot-checked), `src/sttbot/venues/polymarket.py`, `src/sttbot/venues/prediction.py`, `src/sttbot/execution/oms.py`, `src/sttbot/execution/order_manager.py`, `src/sttbot/risk/circuit_breaker.py`, `src/sttbot/paper/account.py`, `src/sttbot/paper/runner.py`, `src/sttbot/data/storage.py`, `src/sttbot/monitoring/alerts.py`, `src/sttbot/backtest/walk_forward.py`, `src/sttbot/backtest/mm_simulator.py`, `src/sttbot/backtest/clv.py`, `src/sttbot/backtest/cohort.py`, `src/sttbot/backtest/event_study.py`, `src/sttbot/backtest/metrics.py`, `src/sttbot/economics/friction.py`, `src/sttbot/strategies/dixon_coles.py`, `src/sttbot/strategies/dixon_coles_fit.py`, `src/sttbot/strategies/pead.py`, `src/sttbot/strategies/earnings_market.py`, `src/sttbot/strategies/prob_arbitrage.py`, `src/sttbot/strategies/market_making.py`, `src/sttbot/strategies/amm.py`, `src/sttbot/strategies/breadth.py`, `src/sttbot/strategies/token_screen.py`, `src/sttbot/strategies/base.py`, `src/sttbot/data/datasets.py`, `src/sttbot/data/football.py`, `src/sttbot/data/earnings.py`, `src/sttbot/data/tokens.py`, `src/sttbot/data/probe.py`, `examples/paper_trade_polymarket.py`, `examples/scan_polymarket.py`, `scripts/run_paper_trade_polymarket.sh`, `tests/helpers.py`, `tests/test_execution.py`, plus full-repo `grep` sweeps for secrets, live-trading gates, WebSocket usage, and signing/wallet code, and a full local `pytest` run (484 passed).
