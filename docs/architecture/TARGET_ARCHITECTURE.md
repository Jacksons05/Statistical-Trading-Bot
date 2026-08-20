# Target architecture

This maps the 9-phase target from the assignment brief onto what
`sttbot` actually has today (per `docs/audits/POLYMARKET_BOT_AUDIT.md`) versus
what's genuinely missing. It is intentionally not a rewrite plan: most of the
target is already built, and re-architecting working, tested modules to match
a generic template would be a regression, not an improvement.

| Target capability | Status | Where |
|---|---|---|
| Paper-mode-only, no live path | **Done, structurally** | No live-order code exists anywhere (audit §3, §6) — a stronger guarantee than a runtime gate. |
| Depth-aware, fee-aware execution | **Done** | `venues/polymarket.py:executable_arbitrage` walks real ask depth; `economics/friction.py`, `venues/prediction.py:FeeModel` price fees per-market, not a flat constant. |
| Combinatorial / logical relative value | **Partially done** | `strategies/prob_arbitrage.py` implements the boundary case (sum of outcome prices vs. 1) with fee-aware sizing — the single most common combinatorial structure (exhaustive, mutually-exclusive negRisk events). **Not built**: a general constraint graph handling implication, equivalence, nested thresholds, or temporal containment between *different* markets/events, with provenance and confidence on each edge. See "Deferred: general constraint graph" below. |
| Calibrated probability estimation | **Partially done** | `strategies/earnings_market.py` does shrinkage-toward-base-rate with a Brier-skill control — this *is* the calibration discipline the brief asks for, just scoped to one market type. `strategies/dixon_coles.py` is a full probability model for another. No cross-strategy calibration harness (reliability diagrams, ECE) exists yet. |
| Cross-venue lead-lag | **Not built** | `venues/prediction.py` does cross-venue *fee/arb* math (Kalshi vs. Polymarket) but nothing aligns event/exchange/receive timestamps or estimates a lead-lag relationship against an external reference feed (Binance, Chainlink). See "Deferred: lead-lag" below. |
| Inventory-aware market making | **Done** | `strategies/market_making.py` (Avellaneda-Stoikov skew, fee-aware breakeven spread, one-sided quoting at inventory limits) + `backtest/mm_simulator.py` (markout-based adverse-selection measurement, explicitly not fill-everything). |
| Event-time data layer | **Partially done** | `paper/account.py` is immutable/idempotent/replay-derived — a real strength. `data/storage.py` (`TickStore`) lacks row-level idempotency and gap detection (audit §9, fixed-priority item). No WebSocket/incremental-book capture exists (nothing needs it yet — `venues/polymarket.py` is REST-only by design). |
| Event-driven execution simulator | **Done for the modeled domains** | `backtest/mm_simulator.py` and `backtest/walk_forward.py` both explicitly reject midpoint/unlimited-liquidity/instantaneous-fill assumptions and document their own simplifications (no partial fills in the MM sim, disclosed as an upper bound). |
| Experiment registry | **Now built (this pass)** | `sttbot.research.experiment_log` — see `docs/research/EXPERIMENT_PROTOCOL.md`. |
| Risk engine | **Mostly done** | `risk/circuit_breaker.py`: dual drawdown triggers, latching, real flatten-on-trip, now paging via `Notifier`, now has a manual `trip_manually()` kill switch. **Still open**: no signal-handler/sentinel-file wiring of the manual trip into any run loop (there's no persistent loop yet — the one wired strategy is a single-tick cron job); no cross-strategy portfolio exposure cap. |
| Observability | **Partially done** | `monitoring/alerts.py` is fail-safe and now wired into the circuit breaker. Still no structured (JSON) logging, no sequence-gap/feed-staleness alerting — not yet needed because there's no persistent feed consumer. |

## Deferred: general constraint graph

The brief asks for a formal constraint graph over arbitrary logical relations
(implication, equivalence, nested thresholds, temporal containment) across
markets, with human-readable provenance per edge and an optimizer (LP/MIP)
evaluating terminal payoff under every outcome state.

**Why this is deferred rather than built in this pass:** `prob_arbitrage.py`
already covers the highest-value, lowest-risk case — exhaustive/mutually
exclusive outcome sets within one Polymarket negRisk event, which is exactly
the structure Polymarket itself flags via `negRiskMarketID` in the gamma API
(the field `complete_baskets()` groups on, `venues/polymarket.py:273-308`).
The literature review (`docs/research/POLYMARKET_QUANT_RESEARCH.md`) found
that genuinely cross-event combinatorial opportunities (e.g. "candidate A
wins primary" implying "party X wins general") are both rarer and harder to
resolve unambiguously — the resolution-rule risk the brief itself calls out
("refuses execution on ambiguous resolution rules") is real: two markets that
look logically linked can have subtly different resolution criteria, and
building an automated implication-inference engine without a human-verified
relationship registry is how a bot ends up "arbitraging" two markets that
don't actually pay off the same way.

**Extension path**, in order of value-to-risk:
1. A small `Relationship` type (`kind: complement|exhaustive|mutex|implies|equivalent`,
   `markets: tuple[str, ...]`, `confidence: Literal["verified","inferred"]`,
   `rationale: str`, `source: str`) — a data structure, not an inference
   engine. Populate it by hand for a short list of manually verified pairs
   before ever automating discovery.
2. A payoff-enumeration function: given a `Relationship` and live executable
   prices/depth/fees per leg (already available from `fetch_book()` +
   `executable_arbitrage()`), evaluate net payoff under every valid outcome
   vector and reject if any outcome is negative net of fees — this is a
   straightforward extension of `find_boundary_arbitrage`'s existing logic to
   `implies`/`equivalent` relationships, not a new optimizer.
3. Only once (1) and (2) exist and are tested: consider LP/MIP for portfolios
   with more legs than can be enumerated by hand. Given Polymarket negRisk
   events cap in the tens of outcomes, brute-force enumeration will likely
   suffice indefinitely and an LP solver would be premature complexity.

## Deferred: cross-venue lead-lag

**Why this is deferred:** it requires infrastructure this repo does not yet
have at all — a synchronized reference feed (Binance/Chainlink) with measured
clock drift and latency, and no such ingestion path exists in `data/` today.
Building it without first measuring feed latency/drift would produce a
lead-lag "signal" indistinguishable from a timestamp bug, and the assignment
brief itself is explicit not to assume latency arbitrage is feasible absent
measured evidence. The research doc's finding that on-chain `OrderFilled`
events disagree with book-inferred trade direction ~40% of the time
(`docs/research/POLYMARKET_QUANT_RESEARCH.md`) raises the bar further: a
lead-lag study built on the wrong ground truth would produce a confidently
wrong result, which is worse than no result.

**Extension path**: (1) build and test the timestamp-alignment/clock-drift
measurement layer in isolation, with no strategy logic attached, and publish
the measured latency distribution *before* writing any signal code; (2) only
if that shows a persistently exploitable latency margin, build the
lead-lag estimator out-of-sample per the walk-forward discipline already used
elsewhere in this repo (`backtest/walk_forward.py`).
