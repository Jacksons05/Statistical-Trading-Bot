# Live-readiness checklist

**Current status: not ready for live capital, by construction.** This is not
a configuration state — there is no code anywhere in this repository capable
of submitting a signed order to Polymarket, Kalshi, or any other venue
(verified independently in `docs/audits/POLYMARKET_BOT_AUDIT.md` §3, §6: no
`py_clob_client`/`py-sdk` dependency, no `eth_account`/`web3`, no wallet or
signing code, and the only two `OMS` implementations in the codebase are
paper simulations, one of which raises `NotImplementedError` on
`place_limit_order`). Going live requires *writing* a live adapter, not
flipping a flag.

This checklist is for when that changes. Nothing below has been done yet;
each item states what would need to exist and how it would be verified.

## Blocking prerequisites (must all be true before any live order)

- [ ] **A live venue adapter exists and is tested against a testnet/sandbox
      or paper-money account**, using the *current* official SDK
      (`Polymarket/py-sdk` — the legacy `py-clob-client` is confirmed
      archived as of May 2026 per `docs/research/POLYMARKET_QUANT_RESEARCH.md`;
      do not build against it). Verification: integration tests against a
      recorded fixture, plus at least one manual sandbox run with a human
      watching.
- [ ] **Explicit two-stage live-mode gate**: an environment variable or config
      flag that must be set (e.g. `STTBOT_LIVE_TRADING=1`) *and* a separate
      runtime confirmation (e.g. an interactive prompt or a second explicit
      flag) before any order-submission code path is reachable. Verification:
      a test asserting the live path is unreachable with either gate absent.
- [ ] **Kill switch wired into the actual run loop**, not just available as a
      method. `RiskCircuitBreaker.trip_manually()` exists as of this pass but
      nothing calls it — a `SIGTERM`/`SIGINT` handler (or equivalent) must
      call it before any persistent live loop ships. Verification: integration
      test sending a signal to a running process and confirming orders stop.
- [ ] **Cancel-all-on-shutdown**: confirmed to actually cancel resting orders
      at the venue, not just locally. Verification: sandbox test placing an
      order, killing the process, confirming no resting order remains venue-side.
- [ ] **Portfolio-level exposure limits enforced centrally** (not just
      per-strategy), across every strategy that could run concurrently.
      Verification: test that combined exposure across two simultaneously
      active strategies is capped by a single limit.
- [ ] **Reconciliation**: local ledger state compared against venue-reported
      fills/positions, with a defined action (halt + alert) on mismatch.
      Verification: fixture test injecting a deliberate mismatch and
      confirming the halt fires.
- [ ] **Alerting reaches a human reliably**: `Notifier` is wired into the
      circuit breaker as of this pass, but delivery has only been tested
      against a webhook stub, never a real Discord/Telegram endpoint in this
      environment. Verify with a real webhook before relying on it.
- [ ] **Funded capital limits**: a hard ceiling on total capital the live
      adapter can ever deploy, enforced in code (not just documented),
      independent of and in addition to the paper-account cash-solvency
      check that already exists.
- [ ] **Dependency reproducibility**: a lockfile pinning exact tested
      versions (flagged as missing in the audit — floors-only pins today mean
      "what CI tested" and "what pyproject states" can silently diverge).
- [ ] **CI includes lint at minimum** (added this pass); a full `mypy` pass
      is not yet clean (31 errors found across 15 files as of this audit,
      several in live pricing paths like `strategies/market_making.py` and
      `strategies/pead.py`) — these should be resolved and mypy added to CI
      before trusting the type-annotated `Protocol`s that the execution
      surface (`execution/oms.py:OMS`) depends on.

## Evidence-of-edge prerequisites (must all be true before live *sizing* beyond nominal)

- [ ] **At least one strategy has cleared a locked, never-previously-viewed
      holdout period** with the conservative execution-simulation assumptions
      (per `docs/research/EXPERIMENT_PROTOCOL.md`), not just an in-sample or
      repeatedly-revisited validation window.
- [ ] **The experiment registry (`sttbot.research.experiment_log`) has enough
      logged trials** for a Deflated Sharpe Ratio / Probability of Backtest
      Overfitting calculation to be more than noise — not computed from zero
      or a handful of trials, which was explicitly avoided in this pass
      (`docs/research/EXPERIMENT_PROTOCOL.md`).
- [ ] **Capacity has been measured, not assumed** — `strategies/breadth.py`'s
      Monte Carlo is explicitly labeled assumption-driven, not a backtest;
      an actual capacity curve (P&L vs. deployed size, accounting for
      Polymarket's confirmed-rare/short-lived complement arbitrage and
      capacity-constrained combinatorial opportunities per the research doc)
      does not yet exist.
- [ ] **Paper-traded track record exists and is long enough to be more than
      one lucky/unlucky run** — the paper-trading cron job
      (`scripts/run_paper_trade_polymarket.sh`) is running against real
      Polymarket prices as of this pass; it has not yet accumulated a track
      record long enough to draw a conclusion from.

## Explicit non-goals of this checklist

This is not a request to build the above now. Per the assignment's own
priority order, correctness and paper-trading infrastructure come before
live capital, and several of the audit's flagged gaps (no persistent
execution loop exists yet, so no kill-switch wiring or cancel-all-on-shutdown
target exists to wire into) are genuinely lower priority *today* than they
will be once a continuously-running strategy is built. Treat each unchecked
box as a precondition, not a todo list to race through.
