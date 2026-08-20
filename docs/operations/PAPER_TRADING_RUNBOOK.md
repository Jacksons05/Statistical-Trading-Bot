# Paper trading runbook

Scope: the one strategy currently wired end-to-end into durable paper
trading — Polymarket negRisk boundary arbitrage
(`examples/paper_trade_polymarket.py`, run on a schedule by
`scripts/run_paper_trade_polymarket.sh`). Everything below is paper-only:
fake money, real prices, a real DuckDB ledger. There is no live-order code
in this repository to accidentally trigger (audit §3, §6).

## Running it

```bash
pip install -r requirements.txt && pip install -e .

# one tick: scan Polymarket, size against live depth, record fills
python examples/paper_trade_polymarket.py

# optional: page a Discord/Telegram webhook when the circuit breaker trips
export ALERT_WEBHOOK_URL="https://discord.com/api/webhooks/..."
python examples/paper_trade_polymarket.py

# replay against a captured snapshot instead of hitting the live API
python examples/paper_trade_polymarket.py --cache events.jsonl
```

It is safe to re-run repeatedly (cron, `watch`, a systemd timer) — fills are
idempotent on a caller-supplied `ref`, and the runner does not re-enter a
basket the account already holds a leg of.

## What "healthy" looks like

- `RunResult.summary()` prints fills, rejected-basket count, and equity each
  tick. Rejections are normal — most scanned baskets fail the executable-depth
  or cash checks and that's the fill model working, not a bug.
- The circuit breaker (`RiskCircuitBreaker`, 5% high-water-mark and rolling
  drawdown by default) should almost never trip in paper mode against a
  measured, tiny, fee-aware edge. A trip is a signal to look at the ledger,
  not to just call `reset()`.
- If `ALERT_WEBHOOK_URL` is set, a trip pages the configured channel with the
  drawdown or manual reason. With no webhook set, trips are still logged
  (Python `logging`, logger name `sttbot.alerts`) but nothing pages anyone —
  check logs, or set the webhook, before relying on this unattended.

## Circuit-breaker trip

1. Confirm the trip reason: `breaker.trip_reason` (drawdown magnitude, or
   `"manual: <reason>"` if triggered via `trip_manually()`).
2. Check `PaperAccount.open_positions()` and recent fills — the breaker's
   `_trip()` already calls `flatten_all_positions()`, so open exposure should
   already be closed at the marks supplied to that run. Positions with no
   available mark are left open and reported (`_AccountOms.flatten_all_positions`,
   `paper/runner.py`) — check for those explicitly.
3. Do not call `breaker.reset()` until the cause is understood. The breaker
   is deliberately latched (audit §8) — a recovering equity curve does not
   self-heal it.
4. After `reset()`, the next `run_once()` re-evaluates equity from the
   current snapshot; there's no separate "resume" step.

## Data staleness / no fills for an extended period

There is currently no automated staleness circuit breaker (see
`docs/audits/POLYMARKET_BOT_AUDIT.md` remediation table — this is an
explicit, documented gap, not a silent one). If the gamma/CLOB endpoints
become unreachable, `examples/paper_trade_polymarket.py` will simply find no
candidates and report zero fills; check `python -m sttbot.data.probe` to
re-measure host reachability before assuming a strategy problem rather than
a network one.

## Shutdown

There is no persistent process to shut down today — each invocation is a
single tick that exits when done. If a persistent/continuously-running
strategy (e.g. market making) is wired up later, it must call
`RiskCircuitBreaker.trip_manually(oms, "shutdown")` (or equivalent) from a
`SIGTERM`/`SIGINT` handler before exiting — this does not exist yet and is
flagged in the audit as required before any continuous execution loop ships.

## Database

`paper_polymarket.duckdb` (default path, override via CLI) is the entire
source of truth: cash, positions, and P&L are all recomputed from the
append-only `fills` table, never stored as a running balance
(`paper/account.py`). Back it up like you'd back up any ledger — deleting it
loses the paper track record, not just a cache.
