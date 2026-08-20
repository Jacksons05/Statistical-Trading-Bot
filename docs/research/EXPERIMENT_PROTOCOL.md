# Experiment protocol

## What already enforces validity (before any registry existed)

Most of the statistical-validity discipline the brief asks for was already
built into the backtest modules themselves, not bolted on afterward:

- **No-look-ahead by construction**: `backtest/walk_forward.py` re-derives a
  strict `date < predict_date` training cut on every prediction date, rather
  than a single train/test split (audit §7).
- **Baseline-relative skill, not raw metrics**: `backtest/clv.py` requires
  beating a *measured* random-selection and odds-matched baseline before
  reporting anything as "edge" — raw CLV was found to average +0.004 to
  +0.007 from **random** selections on the same data, which is the whole
  reason the baseline exists (`backtest/clv.py:1-18`).
- **Survivorship handling with disclosed artifacts**: `backtest/cohort.py`
  separates "never traded" / "fetch failed" / "bad entry" rather than merging
  them, and documents a specific measured artifact (70.6% of pools showing
  their peak inside the birth candle) with off-by-default guards so raw
  measurement stays visible rather than silently cleaned.
- **Adverse-selection-aware market-making backtest**: `backtest/mm_simulator.py`
  measures post-fill markout specifically to avoid the standard failure mode
  where a naive MM backtest profits purely because it fills against a
  non-reactive price path.

These are strategy-specific implementations of walk-forward, purging, and
baseline discipline. What was missing was a place to **record every trial**
so that discipline can be checked in aggregate later (multiple-testing
correction, DSR, PBO) rather than only within one strategy's own backtest run.

## The experiment registry

`sttbot.research.experiment_log` (`src/sttbot/research/experiment_log.py`)
adds exactly that: an append-only, newline-delimited JSON log. One
`ExperimentRecord` per trial, written *before* the result is known to be
good or bad — logging only the wins is how an honest walk-forward discipline
still ends up p-hacked at the strategy-selection level.

```python
from sttbot.research.experiment_log import ExperimentRecord, append_experiment

record = ExperimentRecord(
    strategy="pead",
    hypothesis="SUE >= 2.0 micro-caps drift positively over 5 sessions",
    dataset_version="alphavantage-earnings-2026-08-01",
    train_interval="2019-01-01..2023-12-31",
    validation_interval="2024-01-01..2024-12-31",
    test_interval="2025-01-01..2025-12-31",
    parameters={"sue_threshold": 2.0, "holding_days": 5},
    execution_assumptions="conservative: best-of-book entry, full friction model",
    results={"sharpe": 0.4, "n_trades": 112, "brier": None},
    influenced_later_decisions=True,  # honest even when the answer is "yes, I acted on this"
)
append_experiment(record, "experiments/pead.jsonl")
```

`code_version` is captured automatically from `git rev-parse HEAD` (or
recorded as `None`, never a fabricated placeholder, when git isn't
available). Records are append-only JSON lines — diffable, greppable, and
loadable straight into pandas/DuckDB (`load_experiments()` returns plain
dicts, tolerant of schema drift across older entries).

**What this module deliberately does not do**: compute the Deflated Sharpe
Ratio, Probability of Backtest Overfitting, or a multiple-testing correction.
Those statistics require a meaningful trial count (dozens+) to be anything
but noise, and doing them now, over 13 trials, would produce a number with
no real information content — a false precision the assignment brief
specifically warns against.

## Backfilled history

`scripts/backfill_experiment_log.py` populates
`experiments/sttbot_history.jsonl` with **13 real historical trials**,
transcribed from the commit message of the commit that produced each one
(the `code_version` field on every record cites the exact short SHA). This
repo's research predates the registry, so the log would otherwise start
empty as if no trials had happened — that would misrepresent the actual
trial count going into any later multiple-testing correction. Nothing in
the backfill is estimated or invented; several records log clean **negative**
or inconclusive results on purpose (Dixon-Coles 1X2 betting is unprofitable
in all 7 tested divisions; momentum doesn't exist; the token payoff-tail
cohort couldn't resolve its own headline statistic) because the registry's
value is recording failed hypotheses too, not curating the wins.

```bash
python3 -c "
from sttbot.research.experiment_log import load_experiments
for r in load_experiments('experiments/sttbot_history.jsonl'):
    print(r['code_version'], '-', r['strategy'])
"
```

13 trials is still not enough for a meaningful DSR/PBO calculation — that
remains the concrete next step once more trials accumulate through the
registry going forward, not something to compute now for the sake of having
a number.

## Required reporting per trial (already partially met, registry makes it durable)

| Field the brief requires | Where it lives today |
|---|---|
| Net P&L, ROI on deployed capital | `backtest/metrics.py:summarise`, `paper/account.py` snapshot equity |
| Max drawdown | `backtest/metrics.py:max_drawdown` |
| Fees / rebates | `economics/friction.py`, per-fill `Fill.fee` in `paper/account.py` |
| Brier score / log loss | `strategies/earnings_market.py` (Brier-skill vs. base rate); not yet computed for Dixon-Coles or prob-arbitrage forecasts |
| Regime-stratified results | Not yet built centrally — would sit in `backtest/metrics.py` as a `by_regime()` helper once a regime taxonomy (volatility/spread/time-to-expiry buckets) is defined |
| Capacity analysis | `strategies/breadth.py` has depth-capped sizing and a documented Monte Carlo (explicitly labeled as assumption-driven, not backtested) — a formal capacity curve (P&L vs. AUM) is not yet built |

## Walk-forward / purge / embargo status by strategy

- **Dixon-Coles (football)**: walk-forward with strict date cut, no explicit
  embargo needed (bets are on `predict_date` itself, not a window that could
  leak past it).
- **PEAD**: entry timing delegated to `backtest/event_study.py`, which
  enforces `first_tradeable_date` (a report cannot be traded the day it's
  announced after market close) — audit §2.
- **Polymarket boundary arbitrage**: not a walk-forward strategy in the
  conventional sense — it is priced against live executable depth at
  decision time, so there is no train/test split to purge; the relevant
  validity question is whether the measured $66 of executable arbitrage
  (README) reproduces across repeated snapshots, which is exactly what the
  now-running paper-trading cron job (`scripts/run_paper_trade_polymarket.sh`)
  is measuring.
- **Cross-strategy locked holdout**: not yet formalized — there is no
  designated, never-touched final holdout period across strategies. Adding
  one requires picking a cutoff date and refusing to look at data past it
  until a strategy is otherwise finalized; this is a process discipline to
  adopt going forward, not a code change.
