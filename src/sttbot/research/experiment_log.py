"""A machine-readable, append-only experiment registry.

Prediction-market and micro-cap backtests are cheap to run and easy to
p-hack by accident: try five parameter sets, keep the one that looks best,
and the "backtest" has quietly become an unrecorded search over five
hypotheses instead of a test of one. The only defense is a durable record of
every trial -- including the ones that failed -- made *before* the result is
known to be good or bad, so a later Deflated Sharpe Ratio / Probability of
Backtest Overfitting calculation has an honest trial count to condition on.

This module is intentionally small: one dataclass, one append function, one
load function. It records to newline-delimited JSON (one record per line) so
it is diffable, appendable without parsing the whole file, and trivially
loadable into pandas/DuckDB for analysis. It does not compute DSR/PBO itself
-- that belongs with the backtest metrics once enough trials exist to make
those statistics meaningful, and shouldn't be faked with a handful of runs.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    """Best-effort code version. None (never a fake value) if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclasses.dataclass(frozen=True)
class ExperimentRecord:
    """One strategy trial, recorded regardless of outcome.

    ``results`` and the interval/cost fields are left as free-form dicts
    rather than a rigid schema, because a Dixon-Coles walk-forward and a
    Polymarket boundary-arbitrage scan report genuinely different metrics --
    forcing one shape would either drop real fields or invent placeholder
    ones. What is *not* optional is recording the trial at all, and the
    ``influenced_later_decisions`` flag, which is what lets a later
    multiple-testing correction count trials honestly instead of only the
    ones someone remembered to log.
    """

    strategy: str
    hypothesis: str
    dataset_version: str
    train_interval: str  # e.g. "2018-01-01..2022-12-31", or a season/cohort id
    validation_interval: str
    test_interval: str
    parameters: dict[str, Any]
    execution_assumptions: str  # e.g. "conservative sim: best-of-book, fees, no partial fills"
    results: dict[str, Any]
    features: tuple[str, ...] = ()
    search_method: str = "single run"  # e.g. "grid search over N points", "manual"
    costs: dict[str, Any] = dataclasses.field(default_factory=dict)
    influenced_later_decisions: bool = False
    notes: str = ""
    timestamp: str = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    code_version: str | None = dataclasses.field(default_factory=_git_commit)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def append_experiment(record: ExperimentRecord, path: str | Path) -> None:
    """Append one record as a JSON line. Never rewrites or reorders history."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def load_experiments(path: str | Path) -> list[dict[str, Any]]:
    """Load every recorded trial as a plain dict, in the order they were logged.

    Returns dicts rather than :class:`ExperimentRecord` so records written by
    an older schema version (extra or missing fields) still load instead of
    raising -- an experiment log that can't be read is worse than one with a
    slightly stale schema.
    """
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
