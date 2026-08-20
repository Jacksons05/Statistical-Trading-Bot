"""Regression test for the historical experiment-log backfill script.

Runs the backfill against a temp path (never the real experiments/ log,
which is append-only and already populated) and checks the records are
well-formed and traceable back to a real commit.
"""

import importlib
import subprocess
import sys

import pytest

from sttbot.research.experiment_log import load_experiments


def _in_git_repo() -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, check=True, timeout=5,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _in_git_repo(), reason="backfill cites real commit SHAs")
def test_backfill_records_are_well_formed(tmp_path, monkeypatch):
    sys.path.insert(0, "scripts")
    try:
        backfill = importlib.import_module("backfill_experiment_log")
    finally:
        sys.path.pop(0)

    log_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(backfill, "LOG_PATH", log_path)
    backfill.main()

    records = load_experiments(log_path)
    assert len(records) == len(backfill.RECORDS) > 0

    seen_strategies = set()
    for r in records:
        assert r["strategy"]
        assert r["hypothesis"]
        assert r["dataset_version"]
        assert r["results"]
        assert r["code_version"]  # every backfilled trial cites a real commit
        seen_strategies.add(r["strategy"])

    # Every trial is distinct (no accidental duplicate append).
    assert len(seen_strategies) == len(records)


def test_committed_history_log_is_valid_and_nonempty():
    records = load_experiments("experiments/sttbot_history.jsonl")
    assert len(records) >= 10
    for r in records:
        assert r["code_version"] and r["results"]
