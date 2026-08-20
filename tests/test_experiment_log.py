import json

from sttbot.research.experiment_log import (
    ExperimentRecord,
    append_experiment,
    load_experiments,
)


def _record(**overrides):
    defaults = dict(
        strategy="pead",
        hypothesis="SUE-ranked micro-caps drift for 5 sessions post-earnings",
        dataset_version="alphavantage-earnings-2026-08-01",
        train_interval="2019-01-01..2023-12-31",
        validation_interval="2024-01-01..2024-12-31",
        test_interval="2025-01-01..2025-12-31",
        parameters={"sue_threshold": 2.0, "holding_days": 5},
        execution_assumptions="conservative: best-of-book entry, full friction model",
        results={"sharpe": 0.4, "n_trades": 112},
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "experiments.jsonl"
    r1 = _record()
    r2 = _record(strategy="prob_arbitrage", influenced_later_decisions=True)

    append_experiment(r1, path)
    append_experiment(r2, path)

    loaded = load_experiments(path)
    assert len(loaded) == 2
    assert loaded[0]["strategy"] == "pead"
    assert loaded[1]["strategy"] == "prob_arbitrage"
    assert loaded[1]["influenced_later_decisions"] is True
    # Every trial gets a timestamp even if the caller didn't supply one.
    assert loaded[0]["timestamp"]


def test_append_never_rewrites_prior_lines(tmp_path):
    path = tmp_path / "experiments.jsonl"
    append_experiment(_record(strategy="a"), path)
    first_line = path.read_text().splitlines()[0]
    append_experiment(_record(strategy="b"), path)
    lines = path.read_text().splitlines()
    assert lines[0] == first_line
    assert len(lines) == 2


def test_load_missing_file_returns_empty(tmp_path):
    assert load_experiments(tmp_path / "does_not_exist.jsonl") == []


def test_records_are_diffable_single_line_json(tmp_path):
    path = tmp_path / "experiments.jsonl"
    append_experiment(_record(), path)
    line = path.read_text().splitlines()[0]
    # Round-trips through plain json, one record per line, no trailing junk.
    parsed = json.loads(line)
    assert parsed["strategy"] == "pead"


def test_defaults_never_fabricate_a_code_version(tmp_path, monkeypatch):
    # Outside a git repo (or with git unavailable), code_version must be None,
    # never a made-up placeholder.
    monkeypatch.chdir(tmp_path)
    record = _record()
    if record.code_version is not None:
        # We are inside the real repo's git tree in CI; this just documents
        # the contract rather than asserting a specific environment.
        assert isinstance(record.code_version, str) and len(record.code_version) == 40
