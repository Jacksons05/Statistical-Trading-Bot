"""Research-validity infrastructure: the experiment registry.

Everything a strategy trial needs recorded to make later "did this actually
work" questions answerable without re-litigating memory.
"""

from __future__ import annotations

from .experiment_log import ExperimentRecord, append_experiment, load_experiments

__all__ = ["ExperimentRecord", "append_experiment", "load_experiments"]
