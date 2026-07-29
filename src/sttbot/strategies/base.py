"""Declarative strategy framework.

Strategies separate *signal logic* from *tunable parameters*. Parameters are
declared with :class:`Param`, which lets the optimiser enumerate a search
space without the strategy hard-coding any values. This is the programmatic
equivalent of the ``# @param`` convention: every tunable is discoverable,
typed, and bounded.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Iterator


class Signal(IntEnum):
    """Trade direction. Values are chosen so a signal can be multiplied by a
    size to get a signed position."""

    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(frozen=True)
class Param:
    """A tunable hyperparameter.

    ``low``/``high`` bound a continuous or integer search space; ``choices``
    declares a categorical one. Exactly one of the two styles should be used.
    """

    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None
    kind: str = "float"  # "float" | "int" | "categorical"

    def grid(self, points: int = 5) -> list[Any]:
        """Enumerate candidate values for a grid search."""
        if self.choices is not None:
            return list(self.choices)
        if self.low is None or self.high is None:
            return [self.default]
        if self.step is not None:
            values: list[Any] = []
            v = self.low
            # Guard against floating point drift accumulating past ``high``.
            while v <= self.high + 1e-9:
                values.append(int(round(v)) if self.kind == "int" else round(v, 10))
                v += self.step
            return values
        span = self.high - self.low
        values = [self.low + span * i / (points - 1) for i in range(points)]
        if self.kind == "int":
            # De-duplicate after rounding so an int grid never repeats a value.
            return sorted({int(round(v)) for v in values})
        return values


class StrategyMeta(type):
    """Collects :class:`Param` declarations from the class body into
    ``__params__`` so subclasses inherit and can extend the search space."""

    def __new__(mcls, name, bases, namespace):
        params: dict[str, Param] = {}
        for base in bases:
            params.update(getattr(base, "__params__", {}))
        for key, value in list(namespace.items()):
            if isinstance(value, Param):
                params[key] = value
                namespace[key] = value.default  # class attribute falls back to default
        namespace["__params__"] = params
        return super().__new__(mcls, name, bases, namespace)


class Strategy(metaclass=StrategyMeta):
    """Base class for signal-generating strategies.

    Declared params are set as instance attributes from ``**overrides`` (or
    their declared default). Unknown keyword names raise ``TypeError`` so a
    typo in an optimiser config fails loudly instead of being ignored.
    """

    def __init__(self, **overrides: Any):
        unknown = set(overrides) - set(self.__params__)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected parameter(s): "
                f"{', '.join(sorted(unknown))}"
            )
        for pname, param in self.__params__.items():
            setattr(self, pname, overrides.get(pname, param.default))

    @classmethod
    def params(cls) -> dict[str, Param]:
        return dict(cls.__params__)

    @classmethod
    def grid(cls, points: int = 5, **overrides: Iterable[Any]) -> Iterator[dict[str, Any]]:
        """Yield every parameter combination for a grid search.

        ``overrides`` replaces the auto-generated candidate list for a given
        parameter, e.g. ``grid(sue_threshold=[1.0, 1.5, 2.0])``.
        """
        names = list(cls.__params__)
        axes = [
            list(overrides[n]) if n in overrides else cls.__params__[n].grid(points)
            for n in names
        ]
        for combo in itertools.product(*axes):
            yield dict(zip(names, combo))

    def config(self) -> dict[str, Any]:
        """The strategy's current parameter values, for logging / MLflow."""
        return {name: getattr(self, name) for name in self.__params__}

    def generate_signal(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        raise NotImplementedError
