"""sttbot.paper — persistent paper trading against live prices.

``PaperBroker`` in :mod:`sttbot.execution.oms` is an in-memory simulator for
tests: it forgets everything on exit. This package is the other thing a paper
account has to be — durable, so a run that restarts continues the same book
rather than starting a fresh one and reporting a flattering P&L.
"""
