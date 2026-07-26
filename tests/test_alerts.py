import logging

from sttbot.monitoring.alerts import Level, Notifier


def test_noop_without_webhook_returns_false(caplog):
    n = Notifier(webhook_url=None)
    with caplog.at_level(logging.INFO, logger="sttbot.alerts"):
        assert n.info("hello") is False
    assert "hello" in caplog.text


def test_levels_map_to_log_levels(caplog):
    n = Notifier()
    with caplog.at_level(logging.INFO, logger="sttbot.alerts"):
        n.warning("careful")
        n.critical("boom")
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_delivery_failure_is_swallowed(caplog):
    # An unroutable webhook URL must not raise out of send().
    n = Notifier(webhook_url="http://127.0.0.1:1/webhook", timeout=0.2)
    with caplog.at_level(logging.ERROR, logger="sttbot.alerts"):
        assert n.send("ping", Level.INFO) is False
    assert "alert delivery failed" in caplog.text
