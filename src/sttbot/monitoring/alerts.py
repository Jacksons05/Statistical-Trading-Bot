"""Telemetry and alerting via chat webhooks.

Execution logs and risk warnings are pushed to a private Discord or Telegram
channel. The notifier uses only the standard library (``urllib``) and is
fail-safe: a webhook error is logged and swallowed so a monitoring outage can
never crash the trading loop. With no webhook configured it degrades to a
no-op, which keeps tests and local runs quiet.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("sttbot.alerts")


class Level(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Notifier:
    """Posts messages to a Discord/Telegram-style webhook.

    ``webhook_url`` is optional; when absent, messages are only logged. The
    payload uses Discord's ``{"content": ...}`` shape, the most common format.
    """

    webhook_url: str | None = None
    timeout: float = 5.0

    def send(self, message: str, level: Level = Level.INFO) -> bool:
        """Return ``True`` if the message was delivered, ``False`` otherwise."""
        text = f"[{level.value}] {message}"
        logger.log(_LOG_LEVELS[level], text)
        if not self.webhook_url:
            return False
        try:
            payload = json.dumps({"content": text}).encode("utf-8")
            request = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError) as exc:
            logger.error("alert delivery failed: %s", exc)
            return False

    def info(self, message: str) -> bool:
        return self.send(message, Level.INFO)

    def warning(self, message: str) -> bool:
        return self.send(message, Level.WARNING)

    def critical(self, message: str) -> bool:
        return self.send(message, Level.CRITICAL)


_LOG_LEVELS = {
    Level.INFO: logging.INFO,
    Level.WARNING: logging.WARNING,
    Level.CRITICAL: logging.CRITICAL,
}
