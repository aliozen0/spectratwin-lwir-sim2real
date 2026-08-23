"""Structured logging: machine-readable JSON events plus human-readable CLI output."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_EVENT_ATTR = "spectratwin_event"
_FIELDS_ATTR = "spectratwin_fields"


class JsonFormatter(logging.Formatter):
    """Renders a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, _EVENT_ATTR, record.getMessage()),
        }
        payload.update(getattr(record, _FIELDS_ATTR, {}))
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(level: int = logging.INFO, stream: Any = None) -> None:
    """Configure the root ``spectratwin`` logger to emit JSON lines."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("spectratwin")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"spectratwin.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event with key-value fields."""
    logger.log(level, event, extra={_EVENT_ATTR: event, _FIELDS_ATTR: fields})
