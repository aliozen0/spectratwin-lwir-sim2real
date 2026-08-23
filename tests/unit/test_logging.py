import io
import json
import logging

from spectratwin.logging.structured import JsonFormatter, log_event


def test_log_event_serializes_event_and_fields():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test.spectratwin.logging")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_event(logger, "sample_completed", sample_id="0001", duration_s=1.5)

    payload = json.loads(stream.getvalue())
    assert payload["event"] == "sample_completed"
    assert payload["sample_id"] == "0001"
    assert payload["duration_s"] == 1.5
    assert payload["level"] == "INFO"
