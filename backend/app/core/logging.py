import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON.

    Reads the per-request fields off `record` (set via the `extra=` argument
    when logging) and merges them with the message.
    """

    _RESERVED = set(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge any extra={...} fields (request_id, endpoint, latency_ms, ...).
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and key != "message":
                payload[key] = value
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
