"""Machine-readable observation/decision logs for judging and debugging."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload) -> None:
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": asdict(payload) if is_dataclass(payload) else payload,
        }
        with self.path.open("a", encoding="utf8") as file:
            file.write(json.dumps(record) + "\n")
