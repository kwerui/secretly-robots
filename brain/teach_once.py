"""Small *symbolic* teach-once memory, NOT visual or motor imitation learning.

In the future the perception team can transform a recorded demonstration into
these verified fields. Current scope: store the goal, not an arm trajectory.
"""
from dataclasses import asdict
import json
from pathlib import Path
from .contracts import Demonstration, FetchTask

class RoutineMemory:
    def __init__(self, filename: str | Path):
        self.filename = Path(filename)

    def teach(self, demo: Demonstration) -> FetchTask:
        if not demo.completed_successfully:
            raise ValueError("Cannot learn a task from an unverified demonstration")
        task = FetchTask(item_id=demo.item_id, destination=demo.destination)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.filename.write_text(json.dumps(asdict(task), indent=2), encoding="utf-8")
        return task

    def recall(self) -> FetchTask:
        obj = json.loads(self.filename.read_text(encoding="utf-8"))
        return FetchTask(**obj)
