"""Offline-first orchestration of intent, observed state, planning and safety.

A robot controller calls `next_action` after providing updated perception data.
"""
from pathlib import Path
import json
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Protocol
from .contracts import ActionFeedback, Decision, FetchTask, Observation
from .intent import IntentError, RulesIntentParser
from .planner import choose_action
from .safety import validate

class IntentParser(Protocol):
    def parse(self, request: str) -> FetchTask: ...

class Brain:
    def __init__(self, parser: IntentParser | None = None, log_file: str | Path | None = None):
        self.parser = parser or RulesIntentParser()
        self.log_file = Path(log_file) if log_file else None
        self.task: FetchTask | None = None
        self.failures: dict[tuple[str, str | None], int] = {}

    def start_task(self, user_request: str) -> FetchTask:
        self.task = self.parser.parse(user_request)
        self.failures.clear()
        self._log("task_started", {"request": user_request, "task": asdict(self.task)})
        return self.task

    def next_action(self, obs: Observation) -> Decision:
        if self.task is None:
            decision = Decision("ask_for_help", reason="No active request")
            self._log("decision", {"observation": asdict(obs), "decision": asdict(decision)})
            return decision
        decision = choose_action(self.task, obs)
        signature = (decision.action, decision.target)
        if self.failures.get(signature, 0) >= 2 and decision.action not in ("stop", "ask_for_help", "complete"):
            decision = Decision("ask_for_help", reason="Repeated action failure; request human assistance")
        validate(decision, self.task, obs)
        self._log("decision", {"observation": asdict(obs), "decision": asdict(decision)})
        return decision

    def record_feedback(self, feedback: ActionFeedback) -> None:
        signature = (feedback.action, feedback.target)
        if feedback.succeeded:
            self.failures.pop(signature, None)
        else:
            self.failures[signature] = self.failures.get(signature, 0) + 1
        self._log("feedback", asdict(feedback))

    def _log(self, kind: str, payload: dict) -> None:
        if not self.log_file:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"logged_at": datetime.now(timezone.utc).isoformat(), "event": kind, "payload": payload}) + "\n")
