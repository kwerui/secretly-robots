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
from .safety import RejectedAction, validate

class IntentParser(Protocol):
    def parse(self, request: str) -> FetchTask: ...


class HighLevelPlanner(Protocol):
    def propose(self, task: FetchTask, observation: Observation) -> Decision: ...


class Brain:
    def __init__(
        self,
        parser: IntentParser | None = None,
        log_file: str | Path | None = None,
        gemini_planner: HighLevelPlanner | None = None,
        fallback_parser: IntentParser | None = None,
    ):
        self.parser = parser or RulesIntentParser()
        self.fallback_parser = fallback_parser or RulesIntentParser()
        self.gemini_planner = gemini_planner
        self.log_file = Path(log_file) if log_file else None
        self.task: FetchTask | None = None
        self.failures: dict[tuple[str, str | None], int] = {}
        self._last_cloud_observation: tuple | None = None
        # Evaluation reads this transient value after next_action().  It is intentionally
        # not part of Observation and cannot influence planning or control.
        self.last_decision_provenance = "deterministic_planner"
        self.last_cloud_event: dict[str, str] | None = None

    def start_task(self, user_request: str) -> FetchTask:
        self.last_cloud_event = None
        try:
            self.task = self.parser.parse(user_request)
        except IntentError as exc:
            if self.parser is self.fallback_parser:
                raise
            self.task = self.fallback_parser.parse(user_request)
            self.last_cloud_event = {"event": "cloud_intent_rejected", "reason": str(exc)}
            self._log("cloud_intent_rejected", self.last_cloud_event)
        self.failures.clear()
        self._last_cloud_observation = None
        self._log("task_started", {"request": user_request, "task": asdict(self.task)})
        return self.task

    def next_action(self, obs: Observation) -> Decision:
        self.last_cloud_event = None
        if self.task is None:
            decision = Decision("ask_for_help", reason="No active request")
            self.last_decision_provenance = "deterministic_planner"
            self._log("decision", {"observation": asdict(obs), "decision": asdict(decision)})
            return decision
        local_decision = choose_action(self.task, obs)
        decision, provenance = self._propose_with_fallback(local_decision, obs)
        self.last_decision_provenance = provenance
        signature = (decision.action, decision.target)
        if self.failures.get(signature, 0) >= 2 and decision.action not in ("stop", "ask_for_help", "complete"):
            decision = Decision("ask_for_help", reason="Repeated action failure; request human assistance")
        validate(decision, self.task, obs)
        self._log("decision", {"observation": asdict(obs), "decision": asdict(decision)})
        return decision

    def _propose_with_fallback(self, local_decision: Decision, obs: Observation) -> tuple[Decision, str]:
        """Consult cloud only on meaningful new state, then validate locally."""
        # These safety/terminal outcomes need no cloud interpretation at all.
        if local_decision.action in ("stop", "complete", "ask_for_help"):
            return local_decision, "deterministic_planner"
        signature = self._observation_signature(obs)
        if self.gemini_planner is None or signature == self._last_cloud_observation:
            return local_decision, "deterministic_planner"
        self._last_cloud_observation = signature
        task = self.task
        assert task is not None
        try:
            candidate = self.gemini_planner.propose(task, obs)
            validate(candidate, task, obs)
        except (ValueError, RejectedAction) as exc:
            reason = str(exc)
            self.last_cloud_event = {"event": "cloud_plan_rejected", "reason": reason}
            self._log("cloud_plan_rejected", self.last_cloud_event)
            return local_decision, "fallback_after_cloud_rejection_or_failure"
        self.last_cloud_event = {"event": "cloud_plan_accepted", "reason": candidate.reason}
        self._log("cloud_plan_accepted", {"decision": asdict(candidate)})
        return candidate, "gemini_argon"

    @staticmethod
    def _observation_signature(obs: Observation) -> tuple:
        """Exclude timestamps so physics ticks alone never cause cloud requests."""
        return (
            obs.robot_area,
            tuple((o.item_id, o.area, o.container, o.confidence) for o in obs.visible_objects),
            obs.drawer_open,
            obs.holding_item,
            obs.delivered_item,
            obs.person_too_close,
            obs.sensors_ok,
        )

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
