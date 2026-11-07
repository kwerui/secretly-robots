"""Structured Gemini task and decision adapters.

The model proposes one current high-level decision. It never controls motors,
bypasses local safety checks, or returns a stale open-ended action sequence.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Mapping

from .contracts import ActionName, Decision, FetchTask, Observation
from .gemini_client import GeminiApiError, GeminiArgonClient
from .intent import IntentError, normalize_item


class GeminiPlanError(ValueError):
    """A model result violated the local task/decision contract."""


TASK_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["fetch", "unknown"]},
        "item_id": {"type": "string", "enum": ["water_bottle", "cup", "book", "unknown"]},
        "destination": {"type": "string", "enum": ["sofa"]},
    },
    "required": ["intent", "item_id", "destination"],
    "additionalProperties": False,
}

DECISION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ActionName.__args__)},
        "target": {"type": ["string", "null"]},
        "reason": {"type": "string", "maxLength": 240},
    },
    "required": ["action", "target", "reason"],
    "additionalProperties": False,
}


def _exact_object(value: Mapping[str, object], fields: set[str]) -> None:
    if set(value) != fields:
        raise GeminiPlanError("Gemini response has missing or unexpected fields")


class GeminiIntentParser:
    """Opt-in natural-language parser that returns a locally checked task."""

    def __init__(self, client: GeminiArgonClient | None = None):
        self.client = client or GeminiArgonClient()

    def parse(self, request: str) -> FetchTask:
        try:
            value = self.client.generate_json(
                "Identify whether the user request is a supported household fetch task. "
                "Use unknown when the request is ambiguous or unsupported.\n"
                f"User request: {request}",
                TASK_SCHEMA,
            )
            _exact_object(value, {"intent", "item_id", "destination"})
            if value["intent"] != "fetch" or not isinstance(value["item_id"], str):
                raise GeminiPlanError("Gemini did not identify a supported fetch task")
            if value["destination"] != "sofa":
                raise GeminiPlanError("Gemini proposed an unsupported destination")
            return FetchTask(normalize_item(value["item_id"]), destination="sofa")
        except (GeminiApiError, GeminiPlanError, IntentError) as exc:
            raise IntentError("Cloud task interpretation unavailable or invalid") from exc


class GeminiPlanner:
    """Propose one validated, high-level decision for a fresh observation."""

    def __init__(self, client: GeminiArgonClient | None = None):
        self.client = client or GeminiArgonClient()

    def propose(self, task: FetchTask, observation: Observation) -> Decision:
        prompt = (
            "Choose exactly one next high-level household-robot decision for the "
            "current task and current interpreted observation. Do not assume unobserved "
            "capabilities, do not command joints, and do not claim success without the "
            "observation verifying it.\n"
            f"Task: {json.dumps(asdict(task), sort_keys=True)}\n"
            f"Observation: {json.dumps(asdict(observation), sort_keys=True)}"
        )
        try:
            return self._validate(self.client.generate_json(prompt, DECISION_SCHEMA))
        except GeminiApiError as exc:
            raise GeminiPlanError("Gemini planning unavailable") from exc

    @staticmethod
    def _validate(value: Mapping[str, object]) -> Decision:
        _exact_object(value, {"action", "target", "reason"})
        action, target, reason = value["action"], value["target"], value["reason"]
        if action not in ActionName.__args__:
            raise GeminiPlanError("Gemini proposed an unknown action")
        if target is not None and (not isinstance(target, str) or not target.strip() or len(target) > 128):
            raise GeminiPlanError("Gemini proposed an invalid target")
        if not isinstance(reason, str) or len(reason) > 240:
            raise GeminiPlanError("Gemini proposed an invalid reason")
        return Decision(action=action, target=target, reason=reason, source="gemini_argon")
