"""Offline tests for the optional Gemini boundary; no API key or network needed."""
from __future__ import annotations

import json

import pytest

from brain.contracts import FetchTask, ObjectSighting, Observation
from brain.engine import Brain
from brain.gemini_client import GeminiApiError, GeminiArgonClient
from brain.gemini_optional import GeminiIntentParser, GeminiPlanError, GeminiPlanner
from brain.intent import IntentError


class FakeResponse:
    def __init__(self, value: object):
        self.text = json.dumps(value)


class FakeModels:
    def __init__(self, value: object):
        self.value = value
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.value)


class FakeSdkClient:
    def __init__(self, value: object):
        self.models = FakeModels(value)


class FailingSdkClient:
    class Models:
        def generate_content(self, **kwargs):
            raise TimeoutError("simulated timeout")

    def __init__(self):
        self.models = self.Models()


def world(**kwargs) -> Observation:
    values = {
        "timestamp_s": 0.0,
        "robot_area": "sofa",
        "visible_objects": (ObjectSighting("water_bottle", "cupboard", "drawer", 0.95),),
    }
    values.update(kwargs)
    return Observation(**values)


def test_argon_client_requests_json_schema_from_official_sdk_boundary():
    sdk = FakeSdkClient({"intent": "fetch", "item_id": "water_bottle", "destination": "sofa"})
    client = GeminiArgonClient(client=sdk)

    value = client.generate_json("extract task", {"type": "object"})

    assert value["item_id"] == "water_bottle"
    call = sdk.models.calls[0]
    assert call["model"] == "gemini-4-argon"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == {"type": "object"}


def test_argon_client_rejects_invalid_json_without_returning_text():
    sdk = FakeSdkClient("not an object")
    with pytest.raises(GeminiApiError):
        GeminiArgonClient(client=sdk).generate_json("plan", {"type": "object"})


def test_gemini_intent_parser_validates_model_task():
    parser = GeminiIntentParser(
        GeminiArgonClient(client=FakeSdkClient({
            "intent": "fetch", "item_id": "water_bottle", "destination": "sofa"
        }))
    )

    assert parser.parse("Bring water to the person on the sofa") == FetchTask("water_bottle", "sofa")


def test_gemini_planner_returns_a_validated_high_level_decision():
    planner = GeminiPlanner(
        GeminiArgonClient(client=FakeSdkClient({
            "action": "navigate", "target": "cupboard", "reason": "Bottle is observed there"
        }))
    )

    decision = planner.propose(FetchTask("water_bottle"), world())

    assert decision.action == "navigate"
    assert decision.source == "gemini_argon"


def test_gemini_planner_rejects_unknown_action_before_execution():
    planner = GeminiPlanner(
        GeminiArgonClient(client=FakeSdkClient({
            "action": "teleport", "target": "cupboard", "reason": "no"
        }))
    )

    with pytest.raises(GeminiPlanError):
        planner.propose(FetchTask("water_bottle"), world())


def test_brain_falls_back_locally_when_argon_proposal_fails():
    planner = GeminiPlanner(
        GeminiArgonClient(client=FakeSdkClient({
            "action": "navigate", "target": "moon", "reason": "unsafe target"
        }))
    )
    brain = Brain(gemini_planner=planner)
    brain.start_task("bring me water")

    decision = brain.next_action(world())

    assert decision.action == "navigate"
    assert decision.target == "cupboard"
    assert decision.source == "deterministic_planner"


def test_brain_falls_back_locally_when_argon_times_out():
    brain = Brain(gemini_planner=GeminiPlanner(GeminiArgonClient(client=FailingSdkClient())))
    brain.start_task("bring me water")

    decision = brain.next_action(world())

    assert decision.action == "navigate"
    assert decision.source == "deterministic_planner"


def test_brain_falls_back_to_rules_when_argon_cannot_parse_request():
    class UnavailableParser:
        def parse(self, request: str) -> FetchTask:
            raise IntentError("network unavailable")

    brain = Brain(parser=UnavailableParser())

    assert brain.start_task("bring me water") == FetchTask("water_bottle")


def test_brain_does_not_call_argon_twice_for_unchanged_observation():
    sdk = FakeSdkClient({"action": "navigate", "target": "cupboard", "reason": "observed"})
    brain = Brain(gemini_planner=GeminiPlanner(GeminiArgonClient(client=sdk)))
    brain.start_task("bring me water")

    brain.next_action(world(timestamp_s=1.0))
    brain.next_action(world(timestamp_s=2.0))

    assert len(sdk.models.calls) == 1


def test_brain_keeps_stop_local_without_calling_argon():
    sdk = FakeSdkClient({"action": "navigate", "target": "cupboard", "reason": "observed"})
    brain = Brain(gemini_planner=GeminiPlanner(GeminiArgonClient(client=sdk)))
    brain.start_task("bring me water")

    assert brain.next_action(world(person_too_close=True)).action == "stop"
    assert sdk.models.calls == []


def test_brain_consults_argon_again_when_task_relevant_observation_changes():
    sdk = FakeSdkClient({"action": "navigate", "target": "cupboard", "reason": "observed"})
    brain = Brain(gemini_planner=GeminiPlanner(GeminiArgonClient(client=sdk)))
    brain.start_task("bring me water")

    brain.next_action(world())
    brain.next_action(world(visible_objects=(ObjectSighting("water_bottle", "table", None, 0.95),)))

    assert len(sdk.models.calls) == 2
