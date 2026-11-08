from pathlib import Path
import pytest
from brain.contracts import ActionFeedback, Demonstration, Decision, FetchTask, ObjectSighting, Observation
from brain.engine import Brain
from brain.intent import IntentError, RulesIntentParser, normalize_item, OllamaIntentParser
from brain.planner import choose_action
from brain.safety import validate, RejectedAction
from brain.teach_once import RoutineMemory

TASK = FetchTask("water_bottle")

def seen(area="cupboard", container="drawer", confidence=1.0):
    return (ObjectSighting("water_bottle", area, container, confidence),)

def world(area="sofa", sightings=None, **kwargs):
    return Observation(0, area, seen() if sightings is None else sightings, **kwargs)

def test_offline_parser_understands_request():
    assert RulesIntentParser().parse("Please bring me some water").item_id == "water_bottle"

def test_unsupported_request_asks_for_clarification():
    with pytest.raises(IntentError):
        RulesIntentParser().parse("Switch off the oven")

def test_model_output_cannot_invent_object():
    with pytest.raises(IntentError):
        normalize_item("controlled_medicine")

def test_navigate_to_cupboard_based_on_observation():
    assert choose_action(TASK, world()).action == "navigate"
    assert choose_action(TASK, world()).target == "cupboard"

def test_open_drawer_when_at_cupboard():
    assert choose_action(TASK, world(area="cupboard")).action == "open_drawer"

def test_grasp_after_drawer_is_open():
    assert choose_action(TASK, world(area="cupboard", drawer_open=True)).action == "grasp"

def test_adaptation_to_changed_object_location():
    before = choose_action(TASK, world(area="cupboard", drawer_open=True))
    after = choose_action(TASK, world(area="cupboard", sightings=seen("table", None), drawer_open=True))
    assert before.action == "grasp" and after.action == "navigate" and after.target == "table"

def test_no_sensors_means_stop():
    assert choose_action(TASK, world(sensors_ok=False)).action == "stop"

def test_person_too_close_means_stop():
    assert choose_action(TASK, world(person_too_close=True)).action == "stop"

def test_low_confidence_means_search():
    assert choose_action(TASK, world(sightings=seen(confidence=.2))).action == "search"

def test_holding_item_navigates_to_recipient():
    assert choose_action(TASK, world(area="cupboard", sightings=(), holding_item="water_bottle")).target == "sofa"

def test_deliver_requires_robot_at_recipient():
    assert choose_action(TASK, world(area="sofa", sightings=(), holding_item="water_bottle")).action == "deliver"

def test_task_completion_requires_confirmation():
    obs = world(sightings=(), delivered_item="water_bottle")
    assert choose_action(TASK, obs).action == "complete"
    with pytest.raises(RejectedAction):
        validate(Decision("complete"), TASK, world())

def test_block_unsafe_model_action():
    with pytest.raises(RejectedAction):
        validate(Decision("grasp", "water_bottle"), TASK, world(person_too_close=True))

def test_retry_limit_escalates_to_person():
    brain = Brain()
    brain.start_task("fetch my water bottle")
    obs = world(area="cupboard")
    first = brain.next_action(obs)
    assert first.action == "open_drawer"
    brain.record_feedback(ActionFeedback(first.action, first.target, False))
    brain.record_feedback(ActionFeedback(first.action, first.target, False))
    assert brain.next_action(obs).action == "ask_for_help"

def test_jsonl_log_contains_real_decision_not_claimed_execution(tmp_path):
    brain = Brain(log_file=tmp_path / "events.jsonl")
    brain.start_task("bring my water bottle")
    brain.next_action(world())
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"event": "decision"' in lines[1]

def test_rejected_cloud_plan_records_fallback_provenance(tmp_path):
    class InvalidCloudPlanner:
        def propose(self, task, observation):
            return Decision("deliver", "sofa", "deliver without holding", source="gemini_argon")

    brain = Brain(gemini_planner=InvalidCloudPlanner(), log_file=tmp_path / "events.jsonl")
    brain.start_task("bring my water bottle")
    decision = brain.next_action(world())
    assert decision.source == "deterministic_planner"
    assert brain.last_decision_provenance == "fallback_after_cloud_rejection_or_failure"
    assert brain.last_cloud_event["event"] == "cloud_plan_rejected"
    assert '"event": "cloud_plan_rejected"' in (tmp_path / "events.jsonl").read_text()

def test_teach_once_requires_verified_demo(tmp_path):
    memory = RoutineMemory(tmp_path / "routine.json")
    with pytest.raises(ValueError):
        memory.teach(Demonstration("water_bottle", "sofa", False))
    memory.teach(Demonstration("water_bottle", "sofa", True))
    assert memory.recall() == FetchTask("water_bottle", "sofa")

def test_ollama_endpoint_is_localhost_only():
    with pytest.raises(ValueError):
        OllamaIntentParser(endpoint="https://example.org/api/chat")
