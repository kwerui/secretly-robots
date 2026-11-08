"""The G1 voice demo, headless: hint -> robot asks -> yes -> fetch mode, with the G1 balancing."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

G1 = Path(__file__).resolve().parent.parent / "assistive_g1"
pytestmark = pytest.mark.skipif(not (G1 / "assets" / "policy").exists(),
                                reason="G1 assets not fetched (assistive_g1/fetch_assets.py)")


@pytest.fixture(scope="module")
def demo():
    from brain.speech_to_text import model_dir
    if not (model_dir() / "model.bin").exists():
        pytest.skip("Whisper model not downloaded")
    sys.path.insert(0, str(G1))
    import voice_demo
    args = SimpleNamespace(scenario=0, random=False, seed=0, bottle=None, cup=None, lock_door=None, clutter=0,
                           plan_only=True, max_turns=15, speed=1.0, port=0, no_browser=True, no_gemma=True)
    d = voice_demo.Demo(args)
    d.person.play_audio = False
    d.robot_voice.play_audio = False
    yield d
    d.pipeline.stop()


def run_until(d, cond, seconds=15.0):
    end = time.time() + seconds
    while time.time() < end:
        d.handle_keys()
        d.robot.wait(0.04)
        d.listen()
        d.follow_events()
        if cond():
            return True
        assert not d.robot.wbc.fallen()
    return False


def say(d, phrase):
    d.pending.append(("speak", phrase))


def test_hint_then_yes_enters_fetch_mode_with_a_planned_first_step(demo):
    d = demo
    assert d.mode == "listening"
    say(d, "My mouth is so dry.")
    assert run_until(d, lambda: d.mode == "asking"), "robot never asked"
    assert d.question == "Would you like some water?"
    assert d.robot_voice.said[-1] == "ask_water"
    assert d.brain.task is None  # nothing started yet

    say(d, "Yes, please.")
    assert run_until(d, lambda: d.mode == "fetch"), "never entered fetch mode"
    assert d.brain.task.item_id == "water_bottle" and d.brain.task.destination == "sofa"
    assert d.plan["action"] == "search" and d.plan["target"] == "water_bottle"
    assert d.plan["likely"][0][0] == "kitchen_counter"
    assert d.robot.station == "home"  # planned, not carried out
    modes = [e["mode"] for e in d.events.since(0) if e["kind"] == "mode"]
    assert modes[-3:] == ["listening", "asking", "fetch"]


def test_cancel_then_direct_request_goes_straight_to_fetch_mode(demo):
    d = demo
    d.pending.append(("cancel",))
    d.handle_keys()
    assert d.mode == "listening" and d.plan is None and d.brain.task is None
    asked_before = d.robot_voice.said.count("ask_water")
    say(d, "Can you get me something to drink?")
    assert run_until(d, lambda: d.mode == "fetch")
    assert d.robot_voice.said.count("ask_water") == asked_before  # no question for a direct request
    assert d.robot_voice.said[-1] == "will_fetch_water"  # but it does say it's on it


def test_no_answer_returns_to_listening(demo):
    d = demo
    d.pending.append(("cancel",))
    say(d, "I'm parched.")
    assert run_until(d, lambda: d.mode == "asking")
    say(d, "No, thank you.")
    assert run_until(d, lambda: d.mode == "listening")
    assert d.plan is None


def test_cup_request_enters_fetch_mode_for_the_cup(demo):
    d = demo
    d.pending.append(("cancel",))
    d.handle_keys()
    say(d, "Could you bring me my cup?")
    assert run_until(d, lambda: d.mode == "fetch")
    assert d.brain.task.item_id == "cup" and d.robot_voice.said[-1] == "will_fetch_cup"


def test_full_fetch_adapts_to_a_locked_door_and_clutter():
    """Scenario 14: bottle on the kitchen table (20%), living-kitchen door locked, clutter on the floor:
    the robot searches the likelier places first, rules them out, finds it, walks round, delivers."""
    from brain.speech_to_text import model_dir
    if not (model_dir() / "model.bin").exists():
        pytest.skip("Whisper model not downloaded")
    sys.path.insert(0, str(G1))
    import voice_demo
    args = SimpleNamespace(scenario=14, random=False, seed=0, bottle=None, cup=None, lock_door=None, clutter=0,
                           plan_only=False, max_turns=15, speed=1.0, port=0, no_browser=True, no_gemma=True)
    d = voice_demo.Demo(args)
    d.person.play_audio = False
    d.robot_voice.play_audio = False
    try:
        assert d.doors.truth == {"living_kitchen"} and not d.doors.known and d.env.info["clutter"] >= 1
        say(d, "I need water.")
        assert run_until(d, lambda: d.start_fetch)
        assert d.likely()[0][0] != "kitchen_table"  # likelier places come first
        d.run_fetch()
        # Whether the G1 then delivers varies run to run (walking physics, timing): delivery
        # rates come from validate_scenarios.py. What must always hold is the adaptation:
        assert d.mode in ("done", "failed")
        ruled_out = d.bridge.memory._searched_empty.get("water_bottle", set())
        assert ruled_out or d.found_at, "it should have searched likelier places first"
        assert d.doors.known == {"living_kitchen"} and "door_locked" in d.robot_voice.said  # found out on the way
        if d.mode == "done":
            assert d.found_at == "kitchen_table" and d.robot_voice.said[-1] == "delivered_water"
        else:
            assert d.robot_voice.said[-1] == "search_failed"  # it says so instead of pretending
        assert not d.robot.wbc.fallen()
    finally:
        d.pipeline.stop()
