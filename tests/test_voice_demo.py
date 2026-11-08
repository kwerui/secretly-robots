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
    args = SimpleNamespace(layout="tidy", target="kitchen_counter", seed=0, port=0,
                           no_browser=True, no_gemma=True)
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
