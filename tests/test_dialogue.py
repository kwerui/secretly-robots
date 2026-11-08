"""Hint -> robot asks -> person answers -> act. Onboard and offline."""
from types import SimpleNamespace

import numpy as np
import pytest

from brain.command import CommandListener
from brain.contracts import FetchTask
from brain.intent import classify_answer, classify_need
from brain.robot_voice import LINES, RobotVoice, clip_path
from brain.voice_pipeline import VoicePipeline
from brain.voice_window import VoiceEvents

DIRECT = ["I need water.", "I need a drink.", "Could I have some water, please?",
          "Can you get me something to drink?", "I'd love a glass of water."]
HINTS = ["I am thirsty.", "My mouth is so dry.", "I'm parched.", "my throat's really dry"]


@pytest.mark.parametrize("text", DIRECT)
def test_requests_naming_water_are_direct(text):
    assert classify_need(text) == ("direct", "water_bottle")


@pytest.mark.parametrize("text", HINTS)
def test_hints_need_confirming(text):
    assert classify_need(text) == ("hint", "water_bottle")


@pytest.mark.parametrize("text", ["I'm tired.", "My back hurts.", "What time is it?"])
def test_unrelated_speech_is_neither(text):
    assert classify_need(text) is None


@pytest.mark.parametrize("text,ans", [("Yes, please.", "yes"), ("yeah go on", "yes"), ("Okay", "yes"),
                                      ("No, thank you.", "no"), ("I'm fine", "no"), ("no please", "no"),
                                      ("maybe later", None)])
def test_answers(text, ans):
    assert classify_answer(text) == ans


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeBrain:
    def __init__(self):
        self.tasks = []

    def set_task(self, task, user_request="", source=None):
        self.tasks.append((task, user_request, source))
        return task


def make(clock=None):
    said, brain = [], FakeBrain()
    lis = CommandListener(brain=brain, say=lambda k: (said.append(k), LINES[k])[1],
                          clock=clock or Clock(), confirm_timeout_s=10)
    return lis, said, brain


def test_hint_asks_and_waits_without_starting_a_task():
    lis, said, brain = make()
    s = lis.heard("My mouth is so dry.", stt_ms=230)
    assert s.status == "confirming" and s.question == "Would you like some water?"
    assert s.task == FetchTask("water_bottle") and s.expires_at == 1010.0
    assert said == ["ask_water"] and brain.tasks == []


def test_yes_starts_the_fetch_and_robot_acknowledges():
    lis, said, brain = make()
    lis.heard("My mouth is so dry.")
    s = lis.heard("Yes, please.")
    assert s.status == "understood" and s.parser_source == "confirmed"
    assert s.task == FetchTask("water_bottle")
    assert s.transcript == "My mouth is so dry. -> Yes, please."
    assert said == ["ask_water", "will_fetch_water"]
    assert brain.tasks == [(FetchTask("water_bottle"), s.transcript, "voice_confirmed")]


def test_no_declines_without_a_task():
    lis, said, brain = make()
    lis.heard("I'm parched.")
    s = lis.heard("No, thank you.")
    assert s.status == "declined" and s.reason == "person said no (rules)" and s.task is None
    assert said == ["ask_water", "declined"] and brain.tasks == []


def test_unanswered_question_lapses():
    clock = Clock()
    lis, said, brain = make(clock)
    lis.heard("I am thirsty.")
    clock.t += 9.9
    assert lis.tick() is None
    clock.t += 0.2
    s = lis.tick()
    assert s.status == "declined" and s.reason == "no answer"
    assert lis.tick() is None
    assert lis.heard("Yes, please.").status == "rejected"  # too late: nothing to say yes to
    assert brain.tasks == []


def test_new_request_while_waiting_is_handled_normally():
    lis, said, brain = make()
    lis.heard("My mouth is so dry.")
    s = lis.heard("I need water.")
    assert s.status == "understood" and s.parser_source == "rules"
    assert said == ["ask_water"] and len(brain.tasks) == 1


def test_direct_request_acts_without_asking():
    lis, said, brain = make()
    assert lis.heard("Can you get me something to drink?").status == "understood"
    assert said == [] and len(brain.tasks) == 1


# ---------------------------------------------------------------- pipeline + window events

class ScriptedSTT:
    name = "fake"

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, heard):
        from brain.speech_to_text import Transcript
        return Transcript(self.texts.pop(0), heard.start_s, 1.0, 1.0, "sim", -25.0)


def heard(start):
    from brain.hearing import HeardAudio
    return HeardAudio(np.zeros(1600, np.float32), start, -25.0, "sim")


def test_pipeline_logs_the_conversation_and_the_lapse():
    clock = Clock()
    events = VoiceEvents(log=None)
    lis = CommandListener(say=lambda k: (events.add("robot_said", text=LINES[k]), LINES[k])[1],
                          clock=clock, confirm_timeout_s=5)
    p = VoicePipeline(ScriptedSTT(["My mouth is so dry."]), events, lis, clock=clock)
    p.submit(heard(3.0))
    p.drain()
    clock.t += 6
    p.tick()
    got = [(e["kind"], e.get("status") or e.get("text")) for e in events.since(0)]
    assert got == [("transcript", "My mouth is so dry."), ("robot_said", "Would you like some water?"),
                   ("command", "confirming"), ("command", "declined")]
    assert events.since(3)[0]["start_s"] == 3.0  # the lapse attaches to the hint's card


# ---------------------------------------------------------------- robot voice

def test_robot_voice_marks_when_it_is_speaking(monkeypatch):
    clock = Clock()
    v = RobotVoice(play_audio=False, clock=clock)
    assert v.say("ask_water") == "Would you like some water?"
    if clip_path("ask_water").exists():
        assert v.is_speaking() and v.speaking_until > clock.t + 1.0
        clock.t = v.speaking_until + 0.01
    assert not v.is_speaking()


# ---------------------------------------------------------------- real clips, real Whisper

@pytest.mark.skipif(not clip_path("ask_water").exists(), reason="robot clips not generated")
def test_real_conversation_in_the_house():
    from brain.hearing import SimAudioSource
    from brain.speech_to_text import WhisperSTT, model_dir
    if not (model_dir() / "model.bin").exists():
        pytest.skip("Whisper model not downloaded")
    H = pytest.importorskip("sim.house_env")
    from sim.person_voice import ANSWERS, clip_path as person_clip
    env = H.HouseEnv(seed=0)
    env.reset("tidy", "kitchen_counter", jitter=False)
    ears, stt = SimAudioSource(env), WhisperSTT()
    lis, said, brain = make()

    def say(phrase):
        (h,) = ears.hear_clip(person_clip(phrase), 0.0)
        return lis.heard(stt.transcribe(h).text)

    assert say("My mouth is so dry.").status == "confirming"
    assert say(ANSWERS["yes"]).status == "understood"
    assert say("I'm parched.").status == "confirming"
    assert say(ANSWERS["no"]).status == "declined"
    assert said == ["ask_water", "will_fetch_water", "ask_water", "declined"]
    assert len(brain.tasks) == 1


# ---------------------------------------------------------------- Gemma for what the rules don't know

class FakeGemma:
    """Stands in for OllamaIntentParser.classify; records what it was asked."""
    endpoint = "http://127.0.0.1:11434/api/chat"

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def classify(self, text, expecting_answer=False):
        self.asked.append((text, expecting_answer))
        out = self.answers[text]
        if isinstance(out, Exception):
            raise out
        return out


def make_with_gemma(answers):
    from brain.command import OnboardInterpreter
    gemma = FakeGemma(answers)
    said, brain = [], FakeBrain()
    lis = CommandListener(OnboardInterpreter(gemma=gemma), brain=brain, clock=Clock(),
                          say=lambda k: (said.append(k), LINES[k])[1])
    return lis, gemma, said, brain


def test_known_phrases_never_wait_for_gemma():
    lis, gemma, said, _ = make_with_gemma({})
    assert lis.heard("I need water.").parser_source == "rules"
    lis.heard("My mouth is so dry.")
    assert lis.heard("Yes, please.").status == "understood"
    assert gemma.asked == []


def test_gemma_handles_free_form_hint_and_answer():
    lis, gemma, said, brain = make_with_gemma({
        "I could do with something cold, it's warm in here": ("hint", "water_bottle"),
        "go for it": ("yes", "unknown"),
    })
    s = lis.heard("I could do with something cold, it's warm in here")
    assert s.status == "confirming" and s.parser_source == "gemma"
    s = lis.heard("go for it")
    assert s.status == "understood" and len(brain.tasks) == 1
    assert gemma.asked[1] == ("go for it", True)  # told it's an answer


def test_gemma_request_and_other():
    lis, gemma, _, brain = make_with_gemma({
        "pop over and grab my book would you": ("request", "book"),
        "my knee is aching today": ("other", "unknown"),
    })
    s = lis.heard("pop over and grab my book would you")
    assert s.status == "understood" and s.task == FetchTask("book") and s.parser_source == "gemma"
    s = lis.heard("my knee is aching today")
    assert s.status == "rejected" and "gemma" in s.reason
    assert len(brain.tasks) == 1


def test_gemma_down_falls_back_to_rules_without_crashing():
    from brain.intent import IntentError
    lis, gemma, _, brain = make_with_gemma({"fetch the cup": IntentError("Local model unavailable")})
    s = lis.heard("fetch the cup")
    assert s.status == "understood" and s.parser_source == "rules" and s.task == FetchTask("cup")


def test_gemma_classify_validates_model_output(monkeypatch):
    from brain.intent import IntentError, OllamaIntentParser
    g = OllamaIntentParser()
    monkeypatch.setattr(g, "_chat", lambda *a: {"kind": "request", "item_id": "pill_box"})
    with pytest.raises(IntentError):
        g.classify("bring my pills")  # unsupported item is refused, not guessed
    monkeypatch.setattr(g, "_chat", lambda *a: {"kind": "yes", "item_id": "unknown"})
    assert g.classify("sure")[0] == "other"  # a 'yes' with no question pending means nothing
    assert g.classify("sure", expecting_answer=True)[0] == "yes"


def test_gemma_falls_back_to_prompt_json_when_schema_output_is_501(monkeypatch):
    from urllib.error import HTTPError
    from brain.intent import OllamaIntentParser
    g = OllamaIntentParser()
    calls = []

    def post(system, user, schema):
        calls.append(schema is not None)
        if schema is not None:
            raise HTTPError("u", 501, "Not Implemented", {}, None)
        assert "Reply with exactly one JSON object" in system
        return {"kind": "hint", "item_id": "water_bottle"}

    monkeypatch.setattr(g, "_post", post)
    assert g.classify("I'm feeling a bit dried out") == ("hint", "water_bottle")
    assert g.classify("I'm feeling a bit dried out") == ("hint", "water_bottle")
    assert calls == [True, False, False]  # schema tried once, then remembered as unsupported
