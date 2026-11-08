import json
import subprocess
import sys

import pytest

from brain.command import CommandListener, OnboardInterpreter
from brain.contracts import FetchTask
from brain.engine import Brain
from brain.intent import IntentError, OllamaIntentParser, RulesIntentParser

WATER_PHRASES = [
    "I am thirsty",
    "I'm so thirsty",
    "I need water",
    "I need a drink",
    "Could I have some water?",
    "I could do with a drink",
    "My mouth is dry",
    "I fancy a drink",
    "Some water please",
]
NOT_REQUESTS = ["I'm tired", "My back hurts", "What time is it?", "Hello there"]


@pytest.mark.parametrize("phrase", WATER_PHRASES)
def test_rules_map_stated_thirst_to_water(phrase):
    assert RulesIntentParser().parse(phrase) == FetchTask("water_bottle")


@pytest.mark.parametrize("phrase", NOT_REQUESTS)
def test_rules_reject_non_requests(phrase):
    with pytest.raises(IntentError):
        RulesIntentParser().parse(phrase)


class FakeGemma:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def parse(self, text):
        self.calls.append(text)
        if self.error:
            raise IntentError(self.error)
        return self.result


def test_gemma_result_used_when_available():
    task, source, _, _ = OnboardInterpreter(gemma=FakeGemma(FetchTask("water_bottle"))).interpret("parched, me")
    assert (task, source) == (FetchTask("water_bottle"), "gemma")


def test_falls_back_to_rules_when_gemma_unavailable():
    task, source, _, _ = OnboardInterpreter(gemma=FakeGemma(error="Ollama down")).interpret("I am thirsty")
    assert (task, source) == (FetchTask("water_bottle"), "rules")


def test_non_request_is_rejected_with_reasons_not_guessed():
    listener = CommandListener(OnboardInterpreter(gemma=FakeGemma(error="unknown intent")))
    state = listener.heard("I'm tired", stt_ms=120.0)
    assert state.status == "rejected" and state.task is None
    assert "gemma" in state.reason and "rules" in state.reason
    assert state.stt_ms == 120.0 and state.intent_ms is not None


def test_understood_sets_state_and_starts_brain_task_only(tmp_path):
    log = tmp_path / "events.jsonl"
    brain = Brain(log_file=log)
    seen = []
    listener = CommandListener(OnboardInterpreter(), brain=brain, on_change=seen.append)
    state = listener.heard("I need a drink", stt_ms=80.0)

    assert state.status == "understood"
    assert state.task == FetchTask("water_bottle") and state.parser_source == "rules"
    assert brain.task == FetchTask("water_bottle")
    assert [s.status for s in seen] == ["heard", "understood"]

    events = [json.loads(line) for line in log.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds == ["command_state", "command_state", "task_started"]
    assert events[-1]["payload"]["source"] == "voice_rules"
    assert "decision" not in kinds  # no planner/controller step is taken


def test_empty_transcript_rejected():
    assert CommandListener().heard("   ").status == "rejected"


def test_onboard_interpreter_refuses_non_loopback_model():
    gemma = OllamaIntentParser()
    gemma.endpoint = "https://example.com/api/chat"
    with pytest.raises(ValueError):
        OnboardInterpreter(gemma=gemma)


def test_onboard_path_never_imports_cloud_sdk():
    code = (
        "import sys, brain.command\n"
        "from brain.command import CommandListener\n"
        "CommandListener().heard('I am thirsty')\n"
        "assert not any(m.startswith(('google', 'brain.gemini')) for m in sys.modules), "
        "[m for m in sys.modules if m.startswith(('google', 'brain.gemini'))]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
