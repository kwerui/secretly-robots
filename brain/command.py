"""Onboard command interpretation: heard transcript -> CommandState.

Everything here runs on the robot (this laptop). No cloud SDK is imported and
the language-model path only talks to a loopback Ollama endpoint. Speech-to-text
is upstream of this module; pass its transcript and timing in.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import threading
import time
from typing import Callable, Protocol

from .contracts import CommandState, FetchTask
from .intent import IntentError, OllamaIntentParser, RulesIntentParser, classify_answer, classify_need


class _Parser(Protocol):
    def parse(self, request: str) -> FetchTask: ...


class _TaskSink(Protocol):
    def set_task(self, task: FetchTask, user_request: str = "", source: str | None = None) -> FetchTask: ...


class OnboardInterpreter:
    """Gemma 4 via local Ollama first, deterministic rules as fallback.

    Pass ``gemma=None`` to run rules-only (e.g. Ollama not installed).
    """

    def __init__(
        self,
        gemma: _Parser | None = None,
        rules: _Parser | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if isinstance(gemma, OllamaIntentParser) and not _is_loopback(gemma.endpoint):
            raise ValueError("Onboard interpretation only allows a loopback model endpoint")
        self.gemma = gemma
        self.rules = rules or RulesIntentParser()
        self._clock = clock

    def interpret(self, transcript: str) -> tuple[FetchTask | None, str | None, str, float]:
        """Return (task, source, reason, intent_ms). task is None when rejected."""
        start = self._clock()
        reasons = []
        for source, parser in (("gemma", self.gemma), ("rules", self.rules)):
            if parser is None:
                continue
            try:
                task = parser.parse(transcript)
                return task, source, "", _ms(start, self._clock())
            except IntentError as exc:
                reasons.append(f"{source}: {exc}")
        return None, None, "; ".join(reasons) or "no parser available", _ms(start, self._clock())


class CommandListener:
    """Holds the robot's current CommandState and updates it from utterances.

    A request that names the item ("I need water") is acted on straight away.
    A hint that doesn't ("my mouth is so dry") makes the robot ask
    "Would you like some water?" and wait: status "confirming" until the person
    answers yes (-> understood, task started), no (-> declined), or the question
    lapses after `confirm_timeout_s` (-> declined). Call `tick()` regularly so a
    lapsed question is noticed.
    """

    def __init__(
        self,
        interpreter: OnboardInterpreter | None = None,
        brain: _TaskSink | None = None,
        on_change: Callable[[CommandState], None] | None = None,
        log: Callable[[str, dict], None] | None = None,
        clock: Callable[[], float] = time.time,
        say: Callable[[str], str] | None = None,
        confirm_timeout_s: float = 10.0,
    ):
        self.interpreter = interpreter or OnboardInterpreter()
        self.brain = brain
        self.on_change = on_change
        self._log = log or getattr(brain, "_log", None)
        self._clock = clock
        self.say = say  # robot speech: takes a line key, returns its text
        self.confirm_timeout_s = confirm_timeout_s
        self._pending: FetchTask | None = None
        self._lock = threading.RLock()
        self.state = CommandState()

    def heard(self, transcript: str, stt_ms: float | None = None) -> CommandState:
        """Mark that speech was heard, then interpret it. Returns the final state."""
        with self._lock:
            transcript = transcript.strip()
            if self.state.status == "confirming" and self._pending is not None:
                answered = self._answer(transcript, stt_ms)
                if answered is not None:
                    return answered
            self._pending = None
            self._set(CommandState("heard", transcript, stt_ms=stt_ms, timestamp=self._clock()))
            if not transcript:
                return self._set(replace(self.state, status="rejected", reason="empty transcript"))
            need = classify_need(transcript)
            if need is not None and need[0] == "hint":
                return self._ask(FetchTask(need[1]), transcript, stt_ms)
            task, source, reason, intent_ms = self.interpreter.interpret(transcript)
            if task is None:
                return self._set(replace(self.state, status="rejected", reason=reason, intent_ms=intent_ms))
            state = self._set(replace(self.state, status="understood", task=task,
                                      parser_source=source, intent_ms=intent_ms))
            self._start(task, transcript, f"voice_{source}")
            return state

    def tick(self) -> CommandState | None:
        """Lapse an unanswered question. Returns the new state if it changed."""
        with self._lock:
            s = self.state
            if s.status == "confirming" and s.expires_at is not None and self._clock() >= s.expires_at:
                self._pending = None
                return self._set(replace(s, status="declined", reason="no answer", expires_at=None,
                                         timestamp=self._clock()))
            return None

    def reset(self) -> CommandState:
        with self._lock:
            self._pending = None
            return self._set(CommandState(timestamp=self._clock()))

    # -- confirmation -----------------------------------------------------

    def _ask(self, task: FetchTask, transcript: str, stt_ms: float | None) -> CommandState:
        self._pending = task
        question = self._speak("ask_water") or "Would you like some water?"
        now = self._clock()
        return self._set(CommandState("confirming", transcript, task=task, stt_ms=stt_ms,
                                      question=question, expires_at=now + self.confirm_timeout_s,
                                      timestamp=now))

    def _answer(self, transcript: str, stt_ms: float | None) -> CommandState | None:
        """Handle a reply to the open question; None if it wasn't a yes/no answer."""
        answer = classify_answer(transcript)
        if answer is None:
            return None  # e.g. a fresh request: interpret it normally
        task, asked = self._pending, self.state
        self._pending = None
        both = f"{asked.transcript} -> {transcript}"
        now = self._clock()
        if answer == "no":
            self._speak("declined")
            return self._set(CommandState("declined", both, stt_ms=stt_ms, question=asked.question,
                                          reason="person said no", timestamp=now))
        state = self._set(CommandState("understood", both, task=task, parser_source="confirmed",
                                       stt_ms=stt_ms, question=asked.question, timestamp=now))
        self._speak("will_fetch_water")
        self._start(task, both, "voice_confirmed")
        return state

    def _speak(self, key: str) -> str:
        return self.say(key) if self.say is not None else ""

    def _start(self, task: FetchTask, transcript: str, source: str) -> None:
        if self.brain is not None:
            # Starts the task only. The planner loop / controllers are not run here.
            self.brain.set_task(task, transcript, source=source)

    def _set(self, state: CommandState) -> CommandState:
        self.state = state
        if self._log:
            self._log("command_state", asdict(state))
        if self.on_change:
            self.on_change(state)
        return state


def _is_loopback(endpoint: str) -> bool:
    return endpoint.startswith(("http://127.0.0.1:", "http://localhost:"))


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000, 2)
