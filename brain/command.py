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

    def classify(self, transcript: str, expecting_answer: bool = False):
        """Rules first (instant); Gemma only for what the rules don't recognise.

        Returns (kind, task, source, reason, intent_ms) with kind in
        request / hint / yes / no / other and task set for request and hint.
        """
        start = self._clock()
        if expecting_answer:
            ans = classify_answer(transcript)
            if ans is not None:
                return ans, None, "rules", "", _ms(start, self._clock())
        need = classify_need(transcript)
        if need is not None:
            kind = "request" if need[0] == "direct" else "hint"
            return kind, FetchTask(need[1]), "rules", "", _ms(start, self._clock())
        classify = getattr(self.gemma, "classify", None)
        if classify is None:
            # No local model: the legacy parsers are the only other option.
            task, source, reason, ms = self.interpret(transcript)
            return ("request" if task else "other"), task, source, reason, ms
        try:
            kind, item = classify(transcript, expecting_answer=expecting_answer)
        except IntentError as exc:
            task, source, reason, _ = self.interpret_rules(transcript)
            return ("request" if task else "other"), task, source, f"gemma: {exc}; {reason}".strip("; "), \
                _ms(start, self._clock())
        task = FetchTask(item) if kind in ("request", "hint") else None
        reason = "" if kind != "other" else "gemma: not a request for a supported item"
        return kind, task, "gemma", reason, _ms(start, self._clock())

    def interpret_rules(self, transcript: str):
        start = self._clock()
        try:
            return self.rules.parse(transcript), "rules", "", _ms(start, self._clock())
        except IntentError as exc:
            return None, None, f"rules: {exc}", _ms(start, self._clock())

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
            kind, task, source, reason, intent_ms = self.interpreter.classify(transcript)
            if kind == "hint" and task is not None:
                return self._ask(task, transcript, stt_ms, source, intent_ms)
            if kind != "request" or task is None:
                return self._set(replace(self.state, status="rejected", parser_source=source,
                                         reason=reason or "not a request", intent_ms=intent_ms))
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

    def _ask(self, task: FetchTask, transcript: str, stt_ms: float | None,
             source: str | None = None, intent_ms: float | None = None) -> CommandState:
        self._pending = task
        question = self._speak("ask_water") or "Would you like some water?"
        now = self._clock()
        return self._set(CommandState("confirming", transcript, task=task, stt_ms=stt_ms,
                                      parser_source=source, intent_ms=intent_ms, question=question,
                                      expires_at=now + self.confirm_timeout_s, timestamp=now))

    def _answer(self, transcript: str, stt_ms: float | None) -> CommandState | None:
        """Handle a reply to the open question; None if it wasn't a yes/no answer."""
        kind, _task, source, _reason, intent_ms = self.interpreter.classify(transcript, expecting_answer=True)
        if kind not in ("yes", "no"):
            return None  # e.g. a fresh request: interpret it normally
        answer = kind
        task, asked = self._pending, self.state
        self._pending = None
        both = f"{asked.transcript} -> {transcript}"
        now = self._clock()
        if answer == "no":
            self._speak("declined")
            return self._set(CommandState("declined", both, stt_ms=stt_ms, intent_ms=intent_ms,
                                          question=asked.question, reason=f"person said no ({source})",
                                          timestamp=now))
        state = self._set(CommandState("understood", both, task=task, parser_source="confirmed",
                                       stt_ms=stt_ms, intent_ms=intent_ms, question=asked.question,
                                       timestamp=now))
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
