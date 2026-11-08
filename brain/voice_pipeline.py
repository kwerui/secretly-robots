"""Heard speech -> onboard speech-to-text -> command state, off the sim's thread.

`VoicePipeline.submit(heard, ready_at)` queues a speech segment. A worker thread
waits until `ready_at` (wall time; the sim passes the moment the person finishes
speaking, so the robot doesn't transcribe words before they've been said), runs
Whisper, records a `transcript` event and, if a command listener is attached,
passes the text on and records a `command` event with the resulting state.
"""
from __future__ import annotations

from dataclasses import asdict
import queue
import threading
import time
from typing import Callable, Protocol

from .hearing import HeardAudio
from .speech_to_text import Transcript
from .voice_window import VoiceEvents


class _STT(Protocol):
    def transcribe(self, heard: HeardAudio) -> Transcript: ...


class _Listener(Protocol):
    def heard(self, transcript: str, stt_ms: float | None = None): ...


def local_gemma(model: str = "gemma4:e4b"):
    """Gemma 4 on the local Ollama server, warmed up; None if it isn't running."""
    from .intent import IntentError, OllamaIntentParser
    g = OllamaIntentParser(model=model, timeout_s=20)
    try:
        g.warm_up()
    except IntentError:
        return None
    return g


def default_listener(log_file: str | None = "logs/brain_events.jsonl", say=None, gemma=None):
    """The onboard CommandListener feeding a Brain, if that module is present.

    `say(key) -> text` lets it speak (e.g. ask "Would you like some water?").
    `gemma` (an OllamaIntentParser) handles speech the rules don't recognise.
    """
    try:
        from .command import CommandListener, OnboardInterpreter
        from .engine import Brain
    except ImportError:
        return None
    return CommandListener(OnboardInterpreter(gemma=gemma), brain=Brain(log_file=log_file), say=say)


class VoicePipeline:
    def __init__(self, stt: _STT, events: VoiceEvents, listener: _Listener | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.stt, self.events, self.listener = stt, events, listener
        self._clock, self._sleep = clock, sleep
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._asked_at: float | None = None  # start_s of the speech the robot is asking about

    def submit(self, heard: HeardAudio, ready_at: float | None = None) -> None:
        self._q.put((ready_at if ready_at is not None else self._clock(), heard))

    def start(self) -> "VoicePipeline":
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-pipeline")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def drain(self) -> None:
        """Process everything queued now, on this thread (tests, scripts)."""
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                self.process(*item)

    def _run(self) -> None:
        while (item := self._q.get()) is not None:
            self.process(*item)

    def process(self, ready_at: float, heard: HeardAudio) -> Transcript:
        wait = ready_at - self._clock()
        if wait > 0:
            self._sleep(wait)
        t = self.stt.transcribe(heard)
        self.events.add("transcript", **asdict(t))
        if self.listener is not None and t.is_speech:
            state = self.listener.heard(t.text, stt_ms=t.stt_ms)
            self.events.add("command", start_s=t.start_s, **asdict(state))
            self._asked_at = t.start_s if state.status == "confirming" else None
        return t

    def tick(self) -> None:
        """Let an unanswered question lapse; call this from the main loop."""
        tick = getattr(self.listener, "tick", None)
        if tick is None:
            return
        state = tick()
        if state is not None:
            self.events.add("command", start_s=self._asked_at, **asdict(state))
            self._asked_at = None
