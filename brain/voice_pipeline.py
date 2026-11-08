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


def default_listener(log_file: str | None = "logs/brain_events.jsonl"):
    """The onboard CommandListener feeding a Brain, if that module is present."""
    try:
        from .command import CommandListener
        from .engine import Brain
    except ImportError:
        return None
    return CommandListener(brain=Brain(log_file=log_file))


class VoicePipeline:
    def __init__(self, stt: _STT, events: VoiceEvents, listener: _Listener | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.stt, self.events, self.listener = stt, events, listener
        self._clock, self._sleep = clock, sleep
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

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
        return t
