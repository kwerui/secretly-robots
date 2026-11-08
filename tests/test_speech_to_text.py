"""Onboard speech-to-text, the voice pipeline and the live window. Offline."""
from dataclasses import dataclass
import json
import re
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest

from brain.hearing import SAMPLE_RATE, HeardAudio, SimAudioSource
from brain.speech_to_text import WhisperSTT, model_dir
from brain.voice_pipeline import VoicePipeline
from brain.voice_window import VoiceEvents, serve


def seg(text, nsp=0.01, lp=-0.2):
    return SimpleNamespace(text=text, no_speech_prob=nsp, avg_logprob=lp)


class FakeWhisper:
    def __init__(self, segments):
        self.segments, self.calls = segments, 0

    def transcribe(self, samples, **kw):
        self.calls += 1
        assert kw["language"] == "en" and samples.dtype == np.float32
        return iter(self.segments), None


def heard(seconds=1.0, start=4.0):
    return HeardAudio(np.zeros(int(seconds * SAMPLE_RATE), np.float32), start, -25.0, "sim", 1.98)


def test_transcript_joins_segments_and_carries_audio_details():
    stt = WhisperSTT(model=FakeWhisper([seg(" I am"), seg(" thirsty. ")]))
    t = stt.transcribe(heard(1.5))
    assert t.text == "I am thirsty." and t.is_speech
    assert (t.start_s, t.audio_s, t.source, t.distance_m, t.level_dbfs) == (4.0, 1.5, "sim", 1.98, -25.0)
    assert t.stt_ms >= 0


def test_whisper_hallucination_on_noise_is_dropped():
    stt = WhisperSTT(model=FakeWhisper([seg(" Thank you.", nsp=0.9, lp=-1.4)]))
    t = stt.transcribe(heard())
    assert t.text == "" and not t.is_speech and t.no_speech_prob == 0.9


def test_missing_model_says_how_to_get_it(tmp_path):
    with pytest.raises(RuntimeError, match="--download"):
        WhisperSTT(path=tmp_path)


# ---------------------------------------------------------------- pipeline

@dataclass(frozen=True)
class FakeState:
    status: str
    transcript: str


class FakeListener:
    def __init__(self):
        self.got = []

    def heard(self, transcript, stt_ms=None):
        self.got.append((transcript, stt_ms))
        return FakeState("understood", transcript)


def test_pipeline_records_transcript_then_command(tmp_path):
    events = VoiceEvents(tmp_path / "ev.jsonl")
    listener = FakeListener()
    p = VoicePipeline(WhisperSTT(model=FakeWhisper([seg("I need water.")])), events, listener)
    p.submit(heard())
    p.drain()
    kinds = [e["kind"] for e in events.since(0)]
    assert kinds == ["transcript", "command"]
    assert listener.got[0][0] == "I need water."
    cmd = events.since(1)[0]
    assert cmd["status"] == "understood" and cmd["start_s"] == 4.0
    lines = (tmp_path / "ev.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == kinds


def test_pipeline_skips_command_for_non_speech_and_waits_until_ready():
    events = VoiceEvents(log=None)
    listener = FakeListener()
    slept = []
    p = VoicePipeline(WhisperSTT(model=FakeWhisper([seg("")])), events, listener,
                      clock=lambda: 100.0, sleep=slept.append)
    p.submit(heard(), ready_at=102.5)
    p.drain()
    assert slept == [2.5]
    assert [e["kind"] for e in events.since(0)] == ["transcript"] and listener.got == []


def test_pipeline_thread_processes_in_background():
    events = VoiceEvents(log=None)
    p = VoicePipeline(WhisperSTT(model=FakeWhisper([seg("I'm parched.")])), events).start()
    p.submit(heard())
    p.stop()
    assert events.since(0)[0]["text"] == "I'm parched."


# ---------------------------------------------------------------- window

def test_window_serves_page_and_events_on_loopback_only():
    events = VoiceEvents(log=None)
    events.add("status", message="listening", listening=True)
    events.add("transcript", text="I am thirsty.", start_s=1.0)
    url = serve(events, port=0)
    assert url.startswith("http://127.0.0.1:")
    page = urlopen(url, timeout=5).read().decode()
    assert "<title>Robot Hearing</title>" in page
    got = json.loads(urlopen(url + "events?after=1", timeout=5).read())
    assert [e["text"] for e in got] == ["I am thirsty."]
    assert json.loads(urlopen(url + "events?after=2", timeout=5).read()) == []


# ---------------------------------------------------------------- real model, real clips

def _norm(s):
    return re.sub(r"[^a-z' ]", "", s.lower()).strip()


@pytest.mark.skipif(not (model_dir() / "model.bin").exists(), reason="Whisper model not downloaded")
def test_real_whisper_transcribes_every_phrase_heard_in_the_house():
    H = pytest.importorskip("sim.house_env")
    from sim.person_voice import PHRASES, clip_path
    if not all(clip_path(p).exists() for p in PHRASES):
        pytest.skip("voice clips not generated")
    env = H.HouseEnv(seed=0)
    env.reset("tidy", "kitchen_counter", jitter=False)
    ears = SimAudioSource(env)
    stt = WhisperSTT()
    stt.warm_up()
    for phrase in PHRASES:
        (h,) = ears.hear_clip(clip_path(phrase), 0.0)
        t = stt.transcribe(h)
        assert _norm(t.text) == _norm(phrase), (phrase, t.text)
        assert t.stt_ms < 2000
