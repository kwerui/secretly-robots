"""Tests for the seated person's voice. No network, API key or speakers needed."""
import wave
from types import SimpleNamespace

import pytest

import person_voice as V


class FakeTTS:
    """Stands in for google.genai.Client; returns 0.5 s of silent PCM."""

    def __init__(self):
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        blob = SimpleNamespace(data=b"\x00\x00" * 12_000, mime_type="audio/L16;codec=pcm;rate=24000")
        part = SimpleNamespace(inline_data=blob)
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


@pytest.fixture
def voice_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "VOICE_DIR", tmp_path)
    monkeypatch.setattr(V, "MANIFEST", tmp_path / "manifest.json")
    return tmp_path


def test_phrases_cover_both_examples_and_have_unique_clips():
    assert "I am thirsty." in V.PHRASES and "I need water." in V.PHRASES
    assert len({V.slug(p) for p in V.PHRASES}) == len(V.PHRASES)


def test_write_wav_is_16bit_mono(tmp_path):
    path = tmp_path / "x.wav"
    V.write_wav(path, b"\x00\x00" * 2400, 24_000)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24_000)
    assert V.clip_seconds(path) == pytest.approx(0.1)


def test_generate_clips_makes_missing_only(voice_dir):
    pytest.importorskip("google.genai")
    fake = FakeTTS()
    made = V.generate_clips(client=fake)
    assert len(made) == len(V.ALL_LINES) == len(fake.calls)
    part = fake.calls[0][1][0]["parts"][0]
    assert part["text"] in V.PHRASES and part["speech_metadata"]["style"] == V.STYLE
    assert fake.calls[0][2]["speech_config"]["voice_config"]["voice"] == V.TTS_VOICE
    assert (voice_dir / "manifest.json").exists()
    assert V.generate_clips(client=fake) == []  # nothing left to make


def test_speak_records_utterance_without_repeats(voice_dir):
    for p in V.PHRASES[:3]:
        V.write_wav(V.clip_path(p), b"\x00\x00" * 2400)
    pv = V.PersonVoice(seed=1, play_audio=False)
    said = [pv.speak(sim_time=float(i)) for i in range(20)]
    assert all(u.phrase in V.PHRASES[:3] for u in said)
    assert all(a.phrase != b.phrase for a, b in zip(said, said[1:]))
    assert said[4].sim_time == 4.0 and said[4].speaker == "user"
    assert len(pv.history) == 20


def test_speak_without_clips_does_nothing(voice_dir):
    pv = V.PersonVoice(play_audio=False)
    assert pv.speak() is None and pv.history == []


def test_house_env_still_imports_with_voice():
    import house_env  # noqa: F401  (the viewer imports PersonVoice)
