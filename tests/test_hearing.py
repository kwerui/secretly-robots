"""Robot hearing: voice-activity segments from the sim and from raw audio. Offline."""
import subprocess
import sys

import numpy as np
import pytest

from brain.hearing import (SAMPLE_RATE, EnergyVAD, HeardAudio, SimAudioSource, dbfs,
                           distance_gain, load_wav, resample)

SR = SAMPLE_RATE


def tone(seconds, dbfs_level=-20.0, hz=220.0):
    t = np.arange(int(seconds * SR)) / SR
    amp = 10 ** (dbfs_level / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


def run_vad(audio, chunk=None, vad=None):
    vad = vad or EnergyVAD()
    if chunk is None:
        return vad.feed(audio) + vad.flush()
    out = []
    for i in range(0, len(audio), chunk):
        out += vad.feed(audio[i:i + chunk])
    return out + vad.flush()


def test_dbfs_and_distance_gain():
    assert dbfs(tone(1, -20)) == pytest.approx(-20, abs=0.1)
    assert dbfs(silence(1)) == -120.0
    assert distance_gain(0.5) == 1.0  # no boost when closer than 1 m
    assert 20 * np.log10(distance_gain(2.0)) == pytest.approx(-6.02, abs=0.01)


def test_resample_24k_to_16k_keeps_duration_and_level():
    t = np.arange(24_000) / 24_000
    x = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    y = resample(x, 24_000)
    assert len(y) == SR and y.dtype == np.float32
    assert dbfs(y) == pytest.approx(dbfs(x), abs=0.5)


def test_vad_finds_one_utterance_with_its_start_time():
    segs = run_vad(np.concatenate([silence(1.0), tone(1.2), silence(1.0)]))
    assert len(segs) == 1
    samples, start = segs[0]
    assert start == pytest.approx(1.0 - 0.15, abs=0.04)  # includes pre-roll
    assert 1.2 <= len(samples) / SR <= 1.2 + 0.15 + 0.06


def test_vad_splits_two_utterances_and_drops_clicks_and_quiet_sound():
    audio = np.concatenate([silence(0.5), tone(0.8), silence(1.0), tone(0.6), silence(1.0),
                            tone(0.06), silence(1.0), tone(1.0, dbfs_level=-60), silence(0.5)])
    segs = run_vad(audio)
    assert len(segs) == 2
    assert segs[1][1] > segs[0][1] + 1.0


def test_vad_result_does_not_depend_on_chunk_size():
    audio = np.concatenate([silence(0.7), tone(0.9), silence(0.8), tone(0.5), silence(0.6)])
    whole = run_vad(audio)
    chunked = run_vad(audio, chunk=317)
    assert [round(t, 3) for _, t in whole] == [round(t, 3) for _, t in chunked]
    assert [len(s) for s, _ in whole] == [len(s) for s, _ in chunked]


def test_vad_flush_closes_speech_still_going_at_end():
    assert len(run_vad(np.concatenate([silence(0.5), tone(1.0)]))) == 1


def test_heard_audio_round_trips_through_wav(tmp_path):
    h = HeardAudio(tone(0.5, -12), 3.0, -12.0, "sim", 2.0)
    path = h.save_wav(tmp_path / "h.wav")
    back, rate = load_wav(path)
    assert rate == SR and len(back) == len(h.samples)
    assert h.duration_s == pytest.approx(0.5)


# ---------------------------------------------------------------- in the house

@pytest.fixture(scope="module")
def house():
    H = pytest.importorskip("sim.house_env")
    env = H.HouseEnv(seed=0)
    env.reset("tidy", "kitchen_counter", jitter=False)
    return env


@pytest.fixture
def clip():
    from sim.person_voice import PHRASES, clip_path
    path = clip_path(PHRASES[0])
    if not path.exists():
        pytest.skip("voice clips not generated")
    return path


def say(env, wav, t, phrase="I am thirsty."):
    env.info["last_utterance"] = {"speaker": "user", "phrase": phrase, "wav": str(wav), "sim_time": t}


def test_robot_hears_person_once_per_utterance(house, clip):
    ears = SimAudioSource(house)
    say(house, clip, 12.0)
    heard = ears.poll()
    assert len(heard) == 1
    h = heard[0]
    assert h.source == "sim" and h.sample_rate == SR
    assert 11.5 <= h.start_s <= 13.0
    assert 1.0 < h.duration_s < 3.0
    assert h.distance_m == pytest.approx(ears.mouth_to_ears_m())
    assert ears.poll() == []  # same utterance is not heard twice
    say(house, clip, 20.0)
    assert len(ears.poll()) == 1  # a new one is


def test_robot_works_from_sound_not_the_caption_text(house, clip):
    a = SimAudioSource(house, seed=1).hear_clip(clip, 5.0)[0]
    say(house, clip, 5.0, phrase="something completely different")
    b = SimAudioSource(house, seed=1).poll()[0]
    assert np.array_equal(a.samples, b.samples)


def test_farther_away_is_quieter_and_out_of_range_is_not_heard(house, clip, monkeypatch):
    ears = SimAudioSource(house)
    monkeypatch.setattr(ears, "mouth_to_ears_m", lambda: 2.0)
    near = ears.hear_clip(clip, 0.0)[0]
    monkeypatch.setattr(ears, "mouth_to_ears_m", lambda: 8.0)
    far = ears.hear_clip(clip, 0.0)[0]
    assert far.level_dbfs == pytest.approx(near.level_dbfs - 12.0, abs=1.0)
    monkeypatch.setattr(ears, "mouth_to_ears_m", lambda: 200.0)
    assert ears.hear_clip(clip, 0.0) == []


def test_hearing_is_onboard_no_cloud_or_network_imports():
    code = ("import sys, brain.hearing; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('google', 'requests', 'httpx', 'urllib3', 'socket', 'http')]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_mic_calibrates_threshold_to_the_room(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    from brain.hearing import MicAudioSource
    mic = MicAudioSource(clock=lambda: 100.0)  # fake sound card: fill the queue by hand
    mic.vad.reset(t0=101.0)
    rng = np.random.default_rng(0)
    room = lambda s: rng.normal(0, 10 ** (-50 / 20), int(s * SR)).astype(np.float32)
    # 1 s of room noise to calibrate, then a noise bump at -44 dBFS that would have
    # passed the default -45 threshold, then real speech.
    for part in (room(1.0), room(0.5), room(0.6) * 10 ** (6 / 20), room(0.6),
                 tone(1.0, -20) + room(1.0), room(1.0)):
        mic._q.put(part)
    heard = mic.poll() + [HeardAudio(s, t, dbfs(s), "mic") for s, t in mic.vad.flush()]
    assert mic.room_dbfs == pytest.approx(-50, abs=0.5)
    assert mic.vad.threshold == pytest.approx(-38, abs=0.5)
    assert len(heard) == 1 and heard[0].source == "mic"
    assert heard[0].start_s == pytest.approx(101.0 + 1.7 - 0.15, abs=0.05)


def test_streaming_ears_only_finish_a_phrase_after_the_person_stops(house, clip):
    """Live mode: the robot hears the clip as it plays and can't hand a phrase on early."""
    from brain.hearing import load_wav
    now = {"t": 100.0}
    ears = SimAudioSource(house, stream=True, clock=lambda: now["t"], time_key="wall_time")
    house.info["last_utterance"] = {"speaker": "user", "phrase": "x", "wav": str(clip), "sim_time": 0.0,
                                    "wall_time": 100.0}
    clip_s = len(load_wav(clip)[0]) / load_wav(clip)[1]
    got, t_done = [], None
    while now["t"] < 100.0 + clip_s + 2.0:
        heard = ears.poll()
        if heard and t_done is None:
            got, t_done = heard, now["t"]
        now["t"] += 0.02
    assert len(got) == 1
    h = got[0]
    speech_end = h.start_s + h.duration_s
    assert t_done >= speech_end + 0.3  # the detector waited to hear silence after the speech
    assert t_done <= 100.0 + clip_s + 1.0
    assert not ears.hearing
    assert 0.8 < h.duration_s < clip_s + 0.2


def test_streaming_ears_hear_nothing_before_the_clip_starts(house, clip):
    now = {"t": 50.0}
    ears = SimAudioSource(house, stream=True, clock=lambda: now["t"], time_key="wall_time")
    house.info["last_utterance"] = {"speaker": "user", "phrase": "x", "wav": str(clip), "sim_time": 0.0,
                                    "wall_time": 60.0}  # plays in 10 s
    assert ears.poll() == [] and ears.hearing
    now["t"] = 59.0
    assert ears.poll() == []
