"""The robot's ears: turn sound into speech segments, onboard.

Two sources produce the same thing, a `HeardAudio` segment (16 kHz mono float32,
ready for speech-to-text):

- `SimAudioSource` hears the person in the MuJoCo house. When the sim's person
  speaks, it loads that clip's *audio*, makes it quieter with distance from the
  person's head to the robot's head camera, adds a little room noise, and runs it
  through the same voice-activity detector as the mic. It never reads the
  phrase text the sim keeps for captions: the robot has to work from sound.
- `MicAudioSource` hears the laptop microphone (needs `sounddevice`), so someone
  can speak to the robot live.

Everything here is numpy plus the standard library. Nothing goes to the network.
Speech-to-text (stage 3) takes `HeardAudio.samples` and passes its transcript to
`brain.command.CommandListener.heard()`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import queue
import time
import wave
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

SAMPLE_RATE = 16_000  # what Whisper and Gemma audio expect


@dataclass(frozen=True)
class HeardAudio:
    """One stretch of speech the robot heard."""
    samples: np.ndarray = field(repr=False)  # float32, mono, SAMPLE_RATE
    start_s: float  # when the speech began (sim time for the sim source, wall time for the mic)
    level_dbfs: float  # loudness of the speech as it reached the robot
    source: str  # "sim" or "mic"
    distance_m: float | None = None  # mouth to ears; sim only
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def save_wav(self, path: str | Path) -> Path:
        """Write what the robot heard, e.g. to listen back while tuning."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(self.samples, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return path


# ------------------------------------------------------------------ signal helpers

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> (float32 mono in [-1, 1], sample rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample(audio: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Linear-interpolation resample, with a short box filter first when downsampling.

    Good enough for speech going into speech-to-text; not hi-fi.
    """
    if rate == target or len(audio) == 0:
        return audio.astype(np.float32)
    if target < rate:
        k = max(1, int(round(rate / target)))
        if k > 1:
            audio = np.convolve(audio, np.ones(k) / k, mode="same")
    n = int(round(len(audio) * target / rate))
    t_src = np.arange(len(audio)) / rate
    t_dst = np.arange(n) / target
    return np.interp(t_dst, t_src, audio).astype(np.float32)


def dbfs(x: np.ndarray) -> float:
    """RMS level in dB relative to full scale (silence -> -120)."""
    if len(x) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * np.log10(max(rms, 1e-6))


def distance_gain(distance_m: float, ref_m: float = 1.0) -> float:
    """Inverse-distance law: -6 dB per doubling past ref_m, no boost when closer."""
    return ref_m / max(distance_m, ref_m)


# ------------------------------------------------------------------ voice activity

class EnergyVAD:
    """Streaming voice-activity detector based on loudness.

    Feed audio in any chunk size. A segment starts when a frame is louder than
    `threshold_dbfs`, keeps `pre_roll_ms` of audio before that so the first
    syllable isn't clipped, and ends after `hangover_ms` of quiet. Segments
    shorter than `min_speech_ms` (clicks, bumps) are dropped. Simple, and
    deterministic for tests; swap in Silero VAD if real rooms are too noisy.
    """

    def __init__(
        self,
        threshold_dbfs: float = -45.0,
        frame_ms: int = 30,
        pre_roll_ms: int = 150,
        hangover_ms: int = 400,
        min_speech_ms: int = 250,
        max_speech_s: float = 15.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.threshold = threshold_dbfs
        self.rate = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self._pre_frames = max(1, pre_roll_ms // frame_ms)
        self._hang_frames = max(1, hangover_ms // frame_ms)
        self._min_frames = max(1, min_speech_ms // frame_ms)
        self._max_frames = int(max_speech_s * 1000 / frame_ms)
        self.reset()

    def reset(self, t0: float = 0.0) -> None:
        self._buf = np.zeros(0, np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=self._pre_frames)
        self._speech: list[np.ndarray] = []
        self._voiced = 0  # loud frames in the current segment
        self._quiet = 0  # quiet frames since the last loud one
        self._t = t0  # time of the next frame's start
        self._seg_start = 0.0

    def feed(self, audio: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return finished (samples, start_time) segments."""
        out = []
        self._buf = np.concatenate([self._buf, audio.astype(np.float32)])
        n = len(self._buf) // self.frame
        for i in range(n):
            f = self._buf[i * self.frame:(i + 1) * self.frame]
            seg = self._step(f)
            if seg is not None:
                out.append(seg)
            self._t += self.frame / self.rate
        self._buf = self._buf[n * self.frame:]
        return out

    def flush(self) -> list[tuple[np.ndarray, float]]:
        """End of stream: close any open segment."""
        seg = self._close() if self._speech else None
        return [seg] if seg is not None else []

    def _step(self, f: np.ndarray):
        loud = dbfs(f) > self.threshold
        if not self._speech:
            if loud:
                self._seg_start = self._t - len(self._pre) * self.frame / self.rate
                self._speech = list(self._pre) + [f]
                self._voiced, self._quiet = 1, 0
            else:
                self._pre.append(f)
            return None
        self._speech.append(f)
        if loud:
            self._voiced += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet >= self._hang_frames or len(self._speech) >= self._max_frames:
            return self._close()
        return None

    def _close(self):
        voiced, speech, start = self._voiced, self._speech, self._seg_start
        self._speech, self._voiced, self._quiet = [], 0, 0
        self._pre.clear()
        if voiced < self._min_frames:
            return None
        # Trim the trailing quiet tail down to one frame.
        keep = len(speech) - max(0, self._quiet_tail(speech) - 1)
        return np.concatenate(speech[:keep]), start

    def _quiet_tail(self, frames: list[np.ndarray]) -> int:
        n = 0
        for f in reversed(frames):
            if dbfs(f) > self.threshold:
                break
            n += 1
        return n


def _segments_to_heard(segs, source: str, distance_m: float | None = None) -> list[HeardAudio]:
    return [HeardAudio(s, t, dbfs(s), source, distance_m) for s, t in segs]


# ------------------------------------------------------------------ sources

class SimAudioSource:
    """The robot's microphone inside the MuJoCo house.

    Call `poll()` each control tick; it returns speech segments the robot has finished
    hearing. Speech comes from the sim's `env.info["last_utterance"]`, of which only the clip
    path and start time are used; the phrase text is ignored.

    stream=True (use this in a live demo): the robot hears the clip as it plays, only the
    audio produced up to `clock()`, and a segment ends only after the voice-activity detector
    has heard the speech stop (its hangover), so nothing downstream can act before the person
    has finished. `clock` is the time base of the utterance's `time_key` (sim time by default;
    pass time.time with time_key="wall_time" to follow the audio playing on the speakers).

    stream=False hears the whole clip at once (tests, batch checks).
    """

    def __init__(
        self,
        env,
        mouth: str = "user_head",
        ears: str = "cam_robot_head",
        noise_dbfs: float = -65.0,
        vad: EnergyVAD | None = None,
        seed: int | None = 0,
        stream: bool = False,
        clock: Callable[[], float] | None = None,
        time_key: str = "sim_time",
    ):
        self.env = env
        self.mouth, self.ears = mouth, ears
        self.noise_dbfs = noise_dbfs
        self.vad = vad or EnergyVAD()
        self.rng = np.random.default_rng(seed)
        self.stream = stream
        self.clock = clock or (lambda: float(self.env.data.time))
        self.time_key = time_key
        self._seen: tuple | None = None
        self._buf: np.ndarray | None = None  # what the mic will pick up for the current utterance
        self._fed = 0
        self._t0 = 0.0
        self._dist: float | None = None

    @property
    def hearing(self) -> bool:
        """True while an utterance is still coming in (the robot is listening to it)."""
        return self._buf is not None

    def mouth_to_ears_m(self) -> float:
        m, d = self.env.model, self.env.data
        mouth = d.geom_xpos[m.geom(self.mouth).id]
        ears = d.cam_xpos[m.camera(self.ears).id]
        return float(np.linalg.norm(mouth - ears))

    def _mic_audio(self, wav: Path, before_s: float = 0.5, after_s: float = 0.5) -> tuple[np.ndarray, float]:
        """The clip as it reaches the robot's head: quieter with distance, with room noise."""
        audio, rate = load_wav(wav)
        audio = resample(audio, rate)
        dist = self.mouth_to_ears_m()
        audio = audio * distance_gain(dist)
        audio = np.concatenate([np.zeros(int(SAMPLE_RATE * before_s), np.float32), audio,
                                np.zeros(int(SAMPLE_RATE * after_s), np.float32)])
        noise_rms = 10 ** (self.noise_dbfs / 20)
        return audio + self.rng.normal(0, noise_rms, len(audio)).astype(np.float32), dist

    def poll(self) -> list[HeardAudio]:
        heard: list[HeardAudio] = []
        utt = (getattr(self.env, "info", None) or {}).get("last_utterance")
        if utt:
            key = (utt.get("wav"), utt.get("sim_time"), utt.get("wall_time"))
            if key != self._seen:
                self._seen = key
                start = float(utt.get(self.time_key, utt.get("sim_time", 0.0)))
                if not self.stream:
                    return self.hear_clip(Path(utt["wav"]), start)
                if self._buf is not None:  # cut off by a new utterance: close what we had
                    heard += _segments_to_heard(self.vad.flush(), "sim", self._dist)
                # Room either side; 1 s after so the detector hears the speech end.
                self._buf, self._dist = self._mic_audio(Path(utt["wav"]), 0.5, 1.0)
                self._fed, self._t0 = 0, start - 0.5
                self.vad.reset(t0=self._t0)
        if self._buf is not None:
            upto = int(np.clip((self.clock() - self._t0) * SAMPLE_RATE, 0, len(self._buf)))
            if upto > self._fed:
                heard += _segments_to_heard(self.vad.feed(self._buf[self._fed:upto]), "sim", self._dist)
                self._fed = upto
            if self._fed >= len(self._buf):
                heard += _segments_to_heard(self.vad.flush(), "sim", self._dist)
                self._buf = None
        return heard

    def hear_clip(self, wav: Path, sim_time: float) -> list[HeardAudio]:
        """What the robot's mic picks up when `wav` is spoken at the person's head (all at once)."""
        audio, dist = self._mic_audio(wav)
        self.vad.reset(t0=sim_time - 0.5)
        segs = self.vad.feed(audio) + self.vad.flush()
        return _segments_to_heard(segs, "sim", dist)


class MicAudioSource:
    """The laptop microphone, as the robot's ears. Call `poll()` regularly.

    Uses `sounddevice` (PortAudio). Audio stays in memory on this machine.
    """

    def __init__(self, vad: EnergyVAD | None = None, device: int | str | None = None,
                 clock: Callable[[], float] = time.time, calibrate_s: float = 1.0,
                 margin_db: float = 12.0):
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on host
            raise RuntimeError("Install requirements-hearing.txt to use the microphone") from exc
        self._sd = sd
        self.vad = vad or EnergyVAD()
        self.device = device
        self._clock = clock
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        # The first calibrate_s of audio measures the room; the speech threshold
        # is then set margin_db above it (never below the VAD's own default).
        self.calibrate_s, self.margin_db = calibrate_s, margin_db
        self._calib: list[np.ndarray] = []
        self.room_dbfs: float | None = None

    def _calibrate(self, chunk: np.ndarray) -> np.ndarray:
        """Swallow audio until calibration is done; return what's left over."""
        if self.room_dbfs is not None or self.calibrate_s <= 0:
            return chunk
        self._calib.append(chunk)
        have = np.concatenate(self._calib)
        need = int(self.calibrate_s * SAMPLE_RATE)
        if len(have) < need:
            return np.zeros(0, np.float32)
        self.room_dbfs = dbfs(have[:need])
        self.vad.threshold = max(self.vad.threshold, self.room_dbfs + self.margin_db)
        self._calib = []
        return have[need:]

    @property
    def hearing(self) -> bool:
        """True while an utterance is still coming in (same meaning as SimAudioSource.hearing)."""
        return bool(self.vad._speech)

    def start(self) -> "MicAudioSource":
        self.vad.reset(t0=self._clock() + self.calibrate_s)
        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.03), device=self.device,
            callback=lambda data, frames, t, status: self._q.put(data[:, 0].copy()),
        )
        self._stream.start()
        return self

    def poll(self) -> list[HeardAudio]:
        chunks = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return []
        audio = self._calibrate(np.concatenate(chunks))
        return _segments_to_heard(self.vad.feed(audio), "mic") if len(audio) else []

    def stop(self) -> list[HeardAudio]:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self.poll() + _segments_to_heard(self.vad.flush(), "mic")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def listen(sources: Iterable) -> list[HeardAudio]:
    """Poll every source once."""
    heard: list[HeardAudio] = []
    for s in sources:
        heard.extend(s.poll())
    return heard
