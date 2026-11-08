"""Onboard speech-to-text: HeardAudio -> Transcript, with Whisper on the CPU.

Uses faster-whisper (CTranslate2) with the `base.en` model loaded from
`models/whisper-base.en`. At run time the model is opened with
`local_files_only=True` and Hugging Face is set offline, so transcription never
touches the network. Fetch the model once beforehand:

    sim/.venv/bin/pip install -r requirements-stt.txt
    sim/.venv/bin/python -m brain.speech_to_text --download
    sim/.venv/bin/python -m brain.speech_to_text path/to/clip.wav   # transcribe a file

The transcript then goes to `brain.command.CommandListener.heard(text, stt_ms)`.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .hearing import SAMPLE_RATE, HeardAudio, dbfs, load_wav, resample

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "base.en"
MODELS_DIR = ROOT / "models"


def model_dir(name: str = DEFAULT_MODEL) -> Path:
    return MODELS_DIR / f"whisper-{name}"


@dataclass(frozen=True)
class Transcript:
    text: str  # "" when Whisper thinks it wasn't speech
    start_s: float
    audio_s: float
    stt_ms: float
    source: str
    level_dbfs: float
    distance_m: float | None = None
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    model: str = DEFAULT_MODEL

    @property
    def is_speech(self) -> bool:
        return bool(self.text)


class WhisperSTT:
    """Local Whisper. Loads once (~0.3 s), then ~0.2-0.3 s per short command on CPU.

    `model` can be an already-built object with faster-whisper's `transcribe`
    API, which tests use to avoid loading real weights.
    """

    # Whisper sometimes "hears" words in noise. Drop a result that it is itself
    # unsure is speech and that has low confidence.
    NO_SPEECH_MAX = 0.6
    LOGPROB_MIN = -1.0

    def __init__(self, name: str = DEFAULT_MODEL, path: Path | None = None,
                 compute_type: str = "int8", model=None,
                 clock: Callable[[], float] = time.perf_counter):
        self.name = name
        self._clock = clock
        if model is None:
            path = Path(path or model_dir(name))
            if not (path / "model.bin").exists():
                raise RuntimeError(f"Whisper model not found in {path}; run "
                                   "`python -m brain.speech_to_text --download` once while online")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from faster_whisper import WhisperModel
            model = WhisperModel(str(path), device="cpu", compute_type=compute_type,
                                 local_files_only=True)
        self.model = model

    def warm_up(self) -> None:
        """The first call is slower; do it before anyone speaks."""
        import numpy as np
        self._run(np.zeros(SAMPLE_RATE // 2, np.float32))

    def _run(self, samples):
        segments, _info = self.model.transcribe(
            samples, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
        )
        return list(segments)

    def transcribe(self, heard: HeardAudio) -> Transcript:
        t0 = self._clock()
        segments = self._run(heard.samples)
        stt_ms = round((self._clock() - t0) * 1000, 1)
        kept = [s for s in segments
                if not (s.no_speech_prob > self.NO_SPEECH_MAX and s.avg_logprob < self.LOGPROB_MIN)]
        text = " ".join(s.text.strip() for s in kept).strip()
        nsp = max((s.no_speech_prob for s in segments), default=1.0)
        alp = min((s.avg_logprob for s in segments), default=0.0)
        return Transcript(text, heard.start_s, round(heard.duration_s, 3), stt_ms, heard.source,
                          round(heard.level_dbfs, 1), heard.distance_m, round(nsp, 3), round(alp, 3),
                          self.name)


def download(name: str = DEFAULT_MODEL) -> Path:
    """Fetch the model once (needs internet). Run time never calls this."""
    from faster_whisper import download_model
    return Path(download_model(name, output_dir=str(model_dir(name))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="*", help="WAV files to transcribe")
    ap.add_argument("--download", action="store_true", help="fetch the model (needs internet, once)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="e.g. base.en (default), small.en, tiny.en")
    args = ap.parse_args()
    if args.download:
        print("model in", download(args.model))
    if args.wav:
        stt = WhisperSTT(args.model)
        stt.warm_up()
        for w in args.wav:
            audio, rate = load_wav(w)
            audio = resample(audio, rate)
            t = stt.transcribe(HeardAudio(audio, 0.0, dbfs(audio), "file"))
            print(f"{t.stt_ms:6.0f} ms  {w}: {t.text!r}")


if __name__ == "__main__":
    main()
