"""The seated person's voice: short spoken requests that all mean "get me water".

The clips are made once with Gemini text-to-speech and saved in
`assets/voice/`, so the sim plays them with no network or API key. Press V in
the viewer to have the person on the sofa say one.

    cd sim
    .venv/bin/pip install -r requirements-voice.txt   # only needed to (re)generate
    export GEMINI_API_KEY=...                          # never commit a real key
    .venv/bin/python person_voice.py --generate        # writes assets/voice/*.wav
    .venv/bin/python person_voice.py --say             # play one clip from the terminal

Each spoken line is recorded as an `Utterance` (what was said, which clip,
and when, in sim time). This is the person's side only. The robot's onboard
listening is separate and must work from the audio, not from `phrase`, which
the sim only uses for the caption and logs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE_DIR = HERE / "assets" / "voice"
MANIFEST = VOICE_DIR / "manifest.json"

# Every phrase is a request for water, put in different ways: direct requests and
# stated needs. The robot has to understand all of them.
PHRASES: tuple[str, ...] = (
    "I am thirsty.",
    "I need water.",
    "I need a drink.",
    "Could I have some water, please?",
    "My mouth is so dry.",
    "I'm parched.",
    "Can you get me something to drink?",
    "I'd love a glass of water.",
)

# Replies to the robot's "Would you like some water?" (keys Y and N in the viewer).
ANSWERS: dict[str, str] = {
    "yes": "Yes, please.",
    "no": "No, thank you.",
}
ALL_LINES: tuple[str, ...] = PHRASES + tuple(ANSWERS.values())

TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
# "Gacrux" is one of Gemini's prebuilt voices, described as mature.
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Gacrux")
STYLE = "Say this in the warm, slightly tired voice of an older person resting on a sofa"


def slug(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def clip_path(phrase: str) -> Path:
    return VOICE_DIR / f"{slug(phrase)}.wav"


@dataclass(frozen=True)
class Utterance:
    phrase: str
    wav: Path
    sim_time: float
    speaker: str = "user"


# ---------------------------------------------------------------- generation

def _pcm_rate(mime: str | None, default: int = 24_000) -> int:
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else default


def write_wav(path: Path, pcm: bytes, rate: int = 24_000) -> None:
    """Gemini TTS returns raw 16-bit mono PCM; wrap it in a WAV header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def save_audio(path: Path, audio: bytes, rate: int = 24_000) -> None:
    """Write Gemini audio to a WAV: as-is if it already has a RIFF header, else wrap the PCM."""
    if audio[:4] == b"RIFF":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
    else:
        write_wav(path, audio, rate)


def synthesize(client, phrase: str, model: str = TTS_MODEL, voice: str = TTS_VOICE) -> tuple[bytes, int]:
    """One Gemini TTS call -> (audio bytes, sample rate).

    Gemini 3.8 TTS returns a complete WAV (24 kHz, mono, 16-bit) by default;
    older models returned headerless PCM. `save_audio` handles both.
    """
    resp = client.models.generate_content(
        model=model,
        contents=[{
            "role": "user",
            "parts": [{"text": phrase, "speech_metadata": {"style": STYLE}}],
        }],
        config={
            "response_modalities": ["AUDIO"],
            "speech_config": {"voice_config": {"voice": voice}},
        },
    )
    try:
        blob = resp.candidates[0].content.parts[0].inline_data
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Gemini returned no audio for {phrase!r}") from exc
    if not blob or not blob.data:
        raise RuntimeError(f"Gemini returned no audio for {phrase!r}")
    return blob.data, _pcm_rate(blob.mime_type)


def generate_clips(force: bool = False, model: str = TTS_MODEL, voice: str = TTS_VOICE, client=None) -> list[Path]:
    """Make any missing clips (all of them with force=True) and update the manifest."""
    if client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set; it is only needed to generate clips")
        try:
            from google import genai
        except ImportError:
            raise SystemExit("Install the SDK first: .venv/bin/pip install -r requirements-voice.txt")
        # TTS (speech_config, speech_metadata) only exists on v1beta; the SDK's
        # default v1 endpoint rejects those fields as unknown.
        client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})

    made = []
    for phrase in ALL_LINES:
        path = clip_path(phrase)
        if path.exists() and not force:
            continue
        print(f"  generating {path.name} ...", flush=True)
        audio, rate = synthesize(client, phrase, model=model, voice=voice)
        save_audio(path, audio, rate)
        made.append(path)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "generator": "Gemini text-to-speech",
        "model": model,
        "voice": voice,
        "style": STYLE,
        "clips": {p: clip_path(p).name for p in ALL_LINES},
    }, indent=2) + "\n")
    return made


# ---------------------------------------------------------------- playback

def _player_cmd(path: Path) -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay", str(path)]
    for exe in ("paplay", "aplay", "ffplay"):
        if shutil.which(exe):
            return [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)] if exe == "ffplay" else [exe, str(path)]
    return None


class PersonVoice:
    """Picks a phrase, plays its clip without blocking the sim, and records it."""

    def __init__(self, seed: int | None = None, play_audio: bool = True):
        self.rng = random.Random(seed)
        self.play_audio = play_audio
        self.history: list[Utterance] = []
        self._proc: subprocess.Popen | None = None

    def available(self) -> list[str]:
        return [p for p in PHRASES if clip_path(p).exists()]

    def next_phrase(self) -> str | None:
        """Random phrase with a clip, never the same one twice in a row."""
        pool = self.available()
        if self.history and len(pool) > 1:
            pool = [p for p in pool if p != self.history[-1].phrase]
        return self.rng.choice(pool) if pool else None

    def speak(self, sim_time: float = 0.0, phrase: str | None = None) -> Utterance | None:
        phrase = phrase or self.next_phrase()
        if phrase is None:
            print("person_voice: no clips in sim/assets/voice/; run `python person_voice.py --generate`")
            return None
        path = clip_path(phrase)
        if not path.exists():
            print(f"person_voice: missing clip {path.name}; run `python person_voice.py --generate`")
            return None
        if self.play_audio:
            self._play(path)
        utt = Utterance(phrase, path, float(sim_time))
        self.history.append(utt)
        return utt

    def is_speaking(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _play(self, path: Path) -> None:
        self.stop()
        cmd = _player_cmd(path)
        if cmd is None:
            print("person_voice: no audio player found (afplay/paplay/aplay/ffplay); caption only")
            return
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self.is_speaking():
            self._proc.terminate()
        self._proc = None


def clip_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="make missing clips with Gemini TTS")
    ap.add_argument("--force", action="store_true", help="with --generate: remake every clip")
    ap.add_argument("--model", default=TTS_MODEL, help=f"Gemini TTS model (default {TTS_MODEL})")
    ap.add_argument("--voice", default=TTS_VOICE, help=f"prebuilt voice (default {TTS_VOICE})")
    ap.add_argument("--say", action="store_true", help="play one random clip and wait for it")
    args = ap.parse_args()

    if args.generate:
        made = generate_clips(force=args.force, model=args.model, voice=args.voice)
        print(f"made {len(made)} clip(s); {len(PersonVoice().available())}/{len(PHRASES)} available in {VOICE_DIR}")
    if args.say:
        pv = PersonVoice()
        utt = pv.speak()
        if utt:
            print(f'user: "{utt.phrase}"')
            if pv._proc:
                pv._proc.wait()
    if not (args.generate or args.say):
        for p in PHRASES:
            print(("[ok]     " if clip_path(p).exists() else "[missing]") + f" {p}")


if __name__ == "__main__":
    main()
