"""The robot's spoken replies, in a Gemini text-to-speech voice, played onboard.

The lines are generated once with Gemini TTS and saved in `assets/robot_voice/`.
At run time the robot only plays those local files, so speaking needs no network.

    sim/.venv/bin/python -m brain.robot_voice --generate   # needs GEMINI_API_KEY, once
    sim/.venv/bin/python -m brain.robot_voice --say ask_water
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "assets" / "robot_voice"

LINES: dict[str, str] = {
    "ask_water": "Would you like some water?",
    "will_fetch_water": "Okay, I'll get you some water.",
    "will_fetch_cup": "Okay, I'll get your cup.",
    "will_fetch_book": "Okay, I'll get your book.",
    "declined": "Okay. Just let me know if you need anything.",
}

TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
TTS_VOICE = os.getenv("ROBOT_TTS_VOICE", "Kore")  # distinct from the person's voice
STYLE = "Say this as a calm, friendly home-care helper robot, clearly and gently"


# What the robot says when it takes on a fetch, by item.
ACKNOWLEDGE = {"water_bottle": "will_fetch_water", "cup": "will_fetch_cup", "book": "will_fetch_book"}


def clip_path(key: str) -> Path:
    return VOICE_DIR / f"{key}.wav"


def clip_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


class RobotVoice:
    """Plays a stored line without blocking. `speaking_until` lets the robot's
    own microphone ignore what it says (so it doesn't hear itself)."""

    def __init__(self, play_audio: bool = True, clock=time.time):
        self.play_audio = play_audio
        self._clock = clock
        self._proc: subprocess.Popen | None = None
        self.speaking_until = 0.0
        self.said: list[str] = []

    def say(self, key: str) -> str:
        """Speak line `key`; returns its text. Missing clip -> text only, with a note."""
        text = LINES[key]
        path = clip_path(key)
        self.said.append(key)
        if not path.exists():
            print(f"robot_voice: missing {path.name}; run `python -m brain.robot_voice --generate`")
            return text
        self.speaking_until = self._clock() + clip_seconds(path) + 0.4
        if self.play_audio:
            self.stop()
            cmd = _player_cmd(path)
            if cmd:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return text

    def is_speaking(self) -> bool:
        return self._clock() < self.speaking_until

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


def _player_cmd(path: Path) -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay", str(path)]
    for exe in ("paplay", "aplay"):
        if shutil.which(exe):
            return [exe, str(path)]
    return None


def generate(force: bool = False, model: str = TTS_MODEL, voice: str = TTS_VOICE, client=None) -> list[Path]:
    """Make the robot's clips with Gemini TTS (build time only, needs the network)."""
    if client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set; it is only needed to generate clips")
        from google import genai  # build-time only; the robot's run time never imports this
        client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    made = []
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for k, text in LINES.items():
        path = clip_path(k)
        if path.exists() and not force:
            continue
        print(f"  generating {path.name} ...", flush=True)
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": text, "speech_metadata": {"style": STYLE}}]}],
            config={"response_modalities": ["AUDIO"], "speech_config": {"voice_config": {"voice": voice}}},
        )
        data = resp.candidates[0].content.parts[0].inline_data.data
        if data[:4] == b"RIFF":
            path.write_bytes(data)
        else:  # headerless 24 kHz 16-bit mono PCM
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(24_000)
                w.writeframes(data)
        made.append(path)
    (VOICE_DIR / "manifest.json").write_text(json.dumps(
        {"generator": "Gemini text-to-speech", "model": model, "voice": voice, "style": STYLE,
         "lines": LINES}, indent=2) + "\n")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--say", choices=list(LINES))
    args = ap.parse_args()
    if args.generate:
        print(f"made {len(generate(force=args.force))} clip(s) in {VOICE_DIR}")
    if args.say:
        v = RobotVoice()
        print(f'robot: "{v.say(args.say)}"')
        if v._proc:
            v._proc.wait()


if __name__ == "__main__":
    main()
