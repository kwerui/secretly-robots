"""Show what the robot hears. Onboard only: no network, no API key.

    sim/.venv/bin/python -m scripts.hearing_demo            # the sofa person speaks every phrase in the sim
    sim/.venv/bin/python -m scripts.hearing_demo --mic 15   # listen to the laptop mic for 15 s

Each heard speech segment is printed and saved to logs/heard/ so you can play
back exactly what the robot's ears picked up (e.g. `afplay logs/heard/sim_01.wav`).
Speech-to-text is the next stage; this only finds and captures the speech.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from brain.hearing import MicAudioSource, SimAudioSource

OUT = Path("logs/heard")


def show(h, i):
    path = h.save_wav(OUT / f"{h.source}_{i:02d}.wav")
    where = f" from {h.distance_m:.2f} m" if h.distance_m is not None else ""
    print(f"  heard speech at t={h.start_s:8.2f}s  {h.duration_s:.2f}s long  {h.level_dbfs:6.1f} dBFS{where}  -> {path}")


def run_sim(play: bool) -> None:
    from sim.house_env import HouseEnv
    from sim.person_voice import PHRASES, PersonVoice, clip_seconds

    env = HouseEnv(seed=0)
    env.reset("tidy", "kitchen_counter", jitter=False)
    voice = PersonVoice(seed=0, play_audio=play)
    ears = SimAudioSource(env)
    print(f"robot ears are {ears.mouth_to_ears_m():.2f} m from the person's head")
    n = 0
    for phrase in PHRASES:
        utt = voice.speak(sim_time=env.data.time, phrase=phrase)
        if utt is None:
            raise SystemExit("no voice clips; see sim/README.md (Person's voice)")
        env.info["last_utterance"] = {"speaker": utt.speaker, "phrase": utt.phrase,
                                      "wav": str(utt.wav), "sim_time": utt.sim_time}
        print(f'person (caption only, robot does not read this): "{utt.phrase}"')
        for h in ears.poll():
            n += 1
            show(h, n)
        if play:
            time.sleep(clip_seconds(utt.wav) + 0.3)
        env.step(int(round(3.0 / env.model.opt.timestep)))  # 3 s of sim between lines
    print(f"{n} segment(s) heard for {len(PHRASES)} phrases")


def run_mic(seconds: float) -> None:
    mic = MicAudioSource()
    print("calibrating to the room for 1 s, stay quiet...")
    n = 0
    with mic:
        end = time.time() + seconds
        announced = False
        while time.time() < end:
            if mic.room_dbfs is not None and not announced:
                print(f"room {mic.room_dbfs:.1f} dBFS, speech threshold {mic.vad.threshold:.1f} dBFS. Speak now.")
                announced = True
            for h in mic.poll():
                n += 1
                show(h, n)
            time.sleep(0.05)
    print(f"{n} segment(s) heard")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mic", type=float, metavar="SECONDS", help="listen to the laptop mic instead of the sim")
    ap.add_argument("--quiet", action="store_true", help="sim mode: don't play the clips aloud")
    args = ap.parse_args()
    if args.mic:
        run_mic(args.mic)
    else:
        run_sim(play=not args.quiet)


if __name__ == "__main__":
    main()
