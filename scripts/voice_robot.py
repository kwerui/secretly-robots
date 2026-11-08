"""Run the robot's onboard listening with a live speech-to-text window.

    sim/.venv/bin/mjpython -m scripts.voice_robot           # sim viewer; press V to make the person speak
    sim/.venv/bin/python   -m scripts.voice_robot --mic     # laptop microphone instead of the sim

Both open http://127.0.0.1:8765 showing each transcript (text, how long Whisper
took, loudness, distance) and the command state it produced. Everything runs
on this laptop; nothing is sent anywhere. On macOS the sim viewer needs mjpython.
"""
from __future__ import annotations

import argparse
import time

from brain.hearing import MicAudioSource, SimAudioSource
from brain.speech_to_text import WhisperSTT
from brain.voice_pipeline import VoicePipeline, default_listener
from brain.voice_window import PORT, VoiceEvents, serve


def build(port: int, open_browser: bool, with_commands: bool):
    events = VoiceEvents()
    url = serve(events, port, open_browser=open_browser)
    print(f"speech-to-text window: {url}")
    events.add("status", message="loading Whisper…", listening=False)
    stt = WhisperSTT()
    stt.warm_up()
    listener = default_listener() if with_commands else None
    if with_commands and listener is None:
        print("brain.command not available: showing transcripts only")
    return events, VoicePipeline(stt, events, listener).start()


def run_sim(args) -> None:
    from sim.house_env import HouseEnv, Mode, cmd_view

    events, pipeline = build(args.port, not args.no_browser, not args.no_commands)
    env = HouseEnv(seed=args.seed)
    ears: list[SimAudioSource] = []

    def on_tick(e):
        if not ears:  # the viewer resets the env first; attach the ears after that
            ears.append(SimAudioSource(e))
            events.add("status", message=f"listening in the sim · press V · Whisper {pipeline.stt.name}",
                       listening=True)
        for h in ears[0].poll():
            # Transcribe once the person has finished saying it (sim time ~ wall time in the viewer).
            ready = time.time() + max(0.0, h.start_s + h.duration_s - e.data.time)
            pipeline.submit(h, ready)

    try:
        cmd_view(env, args.layout, args.target, args.seed, mode=Mode(args.mode), on_tick=on_tick)
    finally:
        pipeline.stop()


def run_mic(args) -> None:
    events, pipeline = build(args.port, not args.no_browser, not args.no_commands)
    mic = MicAudioSource()
    print("calibrating to the room for 1 s, stay quiet…  (Ctrl+C to stop)")
    events.add("status", message="calibrating to the room…", listening=False)
    announced = False
    try:
        with mic:
            while True:
                if mic.room_dbfs is not None and not announced:
                    announced = True
                    events.add("status", listening=True,
                               message=f"listening on the mic · room {mic.room_dbfs:.0f} dBFS · Whisper {pipeline.stt.name}")
                    print("listening, speak now")
                for h in mic.poll():
                    pipeline.submit(h)
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mic", action="store_true", help="listen to the laptop microphone, no sim")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true", help="don't open the window automatically")
    ap.add_argument("--no-commands", action="store_true", help="transcripts only, no command state")
    ap.add_argument("--mode", choices=["static", "shuffled", "dynamic"], default="static")
    ap.add_argument("--layout")
    ap.add_argument("--target")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    (run_mic if args.mic else run_sim)(args)


if __name__ == "__main__":
    main()
