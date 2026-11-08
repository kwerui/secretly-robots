"""Run the robot's onboard listening with a live speech-to-text window.

    sim/.venv/bin/mjpython -m scripts.voice_robot           # sim viewer; V: person speaks, Y/N: person answers
    sim/.venv/bin/python   -m scripts.voice_robot --mic     # talk to it: test phrasing, hear its replies

Both open http://127.0.0.1:8765 showing each transcript (text, how long Whisper
took, loudness, distance) and the command state it produced. Everything runs
on this laptop; nothing is sent anywhere. On macOS the sim viewer needs mjpython.
"""
from __future__ import annotations

import argparse
import time

from brain.hearing import MicAudioSource, SimAudioSource
from brain.robot_voice import RobotVoice
from brain.speech_to_text import WhisperSTT
from brain.voice_pipeline import VoicePipeline, default_listener, load_gemma_in_background
from brain.voice_window import PORT, VoiceEvents, serve


def build(port: int, open_browser: bool, with_commands: bool, use_gemma: bool = True):
    events = VoiceEvents()
    url = serve(events, port, open_browser=open_browser)
    print(f"speech-to-text window: {url}")
    events.add("status", message="loading Whisper…", listening=False)
    stt = WhisperSTT()
    stt.warm_up()
    robot = RobotVoice()

    def say(key: str) -> str:
        text = robot.say(key)
        events.add("robot_said", text=text)
        print(f'  robot: "{text}"')
        return text

    listener = default_listener(say=say) if with_commands else None
    if with_commands and listener is None:
        print("brain.command not available: showing transcripts only")
    pipeline = VoicePipeline(stt, events, listener).start()
    pipeline.interpreter_name = "rules"
    if listener is not None and use_gemma:
        pipeline.interpreter_name = "rules (Gemma 4 loading…)"

        def ready(g):
            pipeline.interpreter_name = "rules + Gemma 4" if g else "rules (Gemma 4 unavailable)"
            print("interpreter: rules + Gemma 4 (local Ollama)" if g else
                  "interpreter: rules only (run `ollama serve`; `ollama pull gemma4:e4b`)", flush=True)
            events.add("status", listening=True, message=f"interpreter now: {pipeline.interpreter_name}")

        load_gemma_in_background(listener, ready)
    return events, pipeline, robot


def run_sim(args) -> None:
    from sim.house_env import HouseEnv, Mode, cmd_view

    events, pipeline, _robot = build(args.port, not args.no_browser, not args.no_commands, not args.no_gemma)
    env = HouseEnv(seed=args.seed)
    ears: list[SimAudioSource] = []

    def on_tick(e):
        if not ears:  # the viewer resets the env first; attach the ears after that
            ears.append(SimAudioSource(e, stream=True))  # hears the clip as sim time passes
            events.add("status", message=f"listening in the sim · V: person speaks · Y/N: answers · Whisper {pipeline.stt.name} · {pipeline.interpreter_name}",
                       listening=True)
        pipeline.tick()
        for h in ears[0].poll():  # only once it has heard the person stop talking
            pipeline.submit(h)

    try:
        cmd_view(env, args.layout, args.target, args.seed, mode=Mode(args.mode), on_tick=on_tick)
    finally:
        pipeline.stop()


def describe(cmd: dict) -> str:
    """One line for a command event: what the robot made of what it heard."""
    st, task = cmd.get("status"), cmd.get("task") or {}
    src = cmd.get("parser_source") or "-"
    ms = cmd.get("intent_ms")
    took = f"{src}, {ms:.0f} ms" if ms is not None else src
    if st == "understood":
        return f"FETCH {task.get('item_id')} -> {task.get('destination')}  ({took})"
    if st == "confirming":
        return f"HINT for {task.get('item_id')}: asks and waits for yes/no  ({took})"
    if st == "declined":
        return f"DECLINED: {cmd.get('reason', '')}"
    return f"NOT A REQUEST: {cmd.get('reason', '')}  ({took})"


def run_mic(args) -> None:
    """Talk to the robot through the laptop mic and see what it makes of each phrase."""
    import csv
    from pathlib import Path

    events, pipeline, robot = build(args.port, not args.no_browser, not args.no_commands, not args.no_gemma)
    mic = MicAudioSource()
    print("calibrating to the room for 1 s, stay quiet…  (Ctrl+C to stop and save a summary)")
    print("tip: use headphones, or wait for the robot to finish talking before you speak")
    events.add("status", message="calibrating to the room…", listening=False)
    announced, seen, rows = False, 0, []
    try:
        with mic:
            while True:
                if mic.room_dbfs is not None and not announced:
                    announced = True
                    events.add("status", listening=True,
                               message=f"listening on the mic · room {mic.room_dbfs:.0f} dBFS · Whisper {pipeline.stt.name} · {pipeline.interpreter_name}")
                    print(f"listening (interpreter: {pipeline.interpreter_name}), speak now\n")
                pipeline.tick()
                for h in mic.poll():
                    if h.start_s < robot.speaking_until:
                        continue  # the robot's own voice through the speakers
                    pipeline.submit(h)
                for e in events.since(seen):
                    seen = e["seq"] + 1
                    if e["kind"] == "transcript":
                        print(f'you said:  "{e["text"] or "(not speech)"}"   [speech-to-text {e["stt_ms"]:.0f} ms]')
                        if not e["text"]:
                            print()
                    elif e["kind"] == "command":
                        line = describe(e)
                        print(f"  robot:   {line}")
                        if e.get("status") != "heard":
                            rows.append({"said": e.get("transcript", ""), "result": line})
                        if e.get("status") in ("understood", "declined", "rejected"):
                            print()
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if rows:
            out = Path(__file__).resolve().parent.parent / "logs" / "phrase_test.csv"
            out.parent.mkdir(exist_ok=True)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["said", "result"])
                w.writeheader()
                w.writerows(rows)
            print(f"\n{len(rows)} phrase(s):")
            for r in rows:
                print(f'  {r["said"][:50]:52} {r["result"]}')
            print(f"saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mic", action="store_true", help="listen to the laptop microphone, no sim")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true", help="don't open the window automatically")
    ap.add_argument("--no-commands", action="store_true", help="transcripts only, no command state")
    ap.add_argument("--no-gemma", action="store_true", help="rules-only interpreter, skip the local Gemma model")
    ap.add_argument("--mode", choices=["static", "shuffled", "dynamic"], default="static")
    ap.add_argument("--layout")
    ap.add_argument("--target")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    (run_mic if args.mic else run_sim)(args)


if __name__ == "__main__":
    main()
