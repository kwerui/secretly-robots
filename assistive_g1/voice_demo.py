"""Voice demo: the G1 stands in Jack's house, hears the person on the sofa, and goes into fetch mode.

    cd assistive_g1
    ../sim/.venv/bin/mjpython voice_demo.py          # macOS (the viewer needs mjpython)
    ../sim/.venv/bin/python   voice_demo.py          # Linux

Keys in the viewer:
    V   the person says one of eight "I want water" lines (some direct, some hints)
    Y/N the person answers the robot's question
    X   cancel fetch mode, back to listening

What happens, all onboard (no network):
    person speaks (Gemini TTS clip) -> the G1's head microphone hears it (distance falloff)
    -> Whisper speech-to-text -> rules / local Gemma 4 interpreter
    -> a hint ("My mouth is so dry") makes the robot ask "Would you like some water?" and wait
    -> a request, or a yes, puts the robot in FETCH MODE: the brain gets the task and the
       planner's first step is shown (not carried out yet) with a line to where it would search.

The robot keeps balancing with its GR00T policy the whole time. A light above its head shows
the mode (grey listening, amber asking, green fetch mode) and the viewer shows captions. The
speech-to-text window at http://127.0.0.1:8765 shows the same, with timings.
"""
from __future__ import annotations

from viewer_env import relaunch_for_viewer

relaunch_for_viewer()

import functools  # noqa: E402
import builtins  # noqa: E402

print = functools.partial(builtins.print, flush=True)  # console stays live under mjpython and pipes

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root: brain/, scripts/, sim/

from house import G1House  # noqa: E402
from skills import AssistiveRobot  # noqa: E402
from brain_bridge import RobotBridge  # noqa: E402
from house_brain_demo import learned_memory  # noqa: E402

from brain.hearing import SimAudioSource  # noqa: E402
from scripts.voice_robot import build  # noqa: E402
from sim.person_voice import ANSWERS, PersonVoice, clip_seconds  # noqa: E402

LIGHT = {"listening": (0.55, 0.55, 0.55, 0.9), "asking": (1.0, 0.7, 0.1, 0.95), "fetch": (0.15, 0.9, 0.35, 0.95)}
FONT = mujoco.mjtFontScale.mjFONTSCALE_150


class Demo:
    def __init__(self, args):
        self.args = args
        self.env = G1House(seed=args.seed)
        self.env.reset(args.layout, args.target, seed=args.seed)
        print("standing the G1 up…")
        self.robot = AssistiveRobot(world=self.env)
        self.bridge = RobotBridge(self.robot, learned_memory(), containers={}, say=lambda m: None)
        self.events, self.pipeline, self.robot_voice = build(args.port, not args.no_browser, True, not args.no_gemma)
        self.listener = self.pipeline.listener
        self.brain = self.listener.brain
        self.ears = SimAudioSource(self.env, ears="head_cam")
        self.person = PersonVoice(seed=args.seed)
        self.mode, self.question, self.plan = "listening", "", None
        self.captions: dict[str, tuple[str, float]] = {}  # who -> (text, until)
        self.pending: list[tuple] = []
        self.seen = 0
        self.events.add("mode", mode="listening")
        self.events.add("status", listening=True, message=(
            f"G1 in the house · V speak · Y/N answer · X cancel · Whisper {self.pipeline.stt.name} · "
            f"{self.pipeline.interpreter_name}"))

    # ------------------------------------------------------------------ input
    def on_key(self, key: int) -> None:
        if key == ord("V"):
            self.pending.append(("speak", None))
        elif key == ord("Y"):
            self.pending.append(("speak", ANSWERS["yes"]))
        elif key == ord("N"):
            self.pending.append(("speak", ANSWERS["no"]))
        elif key == ord("X"):
            self.pending.append(("cancel",))

    def handle_keys(self) -> None:
        while self.pending:
            p = self.pending.pop(0)
            if p[0] == "speak":
                utt = self.person.speak(sim_time=self.env.data.time, phrase=p[1])
                if utt:
                    self.env.info["last_utterance"] = {"speaker": utt.speaker, "phrase": utt.phrase,
                                                       "wav": str(utt.wav), "sim_time": utt.sim_time}
                    self.caption("person", utt.phrase, clip_seconds(utt.wav) + 2.0)
                    print(f'  person: "{utt.phrase}"')
            elif p[0] == "cancel":
                self.listener.reset()
                self.brain.task = None  # fetch mode off: the brain has nothing to do
                self.set_mode("listening")

    # ------------------------------------------------------------------ hearing
    def listen(self) -> None:
        self.pipeline.tick()
        for h in self.ears.poll():
            ready = time.time() + max(0.0, h.start_s + h.duration_s - self.env.data.time)
            self.pipeline.submit(h, ready)

    def follow_events(self) -> None:
        """React to what the pipeline decided (it runs on its own thread)."""
        for e in self.events.since(self.seen):
            self.seen = e["seq"] + 1
            if e["kind"] == "robot_said":
                self.caption("robot", e["text"], 4.0)
            elif e["kind"] == "command":
                st = e.get("status")
                if st == "confirming":
                    self.question = e.get("question", "")
                    self.set_mode("asking")
                elif st == "understood" and e.get("task"):
                    self.enter_fetch_mode(e["task"]["item_id"], e["task"]["destination"], e.get("transcript", ""))
                elif st in ("declined", "rejected") and self.mode == "asking":
                    self.set_mode("listening")

    def enter_fetch_mode(self, item: str, destination: str, said: str) -> None:
        """The brain already has the task; work out (don't do) the planner's first step."""
        self.bridge.new_task(item)
        decision = self.brain.next_action(self.bridge.latest_observation())
        likely = [(st, round(p, 2)) for st, p in self.bridge.memory.ranked(item)]
        self.plan = {"item": item, "destination": destination, "action": decision.action,
                     "target": decision.target, "reason": decision.reason, "likely": likely}
        self.set_mode("fetch")
        print(f"  FETCH MODE: {item} -> {destination}; first step: {decision.action} {decision.target or ''}"
              f" (likely: {', '.join(f'{s} {p:.0%}' for s, p in likely[:2])})")

    def set_mode(self, mode: str) -> None:
        if mode != "fetch":
            self.plan = None
        self.mode = mode
        self.events.add("mode", mode=mode, question=self.question if mode == "asking" else "",
                        plan=self.plan)

    def caption(self, who: str, text: str, seconds: float) -> None:
        self.captions[who] = (text, time.time() + seconds)

    # ------------------------------------------------------------------ drawing
    def draw(self, viewer) -> None:
        now = time.time()
        texts = []
        if self.mode == "fetch" and self.plan:
            p = self.plan
            likely = ", ".join(f"{s} {pr:.0%}" for s, pr in p["likely"][:3])
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, "FETCH MODE",
                          f"{p['item']} -> {p['destination']}"))
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                          f"next step: {p['action']} {p['target'] or ''}\nlikely places: {likely}",
                          "planned, not carried out"))
        elif self.mode == "asking":
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, "ASKING", self.question))
        else:
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, "LISTENING", "V speak   Y/N answer   X cancel"))
        lines = [f'{"Person" if w == "person" else "Robot"}:  "{t}"'
                 for w, (t, until) in sorted(self.captions.items()) if until > now]
        if lines:
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_BOTTOM, "\n".join(lines), None))
        viewer.set_texts(texts)

        scn, m, d = viewer.user_scn, self.env.model, self.env.data
        scn.ngeom = 0
        head = d.xpos[m.body("torso_link").id] + np.array([0, 0, 0.75])
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.07, 0, 0]), head,
                            np.eye(3).flatten(), np.array(LIGHT[self.mode], dtype=np.float32))
        g.emission = 0.8
        scn.ngeom += 1
        if self.mode == "fetch" and self.plan:
            # A line from the robot to the most likely place it would search first.
            st = self.robot.stations.get(self.plan["likely"][0][0]) if self.plan["likely"] else None
            if st is not None:
                a = d.xpos[m.body("pelvis").id].copy()
                b = np.array([st.x, st.y, a[2]])
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                    np.array(LIGHT["fetch"], dtype=np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 6, a, b)
                scn.ngeom += 1

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=self.on_key) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            pelvis = self.env.data.xpos[self.env.model.body("pelvis").id]
            v.cam.lookat[:] = [pelvis[0] - 0.8, pelvis[1] - 0.3, 0.7]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 4.5, 200, -25
            print("keys: V person speaks | Y/N person answers | X cancel fetch mode")
            t0, s0 = time.time(), self.env.data.time
            while v.is_running():
                self.handle_keys()
                self.robot.wait(0.04)  # two 50 Hz control ticks: the G1 keeps balancing
                self.listen()
                self.follow_events()
                with v.lock():
                    self.draw(v)
                v.sync()
                ahead = (self.env.data.time - s0) - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
                if self.robot.wbc.fallen():
                    print("the G1 fell over; restart the demo")
                    break
        self.pipeline.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="tidy")
    ap.add_argument("--target", default="kitchen_counter", help="where the bottle starts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-gemma", action="store_true")
    Demo(ap.parse_args()).run()


if __name__ == "__main__":
    main()
