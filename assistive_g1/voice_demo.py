"""Voice demo: the person on the sofa asks, the G1 hears, interprets, answers, and fetches.

    cd assistive_g1
    ../sim/.venv/bin/mjpython voice_demo.py --list                        # the 25 demo scenarios
    ../sim/.venv/bin/mjpython voice_demo.py --scenario 18                 # macOS (viewer needs mjpython)
    ../sim/.venv/bin/mjpython voice_demo.py --random --seed 7 --clutter 4 --lock-door living_kitchen

Keys in the viewer:
    V   the person asks for water (one of eight lines: some direct, some hints)
    C   the person asks for their cup
    Y/N the person answers the robot's question
    F   toggle: carry the fetch out (default) / plan only
    X   cancel, back to listening

Scenarios 0-24 are 5 item placements x 5 floor setups (tidy, light clutter, heavy clutter,
locked door, locked door + clutter), all checked with the real G1. The bottle and the cup each
have five spots with set probabilities (scenario.SPAWNS), and the robot's location memory starts
from the same odds, so it searches the likeliest place first and adapts when the item isn't
there: the belief markers over each place shrink and turn red as places are ruled out, and the
walked route is drawn. It doesn't know which door is locked: it finds out at the doorway (red bar),
says so, and reroutes through the other doors. Clutter is walked round.

Everything the robot does runs onboard: hearing (head-camera microphone), Whisper, the rules /
local Gemma 4 interpreter, its voice (pre-made Gemini TTS clips), planning and control.
"""
from __future__ import annotations

from viewer_env import relaunch_for_viewer

relaunch_for_viewer()

import functools  # noqa: E402
import builtins  # noqa: E402

print = functools.partial(builtins.print, flush=True)  # console stays live under mjpython and pipes

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ.setdefault("G1_CHEAT_GRASP", "fallback")  # real grasp first, abstracted contact if it fails

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root: brain/, scripts/, sim/

from articulation import ArticulationModel  # noqa: E402
from brain_bridge import Container, RobotBridge  # noqa: E402
from house import G1House, KITCHEN_TOP  # noqa: E402
from scenario import (DOORS, N_SCENARIOS, SPAWNS, DoorDiscovery, Scenario, apply, describe_scenario,  # noqa: E402
                      prior_memory, scenario)
from skills import AssistiveRobot  # noqa: E402

from brain.hearing import SimAudioSource  # noqa: E402
from brain.safety import RejectedAction  # noqa: E402
from scripts.voice_robot import build  # noqa: E402
from sim import house_env as H  # noqa: E402
from sim.person_voice import ANSWERS, CUP_PHRASES, PersonVoice, clip_seconds  # noqa: E402

LIGHT = {"listening": (0.55, 0.55, 0.55, 0.9), "asking": (1.0, 0.7, 0.1, 0.95), "fetch": (0.15, 0.9, 0.35, 0.95),
         "done": (0.2, 0.55, 1.0, 0.95), "failed": (0.9, 0.2, 0.2, 0.95)}
FONT = mujoco.mjtFontScale.mjFONTSCALE_150
DELIVERED = {"water_bottle": "delivered_water", "cup": "delivered_cup"}
PLAYER_LATENCY_S = 0.15  # afplay/paplay start-up: the sound starts this long after the key press


class Cancelled(Exception):
    pass


class Demo:
    def __init__(self, args):
        self.args = args
        if args.random:
            rng = np.random.default_rng(args.seed)
            fixed = {k: v for k, v in (("water_bottle", args.bottle), ("cup", args.cup)) if v}
            self.scenario = Scenario.sample(rng, lock_door=args.lock_door, clutter=args.clutter, fixed=fixed,
                                            seed=args.seed)
            label = f"random (seed {args.seed})"
        else:
            self.scenario = scenario(args.scenario)
            label = describe_scenario(args.scenario)
        self.env = G1House(seed=self.scenario.seed)
        info = apply(self.env, self.scenario)
        print(f"scenario {label}: {info['scenario']}")
        print("standing the G1 up…")
        self.robot = AssistiveRobot(world=self.env)
        self.doors = DoorDiscovery(self.env, self.robot, say=lambda m: self.robot_step(m), on_found=self.door_found)
        drawer = ArticulationModel.load(HERE / "drawer_model.json")
        containers = {"kitchen_drawers": Container(drawer, KITCHEN_TOP.anchor_marker, f"grasp_{KITCHEN_TOP.body}",
                                                   (KITCHEN_TOP.front_marker,), drawer=KITCHEN_TOP)}
        self.bridge = RobotBridge(self.robot, prior_memory(), containers, say=self.robot_step)
        self.events, self.pipeline, self.robot_voice = build(args.port, not args.no_browser, True, not args.no_gemma)
        self.listener = self.pipeline.listener
        self.brain = self.listener.brain
        # The robot hears the clip as it plays on the speakers (wall clock), and a phrase only
        # reaches speech-to-text once it has heard the person stop talking.
        self.ears = SimAudioSource(self.env, ears="head_cam", stream=True, clock=time.time, time_key="wall_time")
        self.person = PersonVoice(seed=args.seed)
        self.execute = not args.plan_only
        self.mode, self.question, self.plan, self.item = "listening", "", None, None
        self.step_text, self.found_at, self.route = "", None, None
        self.captions: dict[str, tuple[str, float]] = {}  # who -> (text, until)
        self.pending: list[tuple] = []
        self.seen = 0
        self.start_fetch = False
        self.busy = False
        self.viewer = None
        self._shown = None
        self._clock = None
        self._wrap_go_to()
        self.events.add("scenario", **{k: v for k, v in info.items() if k in ("scenario", "spots", "locked", "clutter")})
        self.events.add("mode", mode="listening")
        self.events.add("status", listening=True, message=(
            f"G1 in the house · {info['scenario']} · V water · C cup · Y/N · F fetch/plan · X cancel · "
            f"Whisper {self.pipeline.stt.name} · {self.pipeline.interpreter_name}"))

    def door_found(self, door: str) -> None:
        """The robot just found a locked door: say so, and redraw the route it now takes."""
        if self.listener.say:
            self.listener.say("door_locked")
        self._replan_route()
        self.events.add("step", text=f"found the {door.replace('_', '-')} door locked: rerouting", likely=self.likely())

    def _wrap_go_to(self) -> None:
        """Record the route each time the robot sets off, so it can be drawn."""
        go_to = self.robot.go_to

        def wrapped(station: str):
            self._goal = station
            self._replan_route()
            return go_to(station)
        self.robot.go_to = wrapped

    def _replan_route(self) -> None:
        st, p = self.env.stations.get(getattr(self, "_goal", "")), self.robot.wbc.base_pos
        if st is not None:
            self.route = self.env.plan_path((float(p[0]), float(p[1])), (st.x, st.y))

    # ------------------------------------------------------------------ input
    def on_key(self, key: int) -> None:
        k = {ord("V"): ("speak", None), ord("C"): ("speak", "cup"), ord("Y"): ("speak", ANSWERS["yes"]),
             ord("N"): ("speak", ANSWERS["no"]), ord("X"): ("cancel",), ord("F"): ("toggle",)}.get(key)
        if k:
            self.pending.append(k)

    def handle_keys(self) -> None:
        while self.pending:
            p = self.pending.pop(0)
            if p[0] == "speak":
                phrase = self.person.rng.choice(CUP_PHRASES) if p[1] == "cup" else p[1]
                utt = self.person.speak(sim_time=self.env.data.time, phrase=phrase)
                if utt:
                    self.env.info["last_utterance"] = {"speaker": utt.speaker, "phrase": utt.phrase,
                                                       "wav": str(utt.wav), "sim_time": utt.sim_time,
                                                       "wall_time": time.time() + PLAYER_LATENCY_S}
                    self.caption("person", utt.phrase, clip_seconds(utt.wav) + 2.0)
                    print(f'  person: "{utt.phrase}"')
            elif p[0] == "toggle":
                self.execute = not self.execute
                print(f"  fetch mode will {'carry the fetch out' if self.execute else 'only plan'}")
            elif p[0] == "cancel":
                self.listener.reset()
                self.brain.task = None
                self.start_fetch = False
                self.set_mode("listening")
                if self.busy:
                    raise Cancelled()

    # ------------------------------------------------------------------ hearing and deciding
    def listen(self) -> None:
        self.pipeline.tick()
        for h in self.ears.poll():  # a finished phrase: the person has stopped talking
            self.pipeline.submit(h)

    def follow_events(self) -> None:
        """React to what the voice pipeline decided (it runs on its own thread)."""
        for e in self.events.since(self.seen):
            self.seen = e["seq"] + 1
            if e["kind"] == "robot_said":
                self.caption("robot", e["text"], 4.0)
            elif e["kind"] == "command" and not self.busy:
                st = e.get("status")
                if st == "confirming":
                    self.question = e.get("question", "")
                    self.set_mode("asking")
                elif st == "understood" and e.get("task"):
                    self.enter_fetch_mode(e["task"]["item_id"], e["task"]["destination"])
                elif st in ("declined", "rejected") and self.mode == "asking":
                    self.set_mode("listening")

    def enter_fetch_mode(self, item: str, destination: str) -> None:
        self.item, self.found_at, self.route = item, None, None
        self.bridge.new_task(item)
        decision = self.brain.next_action(self.bridge.latest_observation())
        self.plan = {"item": item, "destination": destination, "action": decision.action,
                     "target": decision.target, "reason": decision.reason, "likely": self.likely()}
        self.set_mode("fetch")
        print(f"  FETCH MODE: {item} -> {destination}; first step: {decision.action} {decision.target or ''}"
              f" (likely: {', '.join(f'{s} {p:.0%}' for s, p in self.plan['likely'][:3])})")
        self.start_fetch = self.execute

    def likely(self):
        return [(st, round(p, 3)) for st, p in self.bridge.memory.ranked(self.item)] if self.item else []

    def robot_step(self, msg: str) -> None:
        """RobotBridge narrates each step ("searching for the cup at the study_desk (belief 15%)")."""
        msg = msg.strip()
        self.step_text = msg
        self.caption("step", msg, 6.0)
        print(f"  [{self.env.data.time:6.1f}s] {msg}")
        self.events.add("step", text=msg, likely=self.likely())

    # ------------------------------------------------------------------ carrying the fetch out
    def run_fetch(self) -> None:
        """Eren's RobotBridge loop: observe -> brain decides -> safety -> G1 acts -> feedback."""
        self.start_fetch, self.busy = False, True
        outcome = "turn limit"
        try:
            for _ in range(self.args.max_turns):
                try:
                    decision = self.brain.next_action(self.bridge.latest_observation())
                except RejectedAction as e:
                    outcome = f"safety stopped it: {e}"
                    break
                self.plan.update(action=decision.action, target=decision.target, reason=decision.reason,
                                 likely=self.likely())
                self.events.add("step", text=f"brain: {decision.action}({decision.target or ''}) - {decision.reason}",
                                likely=self.likely())
                if decision.action in ("complete", "ask_for_help", "stop"):
                    outcome = decision.action
                    break
                fb = self.bridge.execute_high_level(decision)
                self.brain.record_feedback(fb)
                if decision.action == "search" and fb.succeeded:
                    s = self.bridge.sightings.get(self.item)
                    self.found_at = s.area if s else None
                self.events.add("step", text=f"-> {fb.action}: {'ok' if fb.succeeded else 'FAILED'} {fb.detail}",
                                likely=self.likely())
        except Cancelled:
            outcome = "cancelled"
        finally:
            self.busy = False
        self.route = None
        if outcome == "complete":
            self.robot_voice and self.listener.say and self.listener.say(DELIVERED.get(self.item, "delivered_water"))
            self.set_mode("done")
        elif outcome != "cancelled":
            self.listener.say and self.listener.say("search_failed")
            self.set_mode("failed")
        print(f"  fetch {outcome} at {self.env.data.time:.1f}s sim")
        self.events.add("step", text=f"fetch {outcome}", likely=self.likely())

    def set_mode(self, mode: str) -> None:
        if mode in ("listening", "asking"):
            self.plan = None
        self.mode = mode
        self.events.add("mode", mode=mode, question=self.question if mode == "asking" else "",
                        plan=self.plan, executing=self.execute)

    def caption(self, who: str, text: str, seconds: float) -> None:
        self.captions[who] = (text, time.time() + seconds)

    # ------------------------------------------------------------------ drawing
    def texts(self) -> list:
        now = time.time()
        texts = []
        likely = ", ".join(f"{s} {p:.0%}" for s, p in self.likely()[:4])
        if self.mode in ("fetch", "done", "failed") and self.plan:
            p = self.plan
            head = {"fetch": "FETCHING" if self.busy else "FETCH MODE", "done": "DELIVERED", "failed": "FETCH FAILED"}[self.mode]
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, head, f"{p['item']} -> {p['destination']}"))
            what = (f"step: {p['action']} {p['target'] or ''}" if self.busy or self.mode != "fetch"
                    else f"next step: {p['action']} {p['target'] or ''}")
            found = f"\nfound at: {self.found_at}" if self.found_at else ""
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPRIGHT, f"{what}\nbeliefs: {likely}{found}",
                          "" if self.execute or self.busy else "plan only (F to fetch)"))
        elif self.mode == "asking":
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, "ASKING" + (" (listening…)" if self.ears.hearing else ""),
                          self.question))
        elif self.ears.hearing:
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, "LISTENING…", "waiting for the person to finish"))
        else:
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_TOPLEFT, "LISTENING",
                          f"V water  C cup  Y/N answer  F {'fetch' if self.execute else 'plan only'}  X cancel"))
        names = {"person": "Person", "robot": "Robot", "step": "Robot is"}
        lines = [f'{names[w]}:  {t if w == "step" else chr(34) + t + chr(34)}'
                 for w, (t, until) in sorted(self.captions.items()) if until > now]
        if lines:
            texts.append((FONT, mujoco.mjtGridPos.mjGRID_BOTTOM, "\n".join(lines), None))
        return texts

    def _geom(self, scn, kind, size, pos, rgba, emission=0.6):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        g.emission = emission
        scn.ngeom += 1
        return g

    def draw_scene(self, viewer) -> None:
        """Mode light, belief markers, route and locked doors. Call while holding the viewer lock."""
        scn, m, d = viewer.user_scn, self.env.model, self.env.data
        scn.ngeom = 0
        head = d.xpos[m.body("torso_link").id] + np.array([0, 0, 0.75])
        self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.07, 0, 0], head, LIGHT[self.mode], 0.8)
        # Belief over places for the current item: bigger = likelier; red = searched, not there.
        if self.item:
            empty = self.bridge.memory._searched_empty.get(self.item, set())
            for st_name, p in self.likely():
                st = self.env.stations.get(st_name)
                if st is None:
                    continue
                col = ((0.2, 0.9, 0.35, 0.9) if st_name == self.found_at else
                       (0.95, 0.25, 0.2, 0.85) if st_name in empty else (1.0, 0.75, 0.15, 0.85))
                self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.04 + 0.16 * p, 0, 0], (st.x, st.y, 1.9), col)
        # The route the robot is walking (around locked doors and clutter).
        if self.route:
            pts = [np.array([x, y, 0.05]) for x, y in self.route]
            for a, b in zip(pts, pts[1:]):
                g = self._geom(scn, mujoco.mjtGeom.mjGEOM_LINE, [0, 0, 0], [0, 0, 0], LIGHT["fetch"])
                if g is not None:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 5, a, b)
        for dn in self.doors.known:  # only the doors the robot has found locked
            c = H.DOORWAYS[dn]["center"]
            self._geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.06, 0.06, 0.9], (c[0], c[1], 1.0), (0.9, 0.1, 0.1, 0.7))

    # ------------------------------------------------------------------ main loop
    def frame(self, _robot=None) -> None:
        """Every 50 Hz control tick (also while the robot is busy fetching)."""
        v = self.viewer
        if v is None:
            return
        if not v.is_running():
            raise KeyboardInterrupt
        self.handle_keys()
        self.listen()
        self.follow_events()
        with v.lock():
            self.draw_scene(v)
        # set_texts waits for the viewer's GUI thread, which needs the lock: never call it while
        # holding the lock (deadlock), and only when the text actually changes.
        texts = self.texts()
        if texts != self._shown:
            v.set_texts(texts)
            self._shown = texts
        v.sync()
        t0, s0 = self._clock
        ahead = (self.env.data.time - s0) / self.args.speed - (time.time() - t0)
        if ahead > 0:
            time.sleep(ahead)

    def run(self) -> None:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=self.on_key) as v:
            self.viewer = v
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = self.env.model.body("pelvis").id
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 200, -40
            print("keys: V water | C cup | Y/N answer | F fetch/plan only | X cancel")
            self._clock = (time.time(), self.env.data.time)
            self.robot.on_frame = self.frame
            try:
                while v.is_running():
                    self.robot.wait(0.04)  # the G1 keeps balancing; frame() runs every tick
                    if self.start_fetch:
                        self.run_fetch()
                    if self.robot.wbc.fallen():
                        print("the G1 fell over; restart the demo")
                        break
            except KeyboardInterrupt:
                pass
        self.pipeline.stop()


def main() -> None:
    stations = sorted({s.station for spots in SPAWNS.values() for s, _ in spots})
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=int, default=0, choices=range(N_SCENARIOS), metavar="0-24",
                    help="demo scenario: 5 x placement + setup (see --list)")
    ap.add_argument("--list", action="store_true", help="list the 25 scenarios and exit")
    ap.add_argument("--random", action="store_true", help="draw a scenario from the spawn odds instead")
    ap.add_argument("--seed", type=int, default=0, help="with --random: which draw")
    ap.add_argument("--bottle", choices=stations, help="with --random: put the bottle here")
    ap.add_argument("--cup", choices=stations, help="with --random: put the cup here")
    ap.add_argument("--lock-door", choices=DOORS, help="with --random: lock this door")
    ap.add_argument("--clutter", type=int, default=0, help="with --random: pieces of clutter (0-8)")
    ap.add_argument("--plan-only", action="store_true", help="fetch mode only plans (F toggles)")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed vs real time while fetching")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-gemma", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n in range(N_SCENARIOS):
            print(describe_scenario(n))
        return
    Demo(args).run()


if __name__ == "__main__":
    main()
