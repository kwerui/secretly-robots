"""Kristina's brain + our G1 in Jack's house: requests in, decisions out, skills executed.

    python house_brain_demo.py                  # Gemini parses requests and proposes each step
                                                # (needs google-genai and GEMINI_API_KEY); brain/safety.py
                                                # vets every proposal, the rules take over on any failure
    python house_brain_demo.py --offline        # rules parser + deterministic planner only
    python house_brain_demo.py --viewer         # watch it (real time; --speed 2 for faster)
    python house_brain_demo.py --request "I'm thirsty" --request "Could you bring me my cup?"

What happens:
  1. The carer opens the top kitchen drawer once (drawer camera) -> articulation model.
  2. Location memory from "watching the carer for a week": the bottle has mostly been left on
     the kitchen counter, once in the kitchen drawers; the cup on the counter.
  3. Each request -> brain.start_task -> loop of brain/integration.take_turn: the brain looks at
     the robot's observation (head-camera sightings, station, drawer open, holding, delivered),
     decides the next step, safety-checks it, and the bridge runs it on the robot.
     For "I'm thirsty" the bottle isn't in sight -> search: the counter first (memory), not
     there -> the drawers: open (learned model) -> seen -> grasp from above -> close with the
     left hand -> navigate to the sofa -> deliver -> complete.
Decisions and feedback are logged to logs/house_brain.jsonl.
"""

from __future__ import annotations

import os

from viewer_env import relaunch_for_viewer

relaunch_for_viewer()  # --viewer from anaconda: restart with the system libstdc++ (GLX fix)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from brain_bridge import Container, RobotBridge  # noqa: E402  (also puts the repo root on sys.path)
from brain.engine import Brain  # noqa: E402
from brain.integration import take_turn  # noqa: E402
from brain.safety import RejectedAction  # noqa: E402
from articulation import ArticulationModel  # noqa: E402
from demo_capture import capture_sim_drawer  # noqa: E402
from house import G1House, KITCHEN_TOP  # noqa: E402
from memory import LocationMemory  # noqa: E402
from skills import AssistiveRobot  # noqa: E402

DAY = 86400.0
SEARCHABLE = ["kitchen_counter", "kitchen_drawers", "kitchen_table"]


def learned_memory() -> LocationMemory:
    """What the robot picked up watching the carer put things away over the last week."""
    mem = LocationMemory(stations=SEARCHABLE)
    now = time.time()
    for days_ago, item, station in [(6, "water_bottle", "kitchen_counter"), (4, "water_bottle", "kitchen_counter"),
                                    (5, "water_bottle", "kitchen_drawers"), (3, "cup", "kitchen_counter"),
                                    (2, "cup", "kitchen_counter")]:
        mem.observe_placement(item, station, t=now - days_ago * DAY)
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", action="append", default=None)
    ap.add_argument("--offline", action="store_true", help="no Gemini: rules parser + deterministic planner")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--layout", default="tidy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--reuse-demo", action="store_true", help="load drawer_model.json instead of watching the carer")
    args = ap.parse_args()
    requests = args.request or ["I'm thirsty", "Could you bring me my cup?"]

    env = G1House(seed=args.seed)
    env.reset(args.layout, "kitchen_drawer", seed=args.seed)

    sync = None
    viewer = None
    if args.viewer:
        import mujoco
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("pelvis").id
        viewer.cam.distance, viewer.cam.elevation = 3.5, -30
        clock = {"t0": None}

        def _sync(_robot=None):
            if not viewer.is_running():
                raise KeyboardInterrupt
            now, sim = time.time(), env.data.time
            if clock["t0"] is None:
                clock["t0"] = (now, sim)
            ahead = (sim - clock["t0"][1]) / args.speed - (now - clock["t0"][0])
            if ahead > 0:
                time.sleep(ahead)
            viewer.sync()
        sync = _sync

    robot = AssistiveRobot(world=env, on_frame=sync)

    def say(msg):
        print(f"[{robot.data.time:6.1f}s] {msg}", flush=True)

    if args.reuse_demo and Path("drawer_model.json").exists():
        model = ArticulationModel.load("drawer_model.json")
    else:
        say("Carer: opening the top kitchen drawer once (drawer camera watching)")
        model = capture_sim_drawer(robot, KITCHEN_TOP, on_step=sync)
        model.save("drawer_model.json")
    say(f"Drawer model: {model.kind}, {model.amount * 100:.0f} cm, residual {model.residual_m * 1000:.1f} mm")

    containers = {"kitchen_drawers": Container(model, KITCHEN_TOP.anchor_marker, f"grasp_{KITCHEN_TOP.body}",
                                               (KITCHEN_TOP.front_marker,))}
    bridge = RobotBridge(robot, learned_memory(), containers, say=say)

    use_gemini = not args.offline
    if use_gemini:
        # The adapters fall back to the rules silently when Gemini is unreachable; say it up front
        # so a run is never mistaken for a Gemini run.
        missing = []
        if not os.environ.get("GEMINI_API_KEY"):
            missing.append("GEMINI_API_KEY is not set")
        import importlib.util
        if importlib.util.find_spec("google.genai") is None if importlib.util.find_spec("google") else True:
            missing.append("google-genai is not installed (pip install -r ../requirements-gemini.txt)")
        if missing:
            print("WARNING: Gemini unavailable (" + "; ".join(missing) + ") -> running OFFLINE rules only.")
            use_gemini = False
    if use_gemini:
        from brain.gemini_optional import GeminiIntentParser, GeminiPlanner
        brain = Brain(parser=GeminiIntentParser(), gemini_planner=GeminiPlanner(), log_file="logs/house_brain.jsonl")
        print("Brain: Gemini (intent + next-step proposals), rules as fallback, safety.validate on every step")
    else:
        brain = Brain(log_file="logs/house_brain.jsonl")
        print("Brain: offline rules (deterministic planner)")

    summary = []
    try:
        for req in requests:
            say(f'Person: "{req}"')
            try:
                task = brain.start_task(req)
            except Exception as e:  # noqa: BLE001  (IntentError etc.)
                say(f"  couldn't understand: {e}")
                summary.append((req, "not understood"))
                continue
            say(f"  task: fetch {task.item_id} -> {task.destination}")
            bridge.new_task(task.item_id)
            outcome = "turn limit"
            for _ in range(args.max_turns):
                try:
                    decision = take_turn(brain, bridge, bridge)
                except RejectedAction as e:  # brain/safety.py vetoed the planner's own step
                    say(f"  safety rejected the next step: {e}")
                    outcome = "rejected by safety"
                    break
                say(f"  brain [{decision.source}]: {decision.action}({decision.target or ''}) - {decision.reason}")
                if decision.action in ("complete", "ask_for_help"):
                    outcome = decision.action
                    break
            summary.append((req, outcome))
    finally:
        print("\nSummary:")
        for req, outcome in summary:
            print(f"  {req!r}: {outcome}")
        d = env.data
        for item, body in env.item_body.items():
            print(f"  {item}: {[round(float(v), 3) for v in d.body(body).xpos]}")
        print(f"  drawer open: {env.drawer_open_m(KITCHEN_TOP):.3f} m, fallen: {robot.wbc.fallen()}, sim {d.time:.1f} s")
        if viewer is not None:
            print("Close the viewer window (or Ctrl+C) to exit.")
            try:
                while viewer.is_running():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            # Don't tear the window down (viewer.close() can hang on an X error): just exit.
            sys.stdout.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
