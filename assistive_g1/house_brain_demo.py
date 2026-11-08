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
  3. Each request -> brain.start_task -> decision/feedback loop: the brain looks at
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
from brain.safety import RejectedAction  # noqa: E402
from articulation import ArticulationModel  # noqa: E402
from demo_capture import capture_sim_drawer  # noqa: E402
from house import G1House, KITCHEN_TOP  # noqa: E402
from memory import LocationMemory  # noqa: E402
from skills import AssistiveRobot  # noqa: E402
from evaluation import EpisodeMetrics, MujocoTruthReader, TrialRecorder, TruthSnapshot, write_results  # noqa: E402

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
    ap.add_argument("--results-dir", type=Path, help="write evaluation JSON, results.csv and summary.md")
    ap.add_argument("--experiment-id", help="stable ID prefix for evaluation output")
    ap.add_argument("--settle-s", type=float, default=1.0, help="simulated seconds before tray verification")
    ap.add_argument("--drawer-first", action="store_true",
                    help="evaluation only: search kitchen_drawers before learned-memory ranking")
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
                                               (KITCHEN_TOP.front_marker,), drawer=KITCHEN_TOP)}
    bridge = RobotBridge(robot, learned_memory(), containers, say=say,
                         search_first=("kitchen_drawers",) if args.drawer_first else ())

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
    evaluation_records = []
    try:
        for request_index, req in enumerate(requests, start=1):
            experiment_id = args.experiment_id or f"house-brain-{int(time.time())}-{request_index}"
            metrics = EpisodeMetrics.baseline(experiment_id, "unknown")
            metrics.model_mode = "offline" if not use_gemini else "gemini_optional"
            metrics.layout, metrics.seed = args.layout, args.seed
            metrics.bottle_placement = {
                "configured_target": env.target,
                "container": env.target_container,
                "position_xyz": [round(float(x), 4) for x in env.bottle_xyz()],
                "jitter": False,
            }
            metrics.environment_configuration = {
                "mode": env.info.get("mode"),
                "reset_attempts": env.info.get("attempts"),
                "doors_deg": env.info.get("doors"),
                "locked_doors": env.info.get("locked"),
                "search_first": list(bridge.search_first),
            }
            recorder = TrialRecorder(metrics)
            say(f'Person: "{req}"')
            try:
                task = brain.start_task(req)
            except Exception as e:  # noqa: BLE001  (IntentError etc.)
                say(f"  couldn't understand: {e}")
                summary.append((req, "not understood"))
                metrics.failure_stage, metrics.failure_reason = "intent", str(e)
                evaluation_records.append(recorder.finish(TruthSnapshot(robot_fallen=robot.wbc.fallen()),
                                                         simulation_duration_s=float(env.data.time)))
                continue
            metrics.requested_item = task.item_id
            if brain.last_cloud_event is not None:
                recorder.record_cloud_event(**brain.last_cloud_event)
            say(f"  task: fetch {task.item_id} -> {task.destination}")
            bridge.new_task(task.item_id)
            outcome = "turn limit"
            for _ in range(args.max_turns):
                try:
                    observation = bridge.latest_observation()
                    decision = brain.next_action(observation)
                except RejectedAction as e:  # brain/safety.py vetoed the planner's own step
                    say(f"  safety rejected the next step: {e}")
                    outcome = "rejected by safety"
                    metrics.failure_stage, metrics.failure_reason = "safety", str(e)
                    break
                recorder.record_decision(decision.action, decision.source,
                                         fallback=brain.last_decision_provenance == "fallback_after_cloud_rejection_or_failure")
                if brain.last_cloud_event is not None:
                    recorder.record_cloud_event(**brain.last_cloud_event)
                if decision.action in ("complete", "ask_for_help"):  # (executed actions are logged by the bridge, before they run)
                    say(f"  brain [{decision.source}]: {decision.action}({decision.target or ''}) - {decision.reason}")
                    outcome = decision.action
                    break
                feedback = bridge.execute_high_level(decision)
                brain.record_feedback(feedback)
                recorder.record_feedback(feedback.action, feedback.succeeded, feedback.detail, target=feedback.target,
                                         retry_count=int(bridge.last_execution.get("grasp_retry_count", 0)))
                recorder.record_navigation_attempts(bridge.last_execution.get("navigation_attempts", []))
                recorder.record_drawer_opening_attempts(bridge.last_execution.get("drawer_opening_attempts", []))
                if bridge.last_execution.get("drawer_open_attempted"):
                    recorder.record_stage("drawer_opening", bool(bridge.last_execution["drawer_open_succeeded"]))
                if bridge.last_execution.get("drawer_close_attempted"):
                    recorder.record_stage("drawer_closing", bool(bridge.last_execution["drawer_close_succeeded"]))
            summary.append((req, outcome))
            # Ground truth is queried only after planning/control is done.  It is never added
            # to the bridge observation seen by the brain.
            truth_reader = MujocoTruthReader(env, robot, KITCHEN_TOP, task.item_id)
            before_settling = truth_reader.snapshot()
            robot.wait(args.settle_s)
            after_settling = truth_reader.snapshot()
            truth = TruthSnapshot(item_on_tray=before_settling.item_on_tray,
                                  item_on_tray_after_settling=after_settling.item_on_tray,
                                  grasp_attached=after_settling.grasp_attached,
                                  drawer_closed=after_settling.drawer_closed,
                                  robot_fallen=after_settling.robot_fallen)
            evaluation_records.append(recorder.finish(truth, simulation_duration_s=float(env.data.time)))
    finally:
        print("\nSummary:")
        for req, outcome in summary:
            print(f"  {req!r}: {outcome}")
        d = env.data
        for item, body in env.item_body.items():
            print(f"  {item}: {[round(float(v), 3) for v in d.body(body).xpos]}")
        print(f"  drawer open: {env.drawer_open_m(KITCHEN_TOP):.3f} m, fallen: {robot.wbc.fallen()}, sim {d.time:.1f} s")
        if robot.estimator is not None:
            print(f"  localization (estimate vs truth): {robot.estimator.error()}, LiDAR {robot.estimator.stats}")
        if args.results_dir is not None:
            paths = write_results(evaluation_records, args.results_dir)
            print(f"  evaluation: {paths['csv']} and {paths['summary']}")
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
