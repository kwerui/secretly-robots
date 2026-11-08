"""Repeated-trial evaluation of the physical grasp: the honest numbers for the pitch.

Runs the two-request scenario (bottle from the table, cup from the closed cupboard) N times
with the item start positions jittered, and reports per-stage success, pinch forces, slip and
whether each item ended upright on the side table. No weld, no scripted outcome: a trial only
counts when the physics did it.

    python eval_grasp.py --trials 6 --jitter 0.03
    python eval_grasp.py --trials 3 --task table      # bottle only
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time

import mujoco
import numpy as np

from articulation import ArticulationModel
from run_demo import learned_memory
from scene import DEFAULT_PLACEMENT, SIDE_TABLE
from skills import AssistiveRobot
from tasks import fetch


def upright(robot: AssistiveRobot, item: str) -> float:
    b = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, item)
    return float(robot.data.xmat[b].reshape(3, 3)[2, 2])


def on_side_table(robot: AssistiveRobot, item: str) -> bool:
    b = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, item)
    p = robot.data.xpos[b]
    (cx, cy), (hx, hy) = SIDE_TABLE["pos"], SIDE_TABLE["half"]
    return abs(p[0] - cx) < hx and abs(p[1] - cy) < hy and p[2] > SIDE_TABLE["top"]


def run_trial(seed: int, jitter: float, tasks: list[str], model: ArticulationModel) -> dict:
    rng = np.random.default_rng(seed)
    placement = {}
    for name, (surface, x, y) in DEFAULT_PLACEMENT.items():
        dx, dy = rng.uniform(-jitter, jitter, 2)
        if surface == "cupboard":  # stay inside the cupboard
            dx, dy = np.clip(dx, -0.04, 0.04), np.clip(dy, -0.02, 0.02)
        placement[name] = (surface, x + dx, y + dy)
    robot = AssistiveRobot(placement=placement)
    memory = learned_memory()
    out = {"seed": seed, "placement": {k: [round(v[1], 3), round(v[2], 3)] for k, v in placement.items()}, "tasks": {}}
    t0 = time.time()
    for item in tasks:
        res = fetch(robot, memory, item, model, say=lambda s: None)
        out["tasks"][item] = {
            "fetched": bool(res.get("ok")),
            "stage_failed": None if res.get("ok") else res.get("stage"),
            "error": res.get("error"),
            "upright": round(upright(robot, item), 3),
            "on_side_table": on_side_table(robot, item),
            "delivered_upright": bool(res.get("ok")) and on_side_table(robot, item) and upright(robot, item) > 0.95,
        }
        robot.go_to("home")
        robot.wait(0.5)
    out["grasp_log"] = robot.grasp_log
    out["motion"] = robot.motion.metrics()
    out["mppi_plans"] = robot.mppi.stats["plans"] + robot.mppi_palm.stats["plans"]
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.03, help="m, uniform jitter of item start positions")
    ap.add_argument("--task", choices=["both", "table", "cupboard"], default="both")
    ap.add_argument("--out", default="eval_grasp.json")
    args = ap.parse_args()
    tasks = {"both": ["water_bottle", "cup"], "table": ["water_bottle"], "cupboard": ["cup"]}[args.task]
    model = ArticulationModel.load("cupboard_model.json")
    results = []
    for seed in range(args.trials):
        r = run_trial(seed, args.jitter, tasks, model)
        results.append(r)
        summary = ", ".join(f"{k}: {'OK' if v['delivered_upright'] else 'FAIL(' + str(v['stage_failed'] or 'fell') + ')'}"
                            for k, v in r["tasks"].items())
        print(f"trial {seed}: {summary}  ({r['wall_s']} s)")
    n = len(results)
    print("\n=== summary ===")
    for item in tasks:
        ok = sum(r["tasks"][item]["delivered_upright"] for r in results)
        fetched = sum(r["tasks"][item]["fetched"] for r in results)
        print(f"{item}: fetched {fetched}/{n}, delivered upright on the side table {ok}/{n}")
    grasps = [g for r in results for g in r["grasp_log"]]
    good = [g for g in grasps if g["ok"]]
    if grasps:
        print(f"grasp attempts {len(grasps)}, successful {len(good)}; "
              f"pinch force {np.mean([g['normal_force_N'] for g in good]):.0f} N mean "
              f"({np.min([g['normal_force_N'] for g in good]):.0f}-{np.max([g['normal_force_N'] for g in good]):.0f}), "
              f"slip after lift {np.mean([g['slip_mm'] for g in good]):.1f} mm mean, max {np.max([g['slip_mm'] for g in good]):.1f} mm")
    if results and results[0].get("motion"):
        keys = ["joint_acc_rms", "joint_acc_peak", "joint_jerk_rms", "site_acc_rms_mps2", "site_acc_peak_mps2"]
        print("arm smoothness (commanded joint targets; mean over trials): " + ", ".join(
            f"{k} {np.mean([r['motion'][k] for r in results]):.1f}" for k in keys)
              + f"; MPPI {'on' if os.environ.get('ARM_MPPI', '1') != '0' else 'off'}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
