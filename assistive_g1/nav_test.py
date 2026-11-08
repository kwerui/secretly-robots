"""Moved-chair test for the perceived navigation (costmap.py).

    MUJOCO_GL=egl python nav_test.py                 # live map (default)
    MUJOCO_GL=egl python nav_test.py --nav static    # old behaviour: plan once on the true geometry

Two things a stale map gets wrong, in Jack's tidy house:
  A. A chair stands in the living room where the floor plan has nothing (moved there since the
     map was drawn), on the straight line from the robot's start to the kitchen door. The robot
     has never been told about it: it has to see it (LiDAR) before its first plan.
  B. While the robot carries on from the kitchen drawers back to the person, a chair is put down
     in its path (1.3 m ahead, after it has come through the doorway). The route it planned is
     now cut: it has to notice and re-plan while walking.

Reports per leg: arrived, time, number of re-plans, control ticks in contact with a chair, and
writes nav_map_<leg>.png (the robot's map with its path and trail; grey = prior, black = seen).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--nav", default="live", choices=["live", "static"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-chair-a", action="store_true")
ap.add_argument("--no-chair-b", action="store_true")
ap.add_argument("--chair-b", type=float, nargs=2, default=[-1.1, -1.8])
ap.add_argument("--out", default="nav_test")
ap.add_argument("--video", action="store_true", help="record <out>_<nav>.mp4 (follow cam, head cam, the robot's map)")
ap.add_argument("--fps", type=float, default=10)
args = ap.parse_args()
os.environ["G1_NAV"] = args.nav
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from house import G1House  # noqa: E402
from skills import AssistiveRobot  # noqa: E402

CHAIR_A = (-1.0, -1.4)  # 0.9 m ahead-right of the start, on the straight line to the kitchen door


def main():
    t0 = time.time()
    env = G1House(seed=args.seed)
    env.reset("tidy", "kitchen_drawer", seed=args.seed)
    m, d = env.model, env.data
    chair = env.obs_mocap["chair"][0]
    chair_geoms = {g for g in env.mocap_geoms if m.body_mocapid[m.geom_bodyid[g]] == chair}
    robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in env._subtree(env.robot_root)}

    def put_chair(xy):
        d.mocap_pos[chair] = [xy[0], xy[1], 0.0]
        d.mocap_quat[chair] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)

    if not args.no_chair_a:
        put_chair(CHAIR_A)  # moved there since the floor plan was drawn: not in the robot's prior

    stats = {"contact_ticks": 0, "trail": [], "chair_b_t": None}
    state = {"leg": 0}

    def monitor(robot):
        # collisions robot <-> chair (the chair is a mocap body: a hit stops or topples the robot)
        for k in range(d.ncon):
            c = d.contact[k]
            if (c.geom1 in chair_geoms and c.geom2 in robot_geoms) or (c.geom2 in chair_geoms and c.geom1 in robot_geoms):
                stats["contact_ticks"] += 1
                break
        p = robot.wbc.true_base_pos
        stats["trail"].append((float(p[0]), float(p[1])))
        # B: once the robot is back through the kitchen door, a chair is put down in its way
        if state["leg"] == 2 and stats["chair_b_t"] is None and not args.no_chair_b and p[0] < 0.2:
            put_chair(args.chair_b)
            stats["chair_b_t"] = round(float(d.time), 2)
            print(f"[{d.time:6.1f}s] *** chair put down at {tuple(args.chair_b)}, {np.hypot(*(np.asarray(args.chair_b) - p[:2])):.2f} m ahead of the robot")
            if rec is not None:
                rec.caption = "someone puts a chair down in the robot's path"

    rec = None
    if args.video:
        from house_demo import LightRecorder
        rec = LightRecorder(f"{args.out}_{args.nav}.mp4", fps=args.fps, size=(800, 450))
        rec.caption = f"navigation: {'perceived map (LiDAR + depth)' if args.nav == 'live' else 'static map from the simulator'}"

    def on_frame(robot):
        monitor(robot)
        if rec is not None:
            rec(robot)

    robot = AssistiveRobot(world=env, on_frame=on_frame)
    results = {"nav": args.nav}

    def leg(n, station):
        state["leg"] = n
        c0, t_sim, trail0 = stats["contact_ticks"], d.time, len(stats["trail"])
        r = robot.go_to(station)
        out = {"ok": r["ok"], "replans": r.get("replans", 0), "sim_s": round(d.time - t_sim, 1),
               "contact_ticks": stats["contact_ticks"] - c0, "error": r.get("error")}
        if robot.estimator is not None:
            out["loc_err"] = robot.estimator.error()
        print(f"[{d.time:6.1f}s] leg {n} -> {station}: {out}", flush=True)
        results[f"leg{n}_{station}"] = out
        if robot.costmap is not None:
            cm = robot.costmap
            img = cm.image(path=None, robot_xy=robot.wbc.base_pos[:2], scale=4)
            for p, q in zip(stats["trail"][trail0:-1], stats["trail"][trail0 + 1:]):
                a = (int(round((p[0] - cm.x0) / cm.res * 4)), img.shape[0] - 1 - int(round((p[1] - cm.y0) / cm.res * 4)))
                b = (int(round((q[0] - cm.x0) / cm.res * 4)), img.shape[0] - 1 - int(round((q[1] - cm.y0) / cm.res * 4)))
                cv2.line(img, a, b, (230, 120, 30), 2)
            for e in robot.nav_log:
                if e["path"]:
                    pts = [(int(round((x - cm.x0) / cm.res * 4)), img.shape[0] - 1 - int(round((y - cm.y0) / cm.res * 4))) for x, y in e["path"]]
                    cv2.polylines(img, [np.array(pts, np.int32)], False, (40, 90, 220), 1)
            cv2.imwrite(f"{args.out}_map_{n}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            new, gone = cm.changed_vs_prior()
            out["map_cells_new"] = new
            out["map_cells_cleared"] = gone
        return r

    print(f"nav={args.nav}  chair A at {CHAIR_A if not args.no_chair_a else None}", flush=True)
    if rec is not None:
        rec.caption = "to the kitchen drawers: a chair the floor plan doesn't know stands in the way"
    leg(1, "kitchen_drawers")
    robot.nav_log.clear()
    if rec is not None:
        rec.caption = "back to the person"
    leg(2, "person")
    if rec is not None:
        rec.close()
    results["chair_b_t"] = stats["chair_b_t"]
    results["fallen"] = bool(robot.wbc.fallen())
    results["replan_log"] = robot.nav_log
    if robot.costmap is not None:
        results["lidar_scans"] = robot.costmap.scans
        results["depth_frames"] = robot.costmap.depth_frames
    results["sim_s"] = round(float(d.time), 1)
    results["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(results, indent=1, default=str))
    with open(f"{args.out}_{args.nav}.json", "w") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()
