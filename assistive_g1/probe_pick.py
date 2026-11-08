"""Pick robustness probe: start the robot a little off its station (as the laptop/viewer runs
do) and try the pick, for several offsets. Much quicker than the full demo.

    python probe_pick.py --item cup                      # counter side grasp, default offsets
    python probe_pick.py --item water_bottle             # drawer pick from above (drawer opened by the sim)
    python probe_pick.py --item water_bottle --close     # ... then push the drawer shut with the left hand
    G1_STANCE_SEARCH=0 python probe_pick.py --item cup   # the old fixed standoff, for comparison

Offsets are (forward, left, yaw_deg) of the start pose relative to the station.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from house import G1House, KITCHEN_TOP  # noqa: E402
from skills import AssistiveRobot  # noqa: E402

OFFSETS = [(0, 0, 0), (0.08, 0, 0), (-0.08, 0, 0), (0, 0.08, 0), (0, -0.08, 0), (0, 0, 8), (0, 0, -8), (0.06, -0.06, 6)]


def trial(item: str, off, layout: str, seed: int, close: bool = False, opened: float = 0.36) -> dict:
    env = G1House(seed=seed)
    env.reset(layout, "kitchen_drawer", seed=seed)
    station = "kitchen_counter" if item == "cup" else "kitchen_drawers"
    s = env.stations[station]
    if item == "water_bottle":  # open the top drawer as if the robot had (the bottle is in it)
        import mujoco
        j = env.model.joint(KITCHEN_TOP.joint)
        b = env.model.body(KITCHEN_TOP.body).id
        mujoco.mj_forward(env.model, env.data)
        p0 = env.data.xpos[b].copy()
        env.data.qpos[j.qposadr[0]] = opened
        mujoco.mj_forward(env.model, env.data)
        bj = env.model.body(env.item_body["water_bottle"]).jntadr[0]
        env.data.qpos[env.model.jnt_qposadr[bj]:env.model.jnt_qposadr[bj] + 3] += env.data.xpos[b] - p0  # bottle rides along
        mujoco.mj_forward(env.model, env.data)
    th = s.yaw + np.radians(off[2])
    fwd, left = np.array([np.cos(s.yaw), np.sin(s.yaw)]), np.array([-np.sin(s.yaw), np.cos(s.yaw)])
    p = np.array([s.x, s.y])
    if item == "water_bottle":  # the robot ends up backed off by the opening after pulling
        p = p - 0.36 * fwd
    p = p + off[0] * fwd + off[1] * left
    env.stations["_probe"] = type(s)("_probe", float(p[0]), float(p[1]), float(th))
    env.stations["home"] = env.stations["_probe"]
    robot = AssistiveRobot(world=env)
    robot.station = station
    t0, sim0 = time.time(), robot.data.time
    anchor = robot.cam.detect().get(KITCHEN_TOP.anchor_marker)
    r = robot.pick(item, from_above=(item == "water_bottle"))
    held = robot.holding_ok() if r.get("ok") else False
    ok = bool(r.get("ok")) and held
    if close and ok and anchor is not None:
        from articulation import ArticulationModel
        robot.open_state = dict(model=ArticulationModel.load("drawer_model.json"), anchor_marker=KITCHEN_TOP.anchor_marker,
                                weld=f"grasp_{KITCHEN_TOP.body}", opened=opened, stance=(s.x, s.y, s.yaw),
                                anchor_pose=(anchor.rotation, anchor.position), part_markers=(KITCHEN_TOP.front_marker,))
        c = robot.close_opened()
        shut = env.drawer_open_m(KITCHEN_TOP) < 0.03
        r = {**r, "close": {k: v for k, v in c.items() if k in ("ok", "error", "left_open_m")}, "drawer_now": round(env.drawer_open_m(KITCHEN_TOP), 3)}
        ok = ok and shut and robot.holding_ok()
    elif close:
        ok = False
    return dict(off=off, ok=ok, result=r, sim_s=round(robot.data.time - sim0, 1),
                wall_s=round(time.time() - t0, 1), fallen=robot.wbc.fallen())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="cup", choices=["cup", "water_bottle"])
    ap.add_argument("--layout", default="tidy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, nargs="*", help="indices into the offset list")
    ap.add_argument("--close", action="store_true", help="after the drawer pick, push the drawer shut")
    ap.add_argument("--opened", type=float, default=0.36, help="how far the sim opens the drawer (m)")
    args = ap.parse_args()
    offs = [OFFSETS[i] for i in args.only] if args.only else OFFSETS
    n_ok = 0
    for off in offs:
        out = trial(args.item, off, args.layout, args.seed, args.close, args.opened)
        n_ok += out["ok"]
        r = out["result"]
        print(f"{'OK  ' if out['ok'] else 'FAIL'} off {off}  sim {out['sim_s']}s wall {out['wall_s']}s  "
              f"{r.get('error') or r.get('grasp_error_m')} {r.get('close', '')} {r.get('drawer_now', '')}", flush=True)
    print(f"{args.item}: {n_ok}/{len(offs)} (stance search {'on' if os.environ.get('G1_STANCE_SEARCH', '1') != '0' else 'off'})")


if __name__ == "__main__":
    main()
