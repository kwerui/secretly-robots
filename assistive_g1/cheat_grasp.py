"""Abstracted grasp ("cheat grab"): a backup for when the real grasp isn't reliable.

    G1_CHEAT_GRASP=1         always use the abstracted grasp
    G1_CHEAT_GRASP=fallback  try the real grasp (skills.pick) first, abstracted if it fails
    (unset / 0)              off: skills.py behaves exactly as before

What stays real: the item must be SEEN by the head camera (pick() needs a detection), the robot
walks to a grasp stance and moves its arm (IK) to the item, and the hand must get within
MAX_GAP of the *detected* item position or the grasp fails. Opening/closing drawers and doors
is untouched (real handle reach + the learned one-shot motion).

What is abstracted: the contact. Once the hand is near, the item glides into the palm
(SNAP_S), is welded there, and stops colliding with anything while held (no snagging on drawer
fronts or door frames). On release (hand_over / put_down, i.e. when its grasp weld is switched
off) it is set down upright on whatever surface is below it (raycast) and collides again.

Rationale: low-level contact-rich grasping is best learned (e.g. a Dex3 grasp policy trained in
Isaac); a hackathon doesn't have time for that. Everything around it (one-shot learning from
the carer, localization, search, the decision layer) is what we built.

Installed by the hook at the end of skills.py; nothing here runs unless the variable is set.
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

MODE = os.environ.get("G1_CHEAT_GRASP", "0").strip().lower()
ENABLED = MODE in ("1", "true", "yes", "always", "fallback")
FALLBACK = MODE == "fallback"
NEAR = float(os.environ.get("G1_CHEAT_NEAR", "0.20"))  # m, palm-to-item gap: above this, reach again, closer
MAX_GAP = float(os.environ.get("G1_CHEAT_MAX_GAP", "0.40"))  # m, above this after the second reach: fail
SNAP_S = 0.25  # s, the item glides into the hand over this long
UP = np.array([0.0, 0.0, 1.0])
VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # raycast against geom groups 0-2 only


# ---------------------------------------------------------------------------- item helpers
def _body(robot, item: str) -> int:
    return mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, robot.item_body[item])


def _free_adr(model, b: int) -> tuple[int, int]:
    j = model.body_jntadr[b]
    assert j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, "item body has no free joint"
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def _centre_offset(model, data, b: int) -> np.ndarray:
    """Item centre (mean of its collision geoms) in the body frame: Jack's bottle has its body
    origin at its base, our boxes at their centre."""
    gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
          and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not gs:
        return np.zeros(3)
    c = np.mean([data.geom_xpos[g] for g in gs], axis=0)
    return data.xmat[b].reshape(3, 3).T @ (c - data.xpos[b])


def _set_pose(robot, b: int, centre: np.ndarray, quat: np.ndarray, off: np.ndarray) -> None:
    """Put the item's centre at `centre` with orientation `quat` (body origin = centre - R off)."""
    m, d = robot.model, robot.data
    qa, da = _free_adr(m, b)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    d.qpos[qa:qa + 3] = centre - R.reshape(3, 3) @ off
    d.qpos[qa + 3:qa + 7] = quat
    d.qvel[da:da + 6] = 0.0


def _ghost(robot, item: str) -> None:
    """The held item collides with nothing (saved, restored by _unghost)."""
    m = robot.model
    b = _body(robot, item)
    if item in robot._cheat_ghosted:
        return
    saved = [(g, (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))) for g in range(m.ngeom)
             if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
    for g, _ in saved:
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
    robot._cheat_ghosted[item] = saved


def _unghost(robot, item: str) -> bool:
    saved = robot._cheat_ghosted.pop(item, None)
    if saved is None:
        return False
    for g, (ct, ca) in saved:
        robot.model.geom_contype[g], robot.model.geom_conaffinity[g] = ct, ca
    return True


def _grip_centre(robot, item: str, top_mode: bool) -> np.ndarray:
    """Where the item's centre sits in the hand: like the real grasps (side: palm 0.3 h above the
    centre, item just ahead of the palm; top: palm 3 cm below the item's top)."""
    it = robot.items[item]
    p, R = robot.ik.palm_pose()
    if top_mode:
        return p - np.array([0.0, 0.0, it.size[2] - min(0.03, it.size[2])])
    f = R[:, 0].copy()
    f[2] = 0.0
    f /= max(np.linalg.norm(f), 1e-9)
    return p + 0.03 * f - np.array([0.0, 0.0, 0.3 * it.size[2]])


# ---------------------------------------------------------------------------- the grasp
def _dbg(robot, tag: str) -> None:
    S = sys.modules[type(robot).__module__]
    if S.DEBUG:
        print(f"   cheat[{tag}]: base {robot.wbc.base_pos.round(3)} yaw {robot.wbc.base_yaw:.2f} "
              f"palm {robot.ik.palm_pose()[0].round(3)}", flush=True)


def cheat_pick(robot, item: str, from_above: bool = False) -> dict:
    S = sys.modules[type(robot).__module__]  # skills.py constants
    it = robot.items.get(item)
    if it is None:
        return {"ok": False, "error": f"unknown item {item}"}
    det = robot.cam.detect()
    if it.marker_id not in det:
        return {"ok": False, "error": f"{item} not visible from {robot.station}"}
    top = det[it.marker_id].position
    low = top[2] < S.LOW_ITEM_Z
    top_mode = from_above or low

    # 1. Grasp stance (as the real grasps), hand raised first so it can't catch under an edge
    th = robot.wbc.base_yaw
    if low:
        d = top[:2] - robot.wbc.base_pos[:2]
        th = float(np.arctan2(d[1], d[0]))
    fwd = np.array([np.cos(th), np.sin(th), 0.0])
    left = np.array([-np.sin(th), np.cos(th), 0.0])
    if low:
        robot._raise_hand(robot.wbc.base_pos[2] + 0.05)
        goal = top[:2] - 0.20 * fwd[:2] + 0.25 * left[:2]
    else:
        robot._raise_hand(top[2] + (0.25 if from_above else 0.06), toward=top)
        standoff = S.ABOVE_STANDOFF if from_above else S.STANDOFF
        goal = top[:2] - standoff * fwd[:2] + S.GRASP_LATERAL * left[:2]
    if np.linalg.norm(goal - robot.wbc.base_pos[:2]) > 0.03:
        robot._drive_to(goal[0], goal[1], th, tol=0.03, timeout=10.0)
    _dbg(robot, "stance")
    det = robot.cam.detect()  # closer view (falls back to the first detection if occluded)
    if it.marker_id in det:
        top = det[it.marker_id].position
    centre = top - np.array([0.0, 0.0, it.size[2] + 0.001])

    # 2. Posture and an arm reach to near the item. `grip` is where the palm would be in a real
    # grasp; the hand has to get within NEAR / MAX_GAP of it.
    if low:
        robot._set_posture(0.30, 0.4)
        grip = top - np.array([0, 0, min(0.03, it.size[2])])
        pre, axes = grip + np.array([0, 0, 0.08]), [(0, -UP, 0.4)]
        robot.move_hand(pre, axes, duration=1.0)
    elif from_above:
        # Stay ABOVE the drawer front (an arm down in the drawer jams on it when stepping back):
        # use the real top grasp's collision-checked planner for a pose `clear` over the item.
        grip = top - np.array([0, 0, min(0.03, it.size[2])])

        def pitched(deg):
            a = np.deg2rad(deg)
            return -UP * np.cos(a) + fwd * np.sin(a)
        cands = [[(0, pitched(35), 0.6), (1, -left, 0.3)], [(0, pitched(35), 0.6), (1, left, 0.3)],
                 [(0, pitched(20), 0.6), (1, -left, 0.3)], [(0, -UP, 0.6), (1, left, 0.3)], [(0, -UP, 0.6)]]
        axes, best = None, np.inf
        for clear in (0.10, 0.15, 0.22):
            pre = top + np.array([0, 0, clear])
            axes, best = robot._choose(pre + np.array([0, 0, 0.06]), pre, cands,
                                       [(0.74, 0.3), (0.74, 0.15), (0.70, 0.45)], multi_start=True,
                                       elbow_max=S.ABOVE_ELBOW_MAX)
            if best <= 0.05:
                break
        if axes is not None and best <= 0.08:
            q_hi, q_pre = robot._best_q
            robot._reach("right", q_hi, pre + np.array([0, 0, 0.06]), pre, q_pre, axes, duration=0.8)
        else:  # no clean plan: plain IK towards a point well above the item
            robot._set_posture(0.74, 0.3)
            pre, axes = top + np.array([0, 0, 0.18]), [(0, -UP, 0.3)]
            robot.move_hand(pre, axes, duration=1.0)
    else:
        z = centre[2]
        robot._set_posture(*((0.74, 0.0) if z > 0.85 else (0.74, 0.3) if z > 0.6 else (0.60, 0.5)))
        grip = centre + np.array([0, 0, 0.3 * it.size[2]])
        pre, axes = grip - 0.08 * fwd, [(0, fwd, 0.4)]
        robot.move_hand(pre, axes, duration=1.0)
    _dbg(robot, "reached")
    gap = float(np.linalg.norm(robot.ik.palm_pose()[0] - grip))
    if gap > NEAR and not from_above:  # second try, closer (not down into a drawer)
        robot.move_hand(grip + (np.array([0, 0, 0.04]) if top_mode else -0.04 * fwd), axes, duration=0.8)
        gap = float(np.linalg.norm(robot.ik.palm_pose()[0] - grip))
    if gap > MAX_GAP:
        robot._set_posture(0.74, 0.0)
        robot.arm_home()
        return {"ok": False, "error": f"hand couldn't get near the {item} ({gap:.2f} m away)", "grasp": "abstracted"}

    # 3. The abstraction: the item glides into the hand, then the grasp weld holds it
    m, d = robot.model, robot.data
    b = _body(robot, item)
    off = _centre_offset(m, d, b)
    robot._cheat_off[item] = off
    _ghost(robot, item)
    R0 = d.xmat[b].reshape(3, 3)
    c0 = d.xpos[b] + R0 @ off
    # upright, keeping the item's heading (it may have been knocked over)
    yaw = float(np.arctan2(R0[1, 0], R0[0, 0])) if abs(R0[2, 2]) > 0.5 else 0.0
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    robot._cheat_quat[item] = quat
    n = max(1, int(SNAP_S / S.CONTROL_DT))
    for k in range(1, n + 1):
        a = k / n
        a = a * a * (3 - 2 * a)
        _set_pose(robot, b, (1 - a) * c0 + a * _grip_centre(robot, item, top_mode), quat, off)
        robot._tick()
    _set_pose(robot, b, _grip_centre(robot, item, top_mode), quat, off)
    mujoco.mj_forward(m, d)
    robot._weld(f"grasp_{robot.item_body[item]}", True)
    robot.holding = item
    _, R_palm = robot.ik.palm_pose()
    robot.hold_vert = (R_palm.T @ UP, UP.copy())  # keep the item's axis vertical
    robot.hold_mode = "top" if top_mode else "side"
    robot.hold_dz = float(robot.ik.palm_pose()[0][2] - _grip_centre(robot, item, top_mode)[2])

    _dbg(robot, "welded")
    # 4. Lift clear, step back, carry (as the real grasps)
    p, _ = robot.ik.palm_pose()
    if top_mode:
        robot.move_hand(p + np.array([0, 0, 0.12]), axes, duration=0.8, settle=0.1)
    else:
        robot.move_hand(p + np.array([0, 0, 0.04]) - 0.08 * fwd, axes, duration=0.8, settle=0.0)
    _dbg(robot, "lifted")
    if not low:
        robot._step_back(S.STEP_BACK)
    _dbg(robot, "backed")
    robot.carry()
    return {"ok": True, "item": item, "grasp": "abstracted", "hand_gap_m": round(gap, 3),
            "from": "floor" if low else "above" if from_above else "side"}


def _settle_on_surface(robot, item: str) -> None:
    """Released: stand the item upright on the surface below it and give it its collisions back
    (furniture only for now: the arm is still around it; hand_over/put_down then restore the robot)."""
    m, d = robot.model, robot.data
    b = _body(robot, item)
    off = robot._cheat_off.get(item, np.zeros(3))  # measured before the item was ghosted
    c = d.xpos[b] + d.xmat[b].reshape(3, 3) @ off
    quat = robot._cheat_quat.get(item, np.array([1.0, 0, 0, 0]))
    robot_root = m.body_rootid[m.body("pelvis").id]
    gid = np.array([-1], dtype=np.int32)
    start = c.copy()
    hit_z = None
    for _ in range(4):  # skip anything of the robot's below the item (e.g. the forearm)
        dist = mujoco.mj_ray(m, d, start, -UP, VISIBLE_GROUPS, 1, b, gid)
        if dist < 0:
            break
        z = start[2] - dist
        if m.body_rootid[m.geom_bodyid[gid[0]]] != robot_root:
            hit_z = z
            break
        start = np.array([c[0], c[1], z - 1e-3])
    _unghost(robot, item)
    if hit_z is not None:
        c[2] = hit_z + robot.items[item].size[2] + 0.002
        _set_pose(robot, b, c, quat, off)
    robot._real_item_collisions(item, False)
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------- install
def install(cls) -> None:
    """Patch AssistiveRobot (called by skills.py when G1_CHEAT_GRASP is set)."""
    if not ENABLED or getattr(cls, "_cheat_installed", False):
        return
    real_init, real_pick, real_weld, real_coll = cls.__init__, cls.pick, cls._weld, cls._item_collisions

    def __init__(self, *a, **kw):
        self._cheat_ghosted, self._cheat_quat, self._cheat_off = {}, {}, {}
        real_init(self, *a, **kw)

    def pick(self, item, from_above=False):
        if FALLBACK:
            r = real_pick(self, item, from_above)
            if r.get("ok") or self.holding is not None:
                return {**r, "grasp": "real"}
            r2 = cheat_pick(self, item, from_above)
            return {**r2, "real_grasp_error": r.get("error")}
        return cheat_pick(self, item, from_above)

    def _weld(self, name, on, torquescale=1.0, ik=None):
        # releasing an item held by the abstracted grasp: set it down properly
        if (not on and self.holding is not None and self.holding in self._cheat_ghosted
                and name == f"grasp_{self.item_body[self.holding]}"):
            real_weld(self, name, on, torquescale, ik)
            _settle_on_surface(self, self.holding)
            return
        real_weld(self, name, on, torquescale, ik)

    def _item_collisions(self, item, with_robot):
        if item in self._cheat_ghosted:
            if not with_robot:
                return  # held by the abstracted grasp: stays a ghost
            _unghost(self, item)
        real_coll(self, item, with_robot)

    cls.__init__, cls.pick, cls._weld, cls._item_collisions = __init__, pick, _weld, _item_collisions
    cls._real_item_collisions = real_coll
    cls._real_pick = real_pick
    cls._cheat_installed = True
    print(f"[cheat_grasp] abstracted grasp ON (G1_CHEAT_GRASP={MODE})", flush=True)
