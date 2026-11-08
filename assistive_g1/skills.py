"""Robot skills the agent layer calls: go_to, look, pick, open_cupboard, hand_over.

Each skill returns a small result dict (ok + details) so an LLM planner can reason over
outcomes. Perception goes through the head camera only; the simulator's ground truth is
used for the robot's own base pose (localisation is out of scope for the hackathon).
"""

from __future__ import annotations

import os
from typing import Callable

import mujoco
import numpy as np

from g1_wbc import G1WBC, ARM_HOME, SIM_DT
from hand import Gripper, Q_RELEASE
from ik import ArmIK
from mppi import ArmMPPI, MotionLog
from nav import Navigator
from perception import HeadCamera, MarkerPose, ID_TO_ITEM
from articulation import ArticulationModel, rot_axis_angle
from scene import ITEMS, STATIONS, DOOR_MARKER, ANCHOR_MARKER, build_scene

CONTROL_DT = 0.02  # arm/nav update period (50 Hz)
# (pelvis height, torso pitch) tried in order when an item is hard to reach
POSTURES = [(0.74, 0.0), (0.74, 0.3), (0.68, 0.45), (0.60, 0.5)]
# deep squats for items near the floor (GR00T accepts pelvis heights down to ~0.2 m)
LOW_POSTURES = [(0.30, 0.4), (0.25, 0.4), (0.22, 0.4)]
LOW_ITEM_Z = 0.45  # items whose top is below this are picked from a squat (_pick_low)
STANDOFF = 0.40  # preferred forward pelvis-to-item distance for a grasp (m)
GRASP_LATERAL = 0.15  # m, item to the right of the pelvis (in line with the right shoulder)
DOOR_KEEP_REACH = 0.30  # m, walk back while pulling if the handle comes closer than this
BACKOFF_SPEED = 0.15  # m/s
STEP_BACK = 0.30  # m, after a grasp: the carried item's swing circle when turning must clear the furniture
HUB = (0.35, 0.0)  # all station-to-station routes pass through here


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class AssistiveRobot:
    def __init__(self, placement=None, on_frame: Callable[["AssistiveRobot"], None] | None = None,
                 world=None):
        self.world = world
        if world is not None:
            self.model, self.data = world.model, world.data
            self.items = world.items
            self.item_body = world.item_body
            self.stations = world.stations
            self.delivery = world.delivery
            extra_sizes = getattr(world, "marker_sizes", None)
            labels = getattr(world, "marker_labels", None)
            extra_graspable = getattr(world, "graspable", ())
        else:
            from scene import SIDE_TABLE
            self.model, self.data = build_scene(placement)
            self.items = ITEMS
            self.item_body = {k: k for k in ITEMS}
            self.stations = STATIONS
            self.delivery = SIDE_TABLE
            extra_sizes = None
            labels = None
            extra_graspable = ()

        self.id_to_item = {it.marker_id: name for name, it in self.items.items()}
        self.wbc = G1WBC(self.model, self.data)
        self.grip = Gripper(self.model, self.data, "right")
        # Two IK views of the same arm: the palm site (door handle, generic moves) and the grasp
        # site between thumb and finger pads (everything that touches an item).
        self.ik = ArmIK(self.model, self.data, "right", extra_graspable=extra_graspable)
        self.ik_item = ArmIK(self.model, self.data, "right", extra_graspable=extra_graspable, site="right_grasp")
        # Sampling-based MPC for the transport motions (ARM_MPPI=0 falls back to IK tracking
        # of a Cartesian smoothstep everywhere, for A/B comparison).
        self.use_mppi = os.environ.get("ARM_MPPI", "1") != "0"
        self.mppi = ArmMPPI(self.ik_item)
        self.mppi_palm = ArmMPPI(self.ik)
        self.motion = MotionLog(CONTROL_DT)  # arm command history -> smoothness metrics
        self.cam = HeadCamera(self.model, self.data, extra_sizes=extra_sizes, labels=labels)
        self.nav = Navigator()
        self.on_frame = on_frame  # called every CONTROL_DT, e.g. to record video
        self.holding: str | None = None
        self.hold_vert = (2, np.array([0, 0, 1.0]))  # palm axis kept fixed while holding
        self.hold_mode, self.hold_dz = "side", 0.0  # grasp type and grasp-site height above item centre
        self.hold_rel = np.zeros(3)  # item centre in the hand frame at the moment of the grasp
        self.hold_item_axis = np.array([0, 0, 1.0])  # hand-frame direction that is the item's "up" ("pitched" holds)
        self.planned_q = (ARM_HOME[7:].copy(), ARM_HOME[7:].copy())  # (pre-grasp, grasp) joints from _choose
        self.grasp_log: list[dict] = []  # per-grasp contact evidence (forces, pads) for the pitch
        self.placed: list[np.ndarray] = []  # where this robot has set things down (side table)
        self.station = "home"
        self.arm_cmd = ARM_HOME[7:].copy()
        self.last_seen: dict[int, MarkerPose] = {}
        s0 = self.stations["home"]
        self.wbc.place(s0.x, s0.y, s0.yaw)
        self.wait(1.0)

    # ------------------------------------------------------------------ basics
    def _tick(self) -> None:
        self.wbc.step(int(round(CONTROL_DT / SIM_DT)))
        self.motion.add(self.arm_cmd, self.data.site_xpos[self.ik_item.site])
        if self.on_frame is not None:
            self.on_frame(self)

    def wait(self, seconds: float) -> None:
        for _ in range(int(seconds / CONTROL_DT)):
            self._tick()

    def _set_right_arm(self, q: np.ndarray) -> None:
        self.arm_cmd = q
        full = ARM_HOME.copy()
        full[7:] = q
        self.wbc.set_arm_targets(full)

    # ------------------------------------------------------------------ navigation
    def _drive_to(self, x: float, y: float, yaw: float | None, tol: float = 0.04, timeout: float = 25.0,
                  pass_through: bool = False) -> bool:
        """Walk to (x, y[, yaw]) with the smooth PD navigator (nav.py).

        pass_through: a waypoint. Return as soon as we're within `tol`, keeping the current
        velocity (no stop), so the next leg continues smoothly.
        """
        self.nav.p.pos_tol = tol
        self.nav.arrived = False  # keep self.nav.cmd: continue from the current velocity
        for _ in range(int(timeout / CONTROL_DT)):
            p = self.wbc.base_pos
            if pass_through and np.hypot(x - p[0], y - p[1]) < tol:
                return True
            cmd = self.nav.step((p[0], p[1], self.wbc.base_yaw), self.wbc.base_vel_body, (x, y, yaw), CONTROL_DT)
            self.wbc.set_velocity(*cmd)
            self._tick()
            if self.wbc.fallen():
                self.wbc.set_velocity(0, 0, 0)
                return False
            if self.nav.stopped:
                break
        arrived = self.nav.stopped
        self.nav.reset()
        self.wbc.set_velocity(0, 0, 0)
        self.wait(0.8)
        if not arrived:  # timed out (blocked, e.g. a carried item jammed on furniture)
            p = self.wbc.base_pos
            arrived = np.hypot(x - p[0], y - p[1]) < 2 * tol
        return arrived

    def go_to(self, station: str) -> dict:
        """Walk to a named station."""
        if station not in self.stations:
            return {"ok": False, "error": f"unknown station {station}", "stations": list(self.stations)}
        s = self.stations[station]
        p = self.wbc.base_pos
        if self.world is not None and hasattr(self.world, "plan_path"):
            path = self.world.plan_path(p[:2], (s.x, s.y))
            if path is not None and len(path) > 2:
                for wx, wy in path[1:-1]:
                    self._drive_to(wx, wy, None, tol=0.22, pass_through=True)
                    if self.wbc.fallen():
                        return {"ok": False, "error": "fell over"}
        else:
            if np.hypot(p[0] - HUB[0], p[1] - HUB[1]) > 0.3 and station != self.station:
                self._drive_to(*HUB, None, tol=0.25, pass_through=True)
                if self.wbc.fallen():
                    return {"ok": False, "error": "fell over"}
        # Walk more gently with something in the hand: the gait sway is what works a pinch loose.
        v0, w0 = self.nav.p.v_max, self.nav.p.w_max
        if self.holding is not None:
            self.nav.p.v_max, self.nav.p.w_max = min(v0, 0.30), min(w0, 0.7)
        try:
            ok = self._drive_to(s.x, s.y, s.yaw)
        finally:
            self.nav.p.v_max, self.nav.p.w_max = v0, w0
        if ok:
            self.station = station
        out = {"ok": ok, "station": station, "pose": self.wbc.base_pos.round(2).tolist()}
        if not ok:
            out["error"] = f"couldn't reach the {station}"
        return out

    # ------------------------------------------------------------------ perception
    def look(self) -> dict:
        """Detect markers in the head camera. Returns item names -> world position."""
        det = self.cam.detect()
        self.last_seen = det
        items = {self.id_to_item[m]: p.position.round(3).tolist() for m, p in det.items() if m in self.id_to_item}
        extra = {}
        if DOOR_MARKER in det:
            extra["cupboard_door"] = det[DOOR_MARKER].position.round(3).tolist()
        if ANCHOR_MARKER in det:
            extra["cupboard_anchor"] = det[ANCHOR_MARKER].position.round(3).tolist()
        if self.world is not None and hasattr(self.world, "marker_labels"):
            for m, lbl in self.world.marker_labels.items():
                if m in det:
                    extra[lbl] = det[m].position.round(3).tolist()
        return {"ok": True, "station": self.station, "items": items, **extra}

    def look_closer(self) -> dict:
        """Re-approach the current station, then lean in and look again.

        Used when an expected item isn't seen (e.g. after opening the cupboard the robot has
        backed off, or the item sits deep under a shelf). Detections from all views are merged.
        """
        s = self.stations.get(self.station)
        if s is not None:
            self._drive_to(s.x, s.y, s.yaw, tol=0.03, timeout=10.0)
        merged = self.look()
        for pitch in (0.25, 0.45):
            if merged["items"]:
                break
            self._set_posture(0.74, pitch)
            seen = self.look()
            merged["items"].update(seen["items"])
        self._set_posture(0.74, 0.0)
        return merged

    def save_view(self, path) -> None:
        """Write the last head-camera frame, with detections drawn, to `path` (debugging)."""
        import cv2
        if self.cam.last_rgb is None:
            return
        img = self.cam.annotate(self.cam.last_rgb, {})
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------ arm
    def move_hand(self, target: np.ndarray, axes=None, duration: float = 1.0, settle: float = 0.3,
                  ik: ArmIK | None = None, smooth: bool = False, posture_weight: float = 0.02) -> float:
        """Move the control site to `target` (world). Returns the final position error (m).

        ik: which site to control, self.ik (palm) by default or self.ik_item (grasp point).
        smooth: transport motion -> MPPI (receding-horizon, collision-aware, smooth joint
            profiles), with `duration` as a time budget, followed by a short IK settle.
            Otherwise (precise approach, placement) the site follows a Cartesian smoothstep
            tracked by the damped-least-squares IK every tick.
        """
        ik = ik or self.ik
        target = np.asarray(target, float)
        if smooth and self.use_mppi:
            # Moving reference along the straight line (smoothstep timing, as before); MPPI
            # tracks it over its horizon, so the pace is the same as the IK version and the
            # planner's freedom goes into smoothness and clearance.
            mp = self.mppi if ik is self.ik_item else self.mppi_palm
            mp.reset()
            # while transporting, items (and the door) are obstacles for the whole hand too
            ik.hand_may_touch_graspables = False
            start, _ = ik.palm_pose()
            n = max(1, int(duration / CONTROL_DT))

            def ref_at(step: int) -> np.ndarray:
                a = min(1.0, step / n)
                a = a * a * (3 - 2 * a)
                return (1 - a) * start + a * target

            n_max = n + int(0.4 * duration / CONTROL_DT)
            q_ik = self.arm_cmd.copy()
            for k in range(n_max):
                ref = np.array([ref_at(k + 1 + h) for h in range(mp.p.H)])
                # one-step IK towards the next reference point = the nominal the sampler is
                # centred on; MPPI then trades tracking against smoothness and clearance
                q_ik, _ = ik.solve(ref[0], axes, q_init=q_ik, iters=10)
                u_nom = (q_ik - self.arm_cmd) / CONTROL_DT
                q = mp.plan(self.arm_cmd, ref, axes, u_nom=u_nom)
                self._set_right_arm(q)
                self._tick()
                if k >= n and np.linalg.norm(ik.palm_pose()[0] - target) < 0.015:
                    break
            q = self.arm_cmd.copy()
            for _ in range(int(settle / CONTROL_DT)):
                q, _ = ik.solve(target, axes, q_init=q, iters=15)
                self._set_right_arm(q)
                self._tick()
            ik.hand_may_touch_graspables = True
            p, _ = ik.palm_pose()
            return float(np.linalg.norm(p - target))
        start, _ = ik.palm_pose()
        n = max(1, int(duration / CONTROL_DT))
        q = self.arm_cmd.copy()
        for k in range(1, n + int(settle / CONTROL_DT) + 1):
            a = min(1.0, k / n)
            a = a * a * (3 - 2 * a)  # smoothstep
            goal = (1 - a) * start + a * target
            q, _ = ik.solve(goal, axes, q_init=q, iters=15, posture_weight=posture_weight)
            self._set_right_arm(q)
            self._tick()
        p, _ = ik.palm_pose()
        return float(np.linalg.norm(p - target))

    def arm_home(self, duration: float = 1.0) -> None:
        q0 = self.arm_cmd.copy()
        n = int(duration / CONTROL_DT)
        for k in range(1, n + 1):
            a = k / n
            self._set_right_arm((1 - a) * q0 + a * ARM_HOME[7:])
            self._tick()

    def _body(self, item: str) -> str:
        """MuJoCo body name of an item (the house uses Jack's body names, e.g. "waterbottle")."""
        return self.item_body.get(item, item)

    def _set_held(self, item: str | None) -> None:
        """Tell both IK views which body rides in the hand (ignored as an obstacle while held)."""
        b = None if item is None else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._body(item))
        self.ik.held_body = b
        self.ik_item.held_body = b

    def _weld(self, name: str, on: bool, torquescale: float = 1.0) -> None:
        """Switch a grasp weld on/off without anything jumping.

        eq_data layout for welds (checked against the compiler, MuJoCo 3.3+/4.x):
          [0:3] weld point in body2's frame, [3:6] the same point in body1's frame,
          [6:10] orientation of body2 relative to body1, [10] torquescale, [11] = 1.
        torquescale = 0 makes it position-only (a ball joint at the palm).
        """
        eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if on:
            b1 = self.model.eq_obj1id[eq]
            b2 = self.model.eq_obj2id[eq]
            p1, R1, q1 = self.data.xpos[b1], self.data.xmat[b1].reshape(3, 3), self.data.xquat[b1]
            p2, R2, q2 = self.data.xpos[b2], self.data.xmat[b2].reshape(3, 3), self.data.xquat[b2]
            palm, _ = self.ik.palm_pose()
            q1_inv = np.zeros(4)
            mujoco.mju_negQuat(q1_inv, q1)
            rel_q = np.zeros(4)
            mujoco.mju_mulQuat(rel_q, q1_inv, q2)
            self.model.eq_data[eq, 0:3] = R2.T @ (palm - p2)
            self.model.eq_data[eq, 3:6] = R1.T @ (palm - p1)
            self.model.eq_data[eq, 6:10] = rel_q
            self.model.eq_data[eq, 10] = torquescale
            self.model.eq_data[eq, 11] = 1.0
        self.data.eq_active[eq] = 1 if on else 0

    # ------------------------------------------------------------------ manipulation
    def pick(self, item: str, from_above: bool = False) -> dict:
        """Grasp an item seen by the head camera at the current station (side or from_above).

        Physical grasp: the open hand (thumb swung back, fingers forward) is brought in along the
        robot's heading so the item sits between thumb and finger pads, the fingers close under
        position control until the hardware torque limit, the contacts are checked (thumb AND a
        finger, opposing, with a minimum normal force), then the item is lifted and the hold is
        verified before the arm retracts. No constraint is used: the item is held by friction.
        """
        it = self.items.get(item)
        if it is None:
            return {"ok": False, "error": f"unknown item {item}"}
        det = self.cam.detect()
        if it.marker_id not in det:
            return {"ok": False, "error": f"{item} not visible from {self.station}"}
        top = det[it.marker_id].position
        centre = top - np.array([0, 0, it.size[2] + 0.001])
        low = top[2] < LOW_ITEM_Z  # on (or near) the floor
        if low:
            return self._pick_low(item, top, centre)
        if from_above:
            return self._pick_from_above(item, top, centre)
        # Grasp point = where the pinch centre should be: a little above the item's centre so the
        # fingers (which hang ~4.5 cm below the pinch axis) stay clear of the surface.
        grasp = centre + np.array([0, 0, max(0.3 * it.size[2], 0.045 - it.size[2] + 0.012)])
        grasp[2] = min(grasp[2], top[2] - 0.015)

        self.grip.open()
        base = self.wbc.base_pos
        # Raise the hand to chest height, close to the body, before anything else so it can't
        # catch under a table edge while lifting or stepping in.
        self._raise_hand(grasp[2] + 0.06, toward=grasp)
        # Line up so the item is straight ahead of the right shoulder at STANDOFF, facing the
        # furniture squarely (the station's heading, not whatever heading the last walk ended
        # with: after opening the cupboard the robot can be 10-15 deg skewed, and a diagonal
        # entry into the cupboard is out of the arm's reach / drags the item along a side wall).
        # If the stance is blocked (typically by the open cupboard door beside the right leg) or
        # the grasp is not reachable from it, try again standing a little further left.
        th = self.stations[self.station].yaw if self.station in self.stations else self.wbc.base_yaw
        fwd2 = np.array([np.cos(th), np.sin(th)])
        left2 = np.array([-np.sin(th), np.cos(th)])
        up = np.array([0, 0, 1.0])
        axes, best = None, np.inf
        for attempt, lateral in enumerate((GRASP_LATERAL, GRASP_LATERAL + 0.07, GRASP_LATERAL - 0.05)):
            goal = grasp[:2] - STANDOFF * fwd2 + lateral * left2  # pelvis left of the item
            if attempt and self.station in self.stations:  # unstick: back to the station first
                st = self.stations[self.station]
                self._drive_to(st.x, st.y, st.yaw, tol=0.04, timeout=6.0)
            if np.linalg.norm(goal - base[:2]) > 0.03 or abs(wrap(self.wbc.base_yaw - th)) > 0.06:
                self._drive_to(goal[0], goal[1], th, tol=0.025, timeout=8.0)
                base = self.wbc.base_pos
            th_now = self.wbc.base_yaw
            approach = np.array([np.cos(th_now), np.sin(th_now), 0.0])
            reach = float((grasp - base)[:2] @ approach[:2])
            # Pre-grasp back along the approach, but never closer than ~0.27 m to the pelvis:
            # nearer than that the arm would have to fold into the torso.
            back = float(np.clip(reach - 0.27, 0.06, 0.14))
            pre = grasp - back * approach + np.array([0, 0, 0.03])
            # Fingers along the approach, pinch axis horizontal: hand z up (thumb on the body side
            # of the item) or down (thumb on the outside), whichever the arm reaches without
            # touching furniture or the torso.
            candidates = [
                [(0, approach, 0.6), (2, up, 0.4)],
                [(0, approach, 0.6), (2, -up, 0.4)],
            ]
            axes, best = self._choose(pre, grasp, candidates, POSTURES, ik=self.ik_item)
            if best <= 0.05:
                break
        if best > 0.05:
            self._set_posture(0.74, 0.0)
            return {"ok": False, "error": f"no collision-free grasp for {item} from here (score {best:.3f})"}

        # Come in along the approach line: far pre-grasp (outside any cupboard), pre-grasp, grasp.
        self.move_hand(grasp - (back + 0.10) * approach + np.array([0, 0, 0.03]), axes, duration=1.0, settle=0.1,
                       ik=self.ik_item)
        e1 = self.move_hand(pre, axes, duration=0.7, ik=self.ik_item)
        e2 = self.move_hand(grasp, axes, duration=0.9, settle=0.4, ik=self.ik_item)
        if e2 > 0.04:
            self.move_hand(pre, axes, duration=0.8, ik=self.ik_item)
            self.arm_home()
            self._set_posture(0.74, 0.0)
            return {"ok": False, "error": f"could not reach {item} (residual {e2:.3f} m)", "pre_err": round(e1, 3)}
        res = self._close_and_lift(item, axes, lift=np.array([0, 0, 0.05]), ik=self.ik_item)
        if not res["ok"]:
            self.move_hand(pre, axes, duration=0.8, ik=self.ik_item)
            self.arm_home()
            self._set_posture(0.74, 0.0)
            return {**res, "pre_err": round(e1, 3)}
        # The item is held by friction; it stays upright iff this hand axis keeps the world
        # direction it had at the grasp. Record it as measured, not as asked.
        _, R_hand = self.ik_item.palm_pose()
        idx = axes[1][0]
        self.hold_vert = (idx, R_hand[:, idx].copy())
        self.hold_mode, self.hold_dz = "side", grasp[2] - centre[2]
        self.move_hand(grasp - (back + 0.10) * approach + np.array([0, 0, 0.06]), self._hold_axes(approach),
                       duration=1.0, settle=0.0, ik=self.ik_item)
        # Step back with the arm frozen: pulls the item straight out of a cupboard / off the
        # table before the carry motion swings it sideways (it would snag on the frame).
        self._step_back(STEP_BACK)
        if not self.holding_ok():
            return self._lost(item, "slipped out while stepping back")
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e2, 4), **res["grasp"]}

    def _close_and_lift(self, item: str, axes, lift: np.ndarray, ik: ArmIK, min_force: float = 4.0,
                        close_time: float = 0.6, settle_time: float = 0.4) -> dict:
        """Close the fingers on `item` where the hand is, verify the pinch, lift and re-verify."""
        q_hold = self.arm_cmd.copy()
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._body(item))
        R_hand = self.data.xmat[self.grip.palm_body].reshape(3, 3)
        item_in_hand = (R_hand.T @ (self.data.xpos[b] - self.data.xpos[self.grip.palm_body])).round(3).tolist()
        n = int(close_time / CONTROL_DT)
        for k in range(1, n + 1):
            self.grip.blend(k / n)
            self._set_right_arm(q_hold)
            self._tick()
        for _ in range(int(settle_time / CONTROL_DT)):
            self._set_right_arm(q_hold)
            self._tick()
        c = self.grip.contacts(self._body(item))
        evidence = {"pads": c.n_pads, "thumb": c.thumb, "fingers": c.fingers, "normal_force_N": round(c.normal_force, 1),
                    "opposing": c.opposing, "finger_q": self.grip.q.round(2).tolist(), "item_in_hand_before": item_in_hand}
        if not c.opposing or c.normal_force < min_force:
            self.grip.open()
            self.wait(0.3)
            self.grasp_log.append({"item": item, "ok": False, **evidence})
            return {"ok": False, "error": f"fingers closed but no stable pinch on {item}", "grasp": evidence}
        self.holding = item
        self._set_held(item)
        self.hold_rel = self._item_in_hand(item)  # reference for slip detection
        p, _ = ik.palm_pose()
        e = self.move_hand(p + lift, axes, duration=0.7, settle=0.3, ik=ik)
        c2 = self.grip.contacts(self._body(item))
        evidence.update({"lift_err_m": round(e, 3), "pads_after_lift": c2.n_pads,
                         "normal_force_after_lift_N": round(c2.normal_force, 1),
                         "item_in_hand_after": self._item_in_hand(item).round(3).tolist(),
                         "slip_mm": round(1000 * self.slip(), 1)})
        if not self.holding_ok(0.03) or not c2.opposing:
            self.grasp_log.append({"item": item, "ok": False, **evidence})
            return self._lost(item, "slipped during the lift", evidence)
        self.grasp_log.append({"item": item, "ok": True, **evidence})
        return {"ok": True, "grasp": evidence}

    def _lost(self, item: str, why: str, evidence: dict | None = None) -> dict:
        self.grip.open()
        self.holding = None
        self._set_held(None)
        self.wait(0.3)
        out = {"ok": False, "error": f"lost the {item}: {why}"}
        if evidence:
            out["grasp"] = evidence
        return out

    def _pick_from_above(self, item: str, top: np.ndarray, centre: np.ndarray) -> dict:
        """Pinch an item standing inside an open drawer (or on a surface the side approach can't
        reach), coming from above.

        Physical version: the fingers point down and forward at ~45 deg (straight down, the
        fingertips reach 11 cm past the pinch and hit the drawer floor; pitched, 8 cm), the pinch
        axis stays horizontal, and the pinch sits a few cm above the item's centre so the item is
        held near the palm (a pinch near the fingertips squeezes a tall item out along the
        fingers, see the floor pick). Hold verification as in the side grasp.
        """
        it = self.items[item]
        base = self.wbc.base_pos
        self.grip.open()
        self._raise_hand(top[2] + 0.14, toward=top)
        th = self.stations[self.station].yaw if self.station in self.stations else self.wbc.base_yaw
        fwd2 = np.array([np.cos(th), np.sin(th)])
        left2 = np.array([-np.sin(th), np.cos(th)])
        # Stand a little further back than for a side grasp: the arm comes down from above.
        goal = centre[:2] - (STANDOFF + 0.05) * fwd2 + GRASP_LATERAL * left2
        if np.linalg.norm(goal - base[:2]) > 0.03 or abs(wrap(self.wbc.base_yaw - th)) > 0.06:
            self._drive_to(goal[0], goal[1], th, tol=0.025, timeout=8.0)
            det2 = self.cam.detect()
            if it.marker_id in det2:
                top = det2[it.marker_id].position
                centre = top - np.array([0, 0, it.size[2] + 0.001])
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0.0])
        left = np.array([-np.sin(th), np.cos(th), 0.0])
        up = np.array([0, 0, 1.0])
        grasp = centre + np.array([0, 0, min(0.04, 0.4 * it.size[2])])
        grasp[2] = min(grasp[2], top[2] - 0.03)
        pre = grasp + np.array([0, 0, 0.14])

        def pitched(deg):  # finger direction: down, tilted forward by `deg` (0 = straight down)
            a = np.deg2rad(deg)
            return -up * np.cos(a) + fwd * np.sin(a)

        # Steep first: inside a drawer the forearm must clear the drawer front, which a flatter
        # hand cannot (the planner's IK finds a pose, the descent then stalls on the front).
        candidates = [[(0, pitched(20), 0.7), (1, -left, 0.7)], [(0, pitched(20), 0.7), (1, left, 0.7)],
                      [(0, pitched(20), 0.7), (1, fwd, 0.7)], [(0, pitched(35), 0.7), (1, -left, 0.7)],
                      [(0, pitched(35), 0.7), (1, left, 0.7)], [(0, pitched(50), 0.7), (1, -left, 0.7)],
                      [(0, pitched(50), 0.7), (1, fwd, 0.7)]]
        axes, best = self._choose(pre, grasp, candidates, POSTURES, ik=self.ik_item)
        if best > 0.05:
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"no collision-free top grasp for {item} (score {best:.3f})"}

        self.move_hand(pre + np.array([0, 0, 0.05]), axes, duration=1.0, settle=0.1, ik=self.ik_item)
        e1 = self.move_hand(pre, axes, duration=0.6, ik=self.ik_item)
        # The descent runs between the drawer front and the item with the collision rows active,
        # which costs orientation: hold the pinch axis and finger direction hard on the way down
        # and settle longer, then check the pinch axis really is horizontal before closing (a
        # rolled hand pinches the item off-centre and the hold does not survive the walk).
        firm = [(i, v, 1.2) for i, v, _ in axes]
        home0 = self.ik_item.home
        self.ik_item.home = self.planned_q[1]  # null space -> the configuration the planner verified
        try:
            e2 = self.move_hand(grasp, firm, duration=1.2, settle=0.6, ik=self.ik_item, posture_weight=0.15)
        finally:
            self.ik_item.home = home0
        _, R_now = self.ik_item.palm_pose()
        roll = float(abs(R_now[2, 1]))  # vertical component of the pinch axis
        if roll > 0.2 or e2 > 0.025:
            # The tracked descent settled in a rolled configuration; the planner's own solution
            # for this point is collision-free and level, so finish with a short joint-space move
            # to it (re-solved from the planned joints, with the current body pose).
            q_goal, e_goal = self.ik_item.solve(grasp, firm, q_init=self.planned_q[1], iters=200)
            if os.environ.get("ASSISTIVE_DEBUG"):
                print(f"   from-above finish: e_goal {e_goal:.3f} hits {self.ik_item.collisions(q_goal)} "
                      f"axis_err {self.ik_item.axis_error(q_goal, firm):.2f} planned axis_err {self.ik_item.axis_error(self.planned_q[1], firm):.2f}")
            if e_goal < 0.02 and not self.ik_item.collisions(q_goal):
                q0 = self.arm_cmd.copy()
                n = int(0.6 / CONTROL_DT)
                for k in range(1, n + 1):
                    self._set_right_arm((1 - k / n) * q0 + (k / n) * q_goal)
                    self._tick()
                e2 = self.move_hand(grasp, firm, duration=0.3, settle=0.3, ik=self.ik_item, posture_weight=0.15)
                _, R_now = self.ik_item.palm_pose()
                roll = float(abs(R_now[2, 1]))
        # (a pinch axis up to ~35 deg off horizontal still forms a usable pinch on a tall item;
        # beyond that the item ends up between the fingertips and falls out on the walk)
        if e2 > 0.04 or roll > 0.6:
            self.move_hand(pre, axes, duration=0.7, ik=self.ik_item)
            self.arm_home()
            self._set_posture(0.74, 0.0)
            why = f"residual {e2:.3f} m" if e2 > 0.04 else f"pinch axis {np.degrees(np.arcsin(roll)):.0f} deg off horizontal"
            return {"ok": False, "error": f"could not reach {item} from above ({why})", "pre_err": round(e1, 3)}
        res = self._close_and_lift(item, axes, lift=np.array([0, 0, 0.06]), ik=self.ik_item)
        if not res["ok"]:
            self.move_hand(pre, axes, duration=0.7, ik=self.ik_item)
            self.arm_home()
            self._set_posture(0.74, 0.0)
            return {**res, "pre_err": round(e1, 3)}
        # Keep the ITEM's own axis vertical while carrying and placing (a pitched pinch has no
        # principal hand axis to lock, so the IK gets the direction itself). The item can pivot
        # about the pinch axis during the lift, so its axis is read from its pose in the hand now
        # and refreshed before the carry and before placing.
        self.hold_mode, self.hold_dz = "pitched", float(grasp[2] - centre[2])
        self._refresh_hold_axis()
        # Up and out of the drawer with the hand orientation frozen as grasped (levelling the
        # item while still between the drawer walls knocks it out of the pinch); the carry pose
        # after the step back is where the item is brought upright.
        self.move_hand(grasp + np.array([0, 0, 0.20]), firm, duration=1.0, settle=0.1, ik=self.ik_item)
        self._step_back(STEP_BACK)
        if not self.holding_ok():
            return self._lost(item, "slipped out while stepping back")
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e2, 4), "from": "above", **res["grasp"]}

    def _choose(self, pre, grasp, candidates, postures, ik: ArmIK | None = None):
        """Try postures in order; return the hand-axis candidate with the best collision-free IK."""
        ik = ik or self.ik
        axes, best = None, None
        for height, pitch in postures:
            self._set_posture(height, pitch)
            for cand in candidates:
                q_pre, e_pre = ik.solve(pre, cand, q_init=self.arm_cmd, iters=200)
                q_g, e_g = ik.solve(grasp, cand, q_init=q_pre, iters=200)
                hits = ik.collisions(q_pre) + ik.collisions(q_g)
                # orientation residual at the grasp (the axis targets are soft in the IK; a plan
                # that only reaches the point with the hand rolled is not a plan for a pinch)
                o_err = ik.axis_error(q_g, cand)
                score = e_pre + e_g + 0.1 * len(hits) + 0.08 * o_err
                if best is None or score < best:
                    axes, best = cand, score
                    self.planned_q = (q_pre.copy(), q_g.copy())  # the arm configuration the plan assumes
            if best <= 0.03:
                break
        return axes, best

    def _pick_low(self, item: str, top: np.ndarray, centre: np.ndarray) -> dict:
        """Floor pick: deep squat, item beside the right knee, top-down grasp.

        Straight ahead, the thighs come up into the arm's path in a deep squat, so the robot
        stands with the item ~0.2 m ahead and ~0.25 m to its right and reaches down outside
        the knee. Reach is ~0.1 m above the floor: fine for a bottle, marginal for a cup.
        """
        it = self.items[item]
        base = self.wbc.base_pos
        d = top[:2] - base[:2]
        yaw = float(np.arctan2(d[1], d[0]))
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        goal = top[:2] - 0.20 * fwd[:2] + 0.25 * left[:2]
        self._raise_hand(base[2] + 0.05)
        self._drive_to(goal[0], goal[1], yaw, tol=0.025, timeout=12.0)
        # Re-detect from the new stance (closer = more accurate; the first fix from 1.3 m away is
        # 3-4 cm off, more than the pinch tolerates). The item is now almost under the camera, so
        # lean forward until it comes into view.
        for pitch in (0.6, 0.45, 0.7):
            self._set_posture(0.74, pitch)
            det = self.cam.detect()
            if it.marker_id in det:
                top = det[it.marker_id].position
                centre = top - np.array([0, 0, it.size[2] + 0.001])
                break
        # Pinch centre a little above the item's centre, with the fingers pointing down AND forward
        # (~45 deg): the fingertips reach ~11 cm past the pinch, so straight down they would hit the
        # floor; angled, they clear it, and the pinch still cages the item near the palm, which is
        # where it stays put under gravity (a pinch near the fingertips lets it slide out).
        grasp = centre + np.array([0, 0, min(0.03, 0.4 * it.size[2])])
        grasp[2] = max(grasp[2], 0.12)  # fingertips (~8 cm below the pinch at 45 deg) stay off the floor
        pre = grasp + np.array([0, 0, 0.12])
        up = np.array([0, 0, 1.0])

        def pitched(deg):  # finger direction: down, tilted forward by `deg`
            a = np.deg2rad(deg)
            return -up * np.cos(a) + fwd * np.sin(a)

        # The pinch axis (hand y) must really be horizontal, or the thumb side dips into the item
        # on the way down: weight it as strongly as the finger direction.
        candidates = [[(0, pitched(45), 0.7), (1, -fwd, 0.7)], [(0, pitched(45), 0.7), (1, fwd, 0.7)],
                      [(0, pitched(35), 0.7), (1, -fwd, 0.7)], [(0, pitched(35), 0.7), (1, left, 0.7)],
                      [(0, pitched(45), 0.7), (1, left, 0.7)], [(0, pitched(55), 0.7), (1, -fwd, 0.7)]]
        self.grip.open()
        axes, best = self._choose(pre, grasp, candidates, LOW_POSTURES, ik=self.ik_item)
        if best > 0.05:
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"{item} is too low to reach from a squat (score {best:.3f})"}
        self.move_hand(pre + np.array([0, 0, 0.08]), axes, duration=1.0, settle=0.1, ik=self.ik_item)
        self.move_hand(pre, axes, duration=0.6, ik=self.ik_item)
        e = self.move_hand(grasp, axes, duration=0.9, settle=0.4, ik=self.ik_item)
        if e > 0.04:
            self.move_hand(pre, axes, duration=0.6, ik=self.ik_item)
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"could not reach {item} on the floor (residual {e:.3f} m)"}
        res = self._close_and_lift(item, axes, lift=np.array([0, 0, 0.06]), ik=self.ik_item)
        if not res["ok"]:
            self.move_hand(pre, axes, duration=0.6, ik=self.ik_item)
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return res
        _, R_hand = self.ik_item.palm_pose()
        self.hold_vert = (0, R_hand[:, 0].copy())  # fingers (pointing down) keep their direction
        self.hold_mode, self.hold_dz = "top", grasp[2] - centre[2]
        self.move_hand(pre + np.array([0, 0, 0.1]), axes, duration=0.8, settle=0.0, ik=self.ik_item)
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e, 4), "from": "floor", **res["grasp"]}

    def _refresh_hold_axis(self) -> None:
        """For "pitched" holds: the held item's up-axis expressed in the hand frame, from the
        item's current pose in the hand (stand-in for a wrist camera / tactile estimate; the
        simulator's body orientation is used, the way `slip()` uses its position)."""
        if self.holding is None or self.hold_mode != "pitched":
            return
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._body(self.holding))
        R_item = self.data.xmat[b].reshape(3, 3)
        R_hand = self.data.site_xmat[self.ik_item.site].reshape(3, 3)
        self.hold_item_axis = R_hand.T @ R_item[:, 2]

    def _hold_axes(self, fwd: np.ndarray) -> list:
        """IK axis targets while holding: keep the item upright (strong); its yaw is free-ish.

        "pitched" grasps (from above) keep the whole hand orientation relative to the body
        instead, since no single hand axis is vertical then.
        """
        if self.hold_mode == "pitched":
            return [(self.hold_item_axis, np.array([0, 0, 1.0]), 1.0), (0, fwd, 0.15)]
        idx, vert = self.hold_vert
        axes = [(idx, vert, 1.0)]
        if self.hold_mode == "side":
            axes.append((0, fwd, 0.2))  # fingers roughly forward
        return axes

    def _raise_hand(self, z: float, toward: np.ndarray | None = None) -> None:
        """Bring the palm up in front of the chest, lined up with `toward` (a world point)."""
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0.0])
        left = np.array([-np.sin(th), np.cos(th), 0.0])
        base = self.wbc.base_pos
        lateral = -0.14
        if toward is not None:
            # No further right than the shoulder line: with the hand 25 cm out to the right the
            # robot walks it into the open cupboard door and jams against it.
            lateral = float(np.clip((np.asarray(toward) - base) @ left, -0.16, 0.05))
        palm, _ = self.ik.palm_pose()
        up = palm.copy()
        # not above chest height: with the wrist higher and close to the body the IK flips into
        # a shoulder-behind-the-back solution it cannot track out of (seen at the 0.9 m kitchen
        # counter); the approach moves lift the hand the last few cm where it has room
        up[2] = min(max(z, palm[2] + 0.05), base[2] + 0.19)
        self.move_hand(up - 0.06 * fwd, None, duration=0.6, settle=0.0)  # straight up, slightly back
        # The open hand is 22 cm long. With the fingers forward at 18 cm the fingertips reach
        # 40 cm ahead: fine when the item is further than that, otherwise they would knock it
        # over before the grasp even starts, so then the fingers point across the body (left),
        # wrist close to the chest, and the approach turns the hand forward with the wrist yaw.
        # (Fingers-left also pushes the elbow out to the right, which brushes an open cupboard
        # door closed, so it is used only when needed.)
        ahead = float((np.asarray(toward) - base) @ fwd) if toward is not None else 1.0
        if ahead > 0.42:
            tgt = base + 0.20 * fwd + lateral * left
            axes = [(0, fwd, 0.5)]
        else:
            tgt = base + 0.14 * fwd + lateral * left
            axes = [(0, left, 0.5)]
        tgt[2] = up[2]
        # (IK, not MPPI: this move is within 30 cm of the item and the open hand is 22 cm long;
        # the deterministic IK path is the one the pinch tolerance was measured with)
        e = self.move_hand(tgt, axes, duration=0.7, settle=0.1)
        if e > 0.08:
            # jammed against the torso: unfold the arm and try once more from home
            self.arm_home(0.8)
            self.move_hand(tgt, axes, duration=0.8, settle=0.1)

    def _step_back(self, dist: float) -> None:
        th = self.wbc.base_yaw
        p = self.wbc.base_pos[:2] - dist * np.array([np.cos(th), np.sin(th)])
        self._drive_to(p[0], p[1], th, tol=0.04, timeout=4.0)

    def _item_in_hand(self, item: str) -> np.ndarray:
        """Item centre in the hand (wrist-yaw) frame."""
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._body(item))
        hb = self.grip.palm_body
        R = self.data.xmat[hb].reshape(3, 3)
        return R.T @ (self.data.xpos[b] - self.data.xpos[hb])

    def slip(self) -> float:
        """How far the held item has moved in the hand since the grasp (m)."""
        if self.holding is None:
            return float("inf")
        return float(np.linalg.norm(self._item_in_hand(self.holding) - self.hold_rel))

    def holding_ok(self, tol: float = 0.045) -> bool:
        """Is the held item still in the pinch? (It is held by friction only, so it can slip out.)"""
        if self.holding is None:
            return False
        return self.slip() < tol and self.grip.contacts(self._body(self.holding)).n_pads > 0

    def _left(self) -> np.ndarray:
        th = self.wbc.base_yaw
        return np.array([-np.sin(th), np.cos(th), 0.0])

    def _set_posture(self, height: float, pitch: float) -> None:
        if abs(self.wbc.height_cmd - height) < 1e-3 and abs(self.wbc.rpy_cmd[1] - pitch) < 1e-3:
            return
        self.wbc.set_height(height)
        self.wbc.set_torso_rpy(0.0, pitch, 0.0)
        self.wait(1.2)

    def carry(self) -> None:
        """Bring the held item close to the body (pelvis-relative carry pose)."""
        self._set_posture(0.74, 0.0)
        self._refresh_hold_axis()
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0])
        left = np.array([-np.sin(th), np.cos(th), 0])
        # high and close: the item clears anything standing on a table we walk up to
        target = self.wbc.base_pos + 0.30 * fwd - 0.14 * left + np.array([0, 0, 0.18])
        self.move_hand(target, self._hold_axes(fwd), duration=1.2, ik=self.ik_item, smooth=True)

    def hand_over(self) -> dict:
        """At the person: place the held item on the side table / delivery tray."""
        if self.holding is None:
            return {"ok": False, "error": "not holding anything"}
        if not self.holding_ok():
            item = self.holding
            self._lost(item, "on the way")
            self.arm_home()
            return {"ok": False, "error": f"lost the {item} on the way"}
        it = self.items[self.holding]
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0])
        self._refresh_hold_axis()
        axes = self._hold_axes(fwd)
        # Put it down on the delivery surface: a reachable spot, as far as possible from anything
        # already standing there (seen by the head camera).
        base = self.wbc.base_pos
        left = np.array([-np.sin(th), np.cos(th), 0])
        cx, cy = self.delivery["pos"]
        hx, hy = self.delivery["half"]
        # (the carried item can block the camera, so also use where we put things before)
        others = [p.position for m, p in self.cam.detect().items() if m in self.id_to_item and self.id_to_item[m] != self.holding]
        others += self.placed
        # Candidate pinch positions on a grid in front of the robot. What must stay clear of the
        # other items is not the pinch but the hand and forearm behind it. Use the hand's actual
        # orientation now (the carry pose only weakly fixes its yaw): wrist = pinch - R*site, and
        # a capsule from 30 cm back along the forearm to 3 cm past the wrist, radius ~5 cm, plus
        # a 4 cm margin and the other item's radius.
        _, R_h = self.ik_item.palm_pose()
        site_off = (R_h @ self.model.site_pos[self.ik_item.site])[:2]
        wrist_now = self.data.xpos[self.grip.palm_body][:2]
        elbow_now = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")][:2]
        fore = wrist_now - elbow_now
        fore = fore / max(np.linalg.norm(fore), 1e-6)

        def hand_clearance(cand):
            w = cand[:2] - site_off
            a, bseg = w - 0.30 * fore, w + 0.03 * fore
            best = 1.0
            for o in others:
                ab = bseg - a
                t = float(np.clip((o[:2] - a) @ ab / max(ab @ ab, 1e-9), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(o[:2] - (a + t * ab))) - 0.05 - 0.04 - 0.035)
                best = min(best, float(np.linalg.norm(o[:2] - cand[:2])) - it.size[0] - 0.035 - 0.03)  # item vs item
            return best

        best_spot, best_score, best_clear = None, -np.inf, 0.0
        for f in (0.30, 0.33, 0.36, 0.39, 0.42):
            for l in (-0.16, -0.10, -0.04, 0.02, 0.08, 0.14, 0.20):
                cand = base + f * fwd + l * left
                m = 0.09  # keep well clear of the edge (arm tracking error is a few cm)
                cand[0] = np.clip(cand[0], cx - hx + m, cx + hx - m)
                cand[1] = np.clip(cand[1], cy - hy + m, cy + hy - m)
                clear = hand_clearance(cand)
                # Fill the table far-left first: the forearm of every later placement comes in
                # from the robot's side (near-right), so that corner must stay free the longest.
                comfort = -np.hypot(f - 0.42, l - 0.10)
                score = min(clear, 0.06) + 0.3 * comfort  # beyond 6 cm of clearance only comfort counts
                if score > best_score:
                    best_spot, best_score, best_clear = cand, score, clear
        if best_clear < 0.0:
            # nowhere free enough: put it down anyway, but say so
            print(f"hand_over: side table is crowded (best clearance {best_clear:.2f} m)")
        if os.environ.get("ASSISTIVE_DEBUG"):
            print(f"hand_over: others {[np.round(o[:2], 3).tolist() for o in others]} -> spot {np.round(best_spot[:2], 3).tolist()} clearance {best_clear:.3f}")
        spot = best_spot
        # grasp-site height when the item rests on the table (the item may have slipped a little in
        # the pinch since the grasp: measure where it actually is now)
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._body(self.holding))
        site, _ = self.ik_item.palm_pose()
        dz_now = float(site[2] - self.data.xpos[b][2])
        spot[2] = self.delivery["top"] + it.size[2] + dz_now
        # Come in from above (clear of anything already on the table), lower until the item touches
        # the table, open the fingers gradually, then leave straight up (so the fingers can't knock
        # it over) and back.
        # High enough that the held item's bottom clears the tallest thing already on the table
        # (a 20 cm bottle), then straight down.
        site, _ = self.ik_item.palm_pose()
        high = spot + np.array([0, 0, 0.26])
        self.move_hand(np.array([site[0], site[1], max(site[2], high[2])]), axes, duration=0.7, settle=0.0, ik=self.ik_item,
                       smooth=True)
        self.move_hand(high, axes, duration=1.0, ik=self.ik_item, smooth=True)
        self.move_hand(spot + np.array([0, 0, 0.03]), axes, duration=1.0, settle=0.2, ik=self.ik_item)
        item = self.holding
        touched = self._lower_until_contact(item, spot, axes)
        self._release(hold_arm=True)
        self._withdraw(axes)
        self.arm_home()
        self.placed.append(self.data.xpos[b].copy())  # where it actually ended up (it can slide a few cm)
        upright = float(self.data.xmat[b].reshape(3, 3)[2, 2])
        return {"ok": True, "handed_over": item, "placed_on": "delivery_tray" if self.world is not None else "side_table",
                "upright": round(upright, 3), "touched_table": touched}

    def _item_touching(self, item: str, min_force: float = 0.3) -> bool:
        """Is `item` in contact with anything other than the hand (e.g. the table it is set on)?"""
        m, d = self.model, self.data
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self._body(item))
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b not in (b1, b2):
                continue
            other = b2 if b1 == b else b1
            if other in self.grip.hand_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) > min_force:
                return True
        return False

    def _lower_until_contact(self, item: str, spot: np.ndarray, axes, max_drop: float = 0.06,
                             speed: float = 0.06) -> bool:
        """Lower the held item slowly until it rests on the surface (contact felt), then stop.

        Returns True if contact was made within `max_drop` below the nominal spot height.
        """
        site, _ = self.ik_item.palm_pose()
        q = self.arm_cmd.copy()
        z0 = site[2]
        n = int(max_drop / speed / CONTROL_DT)
        # The fingers hang below the pinch, so the item touches down while they are still ~2 cm
        # above the surface: let the IK bring them that close (normally it keeps 3 cm).
        safe = self.ik_item.safe_dist
        self.ik_item.safe_dist = 0.004
        try:
            for k in range(1, n + 1):
                goal = spot.copy()
                goal[2] = z0 - speed * k * CONTROL_DT
                q, _ = self.ik_item.solve(goal, axes, q_init=q, iters=15)
                self._set_right_arm(q)
                self._tick()
                if self._item_touching(item):
                    # a few more mm so the item is really seated, not balanced on the pads
                    for _ in range(5):
                        self._set_right_arm(q)
                        self._tick()
                    return True
            return False
        finally:
            self.ik_item.safe_dist = safe

    def _withdraw(self, axes) -> None:
        """Take the open hand off a released item without touching it.

        The item stands at +y of the hand, 2-3 cm from the finger sides, and in front of the
        swung-back thumb. The arm springs up and the base sways a little when the load goes, so
        first ease the hand towards -y (fingers move away from the item), then back straight out
        along the fingers (-x), and only then swing the thumb fully back and lift.
        """
        site, R_hand = self.ik_item.palm_pose()
        side = -R_hand[:, 1]
        side[2] = 0.0
        side /= max(np.linalg.norm(side), 1e-6)
        out = -R_hand[:, 0]
        out[2] = 0.0
        out /= max(np.linalg.norm(out), 1e-6)
        self.move_hand(site + 0.035 * side, axes, duration=0.5, settle=0.1, ik=self.ik_item)
        # Straight up next (the fingers slide up beside the item, the thumb is behind it): going
        # back at table height would sweep the wrist through anything standing next to it.
        self.move_hand(site + 0.035 * side + np.array([0, 0, 0.22]), axes, duration=0.9, settle=0.1, ik=self.ik_item,
                       smooth=True)
        self.grip.open()
        self.move_hand(site + 0.035 * side + 0.16 * out + np.array([0, 0, 0.22]), axes, duration=0.7, ik=self.ik_item,
                       smooth=True)

    def _release(self, hold_arm: bool = True, seconds: float = 0.5) -> None:
        """Open the fingers gradually (symmetric release: no flick) and forget the item.

        Opens to Q_RELEASE (fingers straight, thumb uncurled but not swung back): the thumb's
        swing-back arc runs through the item, so the full OPEN pose is only commanded once the
        hand has backed away.
        """
        # While the pinch is on, the finger torques load the (softly PD-controlled) wrist; as
        # they come off, the arm springs back ~3 cm and the finger roots can clip the item. So
        # the release is closed-loop: the grasp site is servoed to its current world position
        # (plus a 2 cm ease towards -y, i.e. the fingers move away from the item) while the
        # fingers open.
        site0, R_hand = self.ik_item.palm_pose()
        side = -R_hand[:, 1]
        side[2] = 0.0
        side /= max(np.linalg.norm(side), 1e-6)
        axes = self._hold_axes(R_hand[:, 0]) if hold_arm else None
        q = self.arm_cmd.copy()
        n = int(seconds / CONTROL_DT)
        for k in range(1, n + 1):
            self.grip.blend(1.0 - k / n, open_pose=Q_RELEASE)
            if hold_arm:
                q, _ = self.ik_item.solve(site0 + 0.02 * side * (k / n), axes, q_init=q, iters=15)
                self._set_right_arm(q)
            self._tick()
        self.holding = None
        self._set_held(None)

    def follow(self, targets: list[np.ndarray], axes_list: list, dt: float = CONTROL_DT, max_err: float = 0.05,
               keep_reach: float | None = None) -> int:
        """Track a sequence of palm targets (one per control tick). Returns how many were reached.

        keep_reach: if set, walk backwards whenever the target comes closer than this to the
        pelvis (along the heading), e.g. while pulling a door open towards the body.
        """
        q = self.arm_cmd.copy()
        bad = 0
        v_back = 0.0
        for i, (p, axes) in enumerate(zip(targets, axes_list)):
            if keep_reach is not None:
                th = self.wbc.base_yaw
                fwd = (p - self.wbc.base_pos)[:2] @ np.array([np.cos(th), np.sin(th)])
                want = -BACKOFF_SPEED if fwd < keep_reach else 0.0
                v_back += np.clip(want - v_back, -0.5 * dt, 0.5 * dt)  # 0.5 m/s^2 ramp
                self.wbc.set_velocity(v_back if abs(v_back) > 0.05 else 0.0, 0.0, 0.0)
            q, _ = self.ik.solve(p, axes, q_init=q, iters=25)
            self._set_right_arm(q)
            self._tick()
            err = float(np.linalg.norm(self.ik.palm_pose()[0] - p))
            bad = bad + 1 if err > max_err else 0
            if bad >= 10:  # persistently off the path (out of reach or blocked)
                self.wbc.set_velocity(0, 0, 0)
                return i
        self.wbc.set_velocity(0, 0, 0)
        return len(targets)

    def open_articulated(
        self,
        model: ArticulationModel,
        anchor_marker: int,
        weld_name: str,
        fraction: float = 1.0,
        seconds: float = 5.0,
    ) -> dict:
        """Open a revolute cupboard door or prismatic drawer using the learned ArticulationModel."""
        det = self.cam.detect()
        if anchor_marker not in det:
            return {"ok": False, "error": f"anchor marker {anchor_marker} not visible"}
        A = det[anchor_marker]
        Ra, pa = A.rotation, A.position
        handle = Ra @ np.array(model.grasp_start) + pa
        normal = Ra @ np.array(model.part_normal_start)  # outward normal (towards the robot)
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
        into = -normal
        up = np.array([0, 0, 1.0])

        pre = handle - 0.12 * into
        # Raise the hand first: planning from the hand-by-the-hip pose traps the IK behind the
        # thigh/torso now that the arm avoids the body.
        self._raise_hand(handle[2], toward=handle)
        candidates = [[(0, into, 0.6), (1, up, 0.4)], [(0, into, 0.6), (1, -up, 0.4)],
                      [(0, into, 0.6), (2, up, 0.4)], [(0, into, 0.6), (2, -up, 0.4)]]
        axes, best = None, None
        for cand in candidates:
            q1, e1 = self.ik.solve(pre, cand, q_init=self.arm_cmd, iters=200)
            q2, e2 = self.ik.solve(handle, cand, q_init=q1, iters=200)
            score = e1 + e2 + 0.1 * len(self.ik.collisions(q1) + self.ik.collisions(q2))
            if best is None or score < best:
                axes, best = cand, score
        if best > 0.05:
            self.arm_home()
            return {"ok": False, "error": f"handle not reachable from here (score {best:.3f})"}

        self.move_hand(pre, axes, duration=1.2)
        e = self.move_hand(handle, axes, duration=0.8)
        if e > 0.04:
            self.arm_home()
            return {"ok": False, "error": f"could not reach handle (residual {e:.3f} m)"}
        # Position-only grip on the handle: the door/drawer moves while the wrist keeps a
        # comfortable orientation (like a loose grip around a bar handle). The fingers close
        # around the bar as well, which also keeps the open hand from catching the door edge
        # when it lets go (hand/door and hand/drawer pairs are collision-excluded, so this is
        # visual + compactness, not the holding force).
        q_hold = self.arm_cmd.copy()
        for k in range(1, 16):
            self.grip.blend(k / 15)
            self._set_right_arm(q_hold)
            self._tick()
        self._weld(weld_name, True, torquescale=0.0)

        # Learned arc (revolute) or line (prismatic), mapped to the world
        c = Ra @ np.array(model.axis_point) + pa
        a = Ra @ np.array(model.axis_dir)
        n = int(seconds / CONTROL_DT)
        targets, axes_list = [], []
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)
            amt = fraction * model.amount * s
            if model.kind == "revolute":
                R = rot_axis_angle(a, amt)
                targets.append(R @ (handle - c) + c)
            else:
                targets.append(handle + amt * a)
            # keep only a weak 'palm side up' preference; the fingers may turn freely
            axes_list.append([(axes[1][0], axes[1][1], 0.2)])
        reached = self.follow(targets, axes_list, keep_reach=DOOR_KEEP_REACH)
        self.wait(0.5)
        self._weld(weld_name, False)

        # Let go and back off along the current finger direction (fingers = palm x-axis), then
        # open the hand again once it is clear of the door.
        p, R_palm = self.ik.palm_pose()
        self.move_hand(p - 0.12 * R_palm[:, 0], None, duration=0.7, settle=0.1)
        self.grip.open()
        self.arm_home()
        if model.kind == "revolute":
            s = self.stations.get(self.station)
            if s is not None:
                self._drive_to(s.x, s.y, s.yaw, tol=0.03, timeout=10.0)
        opened = fraction * model.amount * (reached / n)
        return {
            "ok": reached > 0.6 * n,
            "kind": model.kind,
            "opened": round(opened, 3),
            "of_demo": round(model.amount, 3),
            "opened_rad": round(opened, 3),
            "of_demo_rad": round(model.amount, 3),
        }

    def open_cupboard(self, model: ArticulationModel, fraction: float = 1.0, seconds: float = 5.0) -> dict:
        """Open the cupboard the way the carer did, using the learned articulation model."""
        return self.open_articulated(model, ANCHOR_MARKER, "grasp_door", fraction=fraction, seconds=seconds)

    def put_down(self, height_offset: float = 0.0) -> dict:
        """Release the held item where the hand is (e.g. back on a surface)."""
        if self.holding is None:
            return {"ok": False, "error": "not holding anything"}
        item = self.holding
        self._release()
        self._withdraw(None)
        self.arm_home()
        return {"ok": True, "released": item}
