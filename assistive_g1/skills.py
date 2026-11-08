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
from ik import ArmIK
from nav import Navigator
from perception import HeadCamera, MarkerPose
from articulation import ArticulationModel, rot_axis_angle
from scene import ITEMS, STATIONS, SIDE_TABLE, DOOR_MARKER, ANCHOR_MARKER, DOOR_INNER_MARKER, build_scene

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
MAX_ARM_RATE = 3.0  # rad/s, cap on how fast any arm joint target may move (no more jerks)
HOLD_ELBOW_GAIN = 0.3  # IK pulls the elbow down while the right hand holds an item (not while reaching)
ELBOW_W = 0.3  # m of reach error a solution may trade per m of elbow raised above the shoulder band
SETTLE_MAX = 1.0  # s, extra time move_hand may track a target until the palm arrives
# Top-down grasps raise the elbow out to the side ("chicken wing"), as a person's does: allow it
# higher than for side grasps (ik.ELBOW_MAX). The comfort joint ranges still rule out overhead poses.
ABOVE_ELBOW_MAX = float(os.environ.get("G1_ABOVE_ELBOW_MAX", "0.20"))
# How far above the grasp the hand comes in over an open drawer (m), lowest first. Jack's drawer
# front is only ~3.5 cm above the bottle's cap, so 30 cm (the old value) was far more than needed
# and put the hand near shoulder height, i.e. the elbow above the head.
ABOVE_CLEARS = (0.12, 0.18, 0.26)
ABOVE_STANDOFF = 0.45  # m, pelvis behind an item picked from above (body ~0.28 m clear of an open drawer's handle)
DEBUG = bool(os.environ.get("ASSISTIVE_DEBUG"))
GRASP_TOL = 0.05  # m, palm-to-grasp residual accepted before the grasp weld (it pulls in the rest)
# Self-localization (state_estimation.py: IMU + leg odometry + LiDAR scan matching, RGB-D head
# camera) in Jack's house. G1_LOCALIZE=0 falls back to the simulator's root pose.
PUSH_FACE = 0.045  # m, drawer front behind its bar handle (Jack's HANDLE_STANDOFF)
FINGER_REACH = 0.11  # m, Dex3 fingertips past the palm site along the fingers
LOCALIZE = os.environ.get("G1_LOCALIZE", "1") != "0"


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class AssistiveRobot:
    def __init__(self, placement=None, on_frame: Callable[["AssistiveRobot"], None] | None = None, world=None,
                 localize: bool | None = None):
        """world: None = our one-room scene (scene.build_scene); or a house.G1House (Jack's house),
        which supplies the model, stations, items, delivery surface, extra markers and a planner."""
        self.world = world
        if world is None:
            self.model, self.data = build_scene(placement)
            self.stations, self.items = STATIONS, ITEMS
            self.item_body = {n: n for n in ITEMS}
            self.delivery = SIDE_TABLE
            start, graspable, cam_kw = (0.0, 0.0, 0.0), ("cupboard_door",), {}
        else:
            self.model, self.data = world.model, world.data
            self.stations, self.items = world.stations, world.items
            self.item_body, self.delivery = world.item_body, world.delivery
            h = world.stations["home"]
            start, graspable = (h.x, h.y, h.yaw), tuple(world.graspable)
            cam_kw = dict(extra_sizes=world.marker_sizes, labels=world.marker_labels)
        self.marker_to_item = {it.marker_id: n for n, it in self.items.items()}
        self.wbc = G1WBC(self.model, self.data)
        self.ik = ArmIK(self.model, self.data, "right", graspable=graspable)
        self.left_ik = ArmIK(self.model, self.data, "left", graspable=graspable)
        self.left_cmd = ARM_HOME[:7].copy()
        self.open_state: dict | None = None  # what open_articulated() left open (for close_opened)
        self.cam = HeadCamera(self.model, self.data, **cam_kw)
        self.nav = Navigator()
        self.on_frame = on_frame  # called every CONTROL_DT, e.g. to record video
        self.holding: str | None = None
        self.hold_vert = (2, np.array([0, 0, 1.0]))  # palm axis kept fixed while holding
        self.hold_mode, self.hold_dz = "side", 0.0  # grasp type and palm height above item centre
        self.placed: list[np.ndarray] = []  # where this robot has set things down (side table)
        self.station = "home"
        self.arm_cmd = ARM_HOME[7:].copy()
        self.last_seen: dict[int, MarkerPose] = {}
        self._grip_dist = 0.0  # palm-to-body distance when the current grasp weld was made
        self._best_q = None
        self.wbc.place(*start)
        self.estimator = None
        if (LOCALIZE if localize is None else localize) and world is not None:
            self._attach_estimator(world)
        self.wait(1.0)

    def _attach_estimator(self, world) -> None:
        """From here on the robot knows where it is only from its own sensors: the walking
        controller, navigation, the arm IK and the head camera all use the estimate."""
        from state_estimation import StateEstimator
        est = StateEstimator(world, self.model, self.data, self.wbc)
        self.estimator = est
        self.wbc.estimator = est
        self.ik.root_pose = self.left_ik.root_pose = lambda: (est.pos, est.quat)
        self.cam.pose_source = est.camera_pose

    # ------------------------------------------------------------------ basics
    def _tick(self) -> None:
        self.wbc.step(int(round(CONTROL_DT / SIM_DT)))
        if self.estimator is not None:
            self.estimator.tick(CONTROL_DT)
        if self.on_frame is not None:
            self.on_frame(self)

    def wait(self, seconds: float) -> None:
        for _ in range(int(seconds / CONTROL_DT)):
            self._tick()

    def _set_right_arm(self, q: np.ndarray) -> np.ndarray:
        """Set the right arm's joint targets, rate-limited to MAX_ARM_RATE (returns what was set).
        Callers keep seeding the IK with its own previous solution, not with this: the motors
        trail the IK smoothly instead of the IK being dragged through in-between poses."""
        step = MAX_ARM_RATE * CONTROL_DT
        q = self.arm_cmd + np.clip(np.asarray(q) - self.arm_cmd, -step, step)
        self.arm_cmd = q
        full = ARM_HOME.copy()
        full[:7] = self.left_cmd
        full[7:] = q
        self.wbc.set_arm_targets(full)
        return q

    def _set_left_arm(self, q: np.ndarray) -> np.ndarray:
        step = MAX_ARM_RATE * CONTROL_DT
        q = self.left_cmd + np.clip(np.asarray(q) - self.left_cmd, -step, step)
        self.left_cmd = q
        full = ARM_HOME.copy()
        full[:7] = q
        full[7:] = self.arm_cmd
        self.wbc.set_arm_targets(full)
        return q

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
        """Walk to a named station (self.stations).

        In the house: plan on Jack's occupancy grid, walk the waypoints without stopping (pushing
        unlocked doors open on the way), then the fine approach to the station pose. In the
        one-room scene: through the hub.
        """
        if station not in self.stations:
            return {"ok": False, "error": f"unknown station {station}", "stations": list(self.stations)}
        s = self.stations[station]
        p = self.wbc.base_pos
        if self.world is not None:
            if station != self.station:
                self._step_back(0.25)  # off the furniture first: stations sit inside the grid's inflation
            path = self.world.plan_path(self.wbc.base_pos[:2], (s.x, s.y))
            if path is None:
                return {"ok": False, "error": f"no path to the {station}"}
            for wx, wy in path[1:-1]:
                self._drive_to(wx, wy, None, tol=0.20, pass_through=True)
                if self.wbc.fallen():
                    return {"ok": False, "error": "fell over"}
        elif np.hypot(p[0] - HUB[0], p[1] - HUB[1]) > 0.3 and station != self.station:
            self._drive_to(*HUB, None, tol=0.25, pass_through=True)
            if self.wbc.fallen():
                return {"ok": False, "error": "fell over"}
        ok = self._drive_to(s.x, s.y, s.yaw)
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
        items = {self.marker_to_item[m]: p.position.round(3).tolist() for m, p in det.items() if m in self.marker_to_item}
        extra = {}
        if DOOR_MARKER in det:
            extra["cupboard_door"] = det[DOOR_MARKER].position.round(3).tolist()
        if ANCHOR_MARKER in det:
            extra["cupboard_anchor"] = det[ANCHOR_MARKER].position.round(3).tolist()
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
                  q_ref: np.ndarray | None = None) -> float:
        """Interpolate the palm to `target` (world) and track it with IK. Returns final error (m).

        The joint targets are rate-limited (MAX_ARM_RATE), so after the interpolation it keeps
        tracking the target (up to SETTLE_MAX s) until the palm has actually arrived.
        q_ref: planned arm configuration for this move (from _plan): the IK's null space is pulled
        towards it instead of the home pose, so the arm stays on the branch that was checked.
        """
        start, _ = self.ik.palm_pose()
        n = max(1, int(duration / CONTROL_DT))
        q = self.arm_cmd.copy()
        k_min = n + int(settle / CONTROL_DT)
        for k in range(1, k_min + int(SETTLE_MAX / CONTROL_DT) + 1):
            a = min(1.0, k / n)
            a = a * a * (3 - 2 * a)  # smoothstep
            goal = (1 - a) * start + a * np.asarray(target)
            q, _ = self.ik.solve(goal, axes, q_init=q, iters=15, elbow_gain=self._elbow_gain(), q_ref=q_ref)
            self._set_right_arm(q)
            self._tick()
            if k >= k_min and np.linalg.norm(self.ik.palm_pose()[0] - target) < 0.01:
                break
        p, _ = self.ik.palm_pose()
        return float(np.linalg.norm(p - target))

    def _elbow_gain(self) -> float | None:
        """Elbow-down pull for the right arm's IK: on while holding something (the carry with a
        top grasp otherwise lifts the elbow by the head and tilts the item), off while reaching
        (it made the grasps worse). None = ik.py's default."""
        return HOLD_ELBOW_GAIN if self.holding is not None else None

    def _swing(self, side: str, q_goal: np.ndarray, min_duration: float) -> None:
        """Joint-space move of one arm to q_goal (smoothstep), slow enough for the rate limit,
        then a short settle."""
        q0 = (self.arm_cmd if side == "right" else self.left_cmd).copy()
        setter = self._set_right_arm if side == "right" else self._set_left_arm
        dur = max(min_duration, 1.6 * float(np.abs(np.asarray(q_goal) - q0).max()) / MAX_ARM_RATE)
        n = int(dur / CONTROL_DT)
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            setter((1 - a) * q0 + a * np.asarray(q_goal))
            self._tick()
        self.wait(0.3)

    def arm_home(self, duration: float = 1.0) -> None:
        self._swing("right", ARM_HOME[7:], duration)

    def _item_collisions(self, item: str, with_robot: bool) -> None:
        """Held items collide with furniture and other items but not with the robot (the arm
        overlaps what it holds); robot contact is restored once the arm has retracted."""
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.item_body[item])
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == b and (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                self.model.geom_contype[g] = 3 if with_robot else 2
                self.model.geom_conaffinity[g] = 3 if with_robot else 2

    def _weld(self, name: str, on: bool, torquescale: float = 1.0, ik: ArmIK | None = None) -> None:
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
            palm, _ = (ik or self.ik).palm_pose()
            q1_inv = np.zeros(4)
            mujoco.mju_negQuat(q1_inv, q1)
            rel_q = np.zeros(4)
            mujoco.mju_mulQuat(rel_q, q1_inv, q2)
            self.model.eq_data[eq, 0:3] = R2.T @ (palm - p2)
            self.model.eq_data[eq, 3:6] = R1.T @ (palm - p1)
            self.model.eq_data[eq, 6:10] = rel_q
            self.model.eq_data[eq, 10] = torquescale
            self.model.eq_data[eq, 11] = 1.0
            if ik is None and self.holding is None:  # a right-hand grasp of an item (or part)
                self._grip_dist = float(np.linalg.norm(p2 - palm))
        self.data.eq_active[eq] = 1 if on else 0

    # ------------------------------------------------------------------ manipulation
    def pick(self, item: str, from_above: bool = False) -> dict:
        """Side-grasp an item seen by the head camera at the current station.

        from_above: top-down grasp at standing height, for items inside an open drawer (the
        drawer front is in the way of a side grasp).
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
        grasp = centre + np.array([0, 0, 0.3 * it.size[2]])

        base = self.wbc.base_pos
        # Raise the hand to chest height, close to the body, before anything else so it can't
        # catch under a table edge while lifting or stepping in.
        self._raise_hand(grasp[2] + 0.06, toward=grasp)
        # Line up so the item is straight ahead of the right shoulder at STANDOFF, keeping the
        # heading (facing the furniture). The hand then goes in and comes out perpendicular to
        # the furniture: a diagonal path drags the item along a cupboard's side wall.
        th = self.wbc.base_yaw
        fwd2 = np.array([np.cos(th), np.sin(th)])
        left2 = np.array([-np.sin(th), np.cos(th)])
        goal = grasp[:2] - STANDOFF * fwd2 + GRASP_LATERAL * left2  # pelvis left of the item
        if np.linalg.norm(goal - base[:2]) > 0.03:
            self._drive_to(goal[0], goal[1], th, tol=0.025, timeout=8.0)
            base = self.wbc.base_pos
        th = self.wbc.base_yaw
        approach = np.array([np.cos(th), np.sin(th), 0.0])
        reach = float((grasp - base)[:2] @ approach[:2])
        # Pre-grasp back along the approach, but never closer than ~0.27 m to the pelvis:
        # nearer than that the arm would have to fold into the torso.
        back = float(np.clip(reach - 0.27, 0.06, 0.14))
        pre = grasp - back * approach + np.array([0, 0, 0.04])

        # Choose a posture (lean/crouch via the GR00T torso and height commands) and a wrist
        # roll that reach both poses without touching furniture or the torso.
        up = np.array([0, 0, 1.0])
        candidates = [
            [(0, approach, 0.6), (2, up, 0.4)],
            [(0, approach, 0.6), (1, up, 0.4)],
            [(0, approach, 0.6), (1, -up, 0.4)],
            [(0, approach, 0.6), (2, -up, 0.4)],
        ]
        axes, best = self._choose(pre, grasp, candidates, POSTURES)
        if best > 0.05:
            self._set_posture(0.74, 0.0)
            return {"ok": False, "error": f"no collision-free grasp for {item} from here (score {best:.3f})"}

        # Come in along the approach line: far pre-grasp (outside any cupboard), pre-grasp, grasp,
        # all on the planned configurations (joint-space swing to the far pre-grasp).
        q_pre, q_g = self._best_q
        far = grasp - (back + 0.10) * approach + np.array([0, 0, 0.04])
        q_far, _ = self.ik.solve(far, axes, q_init=q_pre, iters=150, q_ref=q_pre)
        if self.ik.elbow_too_high(q_far):
            q_far = q_pre
        self._reach("right", q_far, far, pre, q_pre, axes, duration=0.7)
        e1 = float(np.linalg.norm(self.ik.palm_pose()[0] - pre))
        e2 = self.move_hand(grasp, axes, duration=0.9, q_ref=q_g)
        if e2 > 0.04:
            self.move_hand(pre, axes, duration=0.8, q_ref=q_pre)
            self.arm_home()
            self._set_posture(0.74, 0.0)
            return {"ok": False, "error": f"could not reach {item} (residual {e2:.3f} m)", "pre_err": round(e1, 3)}
        self._weld(f"grasp_{self.item_body[item]}", True)
        self._item_collisions(item, False)
        self.holding = item
        # The weld fixes the item to the hand, so the item stays upright iff this palm axis
        # keeps the world direction it had at the grasp. Record it as measured, not as asked.
        _, R_palm = self.ik.palm_pose()
        idx = axes[1][0] if len(axes) > 1 else 2
        self.hold_vert = (idx, R_palm[:, idx].copy())
        self.hold_mode, self.hold_dz = "side", grasp[2] - centre[2]
        self.move_hand(grasp + np.array([0, 0, 0.04]), axes, duration=0.6, settle=0.0, q_ref=q_g)
        self.move_hand(grasp - (back + 0.10) * approach + np.array([0, 0, 0.05]), axes, duration=0.9, settle=0.0,
                       q_ref=q_far)
        # Step back with the arm frozen: pulls the item straight out of a cupboard / off the
        # table before the carry motion swings it sideways (it would snag on the frame).
        self._step_back(STEP_BACK)
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e2, 4)}

    def _plan(self, ik: ArmIK, q_now: np.ndarray, pre: np.ndarray, target: np.ndarray, candidates,
              multi_start: bool = True, elbow_max: float | None = None):
        """Plan a reach: joint configurations for the pre-pose and the target, for the best of the
        hand-axis candidates. Returns (axes, raw error score, q_pre, q_target), score inf if none.

        Seeds: the current arm, home, and a solution for the target alone (the IK has local
        minima). Plans that lift the elbow above ELBOW_MAX are rejected outright, the rest are
        ranked by error + collisions + a raised-elbow cost. The motion that runs afterwards
        (_reach) follows these configurations, so what was checked here is what happens.
        """
        best = (None, np.inf, None, None)
        rank = np.inf
        for cand in candidates:
            seeds = [q_now]
            if multi_start:
                seeds += [ik.home, ik.solve(target, None, q_init=ik.home, iters=100)[0]]
            for q0 in seeds:
                q_pre, e_pre = ik.solve(pre, cand, q_init=q0, iters=200)
                q_t, e_t = ik.solve(target, cand, q_init=q_pre, iters=200, q_ref=q_pre)
                if ik.elbow_too_high(q_pre, q_t, limit=elbow_max):
                    continue
                raw = e_pre + e_t + 0.1 * len(ik.collisions(q_pre) + ik.collisions(q_t))
                score = raw + ELBOW_W * ik.elbow_cost(q_pre, q_t)
                if score < rank:
                    rank, best = score, (cand, raw, q_pre, q_t)
        return best

    def _path_clear(self, ik: ArmIK, q0: np.ndarray, q1: np.ndarray, n: int = 8) -> bool:
        """No arm collisions along the straight joint-space path q0 -> q1 (sampled)."""
        return all(not ik.collisions((1 - a) * q0 + a * q1, margin=0.0) for a in np.linspace(0, 1, n)[1:])

    def _reach(self, side: str, q_pre: np.ndarray, pre: np.ndarray, target: np.ndarray, q_t: np.ndarray,
               axes, duration: float = 0.9) -> float:
        """Execute a plan: joint-space swing to q_pre (Cartesian with q_ref if the swing would hit
        something), then straight in to the target, staying on the plan's branch. Returns the error."""
        ik, cmd = (self.ik, self.arm_cmd) if side == "right" else (self.left_ik, self.left_cmd)
        mover = self.move_hand if side == "right" else self.move_left_hand
        if self._path_clear(ik, cmd.copy(), q_pre):
            self._swing(side, q_pre, 1.2)
        else:
            mover(pre, axes, duration=1.2, settle=0.1, q_ref=q_pre)
        e = mover(target, axes, duration=duration, q_ref=q_t)
        if e > 0.02:
            # The tracked approach stalls short of the target (collision rows near furniture, or a
            # local minimum). The plan's own target configuration was collision-free: finish with a
            # short joint-space move to it, re-solved for the current body pose.
            q_fin, e_fin = ik.solve(target, axes, q_init=q_t, iters=200, q_ref=q_t)
            if e_fin < e and not ik.collisions(q_fin):
                self._swing(side, q_fin, 0.6)
                e = float(np.linalg.norm(ik.palm_pose()[0] - target))
            if DEBUG:
                print(f"   reach[{side}]: tracked stalled, joint-space finish e_fin {e_fin:.3f} -> residual {e:.3f}")
        return e

    def _choose(self, pre, grasp, candidates, postures, multi_start: bool = True, elbow_max: float | None = None):
        """Try postures in order; return (axes, score) of the best plan (see _plan). The plan's
        configurations are left in self._best_q for _reach."""
        axes, best, posture = None, np.inf, None
        self._best_q = None
        for height, pitch in postures:
            self._set_posture(height, pitch)
            cand, raw, q_pre, q_g = self._plan(self.ik, self.arm_cmd.copy(), pre, grasp, candidates, multi_start,
                                               elbow_max)
            if cand is not None and raw < best:
                axes, best, posture = cand, raw, (height, pitch)
                self._best_q = (q_pre, q_g)
            if best <= 0.03:
                break
        if posture is not None:
            self._set_posture(*posture)  # back to the posture the plan was made in
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
        # re-detect from the new stance (closer = more accurate)
        det = self.cam.detect()
        if it.marker_id in det:
            top = det[it.marker_id].position
            centre = top - np.array([0, 0, it.size[2] + 0.001])
        grasp = top - np.array([0, 0, min(0.03, it.size[2])])  # palm just below the top
        pre = grasp + np.array([0, 0, 0.12])
        up = np.array([0, 0, 1.0])
        candidates = [[(0, -up, 0.6), (1, fwd, 0.3)], [(0, -up, 0.6), (1, left, 0.3)], [(0, -up, 0.6)]]
        axes, best = self._choose(pre, grasp, candidates, LOW_POSTURES)
        if best > 0.05:
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"{item} is too low to reach from a squat (score {best:.3f})"}
        q_pre, q_g = self._best_q
        e = self._reach("right", q_pre, pre, grasp, q_g, axes, duration=0.9)
        if e > 0.04:
            self.move_hand(pre, axes, duration=0.6)
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"could not reach {item} on the floor (residual {e:.3f} m)"}
        self._weld(f"grasp_{self.item_body[item]}", True)
        self._item_collisions(item, False)
        self.holding = item
        _, R_palm = self.ik.palm_pose()
        # The item stood upright: keep its axis (world up, as seen from the palm at the grasp)
        # vertical. Keeping the finger direction instead lets a pitched hand roll about it,
        # which tilts the item (it then fell off the tray at the hand-over).
        self.hold_vert = (R_palm.T @ np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))
        self.hold_mode, self.hold_dz = "top", grasp[2] - centre[2]
        self.move_hand(pre + np.array([0, 0, 0.1]), axes, duration=0.8, settle=0.0, q_ref=q_pre)
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e, 4), "from": "floor"}

    def _pick_from_above(self, item: str, top: np.ndarray, centre: np.ndarray) -> dict:
        """Top-down grasp at standing height, e.g. an item in an open drawer.

        Keeps the heading (facing the furniture), lines the item up with the right shoulder,
        brings the hand in high (over the drawer front), descends onto the item, lifts it
        straight up until it clears the front, then steps back.
        """
        it = self.items[item]
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0.0])
        left = np.array([-np.sin(th), np.cos(th), 0.0])
        self._raise_hand(top[2] + 0.25, toward=top)
        up = np.array([0, 0, 1.0])

        def pitched(deg):  # fingers down, tilted forward (away from us) by `deg`
            a = np.deg2rad(deg)
            return -up * np.cos(a) + fwd * np.sin(a)
        # Pitched first (from the grasp-physical branch): with the fingers tilted forward the
        # forearm comes over the drawer front lower, so the elbow stays down; straight down last.
        candidates = [[(0, pitched(35), 0.6), (1, -left, 0.3)], [(0, pitched(35), 0.6), (1, left, 0.3)],
                      [(0, pitched(20), 0.6), (1, -left, 0.3)], [(0, pitched(50), 0.6), (1, -left, 0.3)],
                      [(0, -up, 0.6), (1, left, 0.3)], [(0, -up, 0.6), (1, fwd, 0.3)], [(0, -up, 0.6)]]
        # The item's distance varies with how far the drawer actually opened: try a few stances
        for standoff in (ABOVE_STANDOFF, ABOVE_STANDOFF + 0.05, ABOVE_STANDOFF - 0.04):
            goal = top[:2] - standoff * fwd[:2] + GRASP_LATERAL * left[:2]
            if np.linalg.norm(goal - self.wbc.base_pos[:2]) > 0.03:
                self._drive_to(goal[0], goal[1], th, tol=0.025, timeout=8.0)
            det = self.cam.detect()  # closer view
            if it.marker_id in det:
                top = det[it.marker_id].position
                centre = top - np.array([0, 0, it.size[2] + 0.001])
            grasp = top - np.array([0, 0, min(0.03, it.size[2])])
            for clear in ABOVE_CLEARS:  # lowest first: the lower the hand, the lower the elbow
                pre = grasp + np.array([0, 0, clear])
                axes, best = self._choose(pre, grasp, candidates, [(0.74, 0.3), (0.74, 0.15), (0.70, 0.45), (0.64, 0.5)],
                                          multi_start=True, elbow_max=ABOVE_ELBOW_MAX)
                if best <= 0.05:
                    break
            if best <= 0.05:
                break
        if best > 0.05:
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"no collision-free top grasp for {item} (score {best:.3f})"}
        # Swing the arm to the chosen pre-grasp configuration in joint space (a Cartesian
        # interpolation from the raised hand would drag the IK back into its local minimum).
        q_pre, q_g = self._best_q
        e = self._reach("right", q_pre, pre, grasp, q_g, axes, duration=1.0)
        if DEBUG:
            print(f"   above: plan {best:.3f} clear {clear} posture {self.wbc.height_cmd:.2f}/{self.wbc.rpy_cmd[1]:.2f} "
                  f"descent residual {e:.3f} hits {self.ik.collisions(self.arm_cmd)}")
        if e > GRASP_TOL:
            self.move_hand(pre, axes, duration=0.8, q_ref=q_pre)
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"could not reach {item} from above (residual {e:.3f} m)"}
        self._weld(f"grasp_{self.item_body[item]}", True)
        self._item_collisions(item, False)
        self.holding = item
        _, R_palm = self.ik.palm_pose()
        # The item stood upright: keep its axis (world up, as seen from the palm at the grasp)
        # vertical. Keeping the finger direction instead lets a pitched hand roll about it,
        # which tilts the item (it then fell off the tray at the hand-over).
        self.hold_vert = (R_palm.T @ np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))
        self.hold_mode, self.hold_dz = "top", grasp[2] - centre[2]
        # Lift until the item's base clears the drawer front (it stands between us and the item),
        # however low the hand came in
        lift = max(clear, it.size[2] * 2 + 0.06) + 0.05
        self.move_hand(grasp + np.array([0, 0, lift]), axes, duration=1.2, settle=0.1, q_ref=q_pre)
        self._step_back(STEP_BACK)
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e, 4), "from": "above"}

    def _hold_axes(self, fwd: np.ndarray) -> list:
        """IK axis targets while holding: keep the item upright (strong); its yaw is free-ish."""
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
            lateral = float(np.clip((np.asarray(toward) - base) @ left, -0.28, 0.05))
        palm, _ = self.ik.palm_pose()
        up = palm.copy()
        up[2] = max(z, palm[2] + 0.05)
        self.move_hand(up - 0.06 * fwd, None, duration=0.6, settle=0.0)  # straight up, slightly back
        tgt = base + 0.18 * fwd + lateral * left
        tgt[2] = up[2]
        self.move_hand(tgt, [(0, fwd, 0.5)], duration=0.6, settle=0.1)

    def _step_back(self, dist: float) -> None:
        th = self.wbc.base_yaw
        p = self.wbc.base_pos[:2] - dist * np.array([np.cos(th), np.sin(th)])
        self._drive_to(p[0], p[1], th, tol=0.04, timeout=4.0)

    def holding_ok(self, tol: float = 0.08) -> bool:
        """Is the held item still at the palm? (A snagged item can be torn out of the weld.)"""
        if self.holding is None:
            return False
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.item_body[self.holding])
        palm, _ = self.ik.palm_pose()
        # compare with the palm-to-body distance at the grasp (a body's origin needn't be its centre)
        return abs(float(np.linalg.norm(self.data.xpos[b] - palm)) - self._grip_dist) < tol

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
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0])
        # High enough that the item clears what stands on a table we walk up to, but chosen by
        # how far the arm stays from its joint limits: the old fixed pose (0.28, -0.18, +0.18)
        # put the wrist and shoulder yaw within 0.05 rad of their limits with a top grasp.
        self.move_hand(self._best_hold_pose([(0.28, -0.18, 0.18), (0.32, -0.20, 0.12), (0.30, -0.22, 0.08),
                                             (0.35, -0.20, 0.10), (0.26, -0.16, 0.14), (0.33, -0.24, 0.15)]),
                       self._hold_axes(fwd), duration=1.0)

    def hand_over(self) -> dict:
        """At the person: place the held item on the side table next to their armchair."""
        if self.holding is None:
            return {"ok": False, "error": "not holding anything"}
        if not self.holding_ok():
            item = self.holding
            self._weld(f"grasp_{self.item_body[item]}", False)
            self.holding = None
            self._item_collisions(item, True)
            self.arm_home()
            return {"ok": False, "error": f"lost the {item} on the way"}
        it = self.items[self.holding]
        surface = self.delivery
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0])
        axes = self._hold_axes(fwd)
        # Put it down on the side table: a reachable spot, as far as possible from anything
        # already standing there (seen by the head camera).
        base = self.wbc.base_pos
        left = np.array([-np.sin(th), np.cos(th), 0])
        cx, cy = surface["pos"]
        hx, hy = surface["half"]
        # (the carried item can block the camera, so also use where we put things before)
        others = [p.position for m, p in self.cam.detect().items()
                  if m in self.marker_to_item and self.marker_to_item[m] != self.holding]
        others += self.placed
        best_spot, best_clear = None, -1.0
        for f, l in [(0.42, -0.14), (0.40, -0.02), (0.45, -0.24), (0.36, -0.20), (0.38, 0.06)]:
            cand = base + f * fwd + l * left
            m = 0.10  # keep well clear of the edge (arm tracking error is a few cm)
            cand[0] = np.clip(cand[0], cx - hx + m, cx + hx - m)
            cand[1] = np.clip(cand[1], cy - hy + m, cy + hy - m)
            clear = min([np.hypot(*(cand[:2] - o[:2])) for o in others], default=1.0)
            if clear > best_clear + 0.02:
                best_spot, best_clear = cand, clear
        spot = best_spot
        spot[2] = surface["top"] + it.size[2] + self.hold_dz + 0.006  # palm height when the item rests on the table
        # Come in from above (clear of anything already on the table), set down, leave upwards.
        self.move_hand(spot + np.array([0, 0, 0.17]), axes, duration=1.2)
        self.move_hand(spot, axes, duration=0.9)
        item = self.holding
        if DEBUG:
            b = self.model.body(self.item_body[item]).id
            print(f"   hand_over: spot {spot.round(3)} palm {self.ik.palm_pose()[0].round(3)} item {self.data.xpos[b].round(3)} "
                  f"item z-axis {self.data.xmat[b].reshape(3, 3)[:, 2].round(2)} surface top {surface['top']:.3f}")
        self._weld(f"grasp_{self.item_body[item]}", False)
        self.holding = None
        self.placed.append(spot.copy())
        self.wait(0.3)
        self.move_hand(spot - 0.06 * fwd + np.array([0, 0, 0.14]), axes, duration=0.8)
        self.move_hand(spot - 0.16 * fwd + np.array([0, 0, 0.14]), None, duration=0.5, settle=0.0)
        self.arm_home()
        self._item_collisions(item, True)
        return {"ok": True, "handed_over": item, "placed_on": "side_table" if self.world is None else "user_tray"}

    def follow(self, targets: list[np.ndarray], axes_list: list, dt: float = CONTROL_DT, max_err: float = 0.05,
               keep_reach: float | None = None) -> int:
        """Track a sequence of palm targets (one per control tick). Returns how many were reached.

        keep_reach: if set, walk backwards whenever the target comes closer than this to the
        pelvis (along the heading), e.g. while pulling a door open towards the body.
        """
        q = self.arm_cmd.copy()
        q_ref = q.copy()  # the grasp configuration: stay on its branch along the whole path
        bad = 0
        v_back = 0.0
        for i, (p, axes) in enumerate(zip(targets, axes_list)):
            if keep_reach is not None:
                th = self.wbc.base_yaw
                fwd = (p - self.wbc.base_pos)[:2] @ np.array([np.cos(th), np.sin(th)])
                want = -BACKOFF_SPEED if fwd < keep_reach else 0.0
                v_back += np.clip(want - v_back, -0.5 * dt, 0.5 * dt)  # 0.5 m/s^2 ramp
                self.wbc.set_velocity(v_back if abs(v_back) > 0.05 else 0.0, 0.0, 0.0)
            q, _ = self.ik.solve(p, axes, q_init=q, iters=25, q_ref=q_ref)
            self._set_right_arm(q)
            self._tick()
            err = float(np.linalg.norm(self.ik.palm_pose()[0] - p))
            bad = bad + 1 if err > max_err else 0
            if bad >= 10:  # persistently off the path (out of reach or blocked)
                self.wbc.set_velocity(0, 0, 0)
                return i
        self.wbc.set_velocity(0, 0, 0)
        return len(targets)

    def open_cupboard(self, model: ArticulationModel, fraction: float = 1.0, seconds: float = 5.0) -> dict:
        """Open the cupboard the way the carer did (see open_articulated)."""
        return self.open_articulated(model, ANCHOR_MARKER, "grasp_door", fraction, seconds,
                                     part_markers=(DOOR_MARKER, DOOR_INNER_MARKER))

    def open_articulated(self, model: ArticulationModel, anchor_marker: int, weld: str,
                         fraction: float = 1.0, seconds: float = 5.0, part_markers: tuple[int, ...] = ()) -> dict:
        """Open a door (revolute) or drawer (prismatic) the way the carer did, using the
        articulation model learned from one demonstration.

        The model lives in the anchor-marker frame, so the robot first finds the anchor with
        its head camera, maps the hinge/slide axis + grasp point into the world, grasps the
        handle and moves its hand along the learned arc or line.
        """
        det = self.cam.detect()
        if anchor_marker not in det:
            return {"ok": False, "error": f"anchor marker {anchor_marker} not visible"}
        A = det[anchor_marker]
        Ra, pa = A.rotation, A.position
        handle = Ra @ np.array(model.grasp_start) + pa
        normal = Ra @ np.array(model.part_normal_start)  # door outward normal (towards the robot)
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
        # Plan and reach from where we stand; if the handle can't be planned for or the reach
        # falls short, step in (whole-body reach: the base is part of the plan) so the handle sits
        # in front of the right shoulder, square to the front, look again and re-plan.
        yaw_in = float(np.arctan2(into[1], into[0]))
        f_in = np.array([np.cos(yaw_in), np.sin(yaw_in)])
        l_in = np.array([-f_in[1], f_in[0]])
        e, why = np.inf, ""
        for dist in (None, 0.36, 0.31):
            if dist is not None:
                self.arm_home(0.6)
                goal = handle[:2] - dist * f_in + GRASP_LATERAL * l_in
                self._drive_to(goal[0], goal[1], yaw_in, tol=0.025, timeout=8.0)
                det = self.cam.detect()  # re-measure from the new stance
                if anchor_marker in det:
                    A = det[anchor_marker]
                    Ra, pa = A.rotation, A.position
                    handle = Ra @ np.array(model.grasp_start) + pa
                    pre = handle - 0.12 * into
                self._raise_hand(handle[2], toward=handle)
            axes, best, q_pre, q_h = self._plan(self.ik, self.arm_cmd.copy(), pre, handle, candidates)
            if best > 0.05:
                th = self.wbc.base_yaw
                d = handle - self.wbc.base_pos
                rel = [d[0] * np.cos(th) + d[1] * np.sin(th), -d[0] * np.sin(th) + d[1] * np.cos(th), d[2]]
                why = (f"handle not reachable from here (score {best:.3f}; handle at "
                       f"fwd {rel[0]:.2f} left {rel[1]:.2f} up {rel[2]:.2f} m from the pelvis)")
                continue
            e = self._reach("right", q_pre, pre, handle, q_h, axes, duration=0.8)
            if e <= GRASP_TOL:
                break
            why = f"could not reach handle (residual {e:.3f} m)"
        if e > GRASP_TOL:
            self.arm_home()
            return {"ok": False, "error": why}
        # Position-only grip on the handle: the door turns about its hinge while the wrist
        # keeps a comfortable orientation (like a loose grip around a bar handle).
        self._weld(weld, True, torquescale=0.0)

        # Learned arc (revolute) or line (prismatic), mapped to the world
        c = Ra @ np.array(model.axis_point) + pa
        a = Ra @ np.array(model.axis_dir)
        n = int(seconds / CONTROL_DT)
        targets, axes_list = [], []
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)
            if model.kind == "revolute":
                R = rot_axis_angle(a, fraction * model.amount * s)
                targets.append(R @ (handle - c) + c)
            else:
                targets.append(handle + a * (fraction * model.amount * s))
            # keep only a weak 'palm side up' preference; the fingers may turn freely
            axes_list.append([(axes[1][0], axes[1][1], 0.2)])
        reached = self.follow(targets, axes_list, keep_reach=DOOR_KEEP_REACH)
        self.wait(0.5)
        self._weld(weld, False)

        # Let go and back off along the current finger direction (fingers = palm x-axis)
        p, R_palm = self.ik.palm_pose()
        self.move_hand(p - 0.10 * R_palm[:, 0], None, duration=0.7, settle=0.1)
        self.arm_home()
        # The pull walked us backwards: step back to the station so the next skill (look,
        # pick) starts from the pose it was designed for.
        s = self.stations.get(self.station)
        opened = fraction * model.amount * (reached / n)
        if s is not None and model.kind == "revolute":
            self._drive_to(s.x, s.y, s.yaw, tol=0.03, timeout=10.0)
        elif s is not None:
            # Drawer: the pull walked us back and usually twisted the base. Square up to the
            # station shifted out by the opening, so the next skill sees the drawer straight on.
            out = a[:2] / max(np.linalg.norm(a[:2]), 1e-9)
            self._drive_to(s.x + out[0] * opened, s.y + out[1] * opened, s.yaw, tol=0.03, timeout=10.0)
        # remembered for close_opened(): what is open, and a stance from which the anchor was in view
        p = self.wbc.base_pos
        self.open_state = dict(model=model, anchor_marker=anchor_marker, weld=weld, opened=opened,
                               stance=(float(p[0]), float(p[1]), float(s.yaw if s is not None else self.wbc.base_yaw)),
                               anchor_pose=(Ra.copy(), pa.copy()),  # the anchor is fixed to the furniture
                               part_markers=part_markers)
        unit = "rad" if model.kind == "revolute" else "m"
        return {"ok": reached > 0.6 * n, f"opened_{unit}": round(opened, 3), f"of_demo_{unit}": round(model.amount, 3)}

    # ------------------------------------------------------------------ closing (left hand)
    def move_left_hand(self, target: np.ndarray, axes=None, duration: float = 1.0, settle: float = 0.3,
                       q_ref: np.ndarray | None = None) -> float:
        """move_hand for the left arm (the right arm keeps its joint targets, e.g. holding an item)."""
        start, _ = self.left_ik.palm_pose()
        n = max(1, int(duration / CONTROL_DT))
        q = self.left_cmd.copy()
        k_min = n + int(settle / CONTROL_DT)
        for k in range(1, k_min + int(SETTLE_MAX / CONTROL_DT) + 1):
            a = min(1.0, k / n)
            a = a * a * (3 - 2 * a)
            q, _ = self.left_ik.solve((1 - a) * start + a * np.asarray(target), axes, q_init=q, iters=15, q_ref=q_ref)
            self._set_left_arm(q)
            self._tick()
            if k >= k_min and np.linalg.norm(self.left_ik.palm_pose()[0] - target) < 0.01:
                break
        p, _ = self.left_ik.palm_pose()
        return float(np.linalg.norm(p - target))

    def left_arm_home(self, duration: float = 1.0) -> None:
        self._swing("left", ARM_HOME[:7], duration)

    def _tuck(self) -> None:
        """Hold the carried item up and out to the right while the left hand works: the most
        reachable of a few poses that keep the item upright (the IK must hold the grasp axis)."""
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0.0])
        # the carry pose, but further out to the right: as the robot steps in to close the
        # part, the item passes beside the furniture instead of meeting its front or handle
        t = self._best_hold_pose([(0.28, -0.30, 0.18), (0.26, -0.34, 0.18), (0.30, -0.28, 0.20),
                                  (0.24, -0.32, 0.20), (0.28, -0.26, 0.18), (0.32, -0.30, 0.10)])
        self.move_hand(t, self._hold_axes(fwd), duration=1.0)

    def _best_hold_pose(self, candidates) -> np.ndarray:
        """Of pelvis-relative (forward, left, up) hand positions, the one the arm holds most
        comfortably with the item upright: reached, upright, collision-free, and away from the
        joint limits (near a limit the IK flips between solutions and the arm jerks)."""
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0.0])
        left = self._left()
        axes = self._hold_axes(fwd)
        idx, vert = self.hold_vert
        best, best_t = None, None
        for f, l, u in candidates:
            t = self.wbc.base_pos + f * fwd + l * left + np.array([0, 0, u])
            q, e = self.ik.solve(t, axes, q_init=self.arm_cmd, iters=150, elbow_gain=self._elbow_gain())
            upright = float(self.scratch_palm_axis(q, idx) @ vert)
            margin = self.ik.limit_margin(q)
            score = (e + 0.3 * (1 - upright) + 0.1 * len(self.ik.collisions(q)) + 0.3 * max(0.0, 0.3 - margin)
                     + ELBOW_W * self.ik.elbow_cost(q))
            if best is None or score < best:
                best, best_t = score, t
        return best_t

    def scratch_palm_axis(self, q: np.ndarray, idx) -> np.ndarray:
        """World direction of palm axis `idx` if the right arm were at q."""
        s = self.ik.scratch
        self.ik._load(s)
        s.qpos[self.ik.qadr] = q
        mujoco.mj_kinematics(self.model, s)
        R = s.site_xmat[self.ik.site].reshape(3, 3)
        return (R[:, idx] if np.isscalar(idx) else R @ idx).copy()

    def _turn_to(self, yaw: float, tol: float = 0.05, timeout: float = 6.0) -> None:
        """Turn in place. GR00T barely responds to small yaw-rate commands when the arms carry a
        load off-centre, so this uses a firm minimum rate (the navigator's PD stalls there)."""
        for _ in range(int(timeout / CONTROL_DT)):
            err = wrap(yaw - self.wbc.base_yaw)
            if abs(err) < tol:
                break
            self.wbc.set_velocity(0.0, 0.0, float(np.sign(err) * np.clip(1.5 * abs(err), 0.5, 0.8)))
            self._tick()
        self.wbc.set_velocity(0.0, 0.0, 0.0)
        self.wait(0.6)

    def _measure_opening(self, st: dict, det: dict, Ra: np.ndarray, pa: np.ndarray) -> float | None:
        """How far the part is open NOW, from a marker on it (it may have been bumped since we
        opened it). Drawer: its front marker projected onto the learned slide axis. Door: the
        normal of the marker on its inside, as a rotation about the learned hinge axis."""
        model = st["model"]
        a = np.array(model.axis_dir)
        markers = st.get("part_markers", ())
        if model.kind == "prismatic" and markers and markers[0] in det:
            p = Ra.T @ (det[markers[0]].position - pa)
            return float((p - np.array(model.axis_point)) @ a)
        if model.kind == "revolute" and len(markers) > 1 and markers[1] in det:
            n = Ra.T @ det[markers[1]].rotation[:, 2]  # inside face normal, anchor frame
            c = -np.array(model.part_normal_start)  # inside face normal when closed
            n, c = n - (n @ a) * a, c - (c @ a) * a
            return float(np.arctan2(np.cross(c, n) @ a, c @ n))
        return None

    def close_opened(self, seconds: float = 4.0, retries: int = 1, push: bool = False) -> dict:
        """Close what open_articulated() opened, with the LEFT hand, while the right hand keeps
        holding whatever was taken out.

        No new data: the one-shot articulation model already describes the whole motion, so
        closing is the learned line (drawer) or arc (door) run from the current opening back
        to zero. The robot re-finds the anchor marker, grips the handle (for a bar handle, on
        its own left half), and walks forward as the part moves away from it.

        push (drawers only): don't grip, push the handle bar shut with the open hand. Used when
        the grip can't be reached or was lost part-way (a drawer doesn't need pulling to close).
        """
        # Drawers are always pushed shut (gripping and walking the bar back lost it part-way in
        # tests); doors are gripped past their free edge.
        push = self.open_state is not None and self.open_state["model"].kind == "prismatic"
        st = self.open_state
        if st is None:
            return {"ok": False, "error": "nothing open"}
        model, opened = st["model"], st["opened"]
        lweld = st["weld"].replace("grasp_", "grasp_left_", 1)
        det = self.cam.detect()
        if st["anchor_marker"] in det:
            Ra, pa = det[st["anchor_marker"]].rotation, det[st["anchor_marker"]].position
        else:
            # Close up and looking down, the anchor on top of the furniture is often out of
            # view; it is fixed to the furniture, so its pose from when we opened still holds.
            Ra, pa = st["anchor_pose"]
        measured = self._measure_opening(st, det, Ra, pa)
        if measured is not None:
            opened = measured
        R_open, t_open = model.transform(opened)
        normal = Ra @ (R_open @ np.array(model.part_normal_start))  # outward face of the part, now
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
        into = -normal
        g0 = np.array(model.grasp_start)  # grasp point on the part, anchor frame, closed
        if model.kind == "revolute":
            # A swung-open door shows the robot its inner face (the handle, on the outer face,
            # points away). Take hold just past the door's free edge instead: on the robot's side
            # while open, and outside the cupboard once shut, so the hand can follow it all the
            # way. The hand comes at it along the door, from beyond the edge.
            a_ax, c_ax = np.array(model.axis_dir), np.array(model.axis_point)
            n0 = np.array(model.part_normal_start)
            u = g0 - c_ax
            u = u - (u @ a_ax) * a_ax - (u @ n0) * n0
            u /= np.linalg.norm(u)  # hinge -> free edge, in the door plane (closed, anchor frame)
            g0 = g0 + 0.09 * u - 0.04 * n0  # 4 cm past the edge, mid-thickness
            u_w = Ra @ (R_open @ u)
            u_w[2] = 0.0
            into = -u_w / np.linalg.norm(u_w)
        if model.kind == "prismatic":
            # bar handle across the front: grip it on the robot's left of centre
            lat = self._left() - (self._left() @ normal) * normal
            lat[2] = 0.0
            lat /= max(np.linalg.norm(lat), 1e-9)
            g0 = g0 + R_open.T @ (Ra.T @ (0.12 * lat))
        handle = Ra @ (R_open @ g0 + t_open) + pa

        # Tuck whatever the right hand holds up and out to the right, clear of the handle and
        # the front (a bottle hanging at carry height snags a drawer's bar handle and twists)
        if self.holding is not None:
            self._tuck()

        # Face the grasp along the hand's approach direction (a drawer: square to its front; a
        # swung door: towards its inner face, where it stands), with the grasp point ahead of
        # the left shoulder. Turning on the spot first keeps the held item clear of the door.
        yaw_goal = float(np.arctan2(into[1], into[0]))
        self._turn_to(yaw_goal)
        fwd = np.array([np.cos(yaw_goal), np.sin(yaw_goal), 0.0])
        left = np.array([-fwd[1], fwd[0], 0.0])
        for _ in range(3):  # walking with a load drifts: look, drive, turn, check, repeat
            # look again: is it still open as far as we think? (it may have been knocked)
            measured = self._measure_opening(st, self.cam.detect(), Ra, pa)
            if measured is not None:
                if measured < 0.05 * model.amount:
                    self.open_state = None
                    if self.holding is not None:
                        self.carry()
                    return {"ok": True, "already_closed": True, "still_holding": self.holding}
                opened = measured
                R_open, t_open = model.transform(opened)
                handle = Ra @ (R_open @ g0 + t_open) + pa
            goal = handle - (0.42 if push else 0.40) * fwd - GRASP_LATERAL * left
            if DEBUG:
                print(f"   close stance: measured {measured} opened {opened:.3f} handle {handle.round(3)} "
                      f"base {self.wbc.base_pos.round(3)} goal {goal[:2].round(3)}")
            if np.linalg.norm(goal[:2] - self.wbc.base_pos[:2]) < 0.06 and abs(wrap(yaw_goal - self.wbc.base_yaw)) < 0.1:
                break
            b = self.wbc.base_pos
            back = float((goal[:2] - b[:2]) @ fwd[:2])
            side = float((goal[:2] - b[:2]) @ left[:2])
            if back < -0.05 and abs(side) > 0.10:
                # Back off first, then sideways: a diagonal move sweeps the held item (tucked out
                # to the right) along the open drawer's front and catches it on the handle bar
                mid = b[:2] + back * fwd[:2]
                self._drive_to(mid[0], mid[1], yaw_goal, tol=0.04, timeout=6.0)
            self._drive_to(goal[0], goal[1], yaw_goal, tol=0.03, timeout=8.0)
            self._turn_to(yaw_goal)

        up = np.array([0, 0, 1.0])
        pre = handle - 0.12 * into
        candidates = [[(0, into, 0.6), (1, up, 0.4)], [(0, into, 0.6), (1, -up, 0.4)],
                      [(0, into, 0.6), (2, up, 0.4)], [(0, into, 0.6), (2, -up, 0.4)]]
        # Pushing: fingertips (~11 cm past the palm site) on the drawer front, 8 cm above the bar
        push_off = (PUSH_FACE - FINGER_REACH) * into + np.array([0, 0, 0.08])
        contact = handle + push_off if push else handle
        if push:
            pre = contact - 0.10 * into
        axes, best, q_best, q_h = self._plan(self.left_ik, self.left_cmd.copy(), pre, contact, candidates)
        if best > 0.05 and not push and model.kind == "prismatic":
            return self.close_opened(seconds, 0, push=True)
        if best > 0.05:
            rel = handle - self.wbc.base_pos
            return {"ok": False, "error": f"handle not reachable with the left hand (score {best:.3f})",
                    "handle_rel_pelvis": rel.round(3).tolist(), "yaw": round(self.wbc.base_yaw, 3),
                    "yaw_goal": round(yaw_goal, 3)}
        # joint-space swing to the pre-grasp (avoids IK local minima from the hand-by-the-hip pose)
        e = self._reach("left", q_best, pre, contact, q_h, axes, duration=0.8)
        if e > 0.04:
            self.left_arm_home()
            if not push and model.kind == "prismatic":
                return {**self.close_opened(seconds, 0, push=True), "grip_failed": round(e, 3)}
            return {"ok": False, "error": f"could not reach the handle with the left hand (residual {e:.3f} m)"}
        if not push:
            self._weld(lweld, True, torquescale=0.0, ik=self.left_ik)

        # The learned path, backwards: opening `opened` -> 0. Paced by the hand's path length:
        # the feet (GR00T, <= 0.15 m/s sideways) have to keep the handle within reach.
        R0, t0 = model.transform(0.0)
        path_len = float(np.linalg.norm((R_open @ g0 + t_open) - (R0 @ g0 + t0)))
        if model.kind == "revolute":
            r = np.linalg.norm(np.cross(np.array(model.axis_dir), g0 - np.array(model.axis_point)))
            path_len = float(r * opened)
        seconds = max(seconds, path_len / 0.05)
        steps = int(seconds / CONTROL_DT)
        targets = []
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)
            if push:  # fingertips pressed 2.5 cm into the front, running 3 cm past shut
                R, t = model.transform(opened * (1 - s) - 0.03 * s)
                targets.append(Ra @ (R @ g0 + t) + pa + push_off + 0.025 * into)
            else:
                R, t = model.transform(opened * (1 - s))
                targets.append(Ra @ (R @ g0 + t) + pa)
        q = self.left_cmd.copy()
        q_ref = q.copy()  # stay on the grasp configuration's branch
        bad, reached = 0, 0
        vel = np.zeros(2)
        hand_home = np.array([0.36, GRASP_LATERAL])  # where the handle should sit in the body frame (well within reach)
        k, i = 0, 0
        while k < len(targets) and i < 3 * len(targets):
            p = targets[k]
            i += 1
            # Follow the handle with the feet: keep it ~0.36 m ahead and in line with the left
            # shoulder (body frame, so a drifted heading doesn't fool it), and hold the heading.
            th = self.wbc.base_yaw
            d = (p - self.wbc.base_pos)[:2]
            rel = np.array([np.cos(th) * d[0] + np.sin(th) * d[1], -np.sin(th) * d[0] + np.cos(th) * d[1]])
            e = rel - hand_home
            lim = np.array([0.25, BACKOFF_SPEED])  # walking forward can be quicker than sideways
            want = np.where(np.abs(e) > 0.03, np.clip(1.2 * e, -lim, lim), 0.0)
            vel += np.clip(want - vel, -0.5 * CONTROL_DT, 0.5 * CONTROL_DT)
            yerr = wrap(yaw_goal - th)
            wz = float(np.clip(1.5 * yerr, -0.5, 0.5)) if abs(yerr) > 0.08 else 0.0
            v_cmd = np.where(np.abs(vel) > 0.05, vel, 0.0)
            self.wbc.set_velocity(float(v_cmd[0]), float(v_cmd[1]), wz)
            # gripping: only a weak palm-up preference (the part turns the hand); pushing: the
            # fingers must keep pointing into the front, or the fingertips miss it
            q, _ = self.left_ik.solve(p, axes if push else [(axes[1][0], axes[1][1], 0.2)], q_init=q, iters=25,
                                      q_ref=q_ref)
            self._set_left_arm(q)
            self._tick()
            err = float(np.linalg.norm(self.left_ik.palm_pose()[0] - p))
            # Pace the path by the feet: only move the target on while the hand keeps up and the
            # target is still comfortably within reach (the feet are slower than the arm)
            if err < 0.05 and rel[0] < hand_home[0] + 0.06:
                k += 1
            bad = bad + 1 if err > 0.07 else 0  # the part's damping makes the soft arm lag a bit
            reached = k
            if DEBUG and i % 50 == 0 and self.world is not None:
                print(f"     close tick {i}: k {k}/{len(targets)} err {err:.3f} rel {rel.round(2)} "
                      f"vel {v_cmd.round(2)} drawer {__import__("house").G1House.drawer_open_m(self.world, __import__("house").KITCHEN_TOP):.3f}")
            if bad >= 60:
                if DEBUG:
                    print(f"   close: lost the path at {k}/{steps}: err {err:.3f} target rel body {rel.round(3)} "
                          f"vel {v_cmd.round(2)} hits {self.left_ik.collisions(self.left_cmd)}")
                break
        self.wbc.set_velocity(0, 0, 0)
        self.wait(0.4)
        if not push:
            self._weld(lweld, False)
        p, R_palm = self.left_ik.palm_pose()
        self.move_left_hand(p - 0.10 * R_palm[:, 0], None, duration=0.6, settle=0.1)
        self.left_arm_home()
        if self.holding is not None:
            self.carry()
        # The commanded path ends at zero, but the part stops at its end stop a little earlier
        # (the hand then stalls): stalling in the last 10% of the motion means it is shut.
        closed_to = opened * (1 - reached / steps)
        # Check by looking (the part can keep going after the hand loses it, or stop short)
        seen = self._measure_opening(st, self.cam.detect(), Ra, pa)
        if seen is not None:
            closed_to = max(seen, 0.0)
        shut = closed_to <= 0.1 * model.amount  # within 10% of the learned travel
        self.open_state = None if shut else {**st, "opened": closed_to}
        if not shut and retries > 0 and reached > 0:
            # lost it part-way: grip it again where it is now and finish
            again = self.close_opened(seconds, retries - 1, push=True)
            return {**again, "retried": True, "first_try_left_open": round(closed_to, 3)}
        unit = "rad" if model.kind == "revolute" else "m"
        return {"ok": shut, "pushed": push, f"closed_from_{unit}": round(opened, 3), f"left_open_{unit}": round(closed_to, 3),
                "checked_by_camera": seen is not None, "still_holding": self.holding}

    def put_down(self, height_offset: float = 0.0) -> dict:
        """Release the held item where the hand is (e.g. back on a surface)."""
        if self.holding is None:
            return {"ok": False, "error": "not holding anything"}
        self._weld(f"grasp_{self.item_body[self.holding]}", False)
        item, self.holding = self.holding, None
        self.arm_home()
        self._item_collisions(item, True)
        return {"ok": True, "released": item}
