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
# Stance selection (inverse reachability, done online): before walking up to something to
# grasp, try candidate foot positions/headings/postures *in the planner's copy of the model*
# and walk to the one where the arm reaches best, instead of a fixed standoff. G1_STANCE_SEARCH=0
# switches back to the fixed standoffs (A/B).
STANCE_SEARCH = os.environ.get("G1_STANCE_SEARCH", "1") != "0"
# Postures whose body shape is measured once at start-up (pelvis height, torso pitch), so the
# stance search can also ask "what if I leaned/crouched there" without doing it.
CALIB_POSTURES = [(0.74, 0.0), (0.74, 0.15), (0.74, 0.3), (0.70, 0.45), (0.68, 0.45), (0.64, 0.5), (0.60, 0.5)]
# After a reach falls short by up to this much, step the feet by the miss and try once more (m)
STEP_CORRECT_MAX = 0.15
REACH_COMFORT = 0.45  # m, forward pelvis-to-target distance beyond which a stance is penalised
# Perceived navigation (costmap.py): the clearance map comes from the robot's 3D LiDAR and depth
# camera (prior: walls + fixed furniture), and go_to re-plans when a new obstacle cuts the route.
# G1_NAV=static plans once on the simulator's true geometry (the old behaviour).
NAV_LIVE = os.environ.get("G1_NAV", "live") != "static"
MAX_REPLANS = 12  # per go_to
REPLAN_MARGIN = 0.08  # m, a route is re-planned when its clearance drops this far below what it was planned with
REPLAN = "replan"  # _drive_to's return value when its stop_if fired


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
        # Perceived clearance map (costmap.py), in the house only
        self.costmap = None
        self.nav_log: list[dict] = []
        self.nav_path: list | None = None
        if world is not None and NAV_LIVE:
            from costmap import HeadLidar, make_costmap
            self.costmap = make_costmap(world)
            self.lidar3d = HeadLidar(self.model, self.data)
            world.costmap = self.costmap
            self._map_ticks = 0
            self._driving = False
            self._lidar_held = None
        self.wait(1.0)
        self._posture_cache: dict = {}
        if STANCE_SEARCH:
            self._calibrate_postures()
        if self.costmap is not None:
            self._map_depth()  # first look at what is in front before the first plan

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
        if self.costmap is not None:
            self._sense_map()
        if self.on_frame is not None:
            self.on_frame(self)

    # ------------------------------------------------------------------ mapping
    def _believed_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Pelvis (position, rotation) the robot believes: the estimator's, or the truth."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.wbc.base_quat)
        return self.wbc.base_pos.copy(), R.reshape(3, 3)

    def _believed_camera(self) -> tuple[np.ndarray, np.ndarray]:
        if self.estimator is not None:
            return self.estimator.camera_pose()
        return (self.data.cam_xpos[self.cam.cam_id].copy(), self.data.cam_xmat[self.cam.cam_id].reshape(3, 3).copy())

    def _sense_map(self) -> None:
        """Feed the clearance map: a 3D LiDAR scan every LIDAR_EVERY ticks, and while walking a
        depth image every DEPTH_EVERY ticks (look() adds one at every look)."""
        from costmap import DEPTH_EVERY, LIDAR_EVERY
        self._map_ticks += 1
        held = self.item_body[self.holding] if self.holding else None
        if held != self._lidar_held:  # a carried item isn't an obstacle
            self.lidar3d.set_held_body(held)
            if self.estimator is not None:
                self.estimator.lidar.set_held_body(held)
            self._lidar_held = held
        if self._map_ticks % LIDAR_EVERY == 0:
            pos, R = self._believed_pose()
            self.costmap.update_lidar(self.lidar3d, pos, R)
        if self._driving and self._map_ticks % DEPTH_EVERY == 0:
            self._map_depth()

    def _map_depth(self) -> None:
        depth = self.cam.render_depth()
        cam_pos, cam_R = self._believed_camera()
        self.costmap.update_depth(depth, self.cam.K, cam_pos, cam_R, self.wbc.base_pos[:2])

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
                  pass_through: bool = False, stop_if: Callable[[], bool] | None = None) -> bool | str:
        """Walk to (x, y[, yaw]) with the smooth PD navigator (nav.py).

        pass_through: a waypoint. Return as soon as we're within `tol`, keeping the current
        velocity (no stop), so the next leg continues smoothly.
        stop_if: checked every tick; when it returns True the leg is abandoned at the current
        velocity (no stop) and REPLAN is returned, so go_to can re-route without a halt.
        """
        self.nav.p.pos_tol = tol
        self.nav.arrived = False  # keep self.nav.cmd: continue from the current velocity
        self._driving = True
        try:
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
                if stop_if is not None and stop_if():
                    return REPLAN
                if self.nav.stopped:
                    break
        finally:
            self._driving = False
        arrived = self.nav.stopped
        self.nav.reset()
        self.wbc.set_velocity(0, 0, 0)
        self.wait(0.8)
        if not arrived:  # timed out (blocked, e.g. a carried item jammed on furniture)
            p = self.wbc.base_pos
            arrived = np.hypot(x - p[0], y - p[1]) < 2 * tol
        return arrived

    def _walk_path(self, path: list, goal_xy, station: str) -> dict:
        """Walk the waypoints of `path` (its last point is the station, approached by the caller).

        With the perceived map: after every map update the rest of the route is checked against
        the live clearance map (at the radius it was planned with); if a new obstacle has cut it,
        a new route is planned from where the robot is and walking continues without stopping.
        """
        replans, cut_at = 0, []
        last_replan = -10 ** 9
        self.nav_path = list(path)  # the route being walked (for video insets / debugging)
        i = 1
        while i < len(path) - 1:
            wx, wy = path[i]
            stop_if = None
            if self.costmap is not None:
                seen = {"v": self.costmap.version}
                # Hysteresis: planned with r_plan of clearance, re-planned only when something
                # comes within REPLAN_MARGIN less than that (a half-seen chair "grows" with every
                # scan; a route that merely grazes its inflation is not cut).
                r_plan = max(self.world.last_plan_radius - REPLAN_MARGIN, self.world.ROBOT_R_MIN)
                since = self.world.plan_clearance  # only obstacles that are new since the plan count

                def stop_if(i=i, seen=seen, r_plan=r_plan, path=path, since=since):
                    if self.costmap.version == seen["v"] or self._map_ticks - last_replan < 25:
                        return False  # no new map, or a route re-planned less than 0.5 s ago
                    seen["v"] = self.costmap.version
                    # the route ahead: from a little in front of the robot (it weaves +-0.2 m
                    # about the waypoints; its own cell near a wall is not a new obstacle)
                    here = np.asarray(self.wbc.base_pos[:2])
                    to = np.asarray(path[i]) - here
                    d = float(np.linalg.norm(to))
                    start = here + to / d * min(0.30, 0.5 * d) if d > 1e-6 else here
                    return self.costmap.path_blocked([tuple(start)] + list(path[i:]), r_plan, since=since) >= 0
            res = self._drive_to(wx, wy, None, tol=0.20, pass_through=True, stop_if=stop_if)
            if self.wbc.fallen():
                return {"ok": False, "error": "fell over", "replans": replans}
            if res == REPLAN:
                here = self.wbc.base_pos[:2].copy()
                replans += 1
                last_replan = self._map_ticks
                cut_at.append(here.round(2).tolist())
                new = self.world.plan_path(here, goal_xy)
                if DEBUG:
                    print(f"[nav] route cut at {here.round(2)}; replan {replans}: {new and [tuple(np.round(p, 2)) for p in new]}")
                self.nav_log.append({"t": round(self.data.time, 2), "at": here.round(3).tolist(), "replan": replans,
                                     "path": None if new is None else [list(np.round(p, 3)) for p in new]})
                if new is None or replans > MAX_REPLANS:
                    self.nav.reset()
                    self.wbc.set_velocity(0, 0, 0)
                    self.wait(0.8)
                    return {"ok": False, "error": f"route to the {station} is blocked", "replans": replans}
                path = new
                self.nav_path = list(path)
                i = 1
                continue
            i += 1
        return {"ok": True, "replans": replans, "cut_at": cut_at, "path": [list(np.round(p, 3)) for p in path]}

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
        walk = {}
        if self.world is not None:
            if station != self.station:
                self._step_back(0.25)  # off the furniture first: stations sit inside the grid's inflation
            path = self.world.plan_path(self.wbc.base_pos[:2], (s.x, s.y))
            if path is None:
                return {"ok": False, "error": f"no path to the {station}"}
            walk = self._walk_path(path, (s.x, s.y), station)
            if not walk["ok"]:
                return {**walk, "station": station, "pose": self.wbc.base_pos.round(2).tolist()}
        elif np.hypot(p[0] - HUB[0], p[1] - HUB[1]) > 0.3 and station != self.station:
            self._drive_to(*HUB, None, tol=0.25, pass_through=True)
            if self.wbc.fallen():
                return {"ok": False, "error": "fell over"}
        ok = bool(self._drive_to(s.x, s.y, s.yaw))
        self.nav_path = None
        if ok:
            self.station = station
        out = {"ok": ok, "station": station, "pose": self.wbc.base_pos.round(2).tolist()}
        if "replans" in walk:
            out["replans"] = walk["replans"]
        if not ok:
            out["error"] = f"couldn't reach the {station}"
        return out

    # ------------------------------------------------------------------ perception
    def look(self) -> dict:
        """Detect markers in the head camera. Returns item names -> world position.

        A single marker frame does not establish full station coverage, so absence remains
        unconfirmed unless a future perception scan explicitly supplies that evidence.
        """
        det = self.cam.detect()
        self.last_seen = det
        if self.costmap is not None:  # every look also updates the clearance map (RGB-D)
            self._map_depth()
        items = {self.marker_to_item[m]: p.position.round(3).tolist() for m, p in det.items() if m in self.marker_to_item}
        extra = {}
        if DOOR_MARKER in det:
            extra["cupboard_door"] = det[DOOR_MARKER].position.round(3).tolist()
        if ANCHOR_MARKER in det:
            extra["cupboard_anchor"] = det[ANCHOR_MARKER].position.round(3).tolist()
        return {"ok": True, "station": self.station, "items": items,
                "marker_ids": sorted(int(marker_id) for marker_id in det),
                "absence_confirmed": False, **extra}

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
        # The hand goes in and comes out perpendicular to the furniture (keep the heading we
        # arrived with, facing it): a diagonal path drags the item along a cupboard's side wall.
        th = self.wbc.base_yaw
        approach = np.array([np.cos(th), np.sin(th), 0.0])
        up = np.array([0, 0, 1.0])
        candidates = [
            [(0, approach, 0.6), (2, up, 0.4)],
            [(0, approach, 0.6), (1, up, 0.4)],
            [(0, approach, 0.6), (1, -up, 0.4)],
            [(0, approach, 0.6), (2, -up, 0.4)],
        ]
        postures = POSTURES
        # Where to stand: searched (reachability in the planner's model), else the item straight
        # ahead of the right shoulder at STANDOFF.
        back0 = float(np.clip(STANDOFF - 0.27, 0.06, 0.14))
        st = self._choose_stance(self.ik, grasp - back0 * approach + np.array([0, 0, 0.04]), grasp, candidates,
                                 th, STANDOFF, GRASP_LATERAL, POSTURES, label=f"pick {item}")
        if st is not None:
            self._go_stance(st)
            postures = self._postures_from(st["posture"], POSTURES)
        else:
            fwd2, left2 = approach[:2], np.array([-np.sin(th), np.cos(th)])
            goal = grasp[:2] - STANDOFF * fwd2 + GRASP_LATERAL * left2  # pelvis left of the item
            if np.linalg.norm(goal - base[:2]) > 0.03:
                self._drive_to(goal[0], goal[1], th, tol=0.025, timeout=8.0)
        e1 = e2 = np.inf
        for attempt in range(2):
            det = self.cam.detect()  # closer view, from where we actually ended up
            if it.marker_id in det:
                top = det[it.marker_id].position
                centre = top - np.array([0, 0, it.size[2] + 0.001])
                grasp = centre + np.array([0, 0, 0.3 * it.size[2]])
            base = self.wbc.base_pos
            reach = float((grasp - base)[:2] @ approach[:2])
            # Pre-grasp back along the approach, but never closer than ~0.27 m to the pelvis:
            # nearer than that the arm would have to fold into the torso.
            back = float(np.clip(reach - 0.27, 0.06, 0.14))
            pre = grasp - back * approach + np.array([0, 0, 0.04])
            # Posture (lean/crouch via the GR00T torso and height commands) and wrist roll that
            # reach both poses without touching furniture or the torso.
            axes, best = self._choose(pre, grasp, candidates, postures)
            if best > 0.05:
                st = self._restance(self.ik, pre, grasp, candidates, STANDOFF, GRASP_LATERAL, POSTURES, None,
                                    f"pick {item} again") if attempt == 0 else None
                if st is not None:
                    postures = self._postures_from(st["posture"], POSTURES)
                    continue
                self._set_posture(0.74, 0.0)
                return {"ok": False, "error": f"no collision-free grasp for {item} from here (score {best:.3f})"}

            # Come in along the approach line: far pre-grasp (outside any cupboard), pre-grasp,
            # grasp, all on the planned configurations (joint-space swing to the far pre-grasp).
            q_pre, q_g = self._best_q
            far = grasp - (back + 0.10) * approach + np.array([0, 0, 0.04])
            q_far, e_far = self.ik.solve(far, axes, q_init=q_pre, iters=150, q_ref=q_pre)
            if self.ik.elbow_too_high(q_far) or e_far > 0.02 or self.ik.collisions(q_far):
                far, q_far = pre, q_pre  # the far point isn't cleanly reachable from this stance: skip it
            if DEBUG:
                rel = self._body_rel(grasp)
                print(f"   pick {item} [{attempt}]: plan {best:.3f} posture {self.wbc.height_cmd:.2f}/{self.wbc.rpy_cmd[1]:.2f} "
                      f"grasp fwd {rel[0]:.2f} left {rel[1]:.2f} up {rel[2]:.2f}  far err {e_far:.3f}")
            self._reach("right", q_far, far, pre, q_pre, axes, duration=0.7)
            e1 = float(np.linalg.norm(self.ik.palm_pose()[0] - pre))
            e2 = self.move_hand(grasp, axes, duration=0.9, q_ref=q_g)
            if DEBUG:
                print(f"   pick {item} [{attempt}]: pre err {e1:.3f} grasp err {e2:.3f} hits {self.ik.collisions(self.arm_cmd)}")
            if e2 <= 0.04:
                break
            miss = grasp - self.ik.palm_pose()[0]
            self.move_hand(pre - 0.06 * approach, axes, duration=0.8, q_ref=q_pre)
            if attempt == 0 and np.linalg.norm(miss[:2]) < STEP_CORRECT_MAX:
                # Short by a few cm: step the feet by the miss and plan again from there
                self._step_by(miss[:2])
                continue
            break
        if e2 > 0.04:
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

    def _free(self, ik: ArmIK, q: np.ndarray) -> bool:
        return not ik.collisions(q, margin=0.0)

    def _seg_free(self, ik: ArmIK, a: np.ndarray, b: np.ndarray, res: float = 0.08) -> bool:
        n = max(2, int(np.ceil(np.abs(b - a).max() / res)) + 1)
        return all(self._free(ik, (1 - t) * a + t * b) for t in np.linspace(0, 1, n)[1:])

    def _joint_path(self, ik: ArmIK, q0: np.ndarray, q1: np.ndarray, max_iters: int = 3000,
                    step: float = 0.25) -> list[np.ndarray] | None:
        """Collision-free joint-space path q0 -> q1 for one arm (RRT-Connect, Kuffner & LaValle
        2000, then shortcut smoothing), checked against the body and furniture with the same
        contact test as the plans. The straight swing is tried first. The start may be touching
        something (it is where the arm is now): its first segment is not checked."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if self._seg_free(ik, q0, q1):
            return [q0, q1]
        if not self._free(ik, q1):
            if DEBUG:
                print(f"   rrt: goal in collision {ik.collisions(q1, margin=0.0)[:2]}")
            return None
        prefix = []
        if not self._free(ik, q0):
            # the arm starts touching something (often the raised hand against the chest): back
            # out along the line to home, or to the goal, until free
            for to in (ik.home, q1):
                for t in np.linspace(0.1, 1.0, 10):
                    qe = (1 - t) * q0 + t * np.asarray(to)
                    if self._free(ik, qe):
                        break
                else:
                    continue
                break
            else:
                return None
            prefix, q0 = [q0], qe
            if self._seg_free(ik, q0, q1):
                return prefix + [q0, q1]
        rng = np.random.default_rng(0)
        lo, hi = ik.lo, ik.hi
        trees = [[(q0, -1)], [(q1, -1)]]  # (config, parent index)

        def nearest(tree, q):
            return int(np.argmin([np.abs(n[0] - q).max() for n in tree]))

        def extend(tree, q, first_free=False):
            i = nearest(tree, q)
            qn = tree[i][0]
            d = q - qn
            dist = np.abs(d).max()
            qs = q if dist <= step else qn + d * (step / dist)
            if (first_free and i == 0) or self._seg_free(ik, qn, qs):
                tree.append((qs, i))
                return len(tree) - 1, dist <= step
            return None, False

        for it in range(max_iters):
            a, b = trees[it % 2], trees[(it + 1) % 2]
            qr = rng.uniform(lo, hi) if rng.random() > 0.2 else b[0][0]
            ia, _ = extend(a, qr, first_free=(a is trees[0]))
            if ia is None:
                continue
            # connect the other tree towards the new node
            while True:
                ib, reached = extend(b, a[ia][0], first_free=(b is trees[0]))
                if ib is None:
                    break
                if reached:
                    def chain(tree, i):
                        out = []
                        while i >= 0:
                            out.append(tree[i][0])
                            i = tree[i][1]
                        return out
                    pa, pb = chain(a, ia), chain(b, ib)
                    path = (pa[::-1] + pb) if a is trees[0] else (pb[::-1] + pa)
                    path = prefix + path
                    # shortcut smoothing
                    for _ in range(40):
                        if len(path) <= 2:
                            break
                        i, j = sorted(rng.choice(len(path), 2, replace=False))
                        if j - i > 1 and self._seg_free(ik, path[i], path[j]):
                            path = path[:i + 1] + path[j:]
                    if DEBUG:
                        print(f"   rrt: path with {len(path)} waypoints after {it} iterations")
                    return path
        if DEBUG:
            print("   rrt: no path")
        return None

    def _swing_path(self, side: str, path: list[np.ndarray], min_duration: float = 1.0) -> None:
        """Joint-space moves through a planned path's waypoints."""
        for k, q in enumerate(path[1:]):
            self._swing(side, q, min_duration / max(1, len(path) - 1)) if k < len(path) - 2 else self._swing(side, q, 0.5)

    def _reach(self, side: str, q_pre: np.ndarray, pre: np.ndarray, target: np.ndarray, q_t: np.ndarray,
               axes, duration: float = 0.9) -> float:
        """Execute a plan: joint-space swing to q_pre (Cartesian with q_ref if the swing would hit
        something), then straight in to the target, staying on the plan's branch. Returns the error."""
        ik, cmd = (self.ik, self.arm_cmd) if side == "right" else (self.left_ik, self.left_cmd)
        mover = self.move_hand if side == "right" else self.move_left_hand
        clear = self._path_clear(ik, cmd.copy(), q_pre)
        b0, t0 = self.wbc.base_pos.copy(), self.wbc.true_base_pos.copy()
        path = None if clear else self._joint_path(ik, cmd.copy(), q_pre)
        if clear:
            self._swing(side, q_pre, 1.2)
        elif path is not None:
            # the straight swing would hit the body or furniture: go around (planned path)
            self._swing_path(side, path, 1.6)
        else:
            mover(pre, axes, duration=1.2, settle=0.1, q_ref=q_pre)
        if DEBUG:
            q_act = self.data.qpos[ik.qadr]
            print(f"   reach[{side}]: {'swing' if clear else 'tracked'} to pre: palm err {np.linalg.norm(ik.palm_pose()[0] - pre):.3f} "
                  f"joint track err {np.abs(q_act - (self.arm_cmd if side == 'right' else self.left_cmd)).max():.3f} "
                  f"plan-vs-cmd {np.abs(q_pre - (self.arm_cmd if side == 'right' else self.left_cmd)).max():.3f} "
                  f"base moved est {np.linalg.norm(self.wbc.base_pos[:2] - b0[:2]):.3f} true {np.linalg.norm(self.wbc.true_base_pos[:2] - t0[:2]):.3f} "
                  f"hits {ik.collisions(q_act)}")
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
        above_postures = [(0.74, 0.3), (0.74, 0.15), (0.70, 0.45), (0.64, 0.5)]
        # The item's distance varies with how far the drawer actually opened: try a few stances,
        # the searched one first (reachability in the planner's model), then fixed standoffs
        goals = [top[:2] - so * fwd[:2] + GRASP_LATERAL * left[:2]
                 for so in (ABOVE_STANDOFF, ABOVE_STANDOFF + 0.05, ABOVE_STANDOFF - 0.04)]
        g0 = top - np.array([0, 0, min(0.03, it.size[2])])
        stn = self._choose_stance(self.ik, g0 + np.array([0, 0, ABOVE_CLEARS[0]]), g0, candidates, th, ABOVE_STANDOFF,
                                  GRASP_LATERAL, above_postures, ABOVE_ELBOW_MAX, label=f"above {item}", yaws=(0.0,))
        if stn is not None and stn["raw"] < 0.05:
            goals.insert(0, np.array([stn["x"], stn["y"]]))
            above_postures = self._postures_from(stn["posture"], above_postures)
        for goal in goals:
            if np.linalg.norm(goal - self.wbc.base_pos[:2]) > 0.03:
                self._drive_to(goal[0], goal[1], th, tol=0.025, timeout=8.0)
            det = self.cam.detect()  # closer view
            if it.marker_id in det:
                top = det[it.marker_id].position
                centre = top - np.array([0, 0, it.size[2] + 0.001])
            grasp = top - np.array([0, 0, min(0.03, it.size[2])])
            for clear in ABOVE_CLEARS:  # lowest first: the lower the hand, the lower the elbow
                pre = grasp + np.array([0, 0, clear])
                axes, best = self._choose(pre, grasp, candidates, above_postures,
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
        miss = grasp - self.ik.palm_pose()[0]
        if e > GRASP_TOL and np.linalg.norm(miss[:2]) < STEP_CORRECT_MAX:
            # A few cm short (the feet landed off the stance): back up over the item, step by the
            # miss, look again, re-plan from there and descend once more
            self.move_hand(pre, axes, duration=0.8, q_ref=q_pre)
            self._step_by(miss[:2])
            det = self.cam.detect()
            if it.marker_id in det:
                top = det[it.marker_id].position
                centre = top - np.array([0, 0, it.size[2] + 0.001])
                grasp = top - np.array([0, 0, min(0.03, it.size[2])])
                pre = grasp + np.array([0, 0, clear])
            axes2, best2 = self._choose(pre, grasp, candidates, above_postures, multi_start=True, elbow_max=ABOVE_ELBOW_MAX)
            if best2 <= 0.05:
                axes = axes2
                q_pre, q_g = self._best_q
                e = self._reach("right", q_pre, pre, grasp, q_g, axes, duration=1.0)
                if DEBUG:
                    print(f"   above (after step): plan {best2:.3f} descent residual {e:.3f}")
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

    # ------------------------------------------------------------------ stance selection
    # Inverse reachability, online: humanoid/mobile-manipulation planners choose where to stand
    # from a reachability map before planning the arm (Burget & Bennewitz, ICRA 2015; Vahrenkamp
    # et al.). Here the map is replaced by asking the arm IK directly, in its own scratch copy of
    # the model with the base (and the measured posture) moved to each candidate stance. Nothing
    # moves in the simulation until the best stance is chosen.

    @property
    def _leg_qadr(self) -> np.ndarray:
        if not hasattr(self, "_leg_qadr_cache"):
            from g1_wbc import BODY_JOINTS
            jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in BODY_JOINTS[:15]]
            self._leg_qadr_cache = np.array([self.model.jnt_qposadr[j] for j in jids])
        return self._leg_qadr_cache

    def _body_shape(self) -> dict:
        """The current posture, heading-free: leg+waist joints, pelvis height and tilt."""
        th = self.wbc.base_yaw
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, np.array([np.cos(-th / 2), 0, 0, np.sin(-th / 2)]), np.asarray(self.wbc.base_quat, dtype=float))
        return dict(legs=self.data.qpos[self._leg_qadr].copy(), z=float(self.wbc.base_pos[2]), quat=rel, xy=np.zeros(2))

    def _calibrate_postures(self) -> None:
        """Measure the body in each lean/crouch posture once (~8 s at start-up), including how far
        the pelvis shifts from where it stood, so stances can be tried in those postures too."""
        p0, th = self.wbc.base_pos[:2].copy(), self.wbc.base_yaw
        c, s = np.cos(th), np.sin(th)
        for h, pitch in CALIB_POSTURES:
            self._set_posture(h, pitch)
            rec = self._body_shape()
            d = self.wbc.base_pos[:2] - p0
            rec["xy"] = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
            self._posture_cache[(h, pitch)] = rec
        self._set_posture(0.74, 0.0)

    def _set_virtual(self, ik: ArmIK, x: float, y: float, yaw: float, posture=(0.74, 0.0)) -> bool:
        """Put the planner's copy of the robot at a hypothetical stance: pelvis (x, y) as it would
        stand there, heading yaw, in a measured posture. False if that posture isn't measured."""
        rec = self._posture_cache.get(tuple(posture))
        if rec is None:
            return False
        c, s = np.cos(yaw), np.sin(yaw)
        px, py = x + c * rec["xy"][0] - s * rec["xy"][1], y + s * rec["xy"][0] + c * rec["xy"][1]
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), rec["quat"])
        ik.virtual_root = (np.array([px, py, rec["z"]]), quat)
        ik.virtual_legs = (self._leg_qadr, rec["legs"])
        return True

    @staticmethod
    def _clear_virtual(ik: ArmIK) -> None:
        ik.virtual_root = None
        ik.virtual_legs = None

    def _stance_blocked(self, ik: ArmIK) -> bool:
        """Would the body (legs, pelvis, torso; not the arms) intersect furniture at the
        virtual stance?"""
        if not hasattr(self, "_trunk_bodies"):
            pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            arms = self.ik.arm_bodies | self.left_ik.arm_bodies
            self._trunk_bodies = {b for b in range(self.model.nbody)
                                  if self.ik._is_descendant(b, pelvis) and b not in arms}
            self._robot_bodies = {b for b in range(self.model.nbody) if self.ik._is_descendant(b, pelvis)}
        s = ik.scratch
        ik._load(s)
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_collision(self.model, s)
        for i in range(s.ncon):
            c = s.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if self.model.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE or \
                    self.model.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if ((b1 in self._trunk_bodies and b2 not in self._robot_bodies) or
                    (b2 in self._trunk_bodies and b1 not in self._robot_bodies)) and c.dist < -0.005:
                return True
        return False

    def _quick_reach(self, ik: ArmIK, pre: np.ndarray, target: np.ndarray, cand, seeds, elbow_max) -> tuple:
        """Cheap reachability score of one hand-axis candidate at the virtual stance (no collision
        rows in the solve, collisions counted after). Returns (score, q_pre, q_t)."""
        best = (np.inf, None, None)
        for q0 in seeds:
            q_t, e_t = ik.solve(target, cand, q_init=q0, iters=80, avoid=False)
            q_p, e_p = ik.solve(pre, cand, q_init=q_t, iters=60, avoid=False, q_ref=q_t)
            if ik.elbow_too_high(q_p, q_t, limit=elbow_max):
                continue
            hits = len(ik.collisions(q_p) + ik.collisions(q_t))
            sc = e_t + e_p + 0.1 * hits + ELBOW_W * ik.elbow_cost(q_p, q_t) + 0.2 * max(0.0, 0.15 - ik.limit_margin(q_t))
            if sc < best[0]:
                best = (sc, q_p, q_t)
        return best

    def _choose_stance(self, ik: ArmIK, pre: np.ndarray, target: np.ndarray, candidates, yaw0: float,
                       standoff: float, lateral: float, postures, elbow_max: float | None = None,
                       label: str = "", yaws=(0.0, -0.15, 0.15)) -> dict | None:
        """Where to stand (and how to lean) to reach `target` (via `pre`) with `ik`'s arm.

        Stances are parameterized relative to the target, in the stance's own heading frame:
        target `f` ahead of the pelvis and `l` to the side (+ = the pelvis is left of it, i.e.
        right arm), heading yaw0 + dyaw. Three stages: (1) all stances x candidates in the first
        posture with quick solves, (2) the 6 best stances in the other measured postures,
        (3) the full collision-aware plan (_plan) for the 3 best, plus a robustness check: the
        same plan from 3 cm off in each direction (the feet never land exactly). Returns
        dict(x, y, yaw, posture, score, raw) or None."""
        if not STANCE_SEARCH:
            return None
        cur = self.wbc.base_pos[:2]
        seeds = [ik.home, self.arm_cmd.copy() if ik is self.ik else self.left_cmd.copy()]

        def pelvis(f, l, yaw):
            fw, lf = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            return np.asarray(target[:2]) - f * fw + l * lf

        def quick(x, y, yaw, posture):
            if not self._set_virtual(ik, x, y, yaw, posture):
                return np.inf, None
            try:
                if self._stance_blocked(ik):
                    return np.inf, None
                return min(((self._quick_reach(ik, pre, target, c, seeds, elbow_max)[0], i)
                            for i, c in enumerate(candidates)), key=lambda t: t[0])
            finally:
                self._clear_virtual(ik)

        stage1 = []
        for dyaw in yaws:
            for df in (0.0, -0.05, 0.05, -0.10, 0.10):
                for dl in (0.0, -0.05, 0.05, -0.10):
                    yaw = yaw0 + dyaw
                    x, y = pelvis(standoff + df, lateral + dl, yaw)
                    sc, ci = quick(x, y, yaw, postures[0])
                    # prefer small moves: stepping is slow and the feet land a few cm off anyway
                    pen = 0.15 * np.hypot(df, dl) + 0.05 * abs(dyaw) + 0.05 * np.hypot(x - cur[0], y - cur[1])
                    # a long reach (arm far out in front) also pulls the balance off: keep it near
                    pen += 0.4 * max(0.0, standoff + df - REACH_COMFORT)
                    stage1.append((sc + pen, x, y, yaw, tuple(postures[0]), pen))
        stage1.sort(key=lambda t: t[0])
        stage2 = [s[:5] for s in stage1[:6]]
        for _, x, y, yaw, _, pen in stage1[:6]:
            for posture in postures[1:]:
                sc, _ = quick(x, y, yaw, posture)
                # Leaning is a last resort: with the arm out in front, a pitched torso moves the
                # centre of mass forward and the balance policy shuffles the feet backwards
                # (measured: 0.36-0.47 m of drift during one reach at 0.3 rad / 0.5 m)
                lean = 0.06 + 0.15 * abs(posture[1]) + 0.5 * max(0.0, 0.74 - posture[0])
                stage2.append((sc + pen + lean, x, y, yaw, tuple(posture)))
        stage2 = sorted((s for s in stage2 if np.isfinite(s[0])), key=lambda t: t[0])[:3]
        best = None
        for sc, x, y, yaw, posture in stage2:
            if not self._set_virtual(ik, x, y, yaw, posture):
                continue
            try:
                cand, raw, q_p, q_t = self._plan(ik, seeds[1], pre, target, candidates, True, elbow_max)
                if cand is None:
                    continue
                worst = 0.0
                for ox, oy in ((0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)):
                    self._set_virtual(ik, x + ox, y + oy, yaw, posture)
                    s2, _, _ = self._quick_reach(ik, pre, target, cand, [q_t], elbow_max)
                    worst = max(worst, s2)
            finally:
                self._clear_virtual(ik)
            total = raw + 0.3 * max(0.0, worst - 0.02)
            if best is None or total < best["score"]:
                best = dict(x=float(x), y=float(y), yaw=float(yaw), posture=posture, score=float(total), raw=float(raw))
        if DEBUG:
            print(f"   stance[{label}]: {len(stage1)} tried, best {best and {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()}}")
        return best

    def _go_stance(self, st: dict) -> None:
        """Walk (standing) to a stance from _choose_stance, then take its posture."""
        if self.wbc.height_cmd < 0.73 or abs(self.wbc.rpy_cmd[1]) > 1e-3:
            self._set_posture(0.74, 0.0)
        if np.hypot(st["x"] - self.wbc.base_pos[0], st["y"] - self.wbc.base_pos[1]) > 0.025 or \
                abs(wrap(st["yaw"] - self.wbc.base_yaw)) > 0.05:
            self._drive_to(st["x"], st["y"], st["yaw"], tol=0.025, timeout=10.0)

    def _restance(self, ik: ArmIK, pre, target, candidates, standoff, lateral, postures, elbow_max, label,
                  yaw0: float | None = None, yaws=(0.0, -0.15, 0.15)):
        """The plan from where we ended up failed: search the stance again from here and step
        there. Returns the stance, or None if there is nowhere better (or the search is off)."""
        if ik is self.left_ik:
            self.left_arm_home(0.6)
        st = self._choose_stance(ik, pre, target, candidates, self.wbc.base_yaw if yaw0 is None else yaw0,
                                 standoff, lateral, postures, elbow_max, label, yaws)
        if st is None or st["raw"] > 0.08:
            return None
        self._go_stance(st)
        return st

    def _left_touching(self, weld: str) -> bool:
        """Is the left hand in contact with the part a weld would hold (e.g. the drawer)? Reads the
        contact list (a real robot: fingertip force / joint torque sensing)."""
        eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld)
        if eq < 0:
            return False
        part = self.model.eq_obj2id[eq]
        hand = self.left_ik.arm_bodies
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in hand and b2 == part) or (b2 in hand and b1 == part):
                return True
        return False

    def _body_rel(self, p: np.ndarray) -> np.ndarray:
        """World point -> (forward, left, up) from the pelvis, in its heading frame."""
        th = self.wbc.base_yaw
        d = np.asarray(p) - self.wbc.base_pos
        return np.array([np.cos(th) * d[0] + np.sin(th) * d[1], -np.sin(th) * d[0] + np.cos(th) * d[1], d[2]])

    def _step_by(self, d_xy: np.ndarray) -> None:
        """Shift the feet by d_xy (world), keeping the heading, standing up for the step."""
        self._set_posture(0.74, 0.0)
        p = self.wbc.base_pos[:2] + np.asarray(d_xy)
        if DEBUG:
            print(f"   step-correct by {np.round(d_xy, 3)}")
        self._drive_to(p[0], p[1], self.wbc.base_yaw, tol=0.02, timeout=6.0)

    @staticmethod
    def _postures_from(first, postures) -> list:
        """`postures` with `first` tried first."""
        return [tuple(first)] + [tuple(p) for p in postures if tuple(p) != tuple(first)]

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
        far = float(np.hypot(*(np.asarray(surface["pos"]) - self.wbc.base_pos[:2])))
        if far > 0.8:  # not at the tray (e.g. the walk there failed): keep holding, don't drop it on the floor
            return {"ok": False, "error": f"not at the delivery tray ({far:.2f} m away); still holding the {self.holding}"}
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
        spots = []
        for f, l in [(0.42, -0.14), (0.40, -0.02), (0.45, -0.24), (0.36, -0.20), (0.38, 0.06)]:
            cand = base + f * fwd + l * left
            # keep clear of the edge (arm tracking error is a few cm), but not so far that a 30 cm
            # tray leaves a 10 cm square for two items (the second one landed on the first)
            m = 0.065
            cand[0] = np.clip(cand[0], cx - hx + m, cx + hx - m)
            cand[1] = np.clip(cand[1], cy - hy + m, cy + hy - m)
            clear = min([np.hypot(*(cand[:2] - o[:2])) for o in others], default=1.0)
            cand[2] = surface["top"] + it.size[2] + self.hold_dz + 0.006  # palm height when the item rests on the table
            if all(np.hypot(*(cand[:2] - c[:2])) > 0.03 for _, c in spots):
                spots.append((clear, cand))
        spots.sort(key=lambda t: -t[0])  # farthest from what is already there first
        item = self.holding
        spot = None
        for clear, cand in spots[:3]:
            # Come in from above (clear of anything already on the table) and set down; only let go
            # once the item is actually down (the hand can be stopped above the tray by something
            # already on it: a cup released 10 cm up knocked the bottle off)
            self.move_hand(cand + np.array([0, 0, 0.17]), axes, duration=1.2)
            self.move_hand(cand, axes, duration=0.9)
            dz = float(self.ik.palm_pose()[0][2] - cand[2])
            if DEBUG:
                b = self.model.body(self.item_body[item]).id
                print(f"   hand_over: spot {cand.round(3)} palm {self.ik.palm_pose()[0].round(3)} item {self.data.xpos[b].round(3)} "
                      f"item z-axis {self.data.xmat[b].reshape(3, 3)[:, 2].round(2)} surface top {surface['top']:.3f} "
                      f"clear {clear:.2f} still {dz:.3f} m above")
            if dz < 0.07:  # a short drop is fine (the old code released here too); 10+ cm knocked things over
                spot = cand
                break
            self.move_hand(cand + np.array([0, 0, 0.17]), axes, duration=0.7)
        if spot is None:
            self.carry()
            return {"ok": False, "error": f"couldn't set the {item} down on the tray (blocked); still holding it"}
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
        if keep_reach is not None:
            return self._follow_paced(targets, axes_list, keep_reach)
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
                if DEBUG:
                    print(f"   follow: lost the path at {i}/{len(targets)}: err {err:.3f} "
                          f"target rel {self._body_rel(p).round(2)} v_back {v_back:.2f} hits {self.ik.collisions(self.arm_cmd)[:2]}")
                self.wbc.set_velocity(0, 0, 0)
                return i
        self.wbc.set_velocity(0, 0, 0)
        return len(targets)

    def _follow_paced(self, targets: list[np.ndarray], axes_list: list, keep_reach: float,
                      stall_s: float = 1.5) -> int:
        """Pull a part (drawer/door) along `targets`, paced by the feet (same scheme as the push
        close): the target only moves on while the hand keeps up, and the feet walk so the
        handle stays where it was in front of the body when grasped (never closer than
        keep_reach), holding the heading. The old fixed-timeline follow gave up after 0.2 s of
        lag: the drawer came at the body faster than GR00T started walking back, the arm folded
        and the pull stopped part-way (15-31 cm of 36). Returns how many targets were reached."""
        q = self.arm_cmd.copy()
        q_ref = q.copy()  # the grasp configuration: stay on its branch along the whole path
        yaw0 = self.wbc.base_yaw
        palm0 = self.ik.palm_pose()[0]
        # The weld held the handle wherever the palm was (up to GRASP_TOL off the handle point):
        # run the learned path from the palm's actual grip point, not the nominal one
        start = 2 * np.asarray(targets[0]) - np.asarray(targets[1]) if len(targets) > 1 else np.asarray(targets[0])
        targets = [np.asarray(t) + (palm0 - start) for t in targets]
        r0 = self._body_rel(palm0)
        hand_home = np.array([max(r0[0], keep_reach + 0.05), r0[1]])  # body-frame (fwd, left) to hold
        vel = np.zeros(2)
        k, i, stall = 0, 0, 0
        n = len(targets)
        while k < n and i < 4 * n:
            p = targets[k]
            i += 1
            rel = self._body_rel(p)[:2]
            e = rel - hand_home
            lim = np.array([BACKOFF_SPEED, 0.10])
            want = np.where(np.abs(e) > 0.02, np.clip(1.5 * e, -lim, lim), 0.0)
            want[0] = min(want[0], 0.05)  # mostly backwards; only creep forward
            vel += np.clip(want - vel, -0.5 * CONTROL_DT, 0.5 * CONTROL_DT)
            yerr = wrap(yaw0 - self.wbc.base_yaw)
            wz = float(np.clip(1.5 * yerr, -0.5, 0.5)) if abs(yerr) > 0.06 else 0.0
            v_cmd = np.where(np.abs(vel) > 0.05, vel, 0.0)
            self.wbc.set_velocity(float(v_cmd[0]), float(v_cmd[1]), wz)
            q, _ = self.ik.solve(p, axes_list[k], q_init=q, iters=25, q_ref=q_ref)
            self._set_right_arm(q)
            self._tick()
            err = float(np.linalg.norm(self.ik.palm_pose()[0] - p))
            # move the target on while the hand keeps up and the handle isn't crowding the body
            if err < 0.05 and rel[0] > keep_reach - 0.03:
                k += 1
                stall = 0
            else:
                stall += 1
            if stall * CONTROL_DT > stall_s:  # no progress (end stop, blocked, or lost the grip)
                if DEBUG:
                    print(f"   follow: stalled at {k}/{n}: err {err:.3f} target rel {np.round(rel, 2)} "
                          f"vel {np.round(v_cmd, 2)} hits {self.ik.collisions(self.arm_cmd)[:2]}")
                break
        self.wbc.set_velocity(0, 0, 0)
        if DEBUG:
            print(f"   follow: reached {k}/{n} in {i * CONTROL_DT:.1f} s")
        return k

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
        up = np.array([0, 0, 1.0])
        candidates = [[(0, into, 0.6), (1, up, 0.4)], [(0, into, 0.6), (1, -up, 0.4)],
                      [(0, into, 0.6), (2, up, 0.4)], [(0, into, 0.6), (2, -up, 0.4)]]
        # Pushing: fingertips (~11 cm past the palm site) on the drawer front, 8 cm above the bar
        push_off = (PUSH_FACE - FINGER_REACH) * into + np.array([0, 0, 0.08])
        searched_for = None  # handle position the stance was searched for
        st_off = None  # chosen stance, relative to the handle (world xy offset)
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
            if STANCE_SEARCH and (searched_for is None or np.linalg.norm(handle - searched_for) > 0.03):
                # where the LEFT arm reaches the contact best (the right arm keeps the item tucked)
                c_pt = handle + push_off if push else handle
                stc = self._choose_stance(self.left_ik, c_pt - 0.10 * into, c_pt, candidates, yaw_goal,
                                         float((c_pt - goal)[:2] @ fwd[:2]), float(-(c_pt - goal)[:2] @ left[:2]),
                                         [(0.74, 0.0)], label="close", yaws=(0.0,))
                searched_for = handle.copy()
                if stc is not None and stc["raw"] < 0.08:
                    st_off = np.array([stc["x"], stc["y"]]) - handle[:2]
            if st_off is not None:
                goal = handle.copy()
                goal[:2] = handle[:2] + st_off
            if DEBUG:
                print(f"   close stance: measured {measured} opened {opened:.3f} handle {handle.round(3)} "
                      f"base {self.wbc.base_pos.round(3)} goal {goal[:2].round(3)}")
            if np.linalg.norm(goal[:2] - self.wbc.base_pos[:2]) < (0.03 if st_off is not None else 0.06) and \
                    abs(wrap(yaw_goal - self.wbc.base_yaw)) < 0.1:
                break
            b = self.wbc.base_pos
            back = float((goal[:2] - b[:2]) @ fwd[:2])
            side = float((goal[:2] - b[:2]) @ left[:2])
            if back < -0.05 and abs(side) > 0.10:
                # Back off first, then sideways: a diagonal move sweeps the held item (tucked out
                # to the right) along the open drawer's front and catches it on the handle bar
                mid = b[:2] + back * fwd[:2]
                self._drive_to(mid[0], mid[1], yaw_goal, tol=0.04, timeout=6.0)
            arrived = self._drive_to(goal[0], goal[1], yaw_goal, tol=0.03, timeout=8.0)
            self._turn_to(yaw_goal)
            if DEBUG:
                print(f"   close drive: arrived {arrived} base {self.wbc.base_pos[:2].round(3)} "
                      f"handle rel {self._body_rel(handle).round(2)}")

        pre = handle - 0.12 * into
        contact = handle + push_off if push else handle
        if push:
            pre = contact - 0.10 * into
        axes, best, q_best, q_h = self._plan(self.left_ik, self.left_cmd.copy(), pre, contact, candidates)
        if best > 0.05:
            # not from here: search the stance again from where we are and step there
            f0 = float(self._body_rel(contact)[0])
            l0 = float(-self._body_rel(contact)[1])
            stn = self._restance(self.left_ik, pre, contact, candidates, f0, l0, [(0.74, 0.0)], None, "close again",
                                 yaw0=yaw_goal, yaws=(0.0,))
            if stn is not None:
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
        miss = contact - self.left_ik.palm_pose()[0]
        if e > 0.04 and np.linalg.norm(miss[:2]) < STEP_CORRECT_MAX and not self._left_touching(st["weld"]):
            # Short by a few cm (the feet landed off the stance): step by the miss, re-plan, again
            self.move_left_hand(pre - 0.05 * into, None, duration=0.6, settle=0.0)
            self._step_by(miss[:2])
            axes2, best2, q_b2, q_h2 = self._plan(self.left_ik, self.left_cmd.copy(), pre, contact, candidates)
            if best2 <= 0.05:
                axes = axes2
                e = self._reach("left", q_b2, pre, contact, q_h2, axes, duration=0.8)
        # Pushing: the fingertips stop on the drawer front (or its bar) a few cm before the
        # planned point, which is just what a push needs; the path below presses on from there.
        if push and e <= 0.09 and self._left_touching(st["weld"]):
            if DEBUG:
                print(f"   close: fingertips on the front {e:.3f} m short of the plan: pushing from here")
            e = 0.0
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


# Backup: abstracted grasp (cheat_grasp.py), only when G1_CHEAT_GRASP is set (1 / fallback)
if os.environ.get("G1_CHEAT_GRASP", "0").strip().lower() not in ("", "0", "false", "no"):
    import cheat_grasp
    cheat_grasp.install(AssistiveRobot)
