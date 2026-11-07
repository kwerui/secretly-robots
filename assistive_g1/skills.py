"""Robot skills the agent layer calls: go_to, look, pick, open_cupboard, hand_over.

Each skill returns a small result dict (ok + details) so an LLM planner can reason over
outcomes. Perception goes through the head camera only; the simulator's ground truth is
used for the robot's own base pose (localisation is out of scope for the hackathon).
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from g1_wbc import G1WBC, ARM_HOME, SIM_DT
from ik import ArmIK
from perception import HeadCamera, MarkerPose, ID_TO_ITEM
from articulation import ArticulationModel, rot_axis_angle
from scene import ITEMS, STATIONS, DOOR_MARKER, ANCHOR_MARKER, build_scene

CONTROL_DT = 0.02  # arm/nav update period (50 Hz)
# (pelvis height, torso pitch) tried in order when an item is hard to reach
POSTURES = [(0.74, 0.0), (0.74, 0.3), (0.68, 0.45), (0.60, 0.5)]
# deep squats for items near the floor (GR00T accepts pelvis heights down to ~0.2 m)
LOW_POSTURES = [(0.30, 0.4), (0.25, 0.4), (0.22, 0.4)]
LOW_ITEM_Z = 0.45  # items whose top is below this are picked from a squat (_pick_low)
STANDOFF = 0.40  # preferred horizontal pelvis-to-item distance for a grasp (m)
HUB = (0.35, 0.0)  # all station-to-station routes pass through here


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class AssistiveRobot:
    def __init__(self, placement=None, on_frame: Callable[["AssistiveRobot"], None] | None = None):
        self.model, self.data = build_scene(placement)
        self.wbc = G1WBC(self.model, self.data)
        self.ik = ArmIK(self.model, self.data, "right")
        self.cam = HeadCamera(self.model, self.data)
        self.on_frame = on_frame  # called every CONTROL_DT, e.g. to record video
        self.holding: str | None = None
        self.hold_up_axis = None
        self.hold_mode, self.hold_dz = "side", 0.0  # grasp type and palm height above item centre
        self.placed: list[np.ndarray] = []  # where this robot has set things down (side table)
        self.station = "home"
        self.arm_cmd = ARM_HOME[7:].copy()
        self.last_seen: dict[int, MarkerPose] = {}
        self.wbc.place(0.0, 0.0, 0.0)
        self.wait(1.0)

    # ------------------------------------------------------------------ basics
    def _tick(self) -> None:
        self.wbc.step(int(round(CONTROL_DT / SIM_DT)))
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
    def _drive_to(self, x: float, y: float, yaw: float | None, tol: float = 0.05, timeout: float = 25.0) -> bool:
        for _ in range(int(timeout / CONTROL_DT)):
            p = self.wbc.base_pos
            th = self.wbc.base_yaw
            dx, dy = x - p[0], y - p[1]
            dist = np.hypot(dx, dy)
            # error in the body frame
            ex = np.cos(th) * dx + np.sin(th) * dy
            ey = -np.sin(th) * dx + np.cos(th) * dy
            if dist > 0.35:
                head_err = wrap(np.arctan2(dy, dx) - th)
                wz = np.clip(2.5 * head_err, -1.2, 1.2)
                vx = 0.0 if abs(head_err) > 0.6 else np.clip(0.9 * dist, 0.15, 0.45) * np.cos(head_err)
                vy = 0.0
            else:
                yaw_err = wrap((yaw if yaw is not None else th) - th)
                if dist < tol and abs(yaw_err) < 0.06:
                    break
                vx = np.clip(1.2 * ex, -0.25, 0.25)
                vy = np.clip(1.2 * ey, -0.2, 0.2)
                wz = np.clip(2.5 * yaw_err, -1.0, 1.0)
                # keep the walking policy engaged (below 0.05 it switches to balance)
                if np.linalg.norm([vx, vy, wz]) < 0.06:
                    s = 0.06 / max(np.linalg.norm([vx, vy, wz]), 1e-6)
                    vx, vy, wz = vx * s, vy * s, wz * s
            self.wbc.set_velocity(vx, vy, wz)
            self._tick()
            if self.wbc.fallen():
                self.wbc.set_velocity(0, 0, 0)
                return False
        self.wbc.set_velocity(0, 0, 0)
        self.wait(0.8)
        return True

    def go_to(self, station: str) -> dict:
        """Walk to a named station (see scene.STATIONS)."""
        if station not in STATIONS:
            return {"ok": False, "error": f"unknown station {station}", "stations": list(STATIONS)}
        s = STATIONS[station]
        p = self.wbc.base_pos
        if np.hypot(p[0] - HUB[0], p[1] - HUB[1]) > 0.3 and station != self.station:
            if not self._drive_to(*HUB, None, tol=0.15):
                return {"ok": False, "error": "fell over"}
        ok = self._drive_to(s.x, s.y, s.yaw)
        if ok:
            self.station = station
        return {"ok": ok, "station": station, "pose": self.wbc.base_pos.round(2).tolist()}

    # ------------------------------------------------------------------ perception
    def look(self) -> dict:
        """Detect markers in the head camera. Returns item names -> world position."""
        det = self.cam.detect()
        self.last_seen = det
        items = {ID_TO_ITEM[m]: p.position.round(3).tolist() for m, p in det.items() if m in ID_TO_ITEM}
        extra = {}
        if DOOR_MARKER in det:
            extra["cupboard_door"] = det[DOOR_MARKER].position.round(3).tolist()
        if ANCHOR_MARKER in det:
            extra["cupboard_anchor"] = det[ANCHOR_MARKER].position.round(3).tolist()
        return {"ok": True, "station": self.station, "items": items, **extra}

    # ------------------------------------------------------------------ arm
    def move_hand(self, target: np.ndarray, axes=None, duration: float = 1.0, settle: float = 0.3) -> float:
        """Interpolate the palm to `target` (world) and track it with IK. Returns final error (m)."""
        start, _ = self.ik.palm_pose()
        n = max(1, int(duration / CONTROL_DT))
        q = self.arm_cmd.copy()
        for k in range(1, n + int(settle / CONTROL_DT) + 1):
            a = min(1.0, k / n)
            a = a * a * (3 - 2 * a)  # smoothstep
            goal = (1 - a) * start + a * np.asarray(target)
            q, _ = self.ik.solve(goal, axes, q_init=q, iters=15)
            self._set_right_arm(q)
            self._tick()
        p, _ = self.ik.palm_pose()
        return float(np.linalg.norm(p - target))

    def arm_home(self, duration: float = 1.0) -> None:
        q0 = self.arm_cmd.copy()
        n = int(duration / CONTROL_DT)
        for k in range(1, n + 1):
            a = k / n
            self._set_right_arm((1 - a) * q0 + a * ARM_HOME[7:])
            self._tick()

    def _item_collisions(self, item: str, on: bool) -> None:
        """While carried, an item passes through things like the hand does (no knock-overs)."""
        g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{item}_geom")
        self.model.geom_contype[g] = 2 if on else 0
        self.model.geom_conaffinity[g] = 2 if on else 0

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
    def pick(self, item: str) -> dict:
        """Side-grasp an item seen by the head camera at the current station."""
        it = ITEMS.get(item)
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
        grasp = centre + np.array([0, 0, 0.3 * it.size[2]])

        base = self.wbc.base_pos
        # Raise the hand to chest height, close to the body, before anything else so it can't
        # catch under a table edge while lifting or stepping in.
        self._raise_hand(grasp[2] + 0.06, toward=grasp)
        # Step in if the item is beyond comfortable reach (e.g. at the back of a cupboard).
        reach = float(np.hypot(*(grasp[:2] - base[:2])))
        if reach > STANDOFF + 0.04:
            d = (grasp[:2] - base[:2]) / reach
            goal = grasp[:2] - STANDOFF * d
            self._drive_to(goal[0], goal[1], self.wbc.base_yaw, tol=0.03, timeout=8.0)
            base = self.wbc.base_pos
        approach = grasp[:2] - base[:2]
        approach = np.append(approach / np.linalg.norm(approach), 0.0)
        pre = grasp - 0.14 * approach + np.array([0, 0, 0.04])  # pre-grasp: 14 cm back, 4 cm up

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

        # Come in along the approach line: far pre-grasp (outside any cupboard), pre-grasp, grasp.
        self.move_hand(grasp - 0.26 * approach + np.array([0, 0, 0.04]), axes, duration=1.0, settle=0.1)
        e1 = self.move_hand(pre, axes, duration=0.7)
        e2 = self.move_hand(grasp, axes, duration=0.9)
        if e2 > 0.04:
            self.move_hand(pre, axes, duration=0.8)
            self.arm_home()
            self._set_posture(0.74, 0.0)
            return {"ok": False, "error": f"could not reach {item} (residual {e2:.3f} m)", "pre_err": round(e1, 3)}
        self._weld(f"grasp_{item}", True)
        self._item_collisions(item, False)
        self.holding = item
        self.hold_up_axis = axes[1] if len(axes) > 1 else None  # keep this wrist roll while holding
        self.hold_mode, self.hold_dz = "side", grasp[2] - centre[2]
        self.move_hand(grasp + np.array([0, 0, 0.04]), axes, duration=0.6, settle=0.0)
        self.move_hand(grasp - 0.26 * approach + np.array([0, 0, 0.05]), axes, duration=0.9, settle=0.0)
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e2, 4)}

    def _choose(self, pre, grasp, candidates, postures):
        """Try postures in order; return the hand-axis candidate with the best collision-free IK."""
        axes, best = None, None
        for height, pitch in postures:
            self._set_posture(height, pitch)
            for cand in candidates:
                q_pre, e_pre = self.ik.solve(pre, cand, q_init=self.arm_cmd, iters=200)
                q_g, e_g = self.ik.solve(grasp, cand, q_init=q_pre, iters=200)
                hits = self.ik.collisions(q_pre) + self.ik.collisions(q_g)
                score = e_pre + e_g + 0.1 * len(hits)
                if best is None or score < best:
                    axes, best = cand, score
            if best <= 0.03:
                break
        return axes, best

    def _pick_low(self, item: str, top: np.ndarray, centre: np.ndarray) -> dict:
        """Floor pick: deep squat, item beside the right knee, top-down grasp.

        Straight ahead, the thighs come up into the arm's path in a deep squat, so the robot
        stands with the item ~0.2 m ahead and ~0.25 m to its right and reaches down outside
        the knee. Reach is ~0.1 m above the floor: fine for a bottle, marginal for a cup.
        """
        it = ITEMS[item]
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
        self.move_hand(pre + np.array([0, 0, 0.08]), axes, duration=1.0, settle=0.1)
        self.move_hand(pre, axes, duration=0.6)
        e = self.move_hand(grasp, axes, duration=0.9)
        if e > 0.04:
            self.move_hand(pre, axes, duration=0.6)
            self._set_posture(0.74, 0.0)
            self.arm_home()
            return {"ok": False, "error": f"could not reach {item} on the floor (residual {e:.3f} m)"}
        self._weld(f"grasp_{item}", True)
        self._item_collisions(item, False)
        self.holding = item
        self.hold_up_axis = axes[1] if len(axes) > 1 else None
        self.hold_mode, self.hold_dz = "top", grasp[2] - centre[2]
        self.move_hand(pre + np.array([0, 0, 0.1]), axes, duration=0.8, settle=0.0)
        self.carry()
        return {"ok": True, "item": item, "grasp_error_m": round(e, 4), "from": "floor"}

    def _hold_axes(self, fwd: np.ndarray) -> list:
        if self.hold_mode == "top":  # fingers down, as grasped
            axes = [(0, np.array([0, 0, -1.0]), 0.6)]
            if self.hold_up_axis is not None:
                axes.append((self.hold_up_axis[0], fwd if self.hold_up_axis[0] == 1 else self.hold_up_axis[1], 0.3))
            return axes
        axes = [(0, fwd, 0.6)]
        if self.hold_up_axis is not None:
            axes.append(self.hold_up_axis)
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
        left = np.array([-np.sin(th), np.cos(th), 0])
        target = self.wbc.base_pos + 0.28 * fwd - 0.18 * left + np.array([0, 0, 0.18])
        self.move_hand(target, self._hold_axes(fwd), duration=1.0)

    def hand_over(self) -> dict:
        """At the person: place the held item on the side table next to their armchair."""
        if self.holding is None:
            return {"ok": False, "error": "not holding anything"}
        from scene import SIDE_TABLE
        it = ITEMS[self.holding]
        th = self.wbc.base_yaw
        fwd = np.array([np.cos(th), np.sin(th), 0])
        axes = self._hold_axes(fwd)
        # Put it down on the side table: a reachable spot, as far as possible from anything
        # already standing there (seen by the head camera).
        base = self.wbc.base_pos
        left = np.array([-np.sin(th), np.cos(th), 0])
        cx, cy = SIDE_TABLE["pos"]
        hx, hy = SIDE_TABLE["half"]
        # (the carried item can block the camera, so also use where we put things before)
        others = [p.position for m, p in self.cam.detect().items() if m in ID_TO_ITEM and ID_TO_ITEM[m] != self.holding]
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
        spot[2] = SIDE_TABLE["top"] + it.size[2] + self.hold_dz + 0.006  # palm height when the item rests on the table
        # Come in from above (clear of anything already on the table), set down, leave upwards.
        self.move_hand(spot + np.array([0, 0, 0.17]), axes, duration=1.2)
        self.move_hand(spot, axes, duration=0.9)
        item = self.holding
        self._item_collisions(item, True)
        self._weld(f"grasp_{item}", False)
        self.holding = None
        self.placed.append(spot.copy())
        self.wait(0.3)
        self.move_hand(spot - 0.06 * fwd + np.array([0, 0, 0.14]), axes, duration=0.8)
        self.move_hand(spot - 0.16 * fwd + np.array([0, 0, 0.14]), None, duration=0.5, settle=0.0)
        self.arm_home()
        return {"ok": True, "handed_over": item, "placed_on": "side_table"}

    def follow(self, targets: list[np.ndarray], axes_list: list, dt: float = CONTROL_DT, max_err: float = 0.05) -> int:
        """Track a sequence of palm targets (one per control tick). Returns how many were reached."""
        q = self.arm_cmd.copy()
        bad = 0
        for i, (p, axes) in enumerate(zip(targets, axes_list)):
            q, _ = self.ik.solve(p, axes, q_init=q, iters=25)
            self._set_right_arm(q)
            self._tick()
            err = float(np.linalg.norm(self.ik.palm_pose()[0] - p))
            bad = bad + 1 if err > max_err else 0
            if bad >= 10:  # persistently off the path (out of reach or blocked)
                return i
        return len(targets)

    def open_cupboard(self, model: ArticulationModel, fraction: float = 1.0, seconds: float = 3.5) -> dict:
        """Open the cupboard the way the carer did, using the learned articulation model.

        The model lives in the anchor-marker frame, so the robot first finds the anchor with
        its head camera, maps hinge + grasp point into the world, grasps the handle and
        moves its hand along the learned arc.
        """
        det = self.cam.detect()
        if ANCHOR_MARKER not in det:
            return {"ok": False, "error": "cupboard anchor marker not visible"}
        A = det[ANCHOR_MARKER]
        Ra, pa = A.rotation, A.position
        handle = Ra @ np.array(model.grasp_start) + pa
        normal = Ra @ np.array(model.part_normal_start)  # door outward normal (towards the robot)
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
        into = -normal
        up = np.array([0, 0, 1.0])

        pre = handle - 0.12 * into
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
            return {"ok": False, "error": f"handle not reachable from here (score {best:.3f})"}

        self._raise_hand(handle[2], toward=handle)
        self.move_hand(pre, axes, duration=1.2)
        e = self.move_hand(handle, axes, duration=0.8)
        if e > 0.04:
            self.arm_home()
            return {"ok": False, "error": f"could not reach handle (residual {e:.3f} m)"}
        # Position-only grip on the handle: the door turns about its hinge while the wrist
        # keeps a comfortable orientation (like a loose grip around a bar handle).
        self._weld("grasp_door", True, torquescale=0.0)

        # Learned arc, mapped to the world
        c = Ra @ np.array(model.axis_point) + pa
        a = Ra @ np.array(model.axis_dir)
        n = int(seconds / CONTROL_DT)
        targets, axes_list = [], []
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)
            R = rot_axis_angle(a, fraction * model.amount * s)
            targets.append(R @ (handle - c) + c)
            # keep only a weak 'palm side up' preference; the fingers may turn freely
            axes_list.append([(axes[1][0], axes[1][1], 0.2)])
        reached = self.follow(targets, axes_list)
        self.wait(0.3)
        self._weld("grasp_door", False)

        # Let go and back off along the current finger direction (fingers = palm x-axis)
        p, R_palm = self.ik.palm_pose()
        self.move_hand(p - 0.10 * R_palm[:, 0], None, duration=0.7, settle=0.1)
        self.arm_home()
        opened = fraction * model.amount * (reached / n)
        return {"ok": reached > 0.6 * n, "opened_rad": round(opened, 3), "of_demo_rad": round(model.amount, 3)}

    def put_down(self, height_offset: float = 0.0) -> dict:
        """Release the held item where the hand is (e.g. back on a surface)."""
        if self.holding is None:
            return {"ok": False, "error": "not holding anything"}
        self._item_collisions(self.holding, True)
        self._weld(f"grasp_{self.holding}", False)
        item, self.holding = self.holding, None
        self.arm_home()
        return {"ok": True, "released": item}
