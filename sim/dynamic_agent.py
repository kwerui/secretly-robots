"""Dynamic human agent simulation: walks into the house and relocates an item.

Supports the dynamic challenge level (Mode 3):
- A realistic human walks along an entry trajectory into a room.
- Approaches the target item, picks it up, and moves it to an unexpected new location.
- Generates natural walking limb kinematics (arm swing, leg strides).
- Detects proximity clearance to the robot for `person_too_close` safety halts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def yaw_to_quat(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


@dataclass
class TrajectoryWaypoint:
    time: float
    pos: tuple[float, float, float]
    yaw_deg: float
    action: str  # "walk", "reach_pick", "carry", "reach_place", "walk_away", "idle"


# Standard dynamic relocation scenario:
# 1. Carer / family member enters kitchen from study doorway (t = 0.0 to 4.0s).
# 2. Reaches kitchen counter and picks up the water bottle (t = 4.0 to 6.0s).
# 3. Carries water bottle across to the dining table (t = 6.0 to 9.0s).
# 4. Sets bottle down on the dining table (t = 9.0 to 11.0s).
# 5. Steps back and observes (t = 11.0 to 15.0s).
DEFAULT_WAYPOINTS: list[TrajectoryWaypoint] = [
    TrajectoryWaypoint(0.0, (2.0, 1.2, 0.0), -90.0, "walk"),
    TrajectoryWaypoint(2.5, (2.0, -0.8, 0.0), -90.0, "walk"),
    TrajectoryWaypoint(4.5, (2.0, -2.1, 0.0), -90.0, "reach_pick"),
    TrajectoryWaypoint(6.5, (2.0, -2.1, 0.0), 45.0, "reach_pick"),
    TrajectoryWaypoint(8.5, (2.8, -1.2, 0.0), 30.0, "carry"),
    TrajectoryWaypoint(10.5, (3.1, -1.0, 0.0), 0.0, "reach_place"),
    TrajectoryWaypoint(12.5, (3.1, -1.0, 0.0), 0.0, "reach_place"),
    TrajectoryWaypoint(14.5, (2.8, -0.4, 0.0), 90.0, "walk_away"),
    TrajectoryWaypoint(18.0, (2.5, 0.5, 0.0), 90.0, "idle"),
]


class DynamicAgentController:
    """Controls the dynamic human agent's trajectory, limb animation, and item manipulation."""

    def __init__(
        self,
        waypoints: Sequence[TrajectoryWaypoint] | None = None,
        item_initial_pos: tuple[float, float, float] = (1.80, -2.62, 0.90),
        item_target_pos: tuple[float, float, float] = (3.10, -0.90, 0.75),
        pick_time: float = 5.5,
        place_time: float = 11.0,
    ):
        self.waypoints = list(waypoints or DEFAULT_WAYPOINTS)
        self.total_duration = self.waypoints[-1].time
        self.item_initial_pos = np.array(item_initial_pos, dtype=float)
        self.item_target_pos = np.array(item_target_pos, dtype=float)
        self.pick_time = pick_time
        self.place_time = place_time
        self.reset()

    def reset(self):
        self.sim_time = 0.0
        self.is_active = True
        self.item_held = False
        self.item_relocated = False

    def state_at(self, t: float) -> dict:
        """Evaluate agent pose, limb gait angles, and item location at time `t`."""
        t_clamped = min(max(t, 0.0), self.total_duration)

        # Interpolate between waypoints
        idx = 0
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].time <= t_clamped <= self.waypoints[i + 1].time:
                idx = i
                break
        else:
            idx = len(self.waypoints) - 2

        wp0, wp1 = self.waypoints[idx], self.waypoints[idx + 1]
        dt = wp1.time - wp0.time
        alpha = (t_clamped - wp0.time) / dt if dt > 1e-4 else 0.0
        # Constant walking speed along a segment; turns on the spot ease in and out
        s = alpha * alpha * (3.0 - 2.0 * alpha)

        p0 = np.array(wp0.pos, dtype=float)
        p1 = np.array(wp1.pos, dtype=float)
        pos = p0 + alpha * (p1 - p0)

        # Yaw interpolation
        y0 = math.radians(wp0.yaw_deg)
        y1 = math.radians(wp1.yaw_deg)
        # shortest angle
        dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
        yaw = y0 + s * dy
        quat = yaw_to_quat(yaw)

        # Walking gait animation (speed-dependent limb swings)
        speed = float(np.linalg.norm(p1[:2] - p0[:2])) / max(dt, 1e-4) if dt > 1e-4 else 0.0
        is_walking = speed > 0.05 and wp0.action in ("walk", "carry", "walk_away")
        stride_freq = 2.0 * math.pi * 1.5  # ~1.5 Hz walking cadence
        gait_phase = t_clamped * stride_freq if is_walking else 0.0

        leg_swing = math.sin(gait_phase) * 0.45 if is_walking else 0.0
        arm_swing = -math.sin(gait_phase) * 0.35 if is_walking else 0.0

        # Item holding & relocation logic
        if t_clamped < self.pick_time:
            item_pos = self.item_initial_pos.copy()
            carrying = False
            relocated = False
            right_arm_reach = 0.0
        elif t_clamped <= self.place_time:
            # Held in person's right hand in front/right of torso
            carrying = True
            relocated = False
            # Forward vector from yaw
            fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
            item_pos = pos + fwd * 0.35 + right * 0.20 + np.array([0.0, 0.0, 0.85])
            right_arm_reach = 0.40  # reaching forward holding item
        else:
            # Set down at target location
            carrying = False
            relocated = True
            item_pos = self.item_target_pos.copy()
            right_arm_reach = 0.0

        action = wp0.action if alpha < 0.5 else wp1.action

        return {
            "time": t_clamped,
            "pos": pos,
            "quat": quat,
            "yaw_deg": math.degrees(yaw),
            "speed": speed,
            "action": action,
            "carrying": carrying,
            "relocated": relocated,
            "item_pos": item_pos,
            "leg_swing": leg_swing,
            "arm_swing": arm_swing,
            "right_arm_reach": right_arm_reach,
        }

    def check_robot_clearance(self, robot_xy: tuple[float, float] | np.ndarray, t: float, clearance_threshold: float = 1.0) -> tuple[bool, float]:
        """True if the dynamic person is within clearance_threshold of the robot."""
        state = self.state_at(t)
        p_xy = state["pos"][:2]
        r_xy = np.array(robot_xy[:2], dtype=float)
        dist = float(np.linalg.norm(p_xy - r_xy))
        too_close = dist < clearance_threshold
        return too_close, dist
