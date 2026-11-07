"""Smooth go-to-pose controller for the G1 on top of the GR00T velocity interface.

    PD on pose error  ->  desired (vx, vy, wz) in the body frame
    -> acceleration limits (slew rate) -> command sent to the walking policy

Why each piece:
  * PD with velocity feedback (D term on the measured body velocity) instead of P only:
    the robot decelerates before the goal instead of overshooting and hunting.
  * Acceleration limits: no velocity steps, so no lurch when starting and no stumble when
    a turn begins or ends.
  * Clean stop: GR00T runs a *walk* policy for |cmd| >= 0.05 and a *balance* policy below.
    The walk policy steps in place for any small non-zero command, so near the goal we
    either command a real velocity (>= MIN_CMD) or exactly zero, and once we have arrived
    we stay arrived (hysteresis) instead of dithering around the target.
  * Turn first, then walk: forward speed is scaled by cos^2 of the heading error and is
    zero while the heading error is large, so the robot never walks and turns hard at once.

Inputs are the robot pose (x, y, yaw) and body-frame velocity, so a localisation estimate
can replace the simulator's ground truth without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class NavParams:
    kp_lin: float = 1.0  # 1/s
    kd_lin: float = 0.4  # on measured body velocity
    kp_yaw: float = 1.8
    kd_yaw: float = 0.15
    v_max: float = 0.40  # m/s forward
    v_back: float = 0.15  # m/s backward
    vy_max: float = 0.15  # m/s sideways
    w_max: float = 1.0  # rad/s (commanded; GR00T tracks roughly half of it)
    a_max: float = 0.5  # m/s^2
    ay_max: float = 0.4
    alpha_max: float = 1.5  # rad/s^2
    far: float = 0.30  # beyond this, face the goal and walk; inside, holonomic fine approach
    walk_heading: float = 0.5  # rad: no forward motion while the heading error exceeds this
    pos_tol: float = 0.04  # m
    yaw_tol: float = 0.05  # rad
    settle_scale: float = 2.0  # accept 2x the tolerances once the PD output is tiny (no dithering)
    min_cmd: float = 0.08  # smallest non-zero command (below ~0.05 GR00T switches to balance)
    vel_tau: float = 0.5  # s, low-pass on measured velocity (the pelvis sways +-0.2 m/s every step)
    dwell: float = 0.6  # s inside the settle band -> arrived, even if the PD output isn't tiny


class Navigator:
    def __init__(self, params: NavParams | None = None):
        self.p = params or NavParams()
        self.cmd = np.zeros(3)
        self.arrived = False
        self.v_filt = np.zeros(3)
        self._in_band = 0.0

    def reset(self) -> None:
        self.cmd[:] = 0.0
        self.v_filt[:] = 0.0
        self.arrived = False
        self._in_band = 0.0

    def step(self, pose: tuple[float, float, float], vel_body: np.ndarray,
             goal: tuple[float, float, float | None], dt: float) -> np.ndarray:
        """One control tick. Returns the (vx, vy, wz) command to send; sets self.arrived."""
        p = self.p
        x, y, th = pose
        gx, gy, gyaw = goal
        dx, dy = gx - x, gy - y
        dist = float(np.hypot(dx, dy))
        ex = np.cos(th) * dx + np.sin(th) * dy  # goal in the body frame
        ey = -np.sin(th) * dx + np.cos(th) * dy
        # D term on low-passed velocity: raw pelvis velocity carries the gait sway
        self.v_filt += (np.asarray(vel_body) - self.v_filt) * min(1.0, dt / p.vel_tau)
        vx_m, vy_m, wz_m = self.v_filt

        if dist > p.far:
            # face the goal, walk forward
            yaw_err = wrap(np.arctan2(dy, dx) - th)
            align = max(0.0, np.cos(yaw_err)) ** 2 if abs(yaw_err) < p.walk_heading else 0.0
            vx = np.clip(p.kp_lin * dist - p.kd_lin * vx_m, 0.0, p.v_max) * align
            vy = 0.0
        else:
            # fine approach: holonomic, towards the goal pose
            yaw_err = wrap((gyaw if gyaw is not None else th) - th)
            vx = np.clip(p.kp_lin * ex - p.kd_lin * vx_m, -p.v_back, p.v_max)
            vy = np.clip(p.kp_lin * ey - p.kd_lin * vy_m, -p.vy_max, p.vy_max)
        wz = np.clip(p.kp_yaw * yaw_err - p.kd_yaw * wz_m, -p.w_max, p.w_max)
        desired = np.array([vx, vy, wz])

        # arrival with hysteresis
        if not self.arrived:
            inside = dist < p.pos_tol and abs(yaw_err) < p.yaw_tol
            in_band = dist < p.settle_scale * p.pos_tol and abs(yaw_err) < p.settle_scale * p.yaw_tol
            self._in_band = self._in_band + dt if in_band else 0.0
            settled = in_band and (np.linalg.norm(desired) < p.min_cmd or self._in_band > p.dwell)
            self.arrived = dist <= p.far and (inside or settled)
        elif dist > 3 * p.pos_tol or abs(yaw_err) > 3 * p.yaw_tol:
            self.arrived = False  # pushed away: resume
        if self.arrived:
            desired[:] = 0.0
        else:
            n = np.linalg.norm(desired)
            if 1e-9 < n < p.min_cmd:  # keep the walk policy engaged with a real command
                desired *= p.min_cmd / n

        # acceleration limits
        lim = np.array([p.a_max, p.ay_max, p.alpha_max]) * dt
        self.cmd += np.clip(desired - self.cmd, -lim, lim)
        if self.arrived and np.linalg.norm(self.cmd) < 0.05:
            self.cmd[:] = 0.0  # snap to zero so GR00T switches cleanly to its balance policy
        return self.cmd.copy()

    @property
    def stopped(self) -> bool:
        return self.arrived and not np.any(self.cmd)
