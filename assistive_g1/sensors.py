"""Onboard sensor suite for the Unitree G1 humanoid in MuJoCo.

Provides realistic, ground-truth-free onboard sensing for self-localization and mapping:
  1. `IMUSensor`: reads pelvis & torso `<gyro>` and `<accelerometer>` channels from
     `data.sensordata` and runs a 500 Hz SO(3) AHRS filter with stationary-gated
     accelerometer leveling to estimate `roll`, `pitch`, `gravity_body`, and `yaw_rate_world`.
  2. `LegOdometry`: uses ONLY the 29 joint encoders (`q`, `dq`) and IMU tilt (`roll`, `pitch`)
     on a private `MjData` kinematic buffer (with base XY/yaw zeroed) to estimate pelvis
     height `z_pelvis`, stance-foot contact weights, body-frame linear velocity `(vx, vy)`,
     and head-camera pose relative to the pelvis.
  3. `LidarSensor`: 360-degree multi-slice horizontal laser scanner using `mujoco.mj_multiRay`.
     Rays are cast in horizontal slices stabilized by `z_pelvis_est` so they hit vertical
     furniture/wall perimeters cleanly without false floor or flat-tabletop interior hits,
     excluding all robot self-geoms (and any held object).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from g1_wbc import BODY_JOINTS


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert (roll, pitch, yaw) ZYX Euler angles (rad) to (w, x, y, z) quaternion."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def rot_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3x3 rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


# =============================================================================
# 1. Onboard IMU + 500 Hz SO(3) AHRS Filter
# =============================================================================


@dataclass
class IMUReading:
    gyro_pelvis: np.ndarray     # (3,) rad/s in pelvis IMU frame
    accel_pelvis: np.ndarray    # (3,) m/s^2 proper acceleration in pelvis IMU frame
    gyro_torso: np.ndarray      # (3,) rad/s in torso IMU frame
    accel_torso: np.ndarray     # (3,) m/s^2 in torso IMU frame
    roll: float                 # estimated pelvis roll (rad)
    pitch: float                # estimated pelvis pitch (rad)
    yaw_rate_world: float       # angular velocity around world vertical +Z axis (rad/s)
    gravity_body: np.ndarray    # (3,) unit downward gravity vector in pelvis frame


class IMUSensor:
    """Reads G1 onboard IMU sensors from `data.sensordata` and estimates 3D attitude."""

    GRAVITY = 9.81

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, kp_acc: float = 0.05):
        self.model = model
        self.data = data
        self.kp_acc = kp_acc

        self._gyro_pelvis_adr = self._sensor_adr("imu-pelvis-angular-velocity")
        self._accel_pelvis_adr = self._sensor_adr("imu-pelvis-linear-acceleration")
        self._gyro_torso_adr = self._sensor_adr("imu-torso-angular-velocity")
        self._accel_torso_adr = self._sensor_adr("imu-torso-linear-acceleration")

        # 3x3 orientation estimate R (pelvis -> world, up to initial yaw)
        self.R = np.eye(3, dtype=np.float64)
        self.g_body = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.roll = 0.0
        self.pitch = 0.0
        self.integrated_yaw = 0.0
        self.last_reading: IMUReading | None = None

    def _sensor_adr(self, name: str) -> int | None:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return None
        return int(self.model.sensor_adr[sid])

    def reset(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> None:
        """Reset AHRS attitude state."""
        self.R = rot_from_euler(roll, pitch, yaw)
        self.g_body = -self.R[2, :].copy()
        self.roll = roll
        self.pitch = pitch
        self.integrated_yaw = yaw
        self.last_reading = None

    def read(self, dt: float) -> IMUReading:
        """Read IMU channels from `data.sensordata` and integrate attitude over `dt`."""
        sd = self.data.sensordata
        if self._gyro_pelvis_adr is not None:
            gyro_p = sd[self._gyro_pelvis_adr : self._gyro_pelvis_adr + 3].copy()
        else:
            gyro_p = np.zeros(3)

        if self._accel_pelvis_adr is not None:
            accel_p = sd[self._accel_pelvis_adr : self._accel_pelvis_adr + 3].copy()
        else:
            accel_p = np.array([0.0, 0.0, self.GRAVITY])

        if self._gyro_torso_adr is not None:
            gyro_t = sd[self._gyro_torso_adr : self._gyro_torso_adr + 3].copy()
        else:
            gyro_t = gyro_p.copy()

        if self._accel_torso_adr is not None:
            accel_t = sd[self._accel_torso_adr : self._accel_torso_adr + 3].copy()
        else:
            accel_t = accel_p.copy()

        w_norm = float(np.linalg.norm(gyro_p))
        acc_norm = float(np.linalg.norm(accel_p))

        if dt > 0.0:
            # Exact SO(3) exponential map update R_{k+1} = R_k @ exp([gyro_p]_x * dt)
            if w_norm > 1e-9:
                axis = gyro_p / w_norm
                angle = w_norm * dt
                K = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ],
                    dtype=np.float64,
                )
                dR = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
                self.R = self.R @ dR

            # Re-orthonormalize R periodically via SVD/Gram-Schmidt
            u0 = self.R[:, 0]
            u0 = u0 / max(float(np.linalg.norm(u0)), 1e-12)
            u1 = self.R[:, 1] - np.dot(self.R[:, 1], u0) * u0
            u1 = u1 / max(float(np.linalg.norm(u1)), 1e-12)
            u2 = np.cross(u0, u1)
            self.R = np.column_stack([u0, u1, u2])

            self.g_body = -self.R[2, :].copy()

            # Stationary-gated accelerometer leveling (prevents walking accelerations from tilting roll/pitch)
            if w_norm < 0.08 and abs(acc_norm - self.GRAVITY) < 0.20 and acc_norm > 1.0:
                g_meas = -accel_p / acc_norm
                alpha = min(1.0, self.kp_acc * dt)
                g_new = (1.0 - alpha) * self.g_body + alpha * g_meas
                g_new /= max(float(np.linalg.norm(g_new)), 1e-12)
                self.g_body = g_new

        gx, gy, gz = self.g_body
        self.pitch = float(math.atan2(gx, math.hypot(gy, gz)))
        self.roll = float(math.atan2(-gy, -gz))
        self.integrated_yaw = float(math.atan2(self.R[1, 0], self.R[0, 0]))

        # World vertical angular rate (yaw rate): project body omega onto -g_body (world +z)
        yaw_rate = float(-np.dot(self.g_body, gyro_p))

        reading = IMUReading(
            gyro_pelvis=gyro_p,
            accel_pelvis=accel_p,
            gyro_torso=gyro_t,
            accel_torso=accel_t,
            roll=self.roll,
            pitch=self.pitch,
            yaw_rate_world=yaw_rate,
            gravity_body=self.g_body.copy(),
        )
        self.last_reading = reading
        return reading


# =============================================================================
# 2. Proprioceptive Leg Odometry & Local Forward Kinematics
# =============================================================================


@dataclass
class LegOdometryReading:
    z_pelvis: float           # estimated pelvis height above floor (m)
    vel_body: np.ndarray      # (3,) estimated [vx, vy, wz] in horizontal body frame
    left_contact_weight: float
    right_contact_weight: float


class LegOdometry:
    """Proprioceptive pelvis height, velocity, and FK using ONLY joint encoders + IMU tilt.

    Uses a private `fk_data` buffer where the pelvis is placed at (0, 0, 0) with only
    (roll, pitch) from the IMU and the 29 encoder joint angles. Never reads the
    simulator's ground-truth root pose or velocity.
    """

    # Exact 4 sole contact sphere positions in ankle_roll_link frame from g1.xml (radius 5 mm, ~2 mm compression)
    FOOT_SPHERES = np.array(
        [
            [-0.05, 0.025, -0.032],
            [-0.05, -0.025, -0.032],
            [0.12, 0.030, -0.032],
            [0.12, -0.030, -0.032],
        ],
        dtype=np.float64,
    )

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.fk_data = mujoco.MjData(model)

        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._root_q = model.jnt_qposadr[root]
        self._root_v = model.jnt_dofadr[root]

        self._qpos_ids = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in BODY_JOINTS
        ])
        self._qvel_ids = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in BODY_JOINTS
        ])

        self._left_foot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self._right_foot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")

        self._jacp_l = np.zeros((3, model.nv))
        self._jacp_r = np.zeros((3, model.nv))

        self.z_pelvis = 0.74
        self.vel_filt = np.zeros(3, dtype=np.float64)
        self._prev_p_l: np.ndarray | None = None
        self._prev_p_r: np.ndarray | None = None

    def reset(self, z_init: float = 0.74) -> None:
        self.z_pelvis = z_init
        self.vel_filt[:] = 0.0
        self._prev_p_l = None
        self._prev_p_r = None

    def update(
        self,
        q_joints: np.ndarray,
        dq_joints: np.ndarray,
        imu: IMUReading,
        dt: float,
    ) -> LegOdometryReading:
        """Compute pelvis height and body-frame linear velocity from leg kinematics."""
        fk = self.fk_data
        qa = self._root_q
        # Place pelvis at origin in the gravity-aligned local frame (yaw = 0)
        fk.qpos[qa : qa + 3] = (0.0, 0.0, 0.0)
        fk.qpos[qa + 3 : qa + 7] = quat_from_euler(imu.roll, imu.pitch, 0.0)
        fk.qpos[self._qpos_ids] = q_joints
        fk.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, fk)
        mujoco.mj_comPos(self.model, fk)

        R_l = fk.xmat[self._left_foot_body].reshape(3, 3)
        R_r = fk.xmat[self._right_foot_body].reshape(3, 3)
        spheres_l = fk.xpos[self._left_foot_body] + (R_l @ self.FOOT_SPHERES.T).T
        spheres_r = fk.xpos[self._right_foot_body] + (R_r @ self.FOOT_SPHERES.T).T

        min_z_l = float(np.min(spheres_l[:, 2]))
        min_z_r = float(np.min(spheres_r[:, 2]))
        p_l = spheres_l[int(np.argmin(spheres_l[:, 2]))]
        p_r = spheres_r[int(np.argmin(spheres_r[:, 2]))]

        # The lower foot sole sphere is in contact with the floor (z_floor = 0)
        z_raw = float(-min(min_z_l, min_z_r))
        z_raw = float(np.clip(z_raw, 0.18, 0.90))
        alpha_z = min(1.0, dt / 0.025) if dt > 0 else 1.0
        self.z_pelvis += alpha_z * (z_raw - self.z_pelvis)

        # Stance weights: foot closer to the ground gets higher weight
        dz = min_z_r - min_z_l  # positive if left foot is lower (left in stance)
        scale = 0.012  # 1.2 cm transition band
        w_l = 1.0 / (1.0 + math.exp(-np.clip(dz / scale, -20.0, 20.0)))
        w_r = 1.0 - w_l

        # Foot linear Jacobians wrt joint velocities
        mujoco.mj_jac(self.model, fk, self._jacp_l, None, p_l, self._left_foot_body)
        mujoco.mj_jac(self.model, fk, self._jacp_r, None, p_r, self._right_foot_body)

        R_tilt = rot_from_euler(imu.roll, imu.pitch, 0.0)
        omega_local = R_tilt @ imu.gyro_pelvis

        v_foot_l = self._jacp_l[:, self._qvel_ids] @ dq_joints + np.cross(omega_local, p_l)
        v_foot_r = self._jacp_r[:, self._qvel_ids] @ dq_joints + np.cross(omega_local, p_r)
        v_jac = -(w_l * v_foot_l + w_r * v_foot_r)

        if self._prev_p_l is not None and self._prev_p_r is not None and dt > 1e-4:
            v_fd_l = -(p_l - self._prev_p_l) / dt
            v_fd_r = -(p_r - self._prev_p_r) / dt
            v_fd = w_l * v_fd_l + w_r * v_fd_r
            v_pelvis = 0.5 * v_jac + 0.5 * v_fd
        else:
            v_pelvis = v_jac

        self._prev_p_l = p_l.copy()
        self._prev_p_r = p_r.copy()

        raw_vel = np.array(
            [
                float(np.clip(v_pelvis[0], -1.2, 1.2)),
                float(np.clip(v_pelvis[1], -1.0, 1.0)),
                float(imu.yaw_rate_world),
            ],
            dtype=np.float64,
        )
        alpha_v = min(1.0, dt / 0.04) if dt > 0 else 1.0
        self.vel_filt += alpha_v * (raw_vel - self.vel_filt)

        return LegOdometryReading(
            z_pelvis=self.z_pelvis,
            vel_body=self.vel_filt.copy(),
            left_contact_weight=w_l,
            right_contact_weight=w_r,
        )

    def camera_pose_for_base(
        self,
        base_pos_3d: np.ndarray,
        roll: float,
        pitch: float,
        yaw: float,
        q_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute `(cam_xpos, cam_xmat)` using ONLY estimated base pose + encoder joints."""
        fk = self.fk_data
        qa = self._root_q
        fk.qpos[qa : qa + 3] = base_pos_3d
        fk.qpos[qa + 3 : qa + 7] = quat_from_euler(roll, pitch, yaw)
        fk.qpos[self._qpos_ids] = q_joints
        mujoco.mj_kinematics(self.model, fk)
        mujoco.mj_camlight(self.model, fk)  # cam_xpos/cam_xmat are only computed here (fix)
        return (
            fk.cam_xpos[self._cam_id].copy(),
            fk.cam_xmat[self._cam_id].reshape(3, 3).copy(),
        )


# =============================================================================
# 3. 360-Degree Multi-Slice Horizontal LiDAR Sensor
# =============================================================================


@dataclass
class LidarScan:
    azimuths: np.ndarray         # (N_az,) angles in [-pi, pi) in robot horizontal frame
    ranges: np.ndarray           # (N_az,) minimum horizontal distance per azimuth (inf if no hit)
    max_heights: np.ndarray      # (N_az,) maximum hit Z height above floor per azimuth
    points_local: np.ndarray     # (M, 2) valid 2D hit points (x_fwd, y_left) in robot horizontal frame
    points_3d_local: np.ndarray  # (K, 3) valid 3D hit points (x_fwd, y_left, z_above_floor)
    timestamp: float


class LidarSensor:
    """360-degree multi-slice horizontal laser scanner using `mujoco.mj_multiRay`.

    Casts horizontal rays across 8 height slices `z_levels` above the floor (stabilized
    using `z_pelvis_est` from `LegOdometry`). Because all rays are horizontal (`dir_z = 0`),
    they never intersect the floor or flat horizontal tabletop interiors, hitting only
    vertical obstacle perimeters (walls, table/shelf/counter edges, cupboards, drawers,
    clutter boxes, and walking humans).
    """

    DEFAULT_Z_LEVELS = (0.25, 0.38, 0.60, 0.73, 0.78, 0.83, 0.98, 1.45)

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        n_azimuth: int = 180,
        z_levels: tuple[float, ...] = DEFAULT_Z_LEVELS,
        max_range: float = 8.0,
        min_range: float = 0.18,
        noise_std: float = 0.001,
        seed: int = 0,
    ):
        self.model = model
        self.data = data
        self.n_azimuth = n_azimuth
        self.z_levels = np.asarray(z_levels, dtype=np.float64)
        self.n_slices = len(self.z_levels)
        self.max_range = max_range
        self.min_range = min_range
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.azimuths = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False, dtype=np.float64)
        self._horiz_dirs = np.column_stack([
            np.cos(self.azimuths),
            np.sin(self.azimuths),
            np.zeros(n_azimuth, dtype=np.float64),
        ])

        # Buffers for mj_multiRay per slice
        self._geomid = np.full(n_azimuth, -1, dtype=np.int32)
        self._dist = np.full(n_azimuth, -1.0, dtype=np.float64)
        # Exclude group 5 (where we temporarily stash robot & held-item geoms during raycast)
        self._geomgroup = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)

        # Identify all robot bodies (descendants of pelvis) so the LiDAR never hits the robot's own arms/legs
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._robot_bodies = self._collect_subtree(self.pelvis_id)
        self._robot_geoms = np.array(
            [g for g in range(model.ngeom) if model.geom_bodyid[g] in self._robot_bodies],
            dtype=np.int32,
        )
        self._held_body_id: int | None = None
        self.last_scan: LidarScan | None = None

    def _collect_subtree(self, root_id: int) -> set[int]:
        out = {root_id}
        for b in range(self.model.nbody):
            cur = b
            while cur > 0:
                if cur == root_id:
                    out.add(b)
                    break
                cur = self.model.body_parentid[cur]
        return out

    def set_held_body(self, body_name: str | None) -> None:
        """Exclude a held object from LiDAR rays so carrying an item doesn't blind the sensor."""
        if body_name is None:
            self._held_body_id = None
        else:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self._held_body_id = bid if bid >= 0 else None

    def scan(self, z_pelvis_est: float = 0.74) -> LidarScan:
        """Perform a 360-degree multi-slice horizontal raycast and return sensor-frame measurements only."""
        m, d = self.model, self.data

        p_pelvis = d.xpos[self.pelvis_id]
        R_pelvis = d.xmat[self.pelvis_id].reshape(3, 3)
        yaw_true = math.atan2(R_pelvis[1, 0], R_pelvis[0, 0])
        cy, sy = math.cos(yaw_true), math.sin(yaw_true)
        R_horiz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        world_dirs = (self._horiz_dirs @ R_horiz.T).ravel()

        exclude_geoms = self._robot_geoms
        if self._held_body_id is not None:
            held_geoms = np.where(m.geom_bodyid == self._held_body_id)[0].astype(np.int32)
            exclude_geoms = np.concatenate([exclude_geoms, held_geoms])

        saved_groups = m.geom_group[exclude_geoms].copy()
        m.geom_group[exclude_geoms] = 5

        dist_grid = np.full((self.n_slices, self.n_azimuth), np.inf, dtype=np.float64)
        try:
            for k, z_lvl in enumerate(self.z_levels):
                dz = float(z_lvl - z_pelvis_est)
                origin = p_pelvis + np.array([0.0, 0.0, dz], dtype=np.float64)
                try:
                    mujoco.mj_multiRay(
                        m,
                        d,
                        origin,
                        world_dirs,
                        self._geomgroup,
                        1,
                        self.pelvis_id,
                        self._geomid,
                        self._dist,
                        None,
                        self.n_azimuth,
                        self.max_range,
                    )
                except TypeError:
                    mujoco.mj_multiRay(
                        m,
                        d,
                        origin,
                        world_dirs,
                        self._geomgroup,
                        1,
                        self.pelvis_id,
                        self._geomid,
                        self._dist,
                        self.n_azimuth,
                        self.max_range,
                    )
                valid = (self._geomid >= 0) & (self._dist >= self.min_range) & (self._dist <= self.max_range)
                if np.any(valid):
                    r = self._dist[valid].copy()
                    if self.noise_std > 0.0:
                        r += self.rng.normal(0.0, self.noise_std, size=len(r))
                        r = np.clip(r, self.min_range, self.max_range)
                    dist_grid[k, valid] = r
        finally:
            m.geom_group[exclude_geoms] = saved_groups

        min_ranges = np.min(dist_grid, axis=0)
        valid_mask = np.isfinite(dist_grid)
        z_broadcast = np.broadcast_to(self.z_levels[:, None], dist_grid.shape)
        max_heights = np.where(np.isfinite(min_ranges), np.max(np.where(valid_mask, z_broadcast, -np.inf), axis=0), 0.0)

        hit_az = np.isfinite(min_ranges)
        points_2d = (
            np.column_stack([
                min_ranges[hit_az] * np.cos(self.azimuths[hit_az]),
                min_ranges[hit_az] * np.sin(self.azimuths[hit_az]),
            ])
            if np.any(hit_az)
            else np.zeros((0, 2), dtype=np.float64)
        )

        # Build 3D local points across all valid slice hits
        ks, az_idx = np.nonzero(valid_mask)
        if len(ks) > 0:
            r_valid = dist_grid[ks, az_idx]
            az_valid = self.azimuths[az_idx]
            pts_3d_valid = np.column_stack([
                r_valid * np.cos(az_valid),
                r_valid * np.sin(az_valid),
                self.z_levels[ks],
            ])
        else:
            pts_3d_valid = np.zeros((0, 3), dtype=np.float64)

        scan = LidarScan(
            azimuths=self.azimuths.copy(),
            ranges=min_ranges,
            max_heights=max_heights,
            points_local=points_2d,
            points_3d_local=pts_3d_valid,
            timestamp=float(d.time),
        )
        self.last_scan = scan
        return scan
