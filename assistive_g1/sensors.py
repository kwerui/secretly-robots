"""Onboard hardware sensors for the Unitree G1 in MuJoCo (zero ground-truth pose access).

Enforces a strict hardware sensor boundary so the robot never reads its world pose
(`data.qpos[0:7]`) or velocity (`data.qvel[0:6]`) from the simulator:

  1. IMUSensor: reads the pelvis and torso <gyro> and <accelerometer> sensors from
     `data.sensordata` (defined in `assets/g1/g1.xml`) and runs a 3D complementary
     AHRS filter on the gravity vector to estimate roll, pitch and world-yaw rate.
  2. LegOdometry: uses a local kinematic MjData with the pelvis fixed at the origin,
     tilted by the IMU's (roll, pitch) and driven solely by the 29 joint encoders
     (q, dq). Estimates pelvis height z above the floor and body-frame linear velocity
     (vx, vy) from stance-foot forward kinematics and Jacobians, plus camera/palm FK.
  3. LidarSensor: simulates a 360-degree multi-layer 3D LiDAR (Livox Mid-360 style)
     using `mujoco.mj_multiRay`. Rays are stabilized in the IMU's horizontal frame so
     walking sway, torso lean, or deep squats do not tilt the scan plane into the floor.
     Returns ONLY sensor-frame distances and local 2D/3D hit coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import mujoco
import numpy as np

from g1_wbc import BODY_JOINTS


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (roll, pitch, yaw in rad) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def rot_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3x3 rotation matrix from ZYX Euler angles (roll, pitch, yaw)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


# =============================================================================
# 1. Onboard IMU + Complementary AHRS Filter
# =============================================================================


@dataclass
class IMUReading:
    gyro_pelvis: np.ndarray   # (3,) rad/s in pelvis frame
    accel_pelvis: np.ndarray  # (3,) m/s^2 in pelvis frame (includes +g upward reaction)
    gyro_torso: np.ndarray    # (3,) rad/s in torso frame
    accel_torso: np.ndarray   # (3,) m/s^2 in torso frame
    roll: float               # rad, estimated pelvis roll from AHRS
    pitch: float              # rad, estimated pelvis pitch from AHRS
    yaw_rate_world: float     # rad/s, angular rate about world vertical (+z)
    gravity_body: np.ndarray  # (3,) unit downward gravity vector in pelvis frame


class IMUSensor:
    """Reads onboard IMU sensors from `data.sensordata` and estimates roll/pitch tilt."""

    GRAVITY = 9.81

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, kp_acc: float = 4.0):
        self.model = model
        self.data = data
        self.kp_acc = kp_acc

        self._gyro_pelvis_adr = self._sensor_adr("imu-pelvis-angular-velocity")
        self._accel_pelvis_adr = self._sensor_adr("imu-pelvis-linear-acceleration")
        self._gyro_torso_adr = self._sensor_adr("imu-torso-angular-velocity")
        self._accel_torso_adr = self._sensor_adr("imu-torso-linear-acceleration")

        # Downward unit gravity vector in the pelvis frame (0, 0, -1 when upright)
        self.g_body = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.roll = 0.0
        self.pitch = 0.0
        self.last_reading: IMUReading | None = None

    def _sensor_adr(self, name: str) -> int | None:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else None

    def reset(self, roll: float = 0.0, pitch: float = 0.0) -> None:
        R = rot_from_euler(roll, pitch, 0.0)
        self.g_body = R.T @ np.array([0.0, 0.0, -1.0])
        self.roll = roll
        self.pitch = pitch
        self.last_reading = None

    def read(self, dt: float = 0.0) -> IMUReading:
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

        # Complementary AHRS update on downward gravity vector in pelvis frame.
        # Accelerometer measures proper acceleration f = a_linear - g_body * G.
        # At rest or constant velocity, -accel_p / ||accel_p|| points along g_body.
        acc_norm = float(np.linalg.norm(accel_p))
        if dt > 0.0:
            # Integrate gyro: dg/dt = -omega x g
            g_pred = self.g_body - np.cross(gyro_p, self.g_body) * dt
            n_pred = np.linalg.norm(g_pred)
            if n_pred > 1e-9:
                g_pred /= n_pred

            if acc_norm > 1.0:
                g_meas = -accel_p / acc_norm
                # Down-weight accelerometer during footstep impact spikes
                dev = abs(acc_norm - self.GRAVITY) / self.GRAVITY
                trust = math.exp(-6.0 * dev * dev)
                alpha = min(1.0, self.kp_acc * trust * dt)
                self.g_body = (1.0 - alpha) * g_pred + alpha * g_meas
            else:
                self.g_body = g_pred
        elif acc_norm > 1.0:
            self.g_body = -accel_p / acc_norm

        gn = np.linalg.norm(self.g_body)
        if gn > 1e-9:
            self.g_body /= gn

        # For R = Rz(yaw) Ry(pitch) Rx(roll), R^T @ [0, 0, -1] = [sin(pitch), -cos(pitch)*sin(roll), -cos(pitch)*cos(roll)]
        gx, gy, gz = self.g_body
        self.pitch = float(math.atan2(gx, math.hypot(gy, gz)))
        self.roll = float(math.atan2(-gy, -gz))

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

    SOLE_OFFSET_Z = 0.035  # distance from ankle_roll_link origin to sole contact spheres

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

    def reset(self, z_init: float = 0.74) -> None:
        self.z_pelvis = z_init
        self.vel_filt[:] = 0.0

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

        # Foot positions in gravity-aligned local frame (pelvis at origin)
        R_l = fk.xmat[self._left_foot_body].reshape(3, 3)
        R_r = fk.xmat[self._right_foot_body].reshape(3, 3)
        p_l = fk.xpos[self._left_foot_body] + R_l @ np.array([0.02, 0.0, -self.SOLE_OFFSET_Z])
        p_r = fk.xpos[self._right_foot_body] + R_r @ np.array([0.02, 0.0, -self.SOLE_OFFSET_Z])

        # The lower foot is in stance on the floor (z_floor = 0), so pelvis height is -min(z_l, z_r)
        z_raw = float(-min(p_l[2], p_r[2]))
        z_raw = float(np.clip(z_raw, 0.18, 0.90))
        alpha_z = min(1.0, dt / 0.04) if dt > 0 else 1.0
        self.z_pelvis += alpha_z * (z_raw - self.z_pelvis)

        # Stance weights: foot closer to the ground gets higher weight (softmax over height difference)
        dz = p_r[2] - p_l[2]  # positive if left foot is lower (left in stance)
        scale = 0.015  # 1.5 cm height transition band
        w_l = 1.0 / (1.0 + math.exp(-np.clip(dz / scale, -20.0, 20.0)))
        w_r = 1.0 - w_l

        # Foot linear Jacobians wrt joint velocities
        mujoco.mj_jac(self.model, fk, self._jacp_l, None, p_l, self._left_foot_body)
        mujoco.mj_jac(self.model, fk, self._jacp_r, None, p_r, self._right_foot_body)

        R_tilt = rot_from_euler(imu.roll, imu.pitch, 0.0)
        omega_local = R_tilt @ imu.gyro_pelvis

        v_foot_l = self._jacp_l[:, self._qvel_ids] @ dq_joints + np.cross(omega_local, p_l)
        v_foot_r = self._jacp_r[:, self._qvel_ids] @ dq_joints + np.cross(omega_local, p_r)

        # Since stance foot is stationary on the ground, v_pelvis = -v_foot
        v_pelvis = -(w_l * v_foot_l + w_r * v_foot_r)
        vel_raw = np.array([v_pelvis[0], v_pelvis[1], imu.yaw_rate_world], dtype=np.float64)

        alpha_v = min(1.0, dt / 0.08) if dt > 0 else 1.0
        self.vel_filt += alpha_v * (vel_raw - self.vel_filt)

        return LegOdometryReading(
            z_pelvis=self.z_pelvis,
            vel_body=self.vel_filt.copy(),
            left_contact_weight=w_l,
            right_contact_weight=w_r,
        )

    def camera_pose_for_base(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        q_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute `head_cam` world (pos, 3x3 rot) via forward kinematics from estimated base pose."""
        fk = self.fk_data
        qa = self._root_q
        fk.qpos[qa : qa + 3] = base_pos
        fk.qpos[qa + 3 : qa + 7] = base_quat
        fk.qpos[self._qpos_ids] = q_joints
        mujoco.mj_kinematics(self.model, fk)
        return fk.cam_xpos[self._cam_id].copy(), fk.cam_xmat[self._cam_id].reshape(3, 3).copy()


# =============================================================================
# 3. 360-Degree Multi-Layer Stabilized LiDAR (mj_multiRay)
# =============================================================================


@dataclass
class LidarScan:
    azimuths: np.ndarray         # (N_az,) angles in [-pi, pi) in robot horizontal frame
    ranges: np.ndarray           # (N_az,) min horizontal distance (m) per azimuth (inf if clear)
    max_heights: np.ndarray      # (N_az,) max hit height z above floor per azimuth (m)
    points_local: np.ndarray     # (M, 2) valid (x, y) obstacle hits in robot horizontal frame
    points_3d_local: np.ndarray  # (K, 3) valid (x, y, z_floor) hits across all elevation rings
    timestamp: float


class LidarSensor:
    """360-degree multi-layer LiDAR sensor using `mujoco.mj_multiRay`.

    Simulates a torso/head-mounted 3D LiDAR (such as the Livox Mid-360 on the G1).
    Rays are cast in a pitch/roll-stabilized horizontal frame so body sway, leaning,
    or squatting does not tilt the scan into the floor.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        n_azimuth: int = 180,
        elevations_deg: tuple[float, ...] = (-24.0, -16.0, -10.0, -4.0, 2.0, 8.0, 16.0),
        max_range: float = 8.0,
        min_range: float = 0.18,
        z_offset_from_pelvis: float = 0.25,
        noise_std: float = 0.002,
        seed: int = 0,
    ):
        self.model = model
        self.data = data
        self.n_azimuth = n_azimuth
        self.elevations_rad = np.deg2rad(np.asarray(elevations_deg, dtype=np.float64))
        self.n_elev = len(self.elevations_rad)
        self.max_range = max_range
        self.min_range = min_range
        self.z_offset = z_offset_from_pelvis
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.azimuths = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False, dtype=np.float64)

        # Precompute unit ray directions in the robot's stabilized horizontal frame:
        # shape (n_elev, n_azimuth, 3) -> flattened to (N_rays, 3)
        ca = np.cos(self.azimuths)[None, :]   # (1, N_az)
        sa = np.sin(self.azimuths)[None, :]   # (1, N_az)
        ce = np.cos(self.elevations_rad)[:, None]  # (N_el, 1)
        se = np.sin(self.elevations_rad)[:, None]  # (N_el, 1)

        dirs = np.empty((self.n_elev, self.n_azimuth, 3), dtype=np.float64)
        dirs[:, :, 0] = ce * ca
        dirs[:, :, 1] = ce * sa
        dirs[:, :, 2] = se
        self._local_dirs = dirs.reshape(-1, 3)
        self.n_rays = self._local_dirs.shape[0]

        # Buffers for mj_multiRay
        self._geomid = np.full(self.n_rays, -1, dtype=np.int32)
        self._dist = np.full(self.n_rays, -1.0, dtype=np.float64)
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
        """Perform a 360-degree multi-layer raycast and return sensor-frame measurements only.

        The raycast itself queries the physics world along rays starting at the physical
        sensor mount, then projects all returned ranges strictly into the robot's local
        horizontal frame. Ground-truth world coordinates and geom IDs are never returned.
        """
        m, d = self.model, self.data

        # Physical sensor origin and true heading for the physics engine raycast
        p_pelvis = d.xpos[self.pelvis_id]
        R_pelvis = d.xmat[self.pelvis_id].reshape(3, 3)
        yaw_true = math.atan2(R_pelvis[1, 0], R_pelvis[0, 0])
        cy, sy = math.cos(yaw_true), math.sin(yaw_true)
        R_horiz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        sensor_origin = p_pelvis + np.array([0.0, 0.0, self.z_offset], dtype=np.float64)
        world_dirs = (self._local_dirs @ R_horiz.T).ravel()

        # Temporarily move robot geoms (and held object geoms) into group 5 so rays pass through
        exclude_geoms = self._robot_geoms
        if self._held_body_id is not None:
            held_geoms = np.where(m.geom_bodyid == self._held_body_id)[0].astype(np.int32)
            exclude_geoms = np.concatenate([exclude_geoms, held_geoms])

        saved_groups = m.geom_group[exclude_geoms].copy()
        m.geom_group[exclude_geoms] = 5
        try:
            mujoco.mj_multiRay(
                m,
                d,
                sensor_origin,
                world_dirs,
                self._geomgroup,
                1,
                self.pelvis_id,
                self._geomid,
                self._dist,
                None,
                self.n_rays,
                self.max_range,
            )
        except TypeError:
            # Compatibility with MuJoCo bindings that omit the normal argument
            mujoco.mj_multiRay(
                m,
                d,
                sensor_origin,
                world_dirs,
                self._geomgroup,
                1,
                self.pelvis_id,
                self._geomid,
                self._dist,
                self.n_rays,
                self.max_range,
            )
        finally:
            m.geom_group[exclude_geoms] = saved_groups

        raw_dist = self._dist.copy()
        valid = (self._geomid >= 0) & (raw_dist >= self.min_range) & (raw_dist <= self.max_range)
        if self.noise_std > 0.0 and np.any(valid):
            raw_dist[valid] += self.rng.normal(0.0, self.noise_std, size=int(np.sum(valid)))
            raw_dist[valid] = np.clip(raw_dist[valid], self.min_range, self.max_range)

        # Compute 3D hit coordinates in the robot's local horizontal frame
        # Height above floor is derived using estimated pelvis height z_pelvis_est (not ground truth)
        sensor_z_est = z_pelvis_est + self.z_offset
        pts_3d = self._local_dirs * raw_dist[:, None]
        pts_3d[:, 2] += sensor_z_est

        # Filter to obstacle height band: above floor (z >= 0.08 m) and below ceiling (z <= 1.70 m)
        obs_mask = valid & (pts_3d[:, 2] >= 0.08) & (pts_3d[:, 2] <= 1.70)
        pts_3d_valid = pts_3d[obs_mask]

        # Collapse vertical rings into 2D minimum horizontal range per azimuth
        horiz_dist = np.hypot(pts_3d[:, 0], pts_3d[:, 1]).reshape(self.n_elev, self.n_azimuth)
        z_grid = pts_3d[:, 2].reshape(self.n_elev, self.n_azimuth)
        mask_grid = obs_mask.reshape(self.n_elev, self.n_azimuth)

        horiz_masked = np.where(mask_grid, horiz_dist, np.inf)
        min_ranges = np.min(horiz_masked, axis=0)

        z_masked = np.where(mask_grid, z_grid, -np.inf)
        max_heights = np.where(np.isfinite(min_ranges), np.max(z_masked, axis=0), 0.0)

        hit_az = np.isfinite(min_ranges)
        points_2d = np.column_stack([
            min_ranges[hit_az] * np.cos(self.azimuths[hit_az]),
            min_ranges[hit_az] * np.sin(self.azimuths[hit_az]),
        ]) if np.any(hit_az) else np.zeros((0, 2), dtype=np.float64)

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
