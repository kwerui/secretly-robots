"""Self-localization for the G1 from its own sensors: no simulator ground-truth pose.

What the robot uses (sensors.py, from the Gemini-G1-Localization branch):
  * IMU (pelvis gyro + accelerometer, `data.sensordata`): attitude (roll, pitch) and yaw rate,
    integrated every physics step (500 Hz).
  * Leg odometry (29 joint encoders + IMU tilt): pelvis height and body-frame velocity from the
    stance foot, every control tick (50 Hz).
  * 360-degree LiDAR (8 horizontal slices, mj_multiRay): scan-to-map matching against the house's
    known static structure (walls and furniture, a 2D distance field built once at reset), every
    LIDAR_EVERY ticks.
  * Head RGB-D camera: depth-refined ArUco detections (perception.py) and the camera pose from
    the *estimated* base + joint encoders (forward kinematics), so items are placed in the map
    frame the robot believes it is in.

Fusion is a complementary filter: odometry + gyro predict, the LiDAR match corrects (blend
LIDAR_GAIN) when it is trustworthy (enough inliers, small residual, small jump). Moved
furniture, open drawers/doors, people and items are not in the static map: their LiDAR points
have large map distance and are trimmed as outliers.

What still reads the simulator: the sensors themselves (a LiDAR/camera has to be simulated from
the true state, as a real one is "computed" by physics), the GR00T policy's joint encoders,
and evaluation/logging (`true_*`). Robot decisions use only `pos`, `yaw`, `vel_body`.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from sensors import IMUSensor, LegOdometry, LidarSensor, quat_from_euler

LIDAR_EVERY = 5        # control ticks between scans (10 Hz)
LIDAR_GAIN = 0.5       # blend towards the scan-match pose per accepted scan
INLIER_DIST = 0.12     # m, map distance beyond which a LiDAR point is an outlier (not in the map)
DF_RES = 0.025         # m, distance-field grid


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class StaticMap:
    """2D distance field (m) to the house's static structure, from the geoms at reset."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, geoms, z_band=(0.2, 1.5),
                 bounds=(-4.2, 4.2, -3.2, 3.2), res: float = DF_RES):
        self.res = res
        self.x0, x1, self.y0, y1 = bounds
        self.gx = np.arange(self.x0, x1 + 1e-9, res)
        self.gy = np.arange(self.y0, y1 + 1e-9, res)
        GX, GY = np.meshgrid(self.gx, self.gy)
        df = np.full(GX.shape, 10.0)
        for g in geoms:
            typ = model.geom_type[g]
            if typ == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            pos, mat, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                half_z = np.abs(mat[2]) @ size
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                half_z = size[1] + (size[0] if typ == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            else:
                half_z = model.geom_rbound[g]
            if pos[2] - half_z > z_band[1] or pos[2] + half_z < z_band[0]:
                continue  # not at any LiDAR slice height
            dx, dy = GX - pos[0], GY - pos[1]
            if typ == mujoco.mjtGeom.mjGEOM_BOX and abs(mat[2, 2]) > 0.99:
                c, s = mat[0, 0], mat[1, 0]
                lx, ly = c * dx + s * dy, -s * dx + c * dy
                d = np.hypot(np.maximum(np.abs(lx) - size[0], 0), np.maximum(np.abs(ly) - size[1], 0))
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE,
                         mujoco.mjtGeom.mjGEOM_SPHERE) and abs(mat[2, 2]) > 0.99:
                d = np.maximum(np.hypot(dx, dy) - size[0], 0)
            elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE and abs(mat[2, 2]) < 0.01:
                ax = mat[:2, 2]
                t = np.clip(dx * ax[0] + dy * ax[1], -size[1], size[1])
                d = np.maximum(np.hypot(dx - t * ax[0], dy - t * ax[1]) - size[0], 0)
            else:
                d = np.maximum(np.hypot(dx, dy) - model.geom_rbound[g], 0)
            np.minimum(df, d, out=df)
        self.df = df
        self.gy_df, self.gx_df = np.gradient(df, res)

    def lookup(self, pts: np.ndarray):
        """Bilinear distance and gradient at world points (N, 2)."""
        fx = np.clip((pts[:, 0] - self.x0) / self.res, 0, len(self.gx) - 1.001)
        fy = np.clip((pts[:, 1] - self.y0) / self.res, 0, len(self.gy) - 1.001)
        i, j = fy.astype(int), fx.astype(int)
        ty, tx = fy - i, fx - j

        def bil(a):
            return ((1 - ty) * ((1 - tx) * a[i, j] + tx * a[i, j + 1])
                    + ty * ((1 - tx) * a[i + 1, j] + tx * a[i + 1, j + 1]))
        return bil(self.df), np.column_stack([bil(self.gx_df), bil(self.gy_df)])


class StateEstimator:
    """IMU + leg odometry + LiDAR scan matching -> (x, y, z, yaw), body velocity, attitude."""

    def __init__(self, env, model: mujoco.MjModel, data: mujoco.MjData, wbc, start_pose=None):
        self.model, self.data, self.wbc = model, data, wbc
        self.imu = IMUSensor(model, data)
        self.odo = LegOdometry(model)
        self.lidar = LidarSensor(model, data)
        geoms = [g for g in env.static_geoms if not self._is_free(model, model.geom_bodyid[g])]
        self.map = StaticMap(model, data, geoms)
        # Initial pose: where the robot was put down (a known start, like a dock). After this
        # nothing reads the simulator's base pose.
        if start_pose is None:
            p = data.qpos[wbc._root_q:wbc._root_q + 3]
            w, x, y, z = data.qpos[wbc._root_q + 3:wbc._root_q + 7]
            start_pose = (p[0], p[1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self.reset(*start_pose)
        self.stats = {"scans": 0, "accepted": 0, "last_inliers": 0, "last_rms": 0.0}

    @staticmethod
    def _is_free(model, b) -> bool:
        return bool(model.body_jntnum[b]) and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE

    def reset(self, x: float, y: float, yaw: float, z: float = 0.75) -> None:
        self.x, self.y, self.yaw, self.z = float(x), float(y), float(yaw), float(z)
        self.vel = np.zeros(3)
        self.imu.reset(0.0, 0.0, yaw)
        self.odo.reset(z)
        self._ticks = 0
        self._yaw_rate = 0.0

    # ---------------------------------------------------------------- per physics step (500 Hz)
    def imu_step(self, dt: float) -> None:
        r = self.imu.read(dt)
        self._yaw_rate = r.yaw_rate_world
        self.yaw = wrap(self.yaw + r.yaw_rate_world * dt)

    # ---------------------------------------------------------------- per control tick (50 Hz)
    def tick(self, dt: float) -> None:
        imu = self.imu.last_reading or self.imu.read(0.0)
        o = self.odo.update(self.wbc.q, self.wbc.dq, imu, dt)
        self.z = o.z_pelvis
        self.vel = o.vel_body.copy()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * o.vel_body[0] - s * o.vel_body[1]) * dt
        self.y += (s * o.vel_body[0] + c * o.vel_body[1]) * dt
        self._ticks += 1
        if self._ticks % LIDAR_EVERY == 0:
            self._lidar_correct()

    def _lidar_correct(self) -> None:
        scan = self.lidar.scan(self.z)
        pts = scan.points_3d_local[:, :2]
        if len(pts) < 30:
            return
        self.stats["scans"] += 1
        x, y, th = self.x, self.y, self.yaw
        w = None
        for it in range(8):  # Gauss-Newton on (x, y, yaw), trimmed + Huber-weighted
            c, s = math.cos(th), math.sin(th)
            pw = np.column_stack([c * pts[:, 0] - s * pts[:, 1] + x, s * pts[:, 0] + c * pts[:, 1] + y])
            d, g = self.map.lookup(pw)
            inl = d < (INLIER_DIST if it > 1 else 2.5 * INLIER_DIST)
            if inl.sum() < 30:
                return
            w = np.where(np.abs(d) < 0.03, 1.0, 0.03 / np.maximum(np.abs(d), 1e-6)) * inl
            dpdth = np.column_stack([-s * pts[:, 0] - c * pts[:, 1], c * pts[:, 0] - s * pts[:, 1]])
            J = np.column_stack([g[:, 0], g[:, 1], (g * dpdth).sum(1)])
            H = (J * w[:, None]).T @ J + np.diag([1e-3, 1e-3, 1e-2])
            delta = -np.linalg.solve(H, (J * w[:, None]).T @ d)
            x, y, th = x + delta[0], y + delta[1], th + delta[2]
            if np.linalg.norm(delta[:2]) < 1e-4 and abs(delta[2]) < 1e-4:
                break
        c, s = math.cos(th), math.sin(th)
        pw = np.column_stack([c * pts[:, 0] - s * pts[:, 1] + x, s * pts[:, 0] + c * pts[:, 1] + y])
        d, _ = self.map.lookup(pw)
        inl = d < INLIER_DIST
        rms = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else 1.0
        self.stats["last_inliers"], self.stats["last_rms"] = int(inl.sum()), rms
        jump = math.hypot(x - self.x, y - self.y)
        # Trust it: enough of the scan explained by the map, tight fit, plausible correction
        if inl.mean() > 0.4 and inl.sum() > 60 and rms < 0.03 and jump < 0.3 and abs(wrap(th - self.yaw)) < 0.2:
            self.stats["accepted"] += 1
            self.x += LIDAR_GAIN * (x - self.x)
            self.y += LIDAR_GAIN * (y - self.y)
            self.yaw = wrap(self.yaw + LIDAR_GAIN * wrap(th - self.yaw))

    # ---------------------------------------------------------------- outputs
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def quat(self) -> np.ndarray:
        return quat_from_euler(self.imu.roll, self.imu.pitch, self.yaw)

    @property
    def gravity_body(self) -> np.ndarray:
        return self.imu.g_body.copy()

    def camera_pose(self, cam: str = "head_cam"):
        """Head-camera pose from the estimated base + joint encoders (forward kinematics)."""
        return self.odo.camera_pose_for_base(self.pos, self.imu.roll, self.imu.pitch, self.yaw, self.wbc.q)

    def error(self) -> dict:
        """Estimate vs simulator truth (evaluation only)."""
        tp = self.wbc.true_base_pos
        return {"xy_m": round(float(np.hypot(tp[0] - self.x, tp[1] - self.y)), 4),
                "yaw_deg": round(float(np.degrees(abs(wrap(self.wbc.true_base_yaw - self.yaw)))), 2),
                "z_m": round(float(abs(tp[2] - self.z)), 4)}
