"""Hybrid two-layer localization and dynamic world mapping for the Unitree G1.

Architecture:
  * Layer 1 (Static Structural Localization):
    - `StaticStructuralMap`: builds a 2 cm Euclidean Distance Field (SDF) and gradients
      from permanent room structure (walls and fixed counters/fixtures), plus static
      ArUco anchor poses.
    - `G1Localizer`: 50 Hz Extended Kalman Filter (EKF) fusing onboard IMU AHRS,
      proprioceptive leg-kinematic odometry, Trimmed Scan-to-SDF Gauss-Newton LiDAR
      registration (with outlier rejection so moved furniture/clutter/people never
      corrupt the pose estimate), and visual ArUco landmark updates.

  * Layer 2 (Live Dynamic World Model):
    - `DynamicWorldMap`: updates a live 5 cm log-odds occupancy grid from LiDAR rays
      (clearing old space when furniture moves and marking new obstacles/clutter),
      clusters non-static LiDAR outliers to track moved furniture and dynamic walking
      people (`person_too_close` safety trigger), and plans collision-free A* paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
import cv2
import mujoco
import numpy as np

from sensors import (
    IMUSensor,
    IMUReading,
    LegOdometry,
    LidarScan,
    LidarSensor,
    quat_from_euler,
    rot_from_euler,
)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# =============================================================================
# Layer 1A: Static Structural Reference Map (Walls & Fixed Fixtures)
# =============================================================================


class StaticStructuralMap:
    """2D Euclidean Distance Field of permanent static structure for scan matching.

    Automatically inspects `MjModel` at initialization using a clean nominal `MjData`
    so runtime shuffling, clutter, doors, drawers, or dynamic people are never baked
    into the static reference map.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        bounds: tuple[float, float, float, float] = (-4.5, -3.5, 4.5, 3.5),
        res: float = 0.02,
        z_min: float = 0.08,
        z_max: float = 1.50,
    ):
        self.model = model
        self.x_min, self.y_min, self.x_max, self.y_max = bounds
        self.res = res
        self.nx = int(round((self.x_max - self.x_min) / res)) + 1
        self.ny = int(round((self.y_max - self.y_min) / res)) + 1
        self.xs = np.linspace(self.x_min, self.x_max, self.nx)
        self.ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.GX, self.GY = np.meshgrid(self.xs, self.ys)

        # Evaluate nominal model kinematics on a clean MjData
        nom_data = mujoco.MjData(model)
        mujoco.mj_forward(model, nom_data)

        self.has_walls = any(
            (model.geom(g).name or "").startswith("wall_") for g in range(model.ngeom)
        )
        self.static_geoms = self._select_static_geoms(model, nom_data, z_min, z_max)
        self.static_anchors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.movable_anchors: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        self._extract_anchors(model, nom_data)

        # Rasterize static structure occupancy and extract surface boundaries for SDF
        occ = self._rasterize_geoms(model, nom_data, self.static_geoms, z_min, z_max)
        self.occ_grid = occ

        if np.any(occ):
            occ_u8 = occ.astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(occ_u8, kernel, iterations=1)
            boundary = occ_u8 & (~eroded)
            # Distance transform expects 0 at target (boundary) cells and non-zero elsewhere
            inv_boundary = np.where(boundary > 0, 0, 1).astype(np.uint8)
            dist_px = cv2.distanceTransform(inv_boundary, cv2.DIST_L2, 5)
            self.dist_grid = dist_px.astype(np.float64) * res
        else:
            self.dist_grid = np.full((self.ny, self.nx), 10.0, dtype=np.float64)

        # Spatial gradients d(dist)/dx and d(dist)/dy for Gauss-Newton scan matching
        gy, gx = np.gradient(self.dist_grid, res, res)
        self.grad_x = gx
        self.grad_y = gy

    def _select_static_geoms(
        self,
        model: mujoco.MjModel,
        nom_data: mujoco.MjData,
        z_min: float,
        z_max: float,
    ) -> list[int]:
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_bodies = set()
        if pelvis >= 0:
            for b in range(model.nbody):
                cur = b
                while cur > 0:
                    if cur == pelvis:
                        robot_bodies.add(b)
                        break
                    cur = model.body_parentid[cur]

        static_geoms = []
        for g in range(model.ngeom):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            gname = model.geom(g).name or ""
            if gname == "floor" or gname.startswith("floor_") or gname.startswith("living_rug"):
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue

            b = int(model.geom_bodyid[g])
            if b in robot_bodies:
                continue
            if model.body_mocapid[b] >= 0:
                continue  # mocap clutter obstacles & dynamic person

            # Skip any body that has a free joint (items) or hinge/slide joint (doors/drawers)
            has_joint = False
            cur = b
            while cur > 0:
                if model.body_jntnum[cur] > 0:
                    has_joint = True
                    break
                cur = model.body_parentid[cur]
            if has_joint:
                continue

            bname = model.body(b).name or ""
            if bname in ("user", "person", "person_dynamic"):
                continue
            # In HouseEnv (where walls exist), `furn_*` bodies are movable in Mode.SHUFFLED,
            # so exclude them from the permanent structural map!
            if self.has_walls and bname.startswith("furn_"):
                continue

            static_geoms.append(g)
        return static_geoms

    def _extract_anchors(self, model: mujoco.MjModel, nom_data: mujoco.MjData) -> None:
        """Find ArUco marker geoms on static or movable furniture bodies."""
        for g in range(model.ngeom):
            gname = model.geom(g).name or ""
            mat_id = model.geom_matid[g]
            mat_name = model.material(mat_id).name if mat_id >= 0 else ""
            marker_id = None
            for token in (gname, mat_name):
                if "marker_" in token or "aruco_" in token:
                    tail = token.split("_")[-1]
                    if tail.isdigit():
                        marker_id = int(tail)
                        break
            if marker_id is None:
                continue

            b = int(model.geom_bodyid[g])
            bname = model.body(b).name or ""
            # Check if attached to a free/hinged/sliding body (like an item, door, or drawer)
            has_joint = False
            cur = b
            while cur > 0:
                if model.body_jntnum[cur] > 0:
                    has_joint = True
                    break
                cur = model.body_parentid[cur]
            if has_joint or model.body_mocapid[b] >= 0:
                continue

            pos = nom_data.geom_xpos[g].copy()
            rot = nom_data.geom_xmat[g].reshape(3, 3).copy()
            if bname.startswith("furn_"):
                # Movable furniture anchor: store relative to furniture body frame
                b_pos = nom_data.xpos[b].copy()
                b_rot = nom_data.xmat[b].reshape(3, 3).copy()
                rel_p = b_rot.T @ (pos - b_pos)
                rel_R = b_rot.T @ rot
                self.movable_anchors[marker_id] = (bname.removeprefix("furn_"), rel_p, rel_R)
            else:
                self.static_anchors[marker_id] = (pos, rot)

    def _rasterize_geoms(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geoms: list[int],
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        occ = np.zeros((self.ny, self.nx), dtype=bool)
        for g in geoms:
            pos = data.geom_xpos[g]
            mat = data.geom_xmat[g].reshape(3, 3)
            size = model.geom_size[g]
            typ = model.geom_type[g]

            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                half_z = float(np.abs(mat[2]) @ size)
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                half_z = float(size[1] + (size[0] if typ == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0))
            else:
                half_z = float(model.geom_rbound[g])

            if pos[2] + half_z < z_min or pos[2] - half_z > z_max:
                continue

            dx = self.GX - pos[0]
            dy = self.GY - pos[1]
            if typ == mujoco.mjtGeom.mjGEOM_BOX and abs(mat[2, 2]) > 0.95:
                c, s = mat[0, 0], mat[1, 0]
                lx = c * dx + s * dy
                ly = -s * dx + c * dy
                occ |= (np.abs(lx) <= size[0]) & (np.abs(ly) <= size[1])
            elif typ in (
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_SPHERE,
            ):
                occ |= np.hypot(dx, dy) <= size[0]
        return occ

    def query_dist_and_grad(
        self, pts_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (dist, grad_x, grad_y, in_bounds) for (N, 2) world points."""
        if len(pts_xy) == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty, np.zeros(0, dtype=bool)

        fx = (pts_xy[:, 0] - self.x_min) / self.res
        fy = (pts_xy[:, 1] - self.y_min) / self.res
        ix = np.rint(fx).astype(np.int32)
        iy = np.rint(fy).astype(np.int32)
        in_bounds = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

        ix_c = np.clip(ix, 0, self.nx - 1)
        iy_c = np.clip(iy, 0, self.ny - 1)
        d = self.dist_grid[iy_c, ix_c]
        gx = self.grad_x[iy_c, ix_c]
        gy = self.grad_y[iy_c, ix_c]
        d = np.where(in_bounds, d, 10.0)
        return d, gx, gy, in_bounds


# =============================================================================
# Layer 2: Live Dynamic World Map (Clutter, Moved Furniture, Dynamic Person)
# =============================================================================


@dataclass
class TrackedObstacle:
    cluster_id: int
    pos: np.ndarray           # (2,) world (x, y) centroid
    vel: np.ndarray           # (2,) estimated world (vx, vy) m/s
    radius: float             # m
    max_z: float              # m, max height above floor
    last_seen: float          # timestamp (s)
    hits: int = 1
    is_person: bool = False


class DynamicWorldMap:
    """Live occupancy grid and dynamic object tracker driven by LiDAR + RGB-D.

    Handles:
      1. Log-odds occupancy updates: clears old space when movable furniture shifts
         and marks new obstacle cells (moved tables, floor clutter boxes/baskets, doors).
      2. Outlier clustering & tracking: separates static structure from novel objects,
         tracks moving clusters (dynamic person) and triggers `person_too_close`.
      3. Furniture pose estimation: tracks relocated furniture via ArUco anchors or LiDAR.
      4. A* path planning on the robot-inflated live grid.
    """

    def __init__(
        self,
        static_map: StaticStructuralMap,
        bounds: tuple[float, float, float, float] = (-4.5, -3.5, 4.5, 3.5),
        res: float = 0.05,
        person_clearance: float = 1.10,
    ):
        self.static_map = static_map
        self.x_min, self.y_min, self.x_max, self.y_max = bounds
        self.res = res
        self.person_clearance = person_clearance

        self.nx = int(round((self.x_max - self.x_min) / res)) + 1
        self.ny = int(round((self.y_max - self.y_min) / res)) + 1
        self.xs = np.linspace(self.x_min, self.x_max, self.nx)
        self.ys = np.linspace(self.y_min, self.y_max, self.ny)
        self.GX, self.GY = np.meshgrid(self.xs, self.ys)

        self.log_odds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.tracks: list[TrackedObstacle] = []
        self._next_id = 1
        self.furniture_poses: dict[str, tuple[float, float, float]] = {}
        self.person_marker_pos: tuple[np.ndarray, float] | None = None
        self.reset()

    def reset(self) -> None:
        self.log_odds.fill(0.0)
        # Seed permanent static structure at moderate positive log-odds so rays can still
        # clear movable objects if they shift in scenes without walls
        for iy in range(self.ny):
            y = self.ys[iy]
            for ix in range(self.nx):
                x = self.xs[ix]
                sx = int(np.clip(round((x - self.static_map.x_min) / self.static_map.res), 0, self.static_map.nx - 1))
                sy = int(np.clip(round((y - self.static_map.y_min) / self.static_map.res), 0, self.static_map.ny - 1))
                if self.static_map.occ_grid[sy, sx]:
                    self.log_odds[iy, ix] = 2.0
        self.tracks.clear()
        self.furniture_poses.clear()
        self.person_marker_pos = None

    def _xy_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(round((x - self.x_min) / self.res), 0, self.nx - 1))
        iy = int(np.clip(round((y - self.y_min) / self.res), 0, self.ny - 1))
        return iy, ix

    def update_from_scan(
        self,
        robot_pose: tuple[float, float, float],
        scan: LidarScan,
        outlier_pts_world: np.ndarray,
        outlier_heights: np.ndarray,
    ) -> None:
        """Update log-odds occupancy grid and dynamic obstacle tracks from a LiDAR scan."""
        rx, ry, ryaw = robot_pose
        cy, sy = math.cos(ryaw), math.sin(ryaw)

        # 1. Log-odds ray updates (subsample azimuths by 2 for fast vectorized grid update)
        step_az = 2
        az_sub = scan.azimuths[::step_az]
        rng_sub = scan.ranges[::step_az]
        max_trace = min(scan.max_heights.shape[0] and 4.5 or 4.5, 4.5)

        for az, r in zip(az_sub, rng_sub):
            world_ang = ryaw + az
            ca, sa = math.cos(world_ang), math.sin(world_ang)
            hit = np.isfinite(r) and r < max_trace
            trace_len = float(r if hit else max_trace)
            # Sample along free ray
            n_steps = max(2, int(trace_len / self.res))
            d_samples = np.linspace(0.20, max(0.20, trace_len - self.res), n_steps)
            xs = rx + d_samples * ca
            ys = ry + d_samples * sa
            ix = np.rint((xs - self.x_min) / self.res).astype(np.int32)
            iy = np.rint((ys - self.y_min) / self.res).astype(np.int32)
            valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            self.log_odds[iy[valid], ix[valid]] = np.clip(
                self.log_odds[iy[valid], ix[valid]] - 0.25, -4.0, 4.0
            )
            if hit:
                hx = rx + r * ca
                hy = ry + r * sa
                hiy, hix = self._xy_to_cell(hx, hy)
                self.log_odds[hiy, hix] = min(4.0, float(self.log_odds[hiy, hix]) + 0.70)

        # 2. Cluster non-static outlier points (clutter, moved furniture, dynamic person)
        clusters = self._cluster_outliers(outlier_pts_world, outlier_heights)
        self._update_tracks(clusters, scan.timestamp)

    def _cluster_outliers(
        self, pts: np.ndarray, heights: np.ndarray, radius: float = 0.25
    ) -> list[tuple[np.ndarray, float, float]]:
        """Return list of (centroid_xy, cluster_radius, max_z) for contiguous outlier groups."""
        if len(pts) == 0:
            return []
        clusters: list[list[int]] = [[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[i - 1]) <= radius:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        # Merge first and last if wrap-around at +-pi
        if len(clusters) > 1 and np.linalg.norm(pts[clusters[0][0]] - pts[clusters[-1][-1]]) <= radius:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()

        out = []
        for idxs in clusters:
            if len(idxs) < 2:
                continue
            group = pts[idxs]
            c = group.mean(axis=0)
            rad = float(np.max(np.linalg.norm(group - c, axis=1)) + 0.05)
            mz = float(np.max(heights[idxs])) if len(heights) == len(pts) else 0.5
            out.append((c, rad, mz))
        return out

    def _update_tracks(
        self, clusters: list[tuple[np.ndarray, float, float]], now: float
    ) -> None:
        used_tracks: set[int] = set()
        for c_pos, c_rad, c_mz in clusters:
            best_t: TrackedObstacle | None = None
            best_dist = 0.45
            for tr in self.tracks:
                if tr.cluster_id in used_tracks:
                    continue
                d = float(np.linalg.norm(c_pos - tr.pos))
                if d < best_dist:
                    best_dist = d
                    best_t = tr

            if best_t is not None:
                dt = max(1e-3, now - best_t.last_seen)
                if dt < 1.0:
                    inst_vel = (c_pos - best_t.pos) / dt
                    best_t.vel = 0.65 * best_t.vel + 0.35 * inst_vel
                best_t.pos = 0.4 * best_t.pos + 0.6 * c_pos
                best_t.radius = 0.5 * best_t.radius + 0.5 * c_rad
                best_t.max_z = max(best_t.max_z * 0.9, c_mz)
                best_t.last_seen = now
                best_t.hits += 1
                speed = float(np.linalg.norm(best_t.vel))
                if best_t.hits >= 3 and speed > 0.14 and best_t.max_z > 0.85 and best_t.radius < 0.55:
                    best_t.is_person = True
                used_tracks.add(best_t.cluster_id)
            else:
                self.tracks.append(
                    TrackedObstacle(
                        cluster_id=self._next_id,
                        pos=c_pos.copy(),
                        vel=np.zeros(2, dtype=np.float64),
                        radius=c_rad,
                        max_z=c_mz,
                        last_seen=now,
                    )
                )
                self._next_id += 1

        # Prune stale tracks not seen for > 2.5 seconds
        self.tracks = [tr for tr in self.tracks if (now - tr.last_seen) <= 2.5]

    def observe_person_marker(self, world_pos: np.ndarray, timestamp: float) -> None:
        """Record a direct camera sighting of dynamic person ArUco marker (#45)."""
        self.person_marker_pos = (world_pos[:2].copy(), timestamp)
        for tr in self.tracks:
            if np.linalg.norm(tr.pos - world_pos[:2]) < 0.50:
                tr.is_person = True
                tr.last_seen = timestamp

    def observe_movable_anchor(
        self, furn_name: str, world_pos: np.ndarray, world_rot: np.ndarray, rel_p: np.ndarray, rel_R: np.ndarray
    ) -> None:
        """Update the live pose (x, y, yaw) of a piece of movable furniture from its ArUco anchor."""
        R_furn = world_rot @ rel_R.T
        p_furn = world_pos - R_furn @ rel_p
        yaw = math.atan2(R_furn[1, 0], R_furn[0, 0])
        self.furniture_poses[furn_name] = (float(p_furn[0]), float(p_furn[1]), float(yaw))

    def check_person_clearance(
        self, robot_xy: np.ndarray, now: float
    ) -> tuple[bool, float, np.ndarray | None]:
        """Sensor-derived check for whether a dynamic person is within `person_clearance`."""
        min_dist = math.inf
        nearest_pos: np.ndarray | None = None

        if self.person_marker_pos is not None:
            p_pos, t_seen = self.person_marker_pos
            if (now - t_seen) <= 2.0:
                d = float(np.linalg.norm(p_pos - robot_xy[:2]))
                if d < min_dist:
                    min_dist = d
                    nearest_pos = p_pos

        for tr in self.tracks:
            if tr.is_person and (now - tr.last_seen) <= 1.5:
                d = float(np.linalg.norm(tr.pos - robot_xy[:2]))
                if d < min_dist:
                    min_dist = d
                    nearest_pos = tr.pos.copy()

        return min_dist < self.person_clearance, min_dist, nearest_pos

    def occupancy_grid(self, robot_radius: float = 0.28) -> np.ndarray:
        """Boolean grid of cells blocked for the robot center (inflated by `robot_radius`)."""
        occ = (self.log_odds > 0.35).astype(np.uint8)
        if not np.any(occ):
            return np.zeros((self.ny, self.nx), dtype=bool)
        inv = np.where(occ > 0, 0, 1).astype(np.uint8)
        dist_m = cv2.distanceTransform(inv, cv2.DIST_L2, 5) * self.res
        return dist_m <= robot_radius

    def plan_path(
        self,
        start_xy: tuple[float, float] | np.ndarray,
        goal_xy: tuple[float, float] | np.ndarray,
        robot_radius: float = 0.28,
    ) -> list[tuple[float, float]]:
        """8-connected A* path on the live occupancy grid with line-of-sight smoothing."""
        blocked = self.occupancy_grid(robot_radius=robot_radius)
        sy, sx = self._xy_to_cell(float(start_xy[0]), float(start_xy[1]))
        gy, gx = self._xy_to_cell(float(goal_xy[0]), float(goal_xy[1]))

        # Ensure start and goal cells themselves are unblocked in case robot stands near a table
        r_clear = max(1, int(round(0.32 / self.res)))
        blocked[max(0, sy - r_clear) : sy + r_clear + 1, max(0, sx - r_clear) : sx + r_clear + 1] = False
        blocked[max(0, gy - r_clear) : gy + r_clear + 1, max(0, gx - r_clear) : gx + r_clear + 1] = False

        if self._line_free(blocked, (sx, sy), (gx, gy)):
            return [(float(goal_xy[0]), float(goal_xy[1]))]

        # A* search
        start = (sy, sx)
        goal = (gy, gx)
        g_cost = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        pq = [(math.hypot(gy - sy, gx - sx), start)]
        nbrs = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
        ]

        found = False
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                found = True
                break
            cy, cx = cur
            cur_g = g_cost[cur]
            for dy, dx, step_c in nbrs:
                ny_, nx_ = cy + dy, cx + dx
                if not (0 <= ny_ < self.ny and 0 <= nx_ < self.nx) or blocked[ny_, nx_]:
                    continue
                ng = cur_g + step_c
                nxt = (ny_, nx_)
                if ng < g_cost.get(nxt, math.inf):
                    g_cost[nxt] = ng
                    parent[nxt] = cur
                    h = math.hypot(gy - ny_, gx - nx_)
                    heapq.heappush(pq, (ng + h, nxt))

        if not found:
            return [(float(goal_xy[0]), float(goal_xy[1]))]

        cells = []
        c: tuple[int, int] | None = goal
        while c is not None:
            cells.append(c)
            c = parent[c]
        cells.reverse()

        # Shortcut smoothing
        simplified = [cells[0]]
        k = 0
        while k < len(cells) - 1:
            nxt = len(cells) - 1
            while nxt > k + 1 and not self._line_free(blocked, (cells[k][1], cells[k][0]), (cells[nxt][1], cells[nxt][0])):
                nxt -= 1
            simplified.append(cells[nxt])
            k = nxt

        waypoints = [(float(self.xs[ix]), float(self.ys[iy])) for iy, ix in simplified[1:]]
        if waypoints:
            waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        else:
            waypoints = [(float(goal_xy[0]), float(goal_xy[1]))]
        return waypoints

    def _line_free(self, blocked: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> bool:
        x0, y0 = p0
        x1, y1 = p1
        n = max(2, int(math.hypot(x1 - x0, y1 - y0)) + 1)
        xs = np.rint(np.linspace(x0, x1, n)).astype(np.int32)
        ys = np.rint(np.linspace(y0, y1, n)).astype(np.int32)
        return not np.any(blocked[ys, xs])


# =============================================================================
# Layer 1B: EKF + Trimmed Scan-to-SDF Localizer
# =============================================================================


class G1Localizer:
    """Fuses IMU AHRS, Leg Odometry, 360° LiDAR Scan Matching, and ArUco Landmarks.

    Provides estimated robot pose `(x, y, z, yaw, roll, pitch)` and body velocity
    `(vx, vy, wz)` at 50 Hz without ever reading ground-truth pose from `MjData`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        inlier_thresh: float = 0.20,
        lidar_every: int = 2,
    ):
        self.model = model
        self.data = data
        self.inlier_thresh = inlier_thresh
        self.lidar_every = lidar_every

        self.imu = IMUSensor(model, data)
        self.leg_odom = LegOdometry(model)
        self.lidar = LidarSensor(model, data)
        self.static_map = StaticStructuralMap(model)
        self.dynamic_map = DynamicWorldMap(self.static_map)

        # EKF state: [x, y, yaw]
        self.state = np.zeros(3, dtype=np.float64)
        self.P = np.diag([1e-4, 1e-4, 1e-4]).astype(np.float64)
        self.vel_body = np.zeros(3, dtype=np.float64)

        self._tick = 0
        self.last_inlier_ratio = 1.0
        self.last_scan_residual = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, z: float = 0.74) -> None:
        self.state[:] = (x, y, wrap_angle(yaw))
        self.P = np.diag([1e-4, 1e-4, 1e-4]).astype(np.float64)
        self.vel_body[:] = 0.0
        self.imu.reset(0.0, 0.0)
        self.leg_odom.reset(z)
        self.dynamic_map.reset()
        self._tick = 0

    # ------------------------------------------------------------------ properties
    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def z(self) -> float:
        return float(self.leg_odom.z_pelvis)

    @property
    def yaw(self) -> float:
        return float(self.state[2])

    @property
    def roll(self) -> float:
        return float(self.imu.roll)

    @property
    def pitch(self) -> float:
        return float(self.imu.pitch)

    @property
    def pos_3d(self) -> np.ndarray:
        return np.array([self.state[0], self.state[1], self.leg_odom.z_pelvis], dtype=np.float64)

    @property
    def quat_wxyz(self) -> np.ndarray:
        return quat_from_euler(self.imu.roll, self.imu.pitch, float(self.state[2]))

    @property
    def pose_2d(self) -> tuple[float, float, float]:
        return float(self.state[0]), float(self.state[1]), float(self.state[2])

    def base_pose_3d(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_3d, quat_wxyz) for ArmIK and kinematic queries."""
        return self.pos_3d, self.quat_wxyz

    def camera_pose(self, q_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute `head_cam` world (pos, 3x3 rot) from estimated base pose + encoder joints."""
        return self.leg_odom.camera_pose_for_base(self.pos_3d, self.quat_wxyz, q_joints)

    # ------------------------------------------------------------------ EKF Predict (50 Hz)
    def step(
        self,
        q_joints: np.ndarray,
        dq_joints: np.ndarray,
        cmd_vel: np.ndarray,
        dt: float,
    ) -> None:
        """Run 50 Hz IMU + Leg Odometry prediction, and periodic 360° LiDAR update."""
        imu_reading = self.imu.read(dt)
        odom = self.leg_odom.update(q_joints, dq_joints, imu_reading, dt)

        # Fuse stance-foot kinematic velocity with command velocity prior to suppress gait sway
        cmd_arr = np.asarray(cmd_vel, dtype=np.float64)
        if np.linalg.norm(cmd_arr[:2]) < 0.02:
            # In balance mode, suppress foot sway jitter
            vx = 0.25 * odom.vel_body[0]
            vy = 0.25 * odom.vel_body[1]
        else:
            vx = 0.70 * odom.vel_body[0] + 0.30 * cmd_arr[0]
            vy = 0.70 * odom.vel_body[1] + 0.30 * cmd_arr[1]
        wz = imu_reading.yaw_rate_world
        self.vel_body[:] = (vx, vy, wz)

        # Propagate 2D state [x, y, yaw]
        th = float(self.state[2])
        c, s = math.cos(th), math.sin(th)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
        dth = wz * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] = wrap_angle(th + dth)

        # EKF Jacobian F and process noise Q
        F = np.array([
            [1.0, 0.0, (-vx * s - vy * c) * dt],
            [0.0, 1.0, (vx * c - vy * s) * dt],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        speed = math.hypot(vx, vy) + abs(wz) * 0.2
        q_xy = (0.015 + 0.04 * speed) * dt
        q_th = (0.010 + 0.03 * abs(wz)) * dt
        Q = np.diag([q_xy ** 2, q_xy ** 2, q_th ** 2])
        self.P = F @ self.P @ F.T + Q

        self._tick += 1
        if self._tick % self.lidar_every == 0:
            scan = self.lidar.scan(z_pelvis_est=self.leg_odom.z_pelvis)
            self.update_lidar(scan)

    # ------------------------------------------------------------------ EKF LiDAR Update
    def update_lidar(self, scan: LidarScan) -> None:
        """Trimmed Gauss-Newton Scan-to-SDF registration + EKF measurement update."""
        pts_local = scan.points_local
        if len(pts_local) < 6:
            return

        pose = self.state.copy()
        inlier_mask = np.zeros(len(pts_local), dtype=bool)
        huber_delta = 0.05

        for _ in range(6):
            x, y, th = pose
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            pts_w = pts_local @ R.T + np.array([x, y])

            d, gx, gy, in_b = self.static_map.query_dist_and_grad(pts_w)
            inlier_mask = in_b & (d <= self.inlier_thresh)
            if int(np.sum(inlier_mask)) < 6:
                break

            d_in = d[inlier_mask]
            gx_in = gx[inlier_mask]
            gy_in = gy[inlier_mask]
            p_loc_in = pts_local[inlier_mask]

            # Derivative of world point wrt [x, y, th]:
            # dp_x / dth = -s * lx - c * ly,  dp_y / dth = c * lx - s * ly
            dp_dth_x = -s * p_loc_in[:, 0] - c * p_loc_in[:, 1]
            dp_dth_y = c * p_loc_in[:, 0] - s * p_loc_in[:, 1]
            j_th = gx_in * dp_dth_x + gy_in * dp_dth_y

            J = np.column_stack([gx_in, gy_in, j_th])  # (M, 3)
            # Huber weights for robustness against partial occlusion
            w = np.where(d_in <= huber_delta, 1.0, huber_delta / np.maximum(d_in, 1e-6))
            Jw = J * w[:, None]
            H = Jw.T @ J + np.diag([1e-3, 1e-3, 1e-3])
            b = -Jw.T @ d_in
            try:
                step = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break

            step = np.clip(step, [-0.08, -0.08, -0.08], [0.08, 0.08, 0.08])
            pose[0] += step[0]
            pose[1] += step[1]
            pose[2] = wrap_angle(pose[2] + step[2])
            if np.linalg.norm(step) < 1e-4:
                break

        # Final evaluation at converged pose
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts_w = pts_local @ R.T + np.array([x, y])
        d, _, _, in_b = self.static_map.query_dist_and_grad(pts_w)
        inlier_mask = in_b & (d <= self.inlier_thresh)
        n_inliers = int(np.sum(inlier_mask))
        self.last_inlier_ratio = n_inliers / max(1, len(pts_local))
        self.last_scan_residual = float(np.mean(d[inlier_mask])) if n_inliers > 0 else 0.0

        if n_inliers >= 6:
            # EKF measurement update with matched pose
            R_meas = np.diag([0.012 ** 2, 0.012 ** 2, 0.015 ** 2]) / max(0.25, self.last_inlier_ratio)
            innov = np.array([
                pose[0] - self.state[0],
                pose[1] - self.state[1],
                wrap_angle(pose[2] - self.state[2]),
            ])
            S = self.P + R_meas
            K = self.P @ np.linalg.inv(S)
            upd = K @ innov
            self.state[0] += upd[0]
            self.state[1] += upd[1]
            self.state[2] = wrap_angle(self.state[2] + upd[2])
            self.P = (np.eye(3) - K) @ self.P

        # Pass scan and non-static outlier points to the Live Dynamic World Map
        c_est, s_est = math.cos(self.state[2]), math.sin(self.state[2])
        R_est = np.array([[c_est, -s_est], [s_est, c_est]], dtype=np.float64)
        pts_w_est = pts_local @ R_est.T + self.state[:2]
        d_est, _, _, _ = self.static_map.query_dist_and_grad(pts_w_est)
        outlier_mask = d_est > self.inlier_thresh
        hit_heights = scan.max_heights[np.isfinite(scan.ranges)]
        outlier_heights = hit_heights[outlier_mask] if len(hit_heights) == len(pts_w_est) else np.zeros(int(np.sum(outlier_mask)))

        self.dynamic_map.update_from_scan(
            self.pose_2d,
            scan,
            pts_w_est[outlier_mask],
            outlier_heights,
        )

    # ------------------------------------------------------------------ EKF Visual Landmark Update
    def update_landmarks(
        self,
        cam_detections: dict[int, tuple[np.ndarray, np.ndarray]],
        q_joints: np.ndarray,
        timestamp: float,
    ) -> None:
        """Fuse ArUco detections `(marker_id -> (p_in_cam, R_in_cam))` with EKF & DynamicMap.

        Static anchors refine the robot's global (x, y, yaw) pose; movable furniture
        anchors update the furniture's pose in `DynamicWorldMap`; Marker #45 updates
        the dynamic person tracker.
        """
        if not cam_detections:
            return

        # Compute camera pose relative to the pelvis in the robot's local horizontal frame
        p_zero = np.array([0.0, 0.0, self.leg_odom.z_pelvis], dtype=np.float64)
        q_zero = quat_from_euler(self.imu.roll, self.imu.pitch, 0.0)
        cam_p_local, cam_R_local = self.leg_odom.camera_pose_for_base(p_zero, q_zero, q_joints)

        for mid, (p_cam, R_cam) in cam_detections.items():
            # Marker in robot's horizontal body frame (yaw=0)
            p_body = cam_p_local + cam_R_local @ p_cam
            R_body = cam_R_local @ R_cam
            dist = float(np.linalg.norm(p_cam))

            if mid in self.static_map.static_anchors and dist < 2.8:
                p_world_ref, R_world_ref = self.static_map.static_anchors[mid]
                # From R_world_ref = Rz(yaw) @ R_body, extract implied robot yaw using marker normal (col 2)
                n_ref = R_world_ref[:2, 2]
                n_body = R_body[:2, 2]
                if np.linalg.norm(n_ref) > 0.3 and np.linalg.norm(n_body) > 0.3:
                    yaw_meas = wrap_angle(
                        math.atan2(n_ref[1], n_ref[0]) - math.atan2(n_body[1], n_body[0])
                    )
                else:
                    yaw_meas = float(self.state[2])

                cy, sy = math.cos(yaw_meas), math.sin(yaw_meas)
                x_meas = float(p_world_ref[0] - (cy * p_body[0] - sy * p_body[1]))
                y_meas = float(p_world_ref[1] - (sy * p_body[0] + cy * p_body[1]))

                sigma_xy = 0.015 * max(1.0, dist)
                sigma_th = 0.035 * max(1.0, dist)
                R_lm = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_th ** 2])
                innov = np.array([
                    x_meas - self.state[0],
                    y_meas - self.state[1],
                    wrap_angle(yaw_meas - self.state[2]),
                ])
                # Gate large outliers
                if np.linalg.norm(innov[:2]) < 0.35:
                    S = self.P + R_lm
                    K = self.P @ np.linalg.inv(S)
                    upd = K @ innov
                    self.state[0] += upd[0]
                    self.state[1] += upd[1]
                    self.state[2] = wrap_angle(self.state[2] + upd[2])
                    self.P = (np.eye(3) - K) @ self.P

            # Recompute world marker pose with updated robot state for dynamic map updates
            cy, sy = math.cos(self.state[2]), math.sin(self.state[2])
            R_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            p_world = np.array([self.state[0], self.state[1], 0.0]) + R_z @ p_body
            R_world = R_z @ R_body

            if mid in self.static_map.movable_anchors:
                furn_name, rel_p, rel_R = self.static_map.movable_anchors[mid]
                self.dynamic_map.observe_movable_anchor(furn_name, p_world, R_world, rel_p, rel_R)
            elif mid == 45:  # Dynamic person badge in HouseEnv
                self.dynamic_map.observe_person_marker(p_world, timestamp)
