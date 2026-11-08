"""Perceived clearance map for navigation: what the robot's own sensors say is in the way.

Until now the planner read the house straight out of MuJoCo (house.plan_path on Jack's grid of
the true geometry), so a chair that had been moved since the map was drawn was either walked into
or, if it had been moved away, still avoided. This module builds the map the way a real robot
does:

  * Prior: the floor plan. At start the map is seeded with the walls and the furniture that
    doesn't move (counter, fridge, sofa, bed, bookshelf, chests of drawers), from the scene as
    it was when the robot was switched on. That is a reasonable thing to give a home robot (it
    is what the localiser already matches its LiDAR against) and it stays true: walls stay put.
    Movable furniture (coffee table, kitchen table, desk, bedside table), clutter (chairs,
    boxes, laundry) and people are NOT in the prior: the robot has to see them.
  * 3D LiDAR (`HeadLidar`, a Livox-style scanner on the torso: 360 deg x -45..+6 deg, 10 Hz,
    mj_multiRay with the robot's own geoms and a held item excluded): every beam carves free
    space up to its hit and marks the hit. Occupancy is kept as log-odds in a coarse voxel grid
    (5 cm cells x 6 height bands over 0.05-1.65 m), so a beam passing *under* a table top clears
    only the band it passed through, and the table top seen by another beam stays. The 2D map
    the planner uses is "any band occupied".
  * RGB-D head camera: the depth image (at every look(), and once a second while walking) is
    back-projected and its points above the floor are added as hits. Dense where the robot
    looks, so table tops and small clutter in front are filled in.
  * Both sensors are placed in the world with the *estimated* pose (state_estimation.py), not
    the simulator's; the estimate is good to a few cm, which at 5 cm cells is fine.

Clearance = Euclidean distance transform of the occupied cells; the planner inflates by the
robot radius (house.plan_path's tiers), and go_to re-checks the remaining path against the live
map every scan and re-plans when a new obstacle has cut it (skills.go_to).

What the map deliberately ignores:
  * Hits on the leaf of an unlocked door: the floor plan says where each push door hangs, and a
    leaf is a thin plank through its hinge, so in a frame its hits all lie at one bearing from
    the hinge; those are dropped (a door leaf is not a wall, the robot pushes it). Anything else
    standing in a door swing is mapped like anything else; locked doors are in the prior.
  * Hits within SELF_R of the robot's own base: the arms, a carried item.
  * Points lower than FLOOR_Z (+ a bit per metre of range, for the attitude error): the floor.

G1_NAV=static restores the old behaviour (plan once on the true geometry, never re-plan).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage

from sensors import rot_from_euler

NAV_MODE = os.environ.get("G1_NAV", "live")  # live | static

# voxel grid: Jack's 5 cm xy grid (sim.house_env.GRID_RES), 6 height bands
BAND_EDGES = np.array([0.05, 0.30, 0.55, 0.80, 1.05, 1.30, 1.65])
# log-odds
L_WALL = 6.0       # prior for walls: practically permanent
L_FIXED = 2.5      # prior for fixed furniture: a few clear scans through it would remove it
L_PRIOR = -1.0     # everything else starts free-ish (two hits make a cell occupied)
L_HIT = 0.7
L_MISS = 0.35
L_DEPTH_HIT = 0.5  # a depth image is one observation however many pixels land in a cell
L_SHADOW = 0.4     # per scan, for the cells just behind a hit on a new object (its unseen volume)
L_SHADOW_CAP = 0.6  # ...never more than weakly occupied: one clear beam through them undoes two scans
SHADOW_DEPTH = 0.15  # m behind the hit (deeper leaks out of the far side of small things)
LEAF_MIN_RUN = 0.12  # m of radius a bearing bin must cover to count as a door leaf
OPENING_HALF = 0.12  # m, half depth of the doorway opening strip that is never marked
LEAF_HALF_T = 0.05   # m, half thickness allowed for the leaf (Jack's leaf is 4 cm)
L_MIN, L_MAX = -2.5, 6.0
FLOOR_Z = 0.06     # m above the floor below which a point is the floor
FLOOR_SLOPE = 0.015  # + this per metre of range (1 deg of attitude error at 4 m is 7 cm)
SELF_R = 0.22      # m round the base: the LiDAR already skips the robot's own geoms and a held item,
                   # this only drops hits at the sensor's minimum range
DEPTH_SELF_R = 0.50  # the depth image can't skip the robot's own hands: ignore points this close
DEPTH_MAX = 3.5    # m, depth points further than this are too coarse to trust
DEPTH_STRIDE = 6   # use every 6th pixel of the 960x720 depth image
LIDAR_EVERY = 5    # control ticks between scans (10 Hz at 50 Hz control)
DEPTH_EVERY = 50   # control ticks between depth integrations while walking (1 Hz)


def _yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


@dataclass
class HeadLidar:
    """Livox-style 3D LiDAR on the torso: azimuths x elevations, fixed in the pelvis frame.

    The physics ray-cast uses the true sensor pose (a LiDAR has to be simulated from the true
    state); the robot only gets back (direction in the sensor frame, range) pairs.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    n_azimuth: int = 120
    elevations_deg: tuple = (-60, -52, -45, -39, -33, -28, -23, -19, -15, -12, -9, -6, -3, 0, 3, 6)
    mount: tuple = (0.0, 0.0, 0.50)  # m, in the pelvis frame (about head height)
    max_range: float = 8.0
    min_range: float = 0.20
    noise_std: float = 0.004

    def __post_init__(self):
        m = self.model
        az = np.linspace(-np.pi, np.pi, self.n_azimuth, endpoint=False)
        el = np.radians(np.asarray(self.elevations_deg, float))
        AZ, EL = np.meshgrid(az, el)
        self.dirs_local = np.column_stack([np.cos(EL.ravel()) * np.cos(AZ.ravel()),
                                           np.cos(EL.ravel()) * np.sin(AZ.ravel()),
                                           np.sin(EL.ravel())])
        self.n = len(self.dirs_local)
        self._geomid = np.full(self.n, -1, np.int32)
        self._dist = np.full(self.n, -1.0)
        self._group = np.array([1, 1, 1, 1, 1, 0], np.uint8)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot = set()
        for b in range(m.nbody):
            cur = b
            while cur > 0:
                if cur == self.pelvis:
                    robot.add(b)
                    break
                cur = m.body_parentid[cur]
        self._robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in robot], np.int32)
        self.held_body: int | None = None
        self.rng = np.random.default_rng(1)
        self.mount = np.asarray(self.mount, float)

    def set_held_body(self, name: str | None) -> None:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
        self.held_body = bid if bid >= 0 else None

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ranges, hit) in the sensor frame: range per beam (max_range where nothing
        was hit within it) and whether the beam hit something."""
        m, d = self.model, self.data
        p = d.xpos[self.pelvis]
        R = d.xmat[self.pelvis].reshape(3, 3)
        origin = p + R @ self.mount
        dirs = (self.dirs_local @ R.T).ravel()
        excl = self._robot_geoms
        if self.held_body is not None:
            excl = np.concatenate([excl, np.where(m.geom_bodyid == self.held_body)[0].astype(np.int32)])
        saved = m.geom_group[excl].copy()
        m.geom_group[excl] = 5
        try:
            mujoco.mj_multiRay(m, d, origin, dirs, self._group, 1, self.pelvis, self._geomid, self._dist, None,
                               self.n, self.max_range)
        finally:
            m.geom_group[excl] = saved
        hit = (self._geomid >= 0) & (self._dist >= self.min_range) & (self._dist <= self.max_range)
        r = np.where(hit, self._dist, self.max_range)
        if self.noise_std > 0:
            r = np.where(hit, np.clip(r + self.rng.normal(0, self.noise_std, self.n), self.min_range, self.max_range), r)
        return r, hit


class Costmap:
    """Voxel log-odds occupancy of the house from the robot's sensors -> 2D clearance grid."""

    def __init__(self, env, prior_geoms, wall_geoms, door_zones, grid_res: float, door_names=None):
        """env: the house (its gx/gy grid, _clearance and planner helpers are reused).
        prior_geoms / wall_geoms: geom ids seeded as fixed furniture / walls.
        door_zones: [(hinge x, hinge y, leaf reach, closed-leaf bearing)] of the unlocked doors
        (see _is_door_leaf). door_names: the door of each zone (for mark_door_locked)."""
        self.env = env
        self.res = grid_res
        self.gx, self.gy = env.gx, env.gy
        self.x0, self.y0 = float(self.gx[0]), float(self.gy[0])
        self.nx, self.ny = len(self.gx), len(self.gy)
        self.nb = len(BAND_EDGES) - 1
        self.L = np.full((self.nb, self.ny, self.nx), L_PRIOR, np.float32)
        self.door_zones = [(float(x), float(y), float(r), float(a)) for x, y, r, a in door_zones]
        self.leaf_bearing: dict[int, float] = {}  # last confident bearing of each door's leaf
        self.door_names = list(door_names) if door_names is not None else [None] * len(self.door_zones)
        self._zone_off: set[int] = set()  # zones of doors found locked (their leaf is a wall now)
        self._seed(wall_geoms, L_WALL)
        self._seed(prior_geoms, L_FIXED)
        self.prior = self.L.copy()
        self.version = 0
        self._cache: dict = {}
        self.scans = 0
        self.depth_frames = 0
        self.log: list[dict] = []

    # ------------------------------------------------------------------ prior
    def mark_door_locked(self, door: str) -> None:
        """The robot found this door locked (scenario.DoorDiscovery): its leaf becomes a wall in
        the prior and in the map, and its hits are no longer dropped as a pushable leaf."""
        geoms = list(self.env.door_geoms.get(door, []))
        self._seed(geoms, L_WALL)
        self._seed(geoms, L_WALL, arr=self.prior)
        self._zone_off |= {n for n, dn in enumerate(self.door_names) if dn == door}
        self.version += 1

    def _seed(self, geoms, value: float, arr: np.ndarray | None = None) -> None:
        arr = self.L if arr is None else arr
        m, d = self.env.model, self.env.data
        for g in geoms:
            typ = m.geom_type[g]
            if typ == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            pos, mat, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                half_z = float(np.abs(mat[2]) @ size)
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                half_z = float(size[1] + (size[0] if typ == mujoco.mjtGeom.mjGEOM_CAPSULE else 0))
            else:
                half_z = float(m.geom_rbound[g])
            z_lo, z_hi = pos[2] - half_z, pos[2] + half_z
            bands = [k for k in range(self.nb) if z_hi > BAND_EDGES[k] and z_lo < BAND_EDGES[k + 1]]
            if not bands:
                continue
            inside = self.env._clearance([g], 10.0) <= 0.0
            for k in bands:
                arr[k][inside] = np.maximum(arr[k][inside], value)

    # ------------------------------------------------------------------ geometry helpers
    def _cells(self, pts: np.ndarray):
        """World points (N, 3) -> (band, i, j) and a validity mask."""
        j = np.rint((pts[:, 0] - self.x0) / self.res).astype(np.int64)
        i = np.rint((pts[:, 1] - self.y0) / self.res).astype(np.int64)
        k = np.searchsorted(BAND_EDGES, pts[:, 2], side="right") - 1
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx) & (k >= 0) & (k < self.nb)
        return k, i, j, ok

    def _is_door_leaf(self, xy: np.ndarray) -> np.ndarray:
        """Which of these hit points (one sensor frame) lie on a door leaf, or in a doorway opening.

        The floor plan says where each unlocked door hangs. A leaf is a thin plank through its
        hinge, so in one frame its hits all sit at one bearing from the hinge over a run of
        radius; anything else in the swing (a chair, a person) spreads over bearings. Per door:
        take the hits within the leaf's reach, in the half-plane the leaf swings through (the
        wall it hangs on is just as radial, the other way); the busiest 3-degree bearing bin,
        if it holds a few hits over some radius, is the leaf, and its bearing is remembered, so
        a frame that only glimpses the leaf's tip round the jamb still knows where it is. Hits
        within the plank's angular width of that bearing are dropped, and so are hits in the
        doorway opening itself (the known gap in the wall, a strip 2*OPENING_HALF deep): a leaf
        swinging through it is not a wall. Everything else near a door is mapped."""
        out = np.zeros(len(xy), bool)
        for n, (x, y, r, closed) in enumerate(self.door_zones):
            if n in self._zone_off:
                continue
            dx, dy = xy[:, 0] - x, xy[:, 1] - y
            rad = np.hypot(dx, dy)
            near = rad <= r
            idx = np.flatnonzero(near)
            if len(idx) == 0:
                continue
            rad = rad[near]
            ang = np.arctan2(dy[near], dx[near])
            rel = (ang - closed + np.pi) % (2 * np.pi) - np.pi
            # the doorway opening: a strip across the wall line through the hinge, DOOR_W wide
            along = rad * np.cos(rel)  # distance from the hinge along the closed leaf
            perp = rad * np.sin(rel)   # across the wall line
            out[idx[(along > -0.02) & (along < r) & (np.abs(perp) <= OPENING_HALF)]] = True
            swing = np.abs(rel) <= np.radians(95.0)
            a_leaf = self.leaf_bearing.get(n)
            if swing.sum() >= 3:
                nb = 120  # 3-degree bins
                b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
                counts = np.bincount(b[swing], minlength=nb)
                # the leaf's hits may straddle two bins: score each bin with its neighbours
                score = counts + np.roll(counts, 1) + np.roll(counts, -1)
                best = int(np.argmax(score))
                sel = swing & ((b == best) | (b == (best - 1) % nb) | (b == (best + 1) % nb))
                if sel.sum() >= 3 and np.ptp(rad[sel]) >= LEAF_MIN_RUN:
                    a_leaf = math.atan2(np.sin(ang[sel]).mean(), np.cos(ang[sel]).mean())
                    self.leaf_bearing[n] = a_leaf
            if a_leaf is None:
                continue
            tol = np.arctan2(LEAF_HALF_T, np.maximum(rad, 0.05)) + np.radians(2.0)
            leaf = np.abs((ang - a_leaf + np.pi) % (2 * np.pi) - np.pi) < tol
            out[idx[leaf]] = True
        return out

    def _vote(self, k, i, j, delta: float, cap: float = L_MAX) -> None:
        if len(k) == 0:
            return
        lin = np.unique((k * self.ny + i) * self.nx + j)
        flat = self.L.reshape(-1)
        v = flat[lin]
        flat[lin] = np.clip(v + delta, L_MIN, np.maximum(v, cap))  # never pull a value down to the cap

    # ------------------------------------------------------------------ sensor updates
    def integrate_rays(self, origin: np.ndarray, dirs: np.ndarray, ranges: np.ndarray, hit: np.ndarray,
                       base_xy: np.ndarray) -> None:
        """Beams from `origin` along unit `dirs` (world, from the *estimated* pose): free up to
        `ranges`, occupied at the end where `hit`."""
        # free space: samples along every beam, one per cell, up to just short of the hit
        n = np.maximum((ranges / self.res).astype(int) - 1, 0)
        total = int(n.sum())
        if total:
            ray = np.repeat(np.arange(len(dirs)), n)
            t = (np.arange(total) - np.repeat(np.cumsum(n) - n, n) + 0.5) * self.res
            pts = origin + dirs[ray] * t[:, None]
            k, i, j, ok = self._cells(pts)
            self._vote(k[ok], i[ok], j[ok], -L_MISS)
        # hits
        if hit.any():
            p = origin + dirs[hit] * ranges[hit][:, None]
            self._add_hits(p, base_xy, L_HIT, SELF_R, dirs=dirs[hit])

    def _add_hits(self, p: np.ndarray, base_xy: np.ndarray, value: float, self_r: float,
                  dirs: np.ndarray | None = None) -> None:
        """Mark hit points. dirs (unit beam directions, same length as p): also give the cells
        up to SHADOW_DEPTH behind a hit on something that is not in the prior a weak "occupied"
        vote, capped at L_SHADOW_CAP. A LiDAR sees only the near face of a chair; the robot
        would otherwise plan right behind that face as if the chair were a sheet of paper. The
        shadow is cleared again by any beam that later passes through it."""
        rh = np.hypot(p[:, 0] - base_xy[0], p[:, 1] - base_xy[1])
        keep = (p[:, 2] > FLOOR_Z + FLOOR_SLOPE * rh) & (p[:, 2] < BAND_EDGES[-1]) & (rh > self_r)
        idx = np.flatnonzero(keep)
        keep[idx[self._is_door_leaf(p[idx, :2])]] = False
        p = p[keep]
        if len(p) == 0:
            return
        k, i, j, ok = self._cells(p)
        self._vote(k[ok], i[ok], j[ok], value)
        if dirs is None:
            return
        dirs = dirs[keep][ok]
        k, i, j, p = k[ok], i[ok], j[ok], p[ok]
        # hits on things the floor plan doesn't have (a hit landing in the skin cell just outside
        # a wall counts as the wall: look one cell further along the beam too)
        k1, i1, j1, ok1 = self._cells(p + dirs * self.res)
        k1, i1, j1 = np.where(ok1, k1, k), np.where(ok1, i1, i), np.where(ok1, j1, j)
        new = (self.prior[k, i, j] <= 0.0) & (self.prior[k1, i1, j1] <= 0.0)
        if not new.any():
            return
        steps = np.arange(1, int(SHADOW_DEPTH / self.res) + 1) * self.res
        q = (p[new][:, None, :] + dirs[new][:, None, :] * steps[None, :, None]).reshape(-1, 3)
        k2, i2, j2, ok2 = self._cells(q)
        self._vote(k2[ok2], i2[ok2], j2[ok2], L_SHADOW, cap=L_SHADOW_CAP)

    def update_lidar(self, lidar: HeadLidar, pose_pos: np.ndarray, pose_R: np.ndarray) -> None:
        """One scan, placed with the estimated pelvis pose (pos, 3x3 rotation)."""
        ranges, hit = lidar.scan()
        origin = np.asarray(pose_pos, float) + pose_R @ lidar.mount
        dirs = lidar.dirs_local @ pose_R.T
        self.integrate_rays(origin, dirs, ranges, hit, np.asarray(pose_pos[:2], float))
        self.scans += 1
        self.version += 1

    def update_depth(self, depth: np.ndarray, K: np.ndarray, cam_pos: np.ndarray, cam_R: np.ndarray,
                     base_xy: np.ndarray) -> None:
        """Depth image (m along the optical axis; MuJoCo camera: x right, y up, looks along -z)."""
        s = DEPTH_STRIDE
        d = depth[s // 2::s, s // 2::s]
        v, u = np.mgrid[s // 2:depth.shape[0]:s, s // 2:depth.shape[1]:s]
        ok = np.isfinite(d) & (d > 0.15) & (d < DEPTH_MAX)
        d, u, v = d[ok], u[ok], v[ok]
        if len(d) == 0:
            return
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        pc = np.column_stack([(u - cx) / f * d, -(v - cy) / f * d, -d])
        pw = np.asarray(cam_pos, float) + pc @ np.asarray(cam_R, float).T
        self._add_hits(pw, np.asarray(base_xy, float), L_DEPTH_HIT, DEPTH_SELF_R)
        self.depth_frames += 1
        self.version += 1

    # ------------------------------------------------------------------ outputs
    def occupied(self) -> np.ndarray:
        """2D: a cell is blocked if any height band in it is occupied."""
        return (self.L > 0.0).any(axis=0)

    def clearance(self) -> np.ndarray:
        """m, distance from each cell to the nearest occupied cell (cached per map version)."""
        c = self._cache.get("clear")
        if c is None or c[0] != self.version:
            occ = self.occupied()
            dist = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, np.inf)
            c = (self.version, dist)
            self._cache["clear"] = c
        return c[1]

    def blocked(self, radius: float) -> np.ndarray:
        """Cells the robot's centre may not be in, for a robot of this radius."""
        return self.clearance() <= radius

    def path_blocked(self, path, radius: float, skip_last: bool = True, since: np.ndarray | None = None) -> int:
        """Index of the first leg of `path` (list of (x, y)) that crosses a blocked cell, or -1.
        The last leg (into a station, which sits inside the inflation) is not checked.
        since: the clearance grid the path was planned on; only cells that have become blocked
        since then count (the planner's own compromises near furniture are not "new")."""
        grid = self.blocked(radius)
        if since is not None:
            grid = grid & ~(since <= radius)
        legs = list(zip(path[:-1], path[1:]))
        if skip_last and len(legs) > 1:
            legs = legs[:-1]
        for n, (p, q) in enumerate(legs):
            if not self.env._line_free(grid, p, q):
                return n
        return -1

    def changed_vs_prior(self) -> tuple[int, int]:
        """(cells occupied now that the prior had free, cells the prior had occupied now free)."""
        now, before = self.occupied(), (self.prior > 0.0).any(axis=0)
        return int((now & ~before).sum()), int((before & ~now).sum())

    def image(self, path=None, robot_xy=None, scale: int = 3) -> np.ndarray:
        """RGB picture of the map (north up): white free, black occupied, grey prior-only,
        blue path, red robot."""
        occ, pri = self.occupied(), (self.prior > 0.0).any(axis=0)
        img = np.full((self.ny, self.nx, 3), 245, np.uint8)
        img[pri] = (150, 150, 150)
        img[occ & ~pri] = (40, 40, 40)
        img[occ & pri] = (0, 0, 0)
        img[(~occ) & pri] = (200, 230, 200)  # prior said occupied, sensors say free
        img = np.repeat(np.repeat(img, scale, 0), scale, 1)

        def px(x, y):
            return int(round((x - self.x0) / self.res * scale)), int(round((y - self.y0) / self.res * scale))

        import cv2
        if path:
            for p, q in zip(path[:-1], path[1:]):
                cv2.line(img, px(*p), px(*q), (40, 90, 220), 2)
        if robot_xy is not None:
            cv2.circle(img, px(*robot_xy), max(2, scale * 2), (220, 40, 40), -1)
        return np.ascontiguousarray(img[::-1])  # y up


def make_costmap(env) -> Costmap:
    """The costmap for Jack's house: prior = walls + fixed furniture + chests (+ the drawers in
    them, closed) + locked door leaves; movable furniture and clutter left for the sensors."""
    from sim import house_env as H
    m = env.model
    movable = {g for gs in env.furn_geoms.values() for g in gs}
    walls, fixed = [], []
    for g in env.static_geoms:
        if g in movable:
            continue
        name = m.geom(g).name
        (walls if name.startswith("wall") or name.startswith("door_frame") else fixed).append(g)
    fixed += list(env.drawer_geoms)
    fixed += [g for dn in env.locked_doors for g in env.door_geoms[dn]]
    zones, names = [], []
    for dn in H.DOORWAYS:
        if dn in env.locked_doors:
            continue
        hinge, closed_deg = H.door_frame(dn)  # closed_deg: bearing of the closed leaf from its hinge
        zones.append((hinge[0], hinge[1], H.DOOR_LEAF_L + 0.05, math.radians(closed_deg)))
        names.append(dn)
    return Costmap(env, fixed, walls, zones, H.GRID_RES, door_names=names)


def pose_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_from_euler(roll, pitch, yaw)
