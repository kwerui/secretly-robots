"""Damped-least-squares IK for one G1 arm (7 joints) using estimated robot state.

Solves on a scratch MjData populated from the localizer's estimated base pose
`(pos_3d, quat_wxyz)` and the 29 joint encoders (never reading ground-truth base pose
from `self.data`). Re-solving every control tick (closed loop) absorbs the small sway
of the balancing lower body.

Targets: palm position (world) plus optional axis alignments, e.g.
    fingers (site x-axis) -> approach direction
    palm side (site z-axis) -> world up
which is enough for side grasps and handle pulls without a full 6D target.

Collision avoidance: every iteration, signed distances (mj_geomDistance) between this arm's
links (upper arm down to the fingers) and nearby obstacles (torso, pelvis, legs, the other
arm, furniture, the door, floor) are computed. Any pair closer than `safe_dist`
becomes an extra row "move the arm point away along the separating normal", weighted above
the hand target, so the arm routes around the body and furniture instead of through them.
The hand itself may touch the item/door it is grasping (same exclusions as the physics).
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from g1_wbc import BODY_JOINTS, ARM_HOME
from scene import RIGHT_HAND_BODIES

ARM_JOINTS = {
    "left": BODY_JOINTS[15:22],
    "right": BODY_JOINTS[22:29],
}


class ArmIK:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str = "right"):
        self.model, self.data = model, data
        self.side = side
        self.scratch = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS[side]]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0] + 0.02
        self.hi = model.jnt_range[jids, 1] - 0.02
        self.home = ARM_HOME[:7] if side == "left" else ARM_HOME[7:]

        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._root_q = model.jnt_qposadr[root] if root >= 0 else 0
        self._body_qpos_ids = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in BODY_JOINTS
        ])
        self._pose_provider: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
        self._joints_provider: Callable[[], np.ndarray] | None = None
        self.articulated_qpos: dict[int, float] = {}  # e.g. estimated door/drawer joint qpos

        # Bodies of this arm (shoulder down to the fingers), for collision checks
        first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")
        self.arm_bodies = {b for b in range(model.nbody) if self._is_descendant(b, first)}
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._jacc = np.zeros((3, model.nv))
        self._ft = np.zeros(6)

        def collidable(g):
            return model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
        names = {b: model.body(b).name for b in range(model.nbody)}
        skip_links = {f"{side}_shoulder_pitch_link", f"{side}_shoulder_roll_link"}
        self.arm_geoms = np.array([g for g in range(model.ngeom) if collidable(g)
                                   and model.geom_bodyid[g] in self.arm_bodies
                                   and names[model.geom_bodyid[g]] not in skip_links])
        hand = set(RIGHT_HAND_BODIES) if side == "right" else {n.replace("right", "left") for n in RIGHT_HAND_BODIES}
        self.hand_geoms = {g for g in self.arm_geoms if names[model.geom_bodyid[g]] in hand}
        # bodies the hand is allowed to touch (things it grasps): items (free bodies) and the door
        self.graspable_bodies = {b for b in range(model.nbody)
                                 if model.body_jntnum[b] and model.jnt_type[model.body_jntadr[b]] == 0
                                 and names[b] != "pelvis"} | {b for b in range(model.nbody) if names[b] == "cupboard_door"}
        self.obstacle_geoms = np.array([g for g in range(model.ngeom) if collidable(g)
                                        and model.geom_bodyid[g] not in self.arm_bodies])
        self.safe_dist = 0.03  # m, keep arm links this far from obstacles
        self.detect_dist = 0.12  # m, broad-phase radius
        self.col_weight = 3.0

    def set_state_provider(
        self,
        pose_provider: Callable[[], tuple[np.ndarray, np.ndarray]],
        joints_provider: Callable[[], np.ndarray],
    ) -> None:
        """Wire estimated base pose `(pos_3d, quat_wxyz)` and 29 encoder joints into IK."""
        self._pose_provider = pose_provider
        self._joints_provider = joints_provider

    def _sync_scratch(self, s: mujoco.MjData, q_arm: np.ndarray | None = None) -> None:
        """Populate scratch MjData using estimated robot state (or data.qpos fallback)."""
        if self._pose_provider is not None and self._joints_provider is not None:
            s.qpos[:] = self.model.qpos0
            for qadr, val in self.articulated_qpos.items():
                s.qpos[qadr] = val
            pos_3d, quat_wxyz = self._pose_provider()
            qa = self._root_q
            s.qpos[qa : qa + 3] = pos_3d
            s.qpos[qa + 3 : qa + 7] = quat_wxyz
            s.qpos[self._body_qpos_ids] = self._joints_provider()
        else:
            s.qpos[:] = self.data.qpos
        if q_arm is not None:
            s.qpos[self.qadr] = q_arm
        s.qvel[:] = 0.0

    def _is_descendant(self, b: int, root: int) -> bool:
        while b > 0:
            if b == root:
                return True
            b = self.model.body_parentid[b]
        return False

    def collisions(self, q: np.ndarray, margin: float = 0.01) -> list[tuple[str, str]]:
        """Contacts between this arm and anything else if the arm were at q (current estimated pose)."""
        s = self.scratch
        self._sync_scratch(s, q)
        mujoco.mj_forward(self.model, s)
        hits = []
        for i in range(s.ncon):
            c = s.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in self.arm_bodies) != (b2 in self.arm_bodies) and c.dist < margin:
                hits.append((self.model.body(b1).name, self.model.body(b2).name))
        return hits

    def _candidate_pairs(self, s: mujoco.MjData) -> list[tuple[int, int]]:
        """Broad phase: arm/obstacle geom pairs whose bounding spheres are within detect_dist."""
        a, o = self.arm_geoms, self.obstacle_geoms
        pa, po = s.geom_xpos[a], s.geom_xpos[o]
        ra, ro = self.model.geom_rbound[a], self.model.geom_rbound[o]
        ro = np.where(ro == 0, 1e6, ro)  # planes (floor) have rbound 0: always a candidate
        gap = np.linalg.norm(pa[:, None, :] - po[None, :, :], axis=2) - ra[:, None] - ro[None, :]
        pairs = []
        for i, j in zip(*np.nonzero(gap < self.detect_dist)):
            ga, go = int(a[i]), int(o[j])
            if ga in self.hand_geoms and self.model.geom_bodyid[go] in self.graspable_bodies:
                continue  # the hand may touch what it grasps
            pairs.append((ga, go))
        return pairs

    def _collision_rows(self, s: mujoco.MjData, pairs) -> tuple[list[np.ndarray], list[float], float]:
        """Rows J (1x7) and errors for pairs closer than safe_dist; also the minimum distance."""
        rows, errs, dmin = [], [], np.inf
        for ga, go in pairs:
            dist = mujoco.mj_geomDistance(self.model, s, ga, go, self.detect_dist, self._ft)
            dmin = min(dmin, dist)
            if dist >= self.safe_dist:
                continue
            p_arm, p_obs = self._ft[:3].copy(), self._ft[3:].copy()
            n = p_arm - p_obs
            nn = np.linalg.norm(n)
            if nn < 1e-6:  # touching: fall back to centre-to-centre direction
                n = s.geom_xpos[ga] - s.geom_xpos[go]
                nn = np.linalg.norm(n)
            n = n / max(nn, 1e-9)
            if dist < 0:
                n = -n  # penetrating: the closest points are swapped
            mujoco.mj_jac(self.model, s, self._jacc, None, p_arm, self.model.geom_bodyid[ga])
            rows.append(self.col_weight * (n @ self._jacc[:, self.vadr]))
            errs.append(self.col_weight * (self.safe_dist - dist))
        return rows, errs, dmin

    def min_clearance(self, q: np.ndarray) -> float:
        """Smallest arm-to-obstacle distance (m, negative = penetration) with the arm at q."""
        s = self.scratch
        self._sync_scratch(s, q)
        mujoco.mj_kinematics(self.model, s)
        return self._collision_rows(s, self._candidate_pairs(s))[2]

    def palm_pose(self, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        if data is not None:
            return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()
        if self._pose_provider is not None:
            s = self.scratch
            self._sync_scratch(s)
            mujoco.mj_kinematics(self.model, s)
            return s.site_xpos[self.site].copy(), s.site_xmat[self.site].reshape(3, 3).copy()
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def solve(
        self,
        target_pos: np.ndarray,
        axes: list[tuple[int, np.ndarray, float]] | None = None,
        q_init: np.ndarray | None = None,
        iters: int = 60,
        damping: float = 0.03,
        posture_weight: float = 0.02,
        tol: float = 0.004,
        avoid: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return (7 arm joint angles, final position error in metres).

        axes: list of (site axis index 0/1/2, desired world direction, weight).
        """
        s = self.scratch
        self._sync_scratch(s)
        q = s.qpos[self.qadr].copy() if q_init is None else np.array(q_init, dtype=float)
        axes = axes or []
        err_pos = np.inf
        s.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, s)
        pairs = self._candidate_pairs(s) if avoid else []
        for it in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
            if avoid and it and it % 10 == 0:
                pairs = self._candidate_pairs(s)  # the arm may have moved a lot
            p = s.site_xpos[self.site]
            R = s.site_xmat[self.site].reshape(3, 3)
            mujoco.mj_jacSite(self.model, s, self._jacp, self._jacr, self.site)
            Jp = self._jacp[:, self.vadr]
            Jr = self._jacr[:, self.vadr]

            e = [target_pos - p]
            J = [Jp]
            for idx, want, w in axes:
                have = R[:, idx]
                want = want / np.linalg.norm(want)
                e.append(w * np.cross(have, want))
                J.append(w * Jr)
            col_rows, col_errs, dmin = self._collision_rows(s, pairs) if pairs else ([], [], np.inf)
            if col_rows:
                J.append(np.array(col_rows))
                e.append(np.array(col_errs))
            e = np.concatenate(e)
            J = np.vstack(J)
            err_pos = float(np.linalg.norm(target_pos - p))
            if not col_rows and err_pos < tol and all(np.linalg.norm(np.cross(R[:, i], v / np.linalg.norm(v))) < 0.08 for i, v, _ in axes):
                break
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            # Null-space pull towards a comfortable posture
            N = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
            dq += N @ (posture_weight * (self.home - q))
            q = np.clip(q + np.clip(dq, -0.25, 0.25), self.lo, self.hi)
        return q, err_pos
