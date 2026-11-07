"""Damped-least-squares IK for one G1 arm (7 joints) on the live MuJoCo model.

Solves on a scratch MjData copy of the current state, so the base/legs/waist are taken
as they are right now and only the arm's joints move. Re-solving every control tick
(closed loop) absorbs the small sway of the balancing lower body.

Targets: palm position (world) plus optional axis alignments, e.g.
    fingers (site x-axis) -> approach direction
    palm side (site z-axis) -> world up
which is enough for side grasps and handle pulls without a full 6D target.
"""

from __future__ import annotations

import mujoco
import numpy as np

from g1_wbc import BODY_JOINTS, ARM_HOME

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
        # Bodies of this arm (shoulder down to the fingers), for collision checks
        first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")
        self.arm_bodies = {b for b in range(model.nbody) if self._is_descendant(b, first)}
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _is_descendant(self, b: int, root: int) -> bool:
        while b > 0:
            if b == root:
                return True
            b = self.model.body_parentid[b]
        return False

    def collisions(self, q: np.ndarray, margin: float = 0.01) -> list[tuple[str, str]]:
        """Contacts between this arm and anything else if the arm were at q (current base pose)."""
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, s)
        hits = []
        for i in range(s.ncon):
            c = s.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in self.arm_bodies) != (b2 in self.arm_bodies) and c.dist < margin:
                hits.append((self.model.body(b1).name, self.model.body(b2).name))
        return hits

    def palm_pose(self, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = data or self.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(
        self,
        target_pos: np.ndarray,
        axes: list[tuple[int, np.ndarray, float]] | None = None,
        q_init: np.ndarray | None = None,
        iters: int = 60,
        damping: float = 0.03,
        posture_weight: float = 0.02,
        tol: float = 0.004,
    ) -> tuple[np.ndarray, float]:
        """Return (7 arm joint angles, final position error in metres).

        axes: list of (site axis index 0/1/2, desired world direction, weight).
        """
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0
        q = s.qpos[self.qadr].copy() if q_init is None else np.array(q_init, dtype=float)
        axes = axes or []
        err_pos = np.inf
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
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
            e = np.concatenate(e)
            J = np.vstack(J)
            err_pos = float(np.linalg.norm(target_pos - p))
            if err_pos < tol and all(np.linalg.norm(np.cross(R[:, i], v / np.linalg.norm(v))) < 0.08 for i, v, _ in axes):
                break
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            # Null-space pull towards a comfortable posture
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (posture_weight * (self.home - q))
            q = np.clip(q + np.clip(dq, -0.25, 0.25), self.lo, self.hi)
        return q, err_pos
