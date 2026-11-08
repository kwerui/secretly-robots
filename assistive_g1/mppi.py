"""Sampling-based MPC (MPPI) for the G1 arm's transport motions.

Model Predictive Path Integral control (Williams et al. 2017) in the style of NVIDIA's STORM
(Bhardwaj et al. 2021): the arm is planned in joint space with a kinematic model, rollouts are
scored on the live MuJoCo kinematics, and the executed command is the exponentially weighted
average of the sampled control sequences. Re-planned every control tick from the warm-started
previous plan, so the result is a receding-horizon controller, not an open-loop trajectory.

    state      q (7 arm joints), commanded at 50 Hz
    control    joint velocities u_t over a horizon of H ticks (time-correlated Gaussian samples)
    dynamics   q_{t+1} = q_t + u_t * dt   (the arm PD + gravity feed-forward tracks q closely)
    cost       grasp-site position error (running + terminal), hand-axis alignment,
               collision clearance of the whole arm (MuJoCo signed distances at a few knots),
               joint limits, velocity, acceleration (smoothness) and jerk-ish (change vs. the
               previous plan)

What it buys over the per-tick damped-least-squares IK used for the final centimetres: the IK
reacts to obstacles only when a link is already within `safe_dist`, which shows up as a kink
in the motion; MPPI sees the obstacle over the whole horizon and bends the path earlier and
more gently, and its explicit acceleration cost gives smooth joint-space profiles instead of a
Cartesian smoothstep forced through an inverse kinematics. The IK is still used for the final
precise settle at a grasp or placement, where accuracy matters more than smoothness.

Cost of the honesty: rollouts call MuJoCo's forward kinematics K*H times per re-plan (pure
Python loop). K=32, H=16 is ~30-50 ms per plan on a laptop core, so the planner re-plans every
`replan_every` ticks and holds the plan in between. Nothing is learned; nothing is a policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ik import ArmIK


@dataclass
class MPPIParams:
    K: int = 32  # samples
    H: int = 16  # horizon, ticks
    dt: float = 0.02
    lam: float = 0.12  # temperature, relative to the cost spread of the batch (see plan())
    sigma: float = 0.7  # rad/s, sampling std of the joint velocities
    noise_smooth: float = 0.6  # one-pole low-pass on the noise along time (0 = white)
    v_max: float = 1.2  # rad/s
    w_pos: float = 80.0  # running cost on tracking the moving reference (m^2)
    w_pos_T: float = 200.0  # terminal
    w_axis: float = 6.0
    w_coll: float = 600.0  # per metre of safe-distance violation, squared
    w_vel: float = 0.01
    w_acc: float = 0.08  # on u_t - u_{t-1} (mild: smoothness mostly comes from the correlated
    # noise and from filtering the updated plan, as in STORM; a heavy acceleration penalty just
    # makes every noisy sample lose to the unperturbed one and the planner freezes)
    w_lim: float = 50.0
    w_post: float = 0.3  # stay near the arm's comfortable posture (keeps the elbow down)
    plan_smooth: float = 0.5  # one-pole low-pass applied to the updated plan along time
    coll_knots: tuple = (2, 5, 8, 11, 14)  # horizon steps at which collisions are checked
    replan_every: int = 2
    seed: int = 0


class ArmMPPI:
    def __init__(self, ik: ArmIK, params: MPPIParams | None = None):
        self.ik = ik
        self.p = params or MPPIParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.U = np.zeros((self.p.H, 7))  # nominal control sequence (warm start)
        self.U_prev = np.zeros((self.p.H, 7))
        self._tick = 0
        self._q_cmd: np.ndarray | None = None
        self.stats = {"plans": 0, "plan_time_s": 0.0}

    def reset(self) -> None:
        self.U[:] = 0.0
        self.U_prev[:] = 0.0
        self._tick = 0
        self._q_cmd = None

    # ------------------------------------------------------------------ costs
    def _site_cost(self, s: mujoco.MjData, target: np.ndarray, axes) -> tuple[float, float]:
        p = s.site_xpos[self.ik.site]
        e = float(np.sum((p - target) ** 2))
        a = 0.0
        if axes:
            R = s.site_xmat[self.ik.site].reshape(3, 3)
            for idx, want, w in axes:
                want = want / np.linalg.norm(want)
                have = R[:, idx] if np.isscalar(idx) else R @ np.asarray(idx, float)
                a += w * float(np.sum(np.cross(have, want) ** 2))
        return e, a

    def _coll_cost(self, s: mujoco.MjData, pairs) -> float:
        if not pairs:
            return 0.0
        cost = 0.0
        safe = self.ik.safe_dist
        for ga, go in pairs:
            d = mujoco.mj_geomDistance(self.ik.model, s, ga, go, self.ik.detect_dist, self.ik._ft)
            if d < safe:
                cost += (safe - d) ** 2
        return cost

    # ------------------------------------------------------------------ plan
    def plan(self, q0: np.ndarray, target: np.ndarray, axes=None, u_nom: np.ndarray | None = None) -> np.ndarray:
        """Return the arm joint target for the next tick.

        target: either one point (3,) the site should reach, or a reference path (H, 3) giving
        the point the site should be at for each step of the horizon (a moving reference makes
        the pace explicit; the planner then spends its freedom on smoothness and clearance).
        u_nom: optional nominal joint velocity (7,) for this tick, e.g. from the one-step IK
        solution. The sampling distribution is centred on a blend of the warm-started plan and
        this nominal, so the planner refines a path that already tracks, instead of having to
        discover the tracking itself from random samples (which is slow with K=32).
        """
        p = self.p
        target = np.asarray(target, float)
        ref = np.repeat(target[None], p.H, axis=0) if target.ndim == 1 else target
        if u_nom is not None:
            u_nom = np.clip(np.asarray(u_nom, float), -p.v_max, p.v_max)
            self.U = 0.3 * self.U + 0.7 * u_nom[None, :]
        self._tick += 1
        if self._q_cmd is not None and (self._tick - 1) % p.replan_every:
            # hold the current plan: advance one step along it
            self.U = np.roll(self.U, -1, axis=0)
            self.U[-1] = self.U[-2]
            self._q_cmd = np.clip(q0 + self.U[0] * p.dt, self.ik.lo, self.ik.hi)
            return self._q_cmd
        t0 = time.perf_counter()
        s = self.ik.scratch
        s.qpos[:] = self.ik.data.qpos  # base/legs/waist as they are now
        s.qpos[self.ik.qadr] = q0
        mujoco.mj_kinematics(self.ik.model, s)
        # broad phase at the current pose and at the end of the warm-started plan, so an obstacle
        # the arm is heading for is costed before it is within the detection radius
        pairs = set(self.ik._candidate_pairs(s))
        s.qpos[self.ik.qadr] = np.clip(q0 + np.sum(self.U, axis=0) * p.dt, self.ik.lo, self.ik.hi)
        mujoco.mj_kinematics(self.ik.model, s)
        pairs = list(pairs | set(self.ik._candidate_pairs(s)))

        # time-correlated noise
        eps = self.rng.normal(0.0, p.sigma, size=(p.K, p.H, 7))
        if p.noise_smooth > 0:
            a = p.noise_smooth
            for t in range(1, p.H):
                eps[:, t] = a * eps[:, t - 1] + (1 - a) * eps[:, t]
            eps *= 1.0 / np.sqrt((1 - a) ** 2 / (1 - a * a))  # keep the std at sigma
        U = np.clip(self.U[None] + eps, -p.v_max, p.v_max)  # K x H x 7
        U[0] = self.U  # always keep the unperturbed warm start as one sample
        Q = q0[None, None, :] + np.cumsum(U * p.dt, axis=1)  # K x H x 7
        lim_viol = np.maximum(0.0, self.ik.lo - Q) + np.maximum(0.0, Q - self.ik.hi)
        Q = np.clip(Q, self.ik.lo, self.ik.hi)

        S = np.zeros(p.K)
        # smoothness / effort (vectorised)
        dU = np.diff(np.concatenate([self.U_prev[None, :1].repeat(p.K, 0), U], axis=1), axis=1)
        S += p.w_acc * np.sum(dU ** 2, axis=(1, 2))
        S += p.w_vel * np.sum(U ** 2, axis=(1, 2))
        S += p.w_lim * np.sum(lim_viol ** 2, axis=(1, 2))
        # kinematic costs on the MuJoCo model
        knots = set(k for k in p.coll_knots if k < p.H)
        S += p.w_post * np.sum((Q - self.ik.home[None, None, :]) ** 2, axis=(1, 2)) / p.H
        for k in range(p.K):
            for t in range(p.H):
                s.qpos[self.ik.qadr] = Q[k, t]
                mujoco.mj_kinematics(self.ik.model, s)
                e, a = self._site_cost(s, ref[t], axes)
                w = p.w_pos_T if t == p.H - 1 else p.w_pos
                S[k] += w * e + p.w_axis * a
                if t in knots:
                    S[k] += p.w_coll * self._coll_cost(s, pairs)
        # path-integral update (temperature relative to the batch's cost spread, so the
        # effective number of contributing samples is stable whatever the cost scale)
        spread = max(float(S.max() - S.min()), 1e-6)
        w = np.exp(-(S - S.min()) / (p.lam * spread))
        w /= w.sum()
        self.U_prev = self.U.copy()
        Unew = np.einsum("k,khj->hj", w, U)
        if p.plan_smooth > 0:  # filter the plan along time (STORM-style), keeps the mean
            a = p.plan_smooth
            for t in range(1, p.H):
                Unew[t] = a * Unew[t - 1] + (1 - a) * Unew[t]
        self.U = Unew
        self._q_cmd = np.clip(q0 + self.U[0] * p.dt, self.ik.lo, self.ik.hi)
        # shift for the next tick's warm start
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = self.U[-2]
        self.stats["plans"] += 1
        self.stats["plan_time_s"] += time.perf_counter() - t0
        return self._q_cmd


@dataclass
class MotionLog:
    """Joint-target history for smoothness metrics (RMS acceleration / jerk of the commands)."""

    dt: float = 0.02
    q: list = field(default_factory=list)
    site: list = field(default_factory=list)

    def add(self, q: np.ndarray, site: np.ndarray) -> None:
        self.q.append(np.array(q, float))
        self.site.append(np.array(site, float))

    def metrics(self) -> dict:
        if len(self.q) < 4:
            return {}
        q = np.array(self.q)
        v = np.diff(q, axis=0) / self.dt
        a = np.diff(v, axis=0) / self.dt
        j = np.diff(a, axis=0) / self.dt
        s = np.array(self.site)
        sv = np.linalg.norm(np.diff(s, axis=0), axis=1) / self.dt
        sa = np.diff(sv) / self.dt
        return {
            "ticks": len(q),
            "joint_acc_rms": float(np.sqrt(np.mean(a ** 2))),
            "joint_acc_peak": float(np.max(np.abs(a))),
            "joint_jerk_rms": float(np.sqrt(np.mean(j ** 2))),
            "site_speed_max_mps": float(sv.max()),
            "site_acc_rms_mps2": float(np.sqrt(np.mean(sa ** 2))),
            "site_acc_peak_mps2": float(np.max(np.abs(sa))),
        }
