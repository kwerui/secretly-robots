"""Learn how a door or drawer moves from ONE demonstration.

Input: a sequence of poses of a marker on the moving part (door/drawer front), expressed
in the frame of a marker fixed to the furniture (the "anchor"). Working in the anchor
frame means the model is valid wherever the robot (or camera) stands later.

Output: an ArticulationModel
  revolute:  hinge axis (point + direction) and the opening angle the carer used
  prismatic: slide direction and the opening distance the carer used
plus the grasp point on the moving part (where the carer's hand was), so the robot can
regenerate its own smooth hand path instead of replaying noisy samples.

Model selection: fit both a circle (in the plane of the trajectory) and a line to the
marker positions; the revolute model wins if it fits better and the arc subtends more
than ~10 degrees. Positions are used rather than marker orientations because ArUco
orientation is noisy at grazing view angles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


@dataclass
class ArticulationModel:
    kind: str  # "revolute" or "prismatic"
    axis_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # anchor frame
    axis_dir: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])  # anchor frame, unit
    amount: float = 0.0  # radians (revolute) or metres (prismatic) the carer opened it
    grasp_in_part: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # grasp point, part-marker frame
    grasp_start: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # grasp point at amount=0, anchor frame
    part_normal_start: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])  # part marker normal at amount=0
    residual_m: float = 0.0  # RMS fit error of the marker positions
    n_samples: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ArticulationModel":
        return cls(**json.loads(Path(path).read_text()))

    # ---------------------------------------------------------------- geometry
    def transform(self, amount: float) -> tuple[np.ndarray, np.ndarray]:
        """Rigid motion (R, t) of the moving part, in the anchor frame, after opening by `amount`."""
        a = np.asarray(self.axis_dir)
        if self.kind == "revolute":
            R = rot_axis_angle(a, amount)
            c = np.asarray(self.axis_point)
            return R, c - R @ c
        return np.eye(3), a * amount

    def path(self, start_point: np.ndarray, n: int = 40, fraction: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
        """(R, point) along the opening motion for a point given in the anchor frame at amount=0."""
        out = []
        for s in np.linspace(0.0, fraction, n):
            R, t = self.transform(s * self.amount)
            out.append((R, R @ start_point + t))
        return out


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def axis_angle_from_R(R: np.ndarray) -> tuple[np.ndarray, float]:
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis / np.linalg.norm(axis), float(angle)


def _fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) circle fit: returns centre (2,) and radius."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre = sol[:2]
    radius = float(np.sqrt(sol[2] + centre @ centre))
    # Geometric refinement (Gauss-Newton on distance-to-circle): Kasa is biased on short arcs.
    for _ in range(30):
        diff = xy - centre
        dist = np.linalg.norm(diff, axis=1) + 1e-12
        r = dist - radius
        J = np.column_stack([-diff / dist[:, None], -np.ones(len(xy))])
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        centre = centre + step[:2]
        radius = float(radius + step[2])
        if np.linalg.norm(step) < 1e-7:
            break
    return centre, abs(radius)


def fit_articulation(
    rotations: np.ndarray,  # (N,3,3) part-marker orientation in the anchor frame
    positions: np.ndarray,  # (N,3) part-marker position in the anchor frame
    grasp_in_part: np.ndarray | None = None,
    revolute_threshold_deg: float = 10.0,
) -> ArticulationModel:
    """Fit from marker positions (robust); orientations only pick the grasp frame.

    Revolute: plane through the trajectory -> axis direction; circle in that plane ->
    hinge point and radius; subtended angle -> how far the carer opened it.
    Prismatic: principal direction of the trajectory -> slide axis.
    The model with the lower residual wins (a straight line is a circle of huge radius,
    so a revolute fit must also subtend more than `revolute_threshold_deg`).
    """
    rotations = np.asarray(rotations, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 5:
        raise ValueError("need at least 5 samples of the moving marker")
    p0 = positions[0]
    centroid = positions.mean(axis=0)
    _, svals, vt = np.linalg.svd(positions - centroid)

    # ---- prismatic candidate
    direction = vt[0] if vt[0] @ (positions[-1] - p0) >= 0 else -vt[0]
    d = positions - p0
    proj = d @ direction
    pris_res = float(np.sqrt(np.mean(np.sum((d - np.outer(proj, direction)) ** 2, axis=1))))
    prismatic = ArticulationModel("prismatic", p0.tolist(), direction.tolist(), float(proj.max()), residual_m=pris_res)

    # ---- revolute candidate
    normal = vt[2]
    e1 = vt[0]
    e2 = np.cross(normal, e1)
    xy = np.column_stack([(positions - centroid) @ e1, (positions - centroid) @ e2])
    c2, radius = _fit_circle_2d(xy)
    centre = centroid + c2[0] * e1 + c2[1] * e2
    rel = xy - c2
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    ang -= ang[0]
    if ang[np.argmax(np.abs(ang))] < 0:  # make opening a positive rotation about the axis
        normal, ang = -normal, -ang
    # Robust opening amount: largest value of the angle smoothed over ~5 samples
    k = min(5, len(ang))
    amount = float(np.convolve(ang, np.ones(k) / k, mode="valid").max())
    revolute = ArticulationModel("revolute", centre.tolist(), normal.tolist(), amount)
    pred = np.array([revolute.transform(a)[0] @ (p0 - centre) + centre for a in ang])
    revolute.residual_m = float(np.sqrt(np.mean(np.sum((pred - positions) ** 2, axis=1))))

    use_revolute = np.rad2deg(amount) >= revolute_threshold_deg and radius < 2.0 and revolute.residual_m < pris_res
    model = revolute if use_revolute else prismatic
    model.n_samples = n

    R0 = rotations[0]
    if grasp_in_part is not None:
        g = np.asarray(grasp_in_part, dtype=float)
        model.grasp_in_part = g.tolist()
        model.grasp_start = (R0 @ g + p0).tolist()
    model.part_normal_start = R0[:, 2].tolist()
    return model


def relative_pose(R_anchor: np.ndarray, p_anchor: np.ndarray, R_part: np.ndarray, p_part: np.ndarray):
    """Pose of the part marker expressed in the anchor marker frame."""
    return R_anchor.T @ R_part, R_anchor.T @ (p_part - p_anchor)
