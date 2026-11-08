"""Articulated Dex3-1 hand for the G1: real finger joints, position actuators, grasp contacts.

The GR00T `g1.xml` ships the Dex3 three-finger hand as rigid meshes (all 7 finger joints and
their motors are commented out), which is why the first version of this project welded items
to the palm. This module puts the joints back at scene-build time (MjSpec), adds position
actuators with the real hardware's torque limits, and tunes the fingertip contacts so a pinch
holds under gravity and walking vibration. Nothing here is a constraint: the item is held by
friction between the finger pads and the item, the same way the physical hand would hold it.

Hand frame = `<side>_wrist_yaw_link`: fingers extend along +x; index at z=+0.0285, middle at
z=-0.0285, both curl towards +y; the thumb sits on the +y side and curls towards -y/+x. An item
pinched between them sits at roughly x=0.14..0.16, y=+0.02..0.03 in that frame.

Contact recipe (MuJoCo docs, "Preventing slip"): condim 4 (torsional friction so a pinched
item cannot spin about the contact normal), elliptic friction cones, impratio ~10, Newton
solver, slightly soft fingertip solref. `scene.py` sets the global options; this module sets the
per-geom values on the finger pads and item geoms.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# (body, joint, axis, range, hardware torque limit) from the commented-out lines of g1.xml.
# Order: thumb_0 (abduction, about y), thumb_1, thumb_2, middle_0, middle_1, index_0, index_1.
RIGHT_FINGERS = [
    ("right_hand_thumb_0_link", "right_hand_thumb_0_joint", (0, 1, 0), (-1.0472, 1.0472), 2.45),
    ("right_hand_thumb_1_link", "right_hand_thumb_1_joint", (0, 0, 1), (-1.0472, 0.724312), 1.4),
    ("right_hand_thumb_2_link", "right_hand_thumb_2_joint", (0, 0, 1), (-1.74533, 0.0), 1.4),
    ("right_hand_middle_0_link", "right_hand_middle_0_joint", (0, 0, 1), (0.0, 1.5708), 1.4),
    ("right_hand_middle_1_link", "right_hand_middle_1_joint", (0, 0, 1), (0.0, 1.74533), 1.4),
    ("right_hand_index_0_link", "right_hand_index_0_joint", (0, 0, 1), (0.0, 1.5708), 1.4),
    ("right_hand_index_1_link", "right_hand_index_1_joint", (0, 0, 1), (0.0, 1.74533), 1.4),
]
FINGER_JOINTS = [f[1] for f in RIGHT_FINGERS]

# Finger postures (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1).
# OPEN: thumb swung away (+thumb_1 = away from the fingers), fingers straight: ~9 cm aperture.
Q_OPEN = np.array([0.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0])
# CLOSE_TARGET: well past where the pads meet a 6-7 cm item. The position actuators saturate at
# the hardware torque limit, so this is "squeeze as hard as the hand can" (~1.4 N*m / 5 cm lever
# ~ 20-30 N per pad), not a precise pose. Thumb curls in (negative), fingers curl in (positive).
Q_CLOSE = np.array([0.0, -0.9, -1.3, 1.1, 0.9, 1.1, 0.9])
# Measured hand geometry (hand frame, OPEN): finger links y in [-0.017, 0.010], x 0.11-0.22;
# the straight thumb swings through y 0.04-0.11, so there is no "thumb in first" pinch for a
# 6 cm item: the working grasp is a wrap where the fingers curl the item against the thumb's
# curled distal pad and the thumb base/palm front (item centre ends near x 0.10, y 0.06-0.075).

# Grasp site in the hand frame: where the centre of a 6-7 cm item ends up when pinched between
# the thumb and the finger pads (measured in sim: x 0.094-0.10, y 0.073-0.079). With the hand
# OPEN the thumb is swung back (-x, +y) and the fingers point along +x, so an item placed here
# clears both and is enclosed when the thumb swings in and the fingers curl.
GRASP_SITE_POS = (0.105, 0.072, 0.0)

# Release posture: fingers straight and thumb uncurled, but thumb_1 NOT swung back: the swing-back
# arc passes through the item, so opening fully would shove it over. Swing back after retreating.
Q_RELEASE = np.array([0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0])

# Finger-pad contact parameters (take priority over the item's own values).
PAD_FRICTION = (1.5, 0.05, 0.002)  # sliding, torsional, rolling
PAD_SOLREF = (0.012, 1.0)  # a touch softer than the 0.02 default: wider contact, no bounce
PAD_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)

KP_FINGER = 4.0  # N*m/rad position gain (the hardware limits the output torque, not the gain)
KV_FINGER = 0.15


def add_dex3_fingers(spec: mujoco.MjSpec, side: str = "right") -> list[str]:
    """Re-enable the Dex3 finger joints on `side` and add position actuators. Returns joint names."""
    fingers = RIGHT_FINGERS if side == "right" else _mirror(RIGHT_FINGERS)
    names = []
    for body, jname, axis, rng, frc in fingers:
        b = spec.body(body)
        b.add_joint(name=jname, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), range=list(rng),
                    damping=0.05, armature=0.0005, frictionloss=0.01)
        a = spec.add_actuator(name=jname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        # (assign whole arrays: MjSpec returns copies, so item assignment would be lost)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE  # position servo: force = kp*(ctrl - q) - kv*qdot
        a.gainprm = [KP_FINGER] + [0.0] * 9
        a.biasprm = [0.0, -KP_FINGER, -KV_FINGER] + [0.0] * 7
        a.ctrlrange = list(rng)
        a.ctrllimited = True
        a.forcerange = [-frc, frc]
        a.forcelimited = True
        names.append(jname)
        # Finger-pad contacts: the collision geoms are the mesh geoms with contype != 0.
        for g in b.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue
            g.condim = 4
            g.priority = 1
            g.friction = list(PAD_FRICTION)
            g.solref = list(PAD_SOLREF)
            g.solimp = list(PAD_SOLIMP)
    # The palm plate too (an item pulled into the palm rests on it).
    for g in spec.body(f"{side}_wrist_yaw_link").geoms:
        if g.contype or g.conaffinity:
            g.condim = 4
            g.priority = 1
            g.friction = list(PAD_FRICTION)
    spec.body(f"{side}_wrist_yaw_link").add_site(name=f"{side}_grasp", pos=list(GRASP_SITE_POS), size=[0.008],
                                                  rgba=[0.1, 0.9, 0.2, 0.0])
    # Hand-internal pairs: MuJoCo only skips parent-child contacts, but the finger meshes sit
    # inside the palm hull at their joints and the thumb links nest. Exclude palm vs every
    # finger link and the grandparent pairs along each finger.
    palm = f"{side}_wrist_yaw_link"
    links = [f[0] for f in fingers]
    for link in links:
        spec.add_exclude(bodyname1=palm, bodyname2=link)
    for a, b in ((0, 2), (3, 4), (5, 6)):
        spec.add_exclude(bodyname1=links[a], bodyname2=links[b])
    spec.add_exclude(bodyname1=links[0], bodyname2=links[1])
    spec.add_exclude(bodyname1=links[1], bodyname2=links[2])
    # the two long fingers lie side by side 5.7 cm apart and never meet
    spec.add_exclude(bodyname1=links[3], bodyname2=links[5])
    spec.add_exclude(bodyname1=links[4], bodyname2=links[6])
    spec.add_exclude(bodyname1=links[3], bodyname2=links[6])
    spec.add_exclude(bodyname1=links[4], bodyname2=links[5])
    return names


def _mirror(fingers):
    out = []
    for body, jname, axis, rng, frc in fingers:
        body, jname = body.replace("right", "left"), jname.replace("right", "left")
        lo, hi = rng
        out.append((body, jname, axis, (-hi, -lo) if axis[2] else rng, frc))
    return out


def item_contact_params(geom) -> None:
    """Contact parameters for a graspable item geom (MjSpec geom)."""
    geom.condim = 4
    geom.friction = [1.2, 0.02, 0.001]
    geom.solref = [0.01, 1.0]


@dataclass
class ContactSummary:
    n_pads: int  # finger pads touching the item
    thumb: bool
    fingers: int  # index/middle links touching
    normal_force: float  # sum of normal forces, N
    opposing: bool  # thumb on one side and at least one finger on the other


class Gripper:
    """Position control of the finger joints on the live model, plus contact queries."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str = "right"):
        self.model, self.data, self.side = model, data, side
        self.joints = FINGER_JOINTS if side == "right" else [j.replace("right", "left") for j in FINGER_JOINTS]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in self.joints])
        self.lo, self.hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
        names = {b: model.body(b).name for b in range(model.nbody)}
        self.thumb_bodies = {b for b, n in names.items() if n.startswith(f"{side}_hand_thumb")}
        self.finger_bodies = {b for b, n in names.items()
                              if n.startswith(f"{side}_hand_index") or n.startswith(f"{side}_hand_middle")}
        self.palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        self.hand_bodies = self.thumb_bodies | self.finger_bodies | {self.palm_body}
        self.target = Q_OPEN.copy()
        self._f6 = np.zeros(6)
        self.set_target(Q_OPEN)

    # ------------------------------------------------------------------ commands
    def set_target(self, q: np.ndarray) -> None:
        self.target = np.clip(np.asarray(q, float), self.lo, self.hi)
        self.data.ctrl[self.act] = self.target

    def open(self) -> None:
        self.set_target(Q_OPEN)

    def close(self) -> None:
        self.set_target(Q_CLOSE)

    def blend(self, a: float, open_pose: np.ndarray | None = None) -> None:
        """0 = open, 1 = closed (gradual close; the fingers reach the item first and press it
        against the thumb pad, which curls in from its swung-back position)."""
        q0 = Q_OPEN if open_pose is None else open_pose
        self.set_target((1 - a) * q0 + a * Q_CLOSE)

    def release_pose(self) -> None:
        self.set_target(Q_RELEASE)

    # ------------------------------------------------------------------ state
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def grasp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def stalled(self, tol: float = 0.05) -> bool:
        """Fingers have stopped short of their targets (they are pressing on something)."""
        q = self.q
        err = np.abs(self.target - q)
        # ignore the abduction joint; a stalled finger is >tol rad short of its target
        return bool(np.any(err[1:] > tol)) and bool(np.all(np.abs(self.data.qvel[self.qadr_v][1:]) < 0.5))

    @property
    def qadr_v(self) -> np.ndarray:
        if not hasattr(self, "_vadr"):
            jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
            self._vadr = np.array([self.model.jnt_dofadr[j] for j in jids])
        return self._vadr

    def contacts(self, body_name: str) -> ContactSummary:
        """Which parts of the hand touch `body_name` right now, and how hard."""
        m, d = self.model, self.data
        target = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        thumb, fingers, force, n = False, 0, 0.0, 0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if target not in (b1, b2):
                continue
            other = b2 if b1 == target else b1
            if other not in self.hand_bodies or c.dist > 0.002:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            force += abs(self._f6[0])
            n += 1
            if other in self.thumb_bodies:
                thumb = True
            elif other in self.finger_bodies:
                fingers += 1
        return ContactSummary(n, thumb, fingers, float(force), thumb and fingers > 0)
