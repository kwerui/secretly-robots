"""Single-process Unitree G1 simulation driven by NVIDIA's GR00T decoupled WBC.

Lower body (legs + waist, 15 joints): GR00T Balance/Walk ONNX policies at 50 Hz.
Upper body (arms, 14 joints): PD to whatever joint targets you set (e.g. from IK).

No DDS, no unitree_sdk2py, no ZMQ: everything runs in one MuJoCo process, so the
commands that LeRobot only exposes through a gamepad are plain method calls here:

    sim.set_velocity(vx, vy, wz)   # m/s, m/s, rad/s in the pelvis frame
    sim.set_height(h)              # pelvis height command, 0.74 = standing
    sim.set_torso_rpy(r, p, y)     # torso orientation command (lean to reach)
    sim.set_arm_targets(q14)       # left 7 then right 7 arm joint targets

Observation layout, scales and gains are copied from
GR00T-WholeBodyControl/decoupled_wbc/control/policy/g1_gear_wbc_policy.py and
lerobot/robots/unitree_g1/controllers/gr00t_locomotion.py (they agree).
"""

from __future__ import annotations

import collections
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ASSET_DIR = Path(__file__).resolve().parent / "assets"

NUM_BODY = 29
NUM_LOWER = 15  # 12 legs + 3 waist
ARM_SLICE = slice(15, 29)

DEFAULT_ANGLES = np.zeros(NUM_BODY, dtype=np.float32)
DEFAULT_ANGLES[[0, 6]] = -0.1  # hip pitch
DEFAULT_ANGLES[[3, 9]] = 0.3  # knee
DEFAULT_ANGLES[[4, 10]] = -0.2  # ankle pitch

# Arm home pose: elbows bent, hands in front of the hips. Left 7 then right 7.
ARM_HOME = np.array(
    [0.2, 0.25, 0.0, 0.9, 0.0, 0.0, 0.0,
     0.2, -0.25, 0.0, 0.9, 0.0, 0.0, 0.0],
    dtype=np.float32,
)

# PD gains from decoupled_wbc/control/main/teleop/configs/g1_gear_wbc.yaml
KP = np.array(
    [150, 150, 150, 300, 40, 40,
     150, 150, 150, 300, 40, 40,
     250, 250, 250,
     100, 100, 40, 40, 20, 20, 20,
     100, 100, 40, 40, 20, 20, 20],
    dtype=np.float64,
)
KD = np.array(
    [2, 2, 2, 4, 2, 2,
     2, 2, 2, 4, 2, 2,
     5, 5, 5,
     5, 5, 2, 2, 2, 2, 2,
     5, 5, 2, 2, 2, 2, 2],
    dtype=np.float64,
)

ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
OBS_DIM = 86
HISTORY = 6

SIM_DT = 0.002
DECIMATION = 10  # 50 Hz policy

HEIGHT_MIN, HEIGHT_MAX = 0.20, 0.80  # the GR00T teleop streamers clip to [0.2, 0.74]


def gravity_in_body(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)],
        dtype=np.float32,
    )


class G1WBC:
    """Owns a MuJoCo model/data pair containing the G1 and steps it under GR00T control."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, policy_dir: Path | str | None = None,
                 arm_gravity_comp: bool = True):
        self.arm_gravity_comp = arm_gravity_comp
        self.model = model
        self.data = data
        model.opt.timestep = SIM_DT

        policy_dir = Path(policy_dir) if policy_dir else ASSET_DIR / "policy"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._balance = ort.InferenceSession(str(policy_dir / "GR00T-WholeBodyControl-Balance.onnx"), so)
        self._walk = ort.InferenceSession(str(policy_dir / "GR00T-WholeBodyControl-Walk.onnx"), so)

        # Indices of the 29 body joints in qpos/qvel and of their actuators in ctrl.
        jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        self._act_ids = []
        self._qpos_ids = []
        self._qvel_ids = []
        for name in BODY_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name} missing from model")
            self._qpos_ids.append(model.jnt_qposadr[jid])
            self._qvel_ids.append(model.jnt_dofadr[jid])
            self._act_ids.append(jnames.index(name))
        self._qpos_ids = np.array(self._qpos_ids)
        self._qvel_ids = np.array(self._qvel_ids)
        self._act_ids = np.array(self._act_ids)

        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._root_q = model.jnt_qposadr[root]
        self._root_v = model.jnt_dofadr[root]
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        self.reset_controller()

    # ------------------------------------------------------------------ commands
    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        self.cmd[:] = (vx, vy, wz)

    def set_height(self, h: float) -> None:
        self.height_cmd = float(np.clip(h, HEIGHT_MIN, HEIGHT_MAX))

    def set_torso_rpy(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> None:
        self.rpy_cmd[:] = (roll, pitch, yaw)

    def set_arm_targets(self, q14: np.ndarray) -> None:
        self.target_q[ARM_SLICE] = q14

    # ------------------------------------------------------------------ state
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self._qpos_ids].copy()

    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self._qpos_ids[ARM_SLICE]].copy()

    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[self._root_q:self._root_q + 3].copy()

    @property
    def base_quat(self) -> np.ndarray:
        return self.data.qpos[self._root_q + 3:self._root_q + 7].copy()

    @property
    def base_yaw(self) -> float:
        w, x, y, z = self.base_quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def base_vel_body(self) -> np.ndarray:
        """(vx, vy, wz) of the pelvis in its own heading frame."""
        v = self.data.qvel[self._root_v:self._root_v + 3]  # world frame for a free joint
        th = self.base_yaw
        vx = np.cos(th) * v[0] + np.sin(th) * v[1]
        vy = -np.sin(th) * v[0] + np.cos(th) * v[1]
        return np.array([vx, vy, self.data.qvel[self._root_v + 5]])

    def fallen(self) -> bool:
        return self.base_pos[2] < 0.25 or gravity_in_body(self.base_quat)[2] > -0.5

    # ------------------------------------------------------------------ loop
    def reset_controller(self) -> None:
        self.cmd = np.zeros(3, dtype=np.float32)
        self.height_cmd = 0.74
        self.rpy_cmd = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(NUM_LOWER, dtype=np.float32)
        self.target_q = DEFAULT_ANGLES.astype(np.float64).copy()
        self.target_q[ARM_SLICE] = ARM_HOME
        self.history = collections.deque([np.zeros(OBS_DIM, np.float32)] * HISTORY, maxlen=HISTORY)
        self._tick = 0

    def place(self, x: float, y: float, yaw: float = 0.0, z: float = 0.793) -> None:
        """Teleport the robot to a standing pose (use before the first step)."""
        qa = self._root_q
        self.data.qpos[qa:qa + 3] = (x, y, z)
        self.data.qpos[qa + 3:qa + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        self.data.qpos[self._qpos_ids] = self.target_q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _observe(self) -> np.ndarray:
        d = self.data
        qj = d.qpos[self._qpos_ids].astype(np.float32)
        dqj = d.qvel[self._qvel_ids].astype(np.float32)
        omega = d.qvel[self._root_v + 3:self._root_v + 6].astype(np.float32)  # body frame
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = self.cmd * CMD_SCALE
        obs[3] = self.height_cmd
        obs[4:7] = self.rpy_cmd
        obs[7:10] = omega * ANG_VEL_SCALE
        obs[10:13] = gravity_in_body(self.base_quat)
        obs[13:42] = qj - DEFAULT_ANGLES
        obs[42:71] = dqj * DOF_VEL_SCALE
        obs[71:86] = self.action
        return obs

    def _policy_step(self) -> None:
        self.history.append(self._observe())
        stacked = np.concatenate(self.history)[None, :]
        sess = self._balance if np.linalg.norm(self.cmd) < 0.05 else self._walk
        self.action = sess.run(None, {sess.get_inputs()[0].name: stacked})[0].squeeze().astype(np.float32)
        self.target_q[:NUM_LOWER] = DEFAULT_ANGLES[:NUM_LOWER] + self.action * ACTION_SCALE

    def step(self, n: int = 1) -> None:
        """Advance n physics steps (2 ms each); the policy runs every 10th."""
        d = self.data
        for _ in range(n):
            if self._tick % DECIMATION == 0:
                self._policy_step()
            q = d.qpos[self._qpos_ids]
            dq = d.qvel[self._qvel_ids]
            tau = KP * (self.target_q - q) - KD * dq
            if self.arm_gravity_comp:
                # Feed-forward gravity/Coriolis on the arms (the real G1 stack does this too,
                # see LeRobot's gravity_compensation option); without it the soft arm gains sag.
                tau[ARM_SLICE] += d.qfrc_bias[self._qvel_ids[ARM_SLICE]]
            d.ctrl[self._act_ids] = tau
            mujoco.mj_step(self.model, d)
            self._tick += 1

    def step_seconds(self, t: float) -> None:
        self.step(int(round(t / SIM_DT)))


BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
