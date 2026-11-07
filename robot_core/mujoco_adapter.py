"""Minimal real MuJoCo adapter: no fake perception or motion control."""
from pathlib import Path
import mujoco
from .contracts import Action, Observation

class MuJoCoAdapter:
    def __init__(self, scene_path: str | Path, robot_body: str):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body) < 0:
            raise ValueError(f"Unknown robot body {robot_body!r}")
        self.robot_body = robot_body
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> Observation:
        position = self.data.body(self.robot_body).xpos.copy()
        return Observation(
            time_s=float(self.data.time),
            robot_position_xyz=tuple(float(x) for x in position),
        )

    def execute(self, action: Action) -> None:
        if action.kind != "stop":
            raise NotImplementedError(
                f"Action {action.kind!r} needs a robot-specific controller; "
                "do not pretend this is implemented yet."
            )
        # 'stop' means no new control command here. For a moving robot,
        # teammates must implement braking/holding behavior in the controller.

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
