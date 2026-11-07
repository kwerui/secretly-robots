"""Translate the teammate-owned ``sim.HouseEnv`` ground truth into observations.

This is a simulation-only bridge. It reads HouseEnv's MuJoCo state directly and
deliberately does not represent camera, depth, or object-detection output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco

from .contracts import ObjectSighting, Observation
from sim.house_env import HouseEnv, ROOMS, TARGETS


@dataclass(frozen=True)
class HouseObservationConfig:
    """Optional metadata for the existing HouseEnv and a future robot model."""

    bottle_body: str = "waterbottle"
    bottle_item_id: str = "water_bottle"
    robot_body: str | None = None


class HouseEnvObservationAdapter:
    """Read ``HouseEnv`` state without controlling its optional robot.

    ``last_changes`` records semantic changes that warrant replanning.  It
    intentionally ignores ordinary physics ticks and small pose jitter. If the
    environment has no robot attached, ``robot_area`` is ``"unknown"`` rather
    than a fabricated pose or start location.
    """

    def __init__(
        self,
        env: HouseEnv,
        config: HouseObservationConfig | None = None,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.config = config or HouseObservationConfig()
        self._require_body(self.config.bottle_body)
        self.robot_body = self.config.robot_body
        if self.robot_body is None and env.robot_root is not None:
            self.robot_body = self.model.body(env.robot_root).name
        if self.robot_body is not None:
            self._require_body(self.robot_body)
        self._previous_location: str | None = None
        self.last_changes: tuple[Literal["bottle_location"], ...] = ()

    def latest_observation(self) -> Observation:
        """Return current HouseEnv ground truth in the brain's typed contract."""
        bottle_area = self._nearest_bottle_target()
        if self._previous_location in (None, bottle_area):
            self.last_changes = ()
        else:
            self.last_changes = ("bottle_location",)
        self._previous_location = bottle_area
        robot_area = self._robot_area()
        note = "MuJoCo HouseEnv ground truth; not camera perception."
        if self.robot_body is None:
            note += " No robot model is attached; robot_area is unknown."

        return Observation(
            timestamp_s=float(self.data.time),
            robot_area=robot_area,
            visible_objects=(
                ObjectSighting(
                    item_id=self.config.bottle_item_id,
                    area=bottle_area,
                    container=None,
                    confidence=1.0,
                ),
            ),
            drawer_open=False,
            scene_note=note,
        )

    def _nearest_bottle_target(self) -> str:
        """Map the actual bottle pose to HouseEnv's nearest named surface."""
        x, y = self.env.bottle_xyz()[:2]
        return min(
            TARGETS,
            key=lambda target: (float(x) - TARGETS[target][0]) ** 2
            + (float(y) - TARGETS[target][1]) ** 2,
        )

    def _robot_area(self) -> str:
        if self.robot_body is None:
            return "unknown"
        x, y = self.data.body(self.robot_body).xpos[:2]
        for room, (x0, y0, x1, y1, _) in ROOMS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return room
        return "unknown"

    def _require_body(self, name: str) -> None:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"Unknown MuJoCo body {name!r}")
