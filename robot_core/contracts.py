"""Shared messages between perception, planning, and control.

These are intentionally generic. Adapt them with your robotics teammates
when you know which robot, sensors, and task will be used.
"""
from dataclasses import dataclass, field
from typing import Literal, Protocol

ActionKind = Literal["stop", "navigate", "inspect", "pick", "place", "report"]

@dataclass(frozen=True)
class DetectedObject:
    id: str
    label: str
    position_xyz: tuple[float, float, float]
    confidence: float | None = None

@dataclass(frozen=True)
class Observation:
    time_s: float
    robot_position_xyz: tuple[float, float, float]
    objects: tuple[DetectedObject, ...] = ()
    sensor_readings: dict[str, float] = field(default_factory=dict)
    user_request: str | None = None

@dataclass(frozen=True)
class Action:
    kind: ActionKind
    target_id: str | None = None
    target_xyz: tuple[float, float, float] | None = None
    reason: str = ""

class Agent(Protocol):
    def decide(self, observation: Observation) -> Action: ...

class Simulator(Protocol):
    def observe(self) -> Observation: ...
    def execute(self, action: Action) -> None: ...
    def step(self) -> None: ...
