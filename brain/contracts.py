"""High-level contracts shared by the AI brain and the MuJoCo team.

These describe *interpreted* observations, not raw camera frames or motor signals.
The perception team must derive them from real simulated sensors.
"""
from dataclasses import dataclass, field
from typing import Literal

ActionName = Literal[
    "search", "navigate", "open_drawer", "grasp", "deliver",
    "stop", "ask_for_help", "complete"
]

@dataclass(frozen=True)
class ObjectSighting:
    item_id: str
    area: str  # e.g. 'cupboard', 'table'
    container: str | None = None  # e.g. 'drawer'
    confidence: float = 1.0

@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    robot_area: str
    visible_objects: tuple[ObjectSighting, ...] = ()
    drawer_open: bool = False
    holding_item: str | None = None
    delivered_item: str | None = None
    person_too_close: bool = False
    sensors_ok: bool = True
    scene_note: str = ""

@dataclass(frozen=True)
class FetchTask:
    item_id: str
    destination: str = "sofa"

@dataclass(frozen=True)
class Decision:
    action: ActionName
    target: str | None = None
    reason: str = ""
    source: str = "deterministic_planner"

@dataclass(frozen=True)
class ActionFeedback:
    action: ActionName
    target: str | None
    succeeded: bool
    detail: str = ""

@dataclass(frozen=True)
class Demonstration:
    """Verified symbolic carer demo; NOT arbitrary video imitation learning."""
    item_id: str
    destination: str
    completed_successfully: bool
    description: str = ""
