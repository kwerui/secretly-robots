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

CommandStatus = Literal["idle", "heard", "confirming", "understood", "declined", "rejected"]
ParserSource = Literal["rules", "gemma", "confirmed"]

@dataclass(frozen=True)
class CommandState:
    """What the robot currently understands it has been asked to do.

    Produced onboard from heard speech. Setting it starts a task in the brain;
    it does NOT move the robot or claim anything was fetched.
    """
    status: CommandStatus = "idle"
    transcript: str = ""
    task: FetchTask | None = None
    parser_source: ParserSource | None = None
    question: str = ""  # what the robot asked while status is "confirming"
    expires_at: float | None = None  # when an unanswered question lapses
    reason: str = ""
    stt_ms: float | None = None
    intent_ms: float | None = None
    timestamp: float = 0.0
