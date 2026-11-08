"""Evaluation-only episode records and MuJoCo ground-truth checks.

This module deliberately has no dependency on the robot's observations or the AI
planner.  It is safe to unit test without MuJoCo, GR00T assets, or a cloud API.
"""
from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from time import time
from typing import Any


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NOT_ATTEMPTED = "not_attempted"
    UNKNOWN = "unknown"


@dataclass
class TruthSnapshot:
    """Simulator-only evidence. ``None`` means the fact was unavailable."""
    item_on_tray: bool | None = None
    item_on_tray_after_settling: bool | None = None
    grasp_attached: bool | None = None
    drawer_closed: bool | None = None
    robot_fallen: bool | None = None


@dataclass
class TruthVerification:
    delivery: Outcome = Outcome.UNKNOWN
    grasp_released: Outcome = Outcome.UNKNOWN
    drawer_closed: Outcome = Outcome.UNKNOWN
    robot_fall: str = "unknown"
    task_outcome: Outcome = Outcome.UNKNOWN
    failure_stage: str | None = None
    failure_reason: str | None = None


def _bool_outcome(value: bool | None, *, true_is_success: bool = True) -> Outcome:
    if value is None:
        return Outcome.UNKNOWN
    return Outcome.SUCCESS if value is true_is_success else Outcome.FAILURE


def verify_truth(snapshot: TruthSnapshot) -> TruthVerification:
    """Classify physical evidence without using skill return values."""
    released = _bool_outcome(snapshot.grasp_attached, true_is_success=False)
    closed = _bool_outcome(snapshot.drawer_closed)
    fall = "unknown" if snapshot.robot_fallen is None else ("fallen" if snapshot.robot_fallen else "not_fallen")

    if snapshot.item_on_tray is False or snapshot.item_on_tray_after_settling is False:
        delivery, stage, reason = Outcome.FAILURE, "delivery", "requested item was not on the tray after settling"
    elif released is Outcome.FAILURE:
        delivery, stage, reason = Outcome.FAILURE, "delivery", "grasp attachment remained active"
    elif snapshot.item_on_tray is True and snapshot.item_on_tray_after_settling is True and released is Outcome.SUCCESS:
        delivery, stage, reason = Outcome.SUCCESS, None, None
    else:
        delivery, stage, reason = Outcome.UNKNOWN, None, None

    if fall == "fallen":
        return TruthVerification(delivery, released, closed, fall, Outcome.FAILURE, "robot_fall", "robot fell")
    if delivery is Outcome.FAILURE:
        return TruthVerification(delivery, released, closed, fall, Outcome.FAILURE, stage, reason)
    if closed is Outcome.FAILURE:
        return TruthVerification(delivery, released, closed, fall, Outcome.FAILURE, "drawer_closing", "drawer remained open")
    if delivery is Outcome.SUCCESS and closed is Outcome.SUCCESS and fall == "not_fallen":
        return TruthVerification(delivery, released, closed, fall, Outcome.SUCCESS)
    return TruthVerification(delivery, released, closed, fall, Outcome.UNKNOWN)


@dataclass
class EpisodeMetrics:
    experiment_id: str
    requested_item: str
    branch: str = "unknown"
    commit: str = "unknown"
    model_mode: str = "offline"
    layout: str = "tidy"
    seed: int | None = None
    bottle_placement: dict[str, Any] = field(default_factory=dict)
    environment_configuration: dict[str, Any] = field(default_factory=dict)
    navigation: Outcome = Outcome.NOT_ATTEMPTED
    drawer_opening: Outcome = Outcome.NOT_ATTEMPTED
    grasp: Outcome = Outcome.NOT_ATTEMPTED
    grasp_retry_count: int = 0
    drawer_closing: Outcome = Outcome.NOT_ATTEMPTED
    delivery: Outcome = Outcome.NOT_ATTEMPTED
    # Simulator-only end-state checks.  They must not reclassify an action that
    # was never attempted.
    physical_delivery: Outcome = Outcome.UNKNOWN
    drawer_closed_in_sim: Outcome = Outcome.UNKNOWN
    grasp_attachment_released: Outcome = Outcome.UNKNOWN
    item_remained_on_tray: Outcome = Outcome.UNKNOWN
    robot_fall: str = "unknown"
    task_outcome: Outcome = Outcome.UNKNOWN
    simulation_duration_s: float | None = None
    wall_clock_duration_s: float | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    first_failed_action: dict[str, str | None] | None = None
    navigation_attempts: list[dict[str, Any]] = field(default_factory=list)
    first_failed_navigation_attempt: dict[str, Any] | None = None
    drawer_opening_attempts: list[dict[str, Any]] = field(default_factory=list)
    first_failed_drawer_opening_attempt: dict[str, Any] | None = None
    executed_actions: list[dict[str, Any]] = field(default_factory=list)
    decision_provenance: list[str] = field(default_factory=list)
    cloud_events: list[dict[str, str]] = field(default_factory=list)
    schema_version: int = 3

    @classmethod
    def baseline(cls, experiment_id: str, requested_item: str) -> "EpisodeMetrics":
        branch, commit = git_identity()
        return cls(experiment_id=experiment_id, requested_item=requested_item, branch=branch, commit=commit)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeMetrics":
        enum_fields = {"navigation", "drawer_opening", "grasp", "drawer_closing", "delivery",
                       "physical_delivery", "drawer_closed_in_sim", "grasp_attachment_released",
                       "item_remained_on_tray", "task_outcome"}
        value = dict(value)
        value.pop("schema_version", None)
        for name in enum_fields:
            if name in value:
                value[name] = Outcome(value[name])
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name, value in list(result.items()):
            if isinstance(value, Outcome):
                result[name] = value.value
        return result


class TrialRecorder:
    """Records skill feedback as execution evidence, then truth evidence separately."""
    ACTION_FIELDS = {"navigate": "navigation", "search": "navigation", "open_drawer": "drawer_opening",
                     "grasp": "grasp", "deliver": "delivery"}

    def __init__(self, metrics: EpisodeMetrics):
        self.metrics = metrics
        self.started_at = time()

    def record_decision(self, action: str, source: str, *, fallback: bool = False) -> None:
        provenance = "fallback_after_cloud_rejection_or_failure" if fallback else source
        if provenance not in self.metrics.decision_provenance:
            self.metrics.decision_provenance.append(provenance)

    def record_cloud_event(self, event: str, reason: str) -> None:
        self.metrics.cloud_events.append({"event": event, "reason": reason[:500]})

    def record_feedback(self, action: str, succeeded: bool, detail: str, *, target: str | None = None,
                        retry_count: int = 0) -> None:
        self.metrics.executed_actions.append({"action": action, "target": target, "succeeded": succeeded,
                                              "detail": detail})
        field_name = self.ACTION_FIELDS.get(action)
        if field_name and action != "search":
            setattr(self.metrics, field_name, Outcome.SUCCESS if succeeded else Outcome.FAILURE)
        if action == "grasp":
            self.metrics.grasp_retry_count += retry_count
        if not succeeded and self.metrics.failure_stage is None:
            self.metrics.failure_stage = action
            self.metrics.failure_reason = detail or "skill reported failure"
            self.metrics.first_failed_action = {"action": action, "target": target,
                                                 "reason": self.metrics.failure_reason}

    def record_navigation_attempts(self, attempts: list[dict[str, Any]]) -> None:
        """Store raw station navigation outcomes, including tolerated near-arrivals."""
        for attempt in attempts:
            recorded = dict(attempt)
            self.metrics.navigation_attempts.append(recorded)
            if not recorded.get("reported_success", False) and self.metrics.first_failed_navigation_attempt is None:
                self.metrics.first_failed_navigation_attempt = recorded
        if self.metrics.navigation_attempts:
            any_rejected = any(not attempt.get("accepted_for_search", False)
                               for attempt in self.metrics.navigation_attempts)
            self.metrics.navigation = Outcome.FAILURE if any_rejected else Outcome.SUCCESS

    def record_drawer_opening_attempts(self, attempts: list[dict[str, Any]]) -> None:
        for attempt in attempts:
            recorded = dict(attempt)
            self.metrics.drawer_opening_attempts.append(recorded)
            if not recorded.get("result_ok", False) and self.metrics.first_failed_drawer_opening_attempt is None:
                self.metrics.first_failed_drawer_opening_attempt = recorded
        if self.metrics.drawer_opening_attempts:
            self.metrics.drawer_opening = (Outcome.FAILURE if self.metrics.first_failed_drawer_opening_attempt
                                           else Outcome.SUCCESS)

    def record_stage(self, field_name: str, succeeded: bool) -> None:
        """Record a nested robot step (for example close during a grasp action)."""
        setattr(self.metrics, field_name, Outcome.SUCCESS if succeeded else Outcome.FAILURE)

    def finish(self, truth: TruthSnapshot, *, simulation_duration_s: float | None = None) -> EpisodeMetrics:
        verified = verify_truth(truth)
        self.metrics.simulation_duration_s = simulation_duration_s
        self.metrics.wall_clock_duration_s = round(time() - self.started_at, 3)
        self.metrics.robot_fall = verified.robot_fall
        self.metrics.physical_delivery = verified.delivery
        self.metrics.drawer_closed_in_sim = verified.drawer_closed
        self.metrics.grasp_attachment_released = (verified.grasp_released
                                                  if self.metrics.grasp is not Outcome.NOT_ATTEMPTED
                                                  else Outcome.NOT_ATTEMPTED)
        self.metrics.item_remained_on_tray = _bool_outcome(truth.item_on_tray_after_settling)
        if verified.robot_fall == "fallen":
            self.metrics.task_outcome = Outcome.FAILURE
            self.metrics.failure_stage, self.metrics.failure_reason = "robot_fall", "robot fell"
        elif self.metrics.first_failed_action is not None:
            self.metrics.task_outcome = Outcome.FAILURE
        elif self.metrics.delivery is Outcome.NOT_ATTEMPTED:
            self.metrics.task_outcome = Outcome.UNKNOWN
        elif self.metrics.delivery is Outcome.FAILURE or verified.task_outcome is Outcome.FAILURE:
            self.metrics.task_outcome = Outcome.FAILURE
            if self.metrics.failure_stage is None:
                self.metrics.failure_stage = (verified.failure_stage
                                              if verified.failure_stage == "drawer_closing"
                                              else "physical_delivery")
                self.metrics.failure_reason = verified.failure_reason
        else:
            self.metrics.task_outcome = verified.task_outcome
        if self.metrics.task_outcome is Outcome.SUCCESS:
            self.metrics.failure_stage = self.metrics.failure_reason = None
        return self.metrics


class MujocoTruthReader:
    """Read ground truth only after the episode; never pass this to the brain."""
    TRAY_INSET_M = 0.01
    TRAY_Z_BELOW_M = 0.04
    TRAY_Z_ABOVE_M = 0.35
    DRAWER_CLOSED_M = 0.02

    def __init__(self, world, robot, drawer, item: str):
        self.world, self.robot, self.drawer, self.item = world, robot, drawer, item

    def snapshot(self) -> TruthSnapshot:
        body = self.world.data.body(self.world.item_body[self.item]).xpos
        delivery = self.world.delivery
        x, y = float(body[0]), float(body[1])
        cx, cy = delivery["pos"]
        hx, hy = delivery["half"]
        on_tray = (abs(x - cx) <= hx - self.TRAY_INSET_M and abs(y - cy) <= hy - self.TRAY_INSET_M
                   and delivery["top"] - self.TRAY_Z_BELOW_M <= float(body[2]) <= delivery["top"] + self.TRAY_Z_ABOVE_M)
        eq = self.world.model.eq(f"grasp_{self.world.item_body[self.item]}").id
        return TruthSnapshot(item_on_tray=on_tray, grasp_attached=bool(self.world.data.eq_active[eq]),
                             drawer_closed=self.world.drawer_open_m(self.drawer) <= self.DRAWER_CLOSED_M,
                             robot_fallen=bool(self.robot.wbc.fallen()))


def git_identity() -> tuple[str, str]:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() or "detached"
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return branch, commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def write_results(records: list[EpisodeMetrics], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.as_dict() for record in records]
    for row in rows:
        (output_dir / f"{row['experiment_id']}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "results.csv"
    fields = list(rows[0]) if rows else ["experiment_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    summary = output_dir / "summary.md"
    lines = ["# G1 baseline evaluation", "",
             "| Experiment | Task outcome | Delivery action | Tray check | First failed action | Fall |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        first = row["first_failed_action"] or {}
        action = first.get("action", "")
        target = first.get("target")
        first_text = f"{action}({target})" if target else action
        lines.append(f"| {row['experiment_id']} | {row['task_outcome']} | {row['delivery']} | "
                     f"{row['physical_delivery']} | {first_text} | {row['robot_fall']} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "summary": summary}
