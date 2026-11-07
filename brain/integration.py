"""Boundary that robotics teammates implement for their actual MuJoCo robot.

A high-level action may take many simulation steps. Call `take_turn` when
controller reports completed/failed, not on every physics tick.
"""
from typing import Protocol
from .contracts import ActionFeedback, Decision, Observation
from .engine import Brain

class PerceptionBridge(Protocol):
    def latest_observation(self) -> Observation:
        """Derive a fresh observation from real MuJoCo simulated sensors."""
        ...

class ControllerBridge(Protocol):
    def execute_high_level(self, decision: Decision) -> ActionFeedback:
        """Translate a validated action into navigation/manipulation and verify it."""
        ...


def take_turn(brain: Brain, perception: PerceptionBridge, controller: ControllerBridge) -> Decision:
    obs = perception.latest_observation()
    decision = brain.next_action(obs)
    if decision.action in ("complete", "ask_for_help"):
        return decision
    feedback = controller.execute_high_level(decision)
    brain.record_feedback(feedback)
    return decision
