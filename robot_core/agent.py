"""Safe default agent; swap with a Gemini-based planner later."""
from .contracts import Action, Observation

class SafeStopAgent:
    def decide(self, observation: Observation) -> Action:
        return Action(kind="stop", reason="No task-specific policy configured")

# Future: GeminiAgent implementing the Agent protocol.
# Gemini should choose high-level actions, NOT raw actuator torques.
