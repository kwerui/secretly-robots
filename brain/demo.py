"""Illustrative *symbolic observations*. Not a MuJoCo autonomous robot demo.

Shows the brain's choices as object position changes. Real sensor/perception
input and action execution require the team's robot integration.
"""
from .contracts import Observation, ObjectSighting
from .engine import Brain


def run() -> None:
    brain = Brain(log_file="logs/brain_demo.jsonl")
    brain.start_task("Please bring me my water bottle")
    observations = [
        Observation(0.0, "sofa", (ObjectSighting("water_bottle", "cupboard", "drawer"),)),
        Observation(1.0, "cupboard", (ObjectSighting("water_bottle", "cupboard", "drawer"),)),
        Observation(2.0, "cupboard", (ObjectSighting("water_bottle", "table", None),), drawer_open=True),
        Observation(3.0, "table", (ObjectSighting("water_bottle", "table"),), drawer_open=True),
        Observation(4.0, "table", (), holding_item="water_bottle"),
        Observation(5.0, "sofa", (), holding_item="water_bottle"),
        Observation(6.0, "sofa", (), delivered_item="water_bottle"),
    ]
    for obs in observations:
        action = brain.next_action(obs)
        print(f"t={obs.timestamp_s:.0f} -> {action.action:12} {str(action.target):14} ({action.reason})")
    print("NOTE: Input observations above are manually authored examples. No robot was controlled.")

if __name__ == "__main__":
    run()
