"""Integration coverage for observations from the teammate-owned HouseEnv."""
from brain.engine import Brain
from brain.mujoco_observation import HouseEnvObservationAdapter
from sim.house_env import HouseEnv


def test_house_env_bottle_move_changes_brain_decision_without_inventing_robot_pose():
    env = HouseEnv(seed=0)
    env.reset("tidy", "tv_unit", jitter=False)
    adapter = HouseEnvObservationAdapter(env)
    brain = Brain()
    brain.start_task("bring my water bottle")

    before = adapter.latest_observation()
    before_decision = brain.next_action(before)

    env.reset("tidy", "study_desk", jitter=False)
    after = adapter.latest_observation()
    after_decision = brain.next_action(after)

    assert before.robot_area == "unknown"
    assert "No robot model is attached" in before.scene_note
    assert before.visible_objects[0].area == "tv_unit"
    assert before.visible_objects[0].container is None
    assert before.drawer_open is False
    assert before_decision.action == "navigate"
    assert before_decision.target == "tv_unit"
    assert after.visible_objects[0].area == "study_desk"
    assert after_decision.action == "navigate"
    assert after_decision.target == "study_desk"
    assert adapter.last_changes == ("bottle_location",)
