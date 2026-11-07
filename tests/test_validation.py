import pytest
from robot_core.contracts import Action, DetectedObject, Observation
from robot_core.validation import InvalidAction, validate_action

observation = Observation(
    time_s=0.0,
    robot_position_xyz=(0.0, 0.0, 0.0),
    objects=(DetectedObject("object_1", "unknown", (1.0, 0.0, 0.0)),),
)

def test_stop_allowed():
    validate_action(Action("stop"), observation)

def test_known_object_inspection_allowed():
    validate_action(Action("inspect", target_id="object_1"), observation)

def test_unknown_object_rejected():
    with pytest.raises(InvalidAction):
        validate_action(Action("pick", target_id="missing"), observation)

def test_navigate_requires_target():
    with pytest.raises(InvalidAction):
        validate_action(Action("navigate"), observation)

def test_navigate_rejects_nonfinite_coordinates():
    with pytest.raises(InvalidAction):
        validate_action(Action("navigate", target_xyz=(float("nan"), 0.0, 0.0)), observation)
