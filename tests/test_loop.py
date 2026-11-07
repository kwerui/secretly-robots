import json
from robot_core.agent import SafeStopAgent
from robot_core.contracts import Observation
from robot_core.logging_utils import EventLogger
from robot_core.loop import run_once

class FakeSimulatorForUnitTests:
    def __init__(self):
        self.steps = 0
    def observe(self):
        return Observation(time_s=0.0, robot_position_xyz=(0.0, 0.0, 0.0))
    def execute(self, action):
        assert action.kind == "stop"
    def step(self):
        self.steps += 1

def test_logs_and_steps(tmp_path):
    sim = FakeSimulatorForUnitTests()
    path = tmp_path / "events.jsonl"
    run_once(sim, SafeStopAgent(), EventLogger(path))
    assert sim.steps == 1
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["event_type"] for e in events] == ["observation", "decision"]
