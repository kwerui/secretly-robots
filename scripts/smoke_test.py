"""Run from project root: python scripts/smoke_test.py"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_core.mujoco_adapter import MuJoCoAdapter
from robot_core.agent import SafeStopAgent
from robot_core.logging_utils import EventLogger
from robot_core.loop import run_once

root = Path(__file__).resolve().parents[1]
sim = MuJoCoAdapter(root / "scenes" / "smoke.xml", robot_body="demo_body")
logger = EventLogger(root / "logs" / "run.jsonl")
for _ in range(10):
    run_once(sim, SafeStopAgent(), logger)
print("MuJoCo physics stepped 10 times. Last observation:")
print(sim.observe())
print("Events written to logs/run.jsonl")
print("This is a physics + logging smoke test, NOT an autonomous robot demo.")
