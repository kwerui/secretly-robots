"""Bottle on the floor -> deep-squat top-down pick -> side table.

    python try_floor_pick.py            # headless, prints results
    python try_floor_pick.py --watch    # live window (no GLX needed)
    python try_floor_pick.py --viewer   # interactive MuJoCo viewer (needs GLX)
"""

from __future__ import annotations

import os
import sys

from viewer_env import relaunch_for_viewer

relaunch_for_viewer()  # --viewer from anaconda: restart with the system libstdc++ (GLX fix)

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering without GLX

import time

from scene import DEFAULT_PLACEMENT
from skills import AssistiveRobot

placement = dict(DEFAULT_PLACEMENT)
placement["water_bottle"] = ("floor", 0.75, -0.12)  # 0.75 m in front of the start pose
robot = AssistiveRobot(placement=placement)
if "--viewer" in sys.argv:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    robot.on_frame = lambda r: (viewer.sync(), time.sleep(0.01))
elif "--watch" in sys.argv:
    from run_demo import Recorder

    robot.on_frame = Recorder(None, show=True)
print(robot.look())
print(robot.pick("water_bottle"))
print(robot.go_to("person"))
print(robot.hand_over())
