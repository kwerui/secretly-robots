"""Measure arm posture during the house scenario: elbow height and joint speed, per skill.

    python arm_probe.py                      # scenario with the cup errand, headless
    G1_ELBOW_GAIN=0.3 python arm_probe.py    # same, with the IK's null-space elbow-down pull on
    python arm_probe.py --viewer             # watch it too

Prints, per skill and arm: highest elbow position relative to the shoulder (m, + = above),
ticks with the elbow above the shoulder, peak joint speed; then each skill's result.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

os.environ.setdefault("MUJOCO_GL", "egl")
sys.argv = [sys.argv[0], "--no-video", "--with-cup"] + [a for a in sys.argv[1:] if a not in ("--no-video",)]
if "--viewer" in sys.argv:
    sys.argv.remove("--no-video")

import numpy as np  # noqa: E402

from skills import AssistiveRobot  # noqa: E402

SKILLS = ["go_to", "pick", "_pick_from_above", "open_articulated", "close_opened", "_tuck", "carry",
          "hand_over", "_raise_hand", "arm_home", "left_arm_home"]
ctx = {"skill": "idle"}
stats = defaultdict(lambda: {"lift": -9.0, "above": 0, "speed": 0.0, "ticks": 0})

for name in SKILLS:
    f = getattr(AssistiveRobot, name)

    def wrap(f=f, name=name):
        def g(self, *a, **k):
            prev, ctx["skill"] = ctx["skill"], name
            try:
                return f(self, *a, **k)
            finally:
                ctx["skill"] = prev
        return g
    setattr(AssistiveRobot, name, wrap())

_tick = AssistiveRobot._tick


def tick(self):
    _tick(self)
    for side, ik in (("R", self.ik), ("L", self.left_ik)):
        q = self.data.qpos[ik.qadr]
        lift = ik.elbow_lift(q)
        st = stats[(ctx["skill"], side)]
        st["lift"] = max(st["lift"], lift)
        st["above"] += lift > 0
        st["speed"] = max(st["speed"], float(np.abs(self.data.qvel[ik.vadr]).max()))
        st["ticks"] += 1


AssistiveRobot._tick = tick

import house_demo  # noqa: E402

try:
    house_demo.main()
finally:
    print(f"\nelbow gain (G1_ELBOW_GAIN) = {os.environ.get('G1_ELBOW_GAIN', '0.0')}")
    print(f"{'skill':18s} arm  max elbow-shoulder (m)  ticks elbow above shoulder  peak joint speed (rad/s)")
    for (skill, side), st in sorted(stats.items(), key=lambda kv: -kv[1]["lift"]):
        if st["ticks"]:
            print(f"{skill:18s} {side}    {st['lift']:+.3f}                  {st['above']:5d}                       {st['speed']:.1f}")
