"""The G1 in Jack's house: watch the carer open a drawer once, then fetch from it.

    MUJOCO_GL=egl python house_demo.py                 # records house_demo.mp4
    MUJOCO_GL=egl python house_demo.py --no-video      # just the numbers (much faster)
    MUJOCO_GL=egl python house_demo.py --with-cup      # then also the cup from the counter

Scenario (steps 1-4 of the integration; the brain/memory come later):
  1. The carer opens the top kitchen drawer once while the drawer camera watches; the robot
     fits the articulation (expected: prismatic) relative to the chest's anchor marker. The
     carer closes it again.
  2. "I'm thirsty": the water bottle is in that drawer (Jack's `kitchen_drawer` target). The
     robot plans across the house on Jack's occupancy grid, walks to the drawers, opens the
     drawer the way the carer did, sees the bottle, picks it from above and brings it to the
     user's tray by the sofa.
  3. (--with-cup) "Could I have my cup?": side grasp from the kitchen counter, same delivery.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from demo_capture import capture_sim_drawer
from house import G1House, KITCHEN_TOP
from run_demo import Recorder
from skills import AssistiveRobot


class LightRecorder(Recorder):
    """run_demo.Recorder with a cheap head-camera inset (own small renderer, no marker
    detection per frame): the house has far more geometry than the one-room scene."""

    def __call__(self, robot) -> None:
        self.k += 1
        if self.k % self.every:
            return
        import mujoco
        if self.renderer is None:
            self.renderer = mujoco.Renderer(robot.model, self.size[1], self.size[0])
            self.inset_r = mujoco.Renderer(robot.model, 240, 320)
            self.cam.distance, self.cam.elevation = 3.6, -35
        target = robot.wbc.base_pos + np.array([0, 0, 0.1])
        self.cam.lookat[:] = 0.92 * self.cam.lookat + 0.08 * target if self.k > 2 else target
        # no shadows/reflections: with the house's 6 lights they cost ~3.5x per frame (software GL)
        self.renderer.update_scene(robot.data, self.cam)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        frame = self.renderer.render().copy()
        self.inset_r.update_scene(robot.data, camera="head_cam")
        self.inset_r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        head = self.inset_r.render()
        iw, ih = self.size[0] * 3 // 10, self.size[0] * 9 // 40
        x0 = self.size[0] - iw - 12
        frame[12:12 + ih, x0:x0 + iw] = cv2.resize(head, (iw, ih))
        cv2.rectangle(frame, (x0, 12), (x0 + iw, 12 + ih), (255, 255, 255), 2)
        cv2.putText(frame, "robot head camera", (x0 + 8, 12 + ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.caption:
            cv2.rectangle(frame, (0, self.size[1] - 40), (self.size[0], self.size[1]), (20, 20, 20), -1)
            cv2.putText(frame, self.caption, (14, self.size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("house_demo.mp4"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--size", type=int, nargs=2, default=[800, 450])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default="tidy", help="Jack's layouts: tidy, laundry_day, door_shut, moving_house, long_way_round")
    ap.add_argument("--with-cup", action="store_true", help="after the bottle, also fetch the cup")
    args = ap.parse_args()

    t0 = time.time()
    env = G1House(seed=args.seed)
    info = env.reset(args.layout, "kitchen_drawer", seed=args.seed)
    rec = None if args.no_video else LightRecorder(args.out, fps=args.fps, size=tuple(args.size))
    robot = AssistiveRobot(world=env, on_frame=rec)
    say_log = []

    def say(msg):
        say_log.append((round(robot.data.time, 1), msg))
        print(f"[{robot.data.time:6.1f}s] {msg}", flush=True)
        if rec is not None:
            rec.caption = msg

    print("doors (deg):", info["doors"], " locked:", info["locked"])
    results = {}

    if True:
        say("Carer: opening the top kitchen drawer once (drawer camera watching)")
        frames = [] if rec is not None else None
        model = capture_sim_drawer(robot, KITCHEN_TOP, video_frames=frames)
        model.save("drawer_model.json")
        say(f"Learned: {model.kind}, opens {model.amount * 100:.0f} cm, fit residual {model.residual_m * 1000:.1f} mm")
        if rec is not None:
            for f in frames[::2]:
                big = cv2.resize(f, tuple(args.size))
                cv2.rectangle(big, (0, args.size[1] - 40), (args.size[0], args.size[1]), (20, 20, 20), -1)
                cv2.putText(big, "Carer demonstration, watched by the drawer camera", (14, args.size[1] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                rec.writer.write(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        results["demo"] = dict(kind=model.kind, amount_m=round(model.amount, 3), residual_mm=round(model.residual_m * 1000, 1),
                               true_open_m=0.36)

        say("'I'm thirsty' - walking to the kitchen drawers")
        r = robot.go_to("kitchen_drawers")
        results["go_drawers"] = r
        seen = robot.look()
        say(f"Seen: {sorted(seen['items']) or 'no items'}; no bottle in sight, drawer closed - opening it")
        r = robot.open_articulated(model, KITCHEN_TOP.anchor_marker, f"grasp_{KITCHEN_TOP.body}")
        results["open"] = {**r, "drawer_now_m": round(env.drawer_open_m(KITCHEN_TOP), 3)}
        say(f"Opened the drawer {env.drawer_open_m(KITCHEN_TOP) * 100:.0f} cm")
        seen = robot.look()
        results["look_after_open"] = sorted(seen["items"])
        if "water_bottle" in seen["items"]:
            say("The bottle is in the drawer - picking it from above")
            r = robot.pick("water_bottle", from_above=True)
            results["pick_bottle"] = r
            if r["ok"]:
                say("Bringing it over")
                results["go_person"] = robot.go_to("person")
                results["hand_over"] = robot.hand_over()
                say("Here you go")
            else:
                say(f"Pick failed: {r.get('error')}")
        else:
            say("Can't see the bottle in the drawer")
            robot.save_view("debug_drawer_view.png")

    if args.with_cup:
        say("'Could I have my cup?' - walking to the kitchen counter")
        results["go_counter"] = robot.go_to("kitchen_counter")
        seen = robot.look()
        results["look_counter"] = sorted(seen["items"])
        r = robot.pick("cup")
        results["pick_cup"] = r
        if r["ok"]:
            say("Bringing the cup over")
            results["go_person_2"] = robot.go_to("person")
            results["hand_over_2"] = robot.hand_over()
            say("Here you go")
        else:
            say(f"Pick failed: {r.get('error')}")

    if rec is not None:
        rec.close()
    d = env.data
    for item, body in env.item_body.items():
        p = d.body(body).xpos
        results.setdefault("final_item_pos", {})[item] = np.round(p, 3).tolist()
    results["fallen"] = robot.wbc.fallen()
    results["sim_s"] = round(d.time, 1)
    results["wall_s"] = round(time.time() - t0, 1)
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
