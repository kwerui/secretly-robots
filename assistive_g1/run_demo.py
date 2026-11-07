"""End-to-end demo: carer demo -> learned cupboard model -> requests -> fetch, recorded to MP4.

    python run_demo.py                      # records demo.mp4 (headless OK)
    python run_demo.py --watch              # live window (OpenCV, no GLX needed) + demo.mp4
    python run_demo.py --viewer             # interactive MuJoCo viewer (needs working GLX)
    python run_demo.py --request "I'd like a cup of tea"

Scenario
  1. The carer is "observed" putting things away for a few days -> location memory.
  2. The carer opens the cupboard once while a camera watches -> articulation model.
  3. "I'm thirsty": the memory says the shelf is most likely, but today the bottle is on
     the table: the robot checks the shelf, doesn't see it, checks the table, fetches it.
  4. "Could I have my cup?": the cup is in the closed cupboard; the robot opens it the way
     the carer did, picks the cup and brings it.
"""

from __future__ import annotations

import os
import sys

from viewer_env import relaunch_for_viewer

relaunch_for_viewer()  # --viewer from anaconda: restart with the system libstdc++ (GLX fix)

# Offscreen rendering (robot camera, video) through EGL: uses the GPU without X/GLX. The
# MuJoCo --viewer window still needs GLX. Must run before anything imports mujoco.
# Override with MUJOCO_GL=glfw/osmesa.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from articulation import ArticulationModel
from memory import LocationMemory
from tasks import fetch, item_for_request

DAY = 86400.0


class Recorder:
    """Follow camera + robot head-camera inset + captions, written to MP4 and/or shown live.

    show=True opens an OpenCV window (no OpenGL/GLX needed: frames come from EGL) and paces
    the sim to real time.
    """

    def __init__(self, path: Path | None, fps: float = 25, size=(960, 540), show: bool = False):
        self.path, self.fps, self.size, self.show = path, fps, size, show
        self.every = max(1, round(50 / fps))  # control runs at 50 Hz
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size) if path else None
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.2, -28, 150
        self.caption = ""
        self.k = 0
        self._t0 = None

    def __call__(self, robot) -> None:
        self.k += 1
        if self.k % self.every:
            return
        if self.renderer is None:
            self.renderer = mujoco.Renderer(robot.model, self.size[1], self.size[0])
        target = robot.wbc.base_pos + np.array([0, 0, 0.1])
        self.cam.lookat[:] = 0.9 * self.cam.lookat + 0.1 * target if self.k > 2 else target
        self.renderer.update_scene(robot.data, self.cam)
        frame = self.renderer.render().copy()
        head = robot.cam.render()
        head = robot.cam.annotate(head, {})
        iw, ih = self.size[0] * 3 // 10, self.size[0] * 9 // 40  # 4:3 inset, 30% of the width
        inset = cv2.resize(head, (iw, ih))
        x0 = self.size[0] - iw - 12
        frame[12:12 + ih, x0:x0 + iw] = inset
        cv2.rectangle(frame, (x0, 12), (x0 + iw, 12 + ih), (255, 255, 255), 2)
        cv2.putText(frame, "robot head camera", (x0 + 8, 12 + ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        if self.caption:
            cv2.rectangle(frame, (0, self.size[1] - 48), (self.size[0], self.size[1]), (20, 20, 20), -1)
            cv2.putText(frame, self.caption, (18, self.size[1] - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.writer is not None:
            self.writer.write(bgr)
        if self.show:
            if self._t0 is None:
                self._t0 = (time.time(), robot.data.time)
            ahead = (robot.data.time - self._t0[1]) - (time.time() - self._t0[0])
            if ahead > 0:  # keep to real time
                time.sleep(ahead)
            cv2.imshow("assistive_g1", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()


def learned_memory() -> LocationMemory:
    """What the robot picked up by watching the carer over the last week."""
    mem = LocationMemory()
    now = time.time()
    for day, (item, station) in enumerate([
        ("water_bottle", "shelf"), ("water_bottle", "shelf"), ("water_bottle", "table"),
        ("water_bottle", "shelf"), ("cup", "cupboard"), ("cup", "cupboard"), ("cup", "table"),
        ("pill_box", "shelf"), ("pill_box", "shelf"),
    ]):
        mem.observe_placement(item, station, t=now - (7 - day % 7) * DAY)
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demo.mp4"))
    ap.add_argument("--viewer", action="store_true", help="interactive MuJoCo viewer (needs GLX)")
    ap.add_argument("--watch", action="store_true", help="live OpenCV window (no GLX needed); q to stop")
    ap.add_argument("--no-video", action="store_true", help="with --watch: don't write the MP4")
    ap.add_argument("--model", type=Path, default=Path("cupboard_model.json"),
                    help="articulation model from demo_capture.py (created from a sim demo if missing)")
    ap.add_argument("--request", action="append", default=None)
    ap.add_argument("--fps", type=float, default=25, help="video frame rate (lower = faster to render)")
    ap.add_argument("--size", type=int, nargs=2, default=[960, 540], help="video width height")
    args = ap.parse_args()
    requests = args.request or ["I'm thirsty", "Could I have my cup, please?"]

    if not args.model.exists():
        from demo_capture import capture_sim
        print("No cupboard model yet: recording the carer demo in simulation...")
        capture_sim(args.model)
    model = ArticulationModel.load(args.model)
    print(f"Cupboard model: {model.kind}, opens {np.degrees(model.amount):.0f} deg, fit residual {model.residual_m*1000:.1f} mm")

    from skills import AssistiveRobot

    rec = None
    viewer = None
    if args.viewer:
        import mujoco.viewer

        robot = AssistiveRobot()
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

        def sync(r):
            viewer.sync()
            time.sleep(0.01)
        robot.on_frame = sync
        say = print
    else:
        out = None if (args.watch and args.no_video) else args.out
        rec = Recorder(out, fps=args.fps, size=tuple(args.size), show=args.watch)
        robot = AssistiveRobot(on_frame=rec)

        def say(msg):
            print(msg)
            rec.caption = msg

    memory = learned_memory()
    try:
        for req in requests:
            say(f'Person: "{req}"')
            robot.wait(1.5)
            item = item_for_request(req)
            if item is None:
                say("Sorry, I didn't understand what you need")
                continue
            result = fetch(robot, memory, item, model, say=say)
            print(result)
            robot.go_to("home")
            robot.wait(1.0)
    except KeyboardInterrupt:
        print("interrupted")
    finally:  # always finalise the video, even after Ctrl-C
        if rec is not None:
            rec.close()
            if rec.writer is not None:
                print(f"wrote {args.out}")
        if viewer is not None:
            viewer.close()

if __name__ == "__main__":
    main()
