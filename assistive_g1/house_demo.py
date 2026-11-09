"""The G1 in Jack's house: watch the carer open a drawer once, then fetch from it.

    python house_demo.py --viewer                      # interactive MuJoCo viewer, real time
    MUJOCO_GL=egl python house_demo.py                 # records house_demo.mp4
    MUJOCO_GL=egl python house_demo.py --no-video      # just the numbers (much faster)
    MUJOCO_GL=egl python house_demo.py --with-cup      # then also the cup from the counter

Scenario (steps 1-4 of the integration; the brain/memory come later):
  1. The carer opens the top kitchen drawer once while the drawer camera watches; the robot
     fits the articulation (expected: prismatic) relative to the chest's anchor marker. The
     carer closes it again.
  2. "I'm thirsty": the water bottle is in that drawer (Jack's `kitchen_drawer` target). The
     robot plans across the house on Jack's occupancy grid, walks to the drawers, opens the
     drawer the way the carer did, sees the bottle, picks it from above, closes the drawer
     with its LEFT hand (the learned motion run backwards) while the right hand keeps the
     bottle, and brings it to the user's tray by the sofa.
  3. (--with-cup) "Could I have my cup?": side grasp from the kitchen counter, same delivery.
"""

from __future__ import annotations

import os

from viewer_env import relaunch_for_viewer

relaunch_for_viewer()  # --viewer from anaconda: restart with the system libstdc++ (GLX fix)
# Offscreen rendering (robot camera, video) through EGL; the --viewer window still needs GLX.
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


PANEL_BG = (34, 36, 40)
PANEL_FG = (230, 230, 230)
PANEL_DIM = (150, 155, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class SidePanel:
    """The robot's-eye side panel, as an RGB image `width` x `height`:
      * the robot's head camera (RGB) with the ArUco markers it detects drawn on,
      * the robot's own clearance map (costmap.py): grey = floor plan, black = seen by its
        LiDAR / depth camera, blue = the route it is walking, red = where it thinks it is,
      * live numbers: LiDAR scans, depth frames, obstacles seen that are not in the floor plan,
        localisation error, re-plans, locked doors known, what it is holding.
    Used by LightRecorder (video) and voice_demo.py (live, overlaid on the MuJoCo viewer)."""

    def __init__(self):
        self.head_r = None

    def render(self, robot, width: int, height: int) -> np.ndarray:
        import mujoco
        if self.head_r is None:
            self.head_r = mujoco.Renderer(robot.model, 240, 320)
        s = width / 320  # text and spacing scale (1 at 320 px wide; 2 on a retina viewer)
        img = np.empty((height, width, 3), np.uint8)
        img[:] = PANEL_BG
        pad = int(8 * s)
        iw = width - 2 * pad
        x0, y = pad, pad
        th = max(1, int(round(s)))

        def text(t, yy, size=0.42, col=PANEL_FG):
            cv2.putText(img, t, (x0, int(yy + 12 * s)), FONT, size * s, col, th, cv2.LINE_AA)

        # head camera + the markers the robot detects in it
        text("Head camera (RGB-D) + ArUco detections", y, 0.4)
        y += int(18 * s)
        self.head_r.update_scene(robot.data, camera="head_cam")
        self.head_r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        head = np.ascontiguousarray(self.head_r.render())
        cam = getattr(robot, "cam", None)
        if cam is not None:
            corners, ids = cam._detect_raw(cv2.cvtColor(head, cv2.COLOR_RGB2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(head, corners, ids, borderColor=(0, 255, 0))
                for c, mid in zip(corners, ids.flatten()):
                    x, yy = c.reshape(4, 2)[0].astype(int)
                    cv2.putText(head, cam.labels.get(int(mid), str(int(mid))), (x, max(10, yy - 5)), FONT, 0.4,
                                (255, 255, 0), 1, cv2.LINE_AA)
        ih = iw * 3 // 4
        img[y:y + ih, x0:x0 + iw] = cv2.resize(head, (iw, ih), interpolation=cv2.INTER_AREA)
        y += ih + int(10 * s)

        # the robot's own map
        cmap = getattr(robot, "costmap", None)
        text("Robot's own map: LiDAR + depth on the floor plan", y, 0.4)
        y += int(18 * s)
        if cmap is not None:
            amap = cmap.image(path=robot.nav_path, robot_xy=robot.wbc.base_pos[:2], scale=2)
            mh = int(round(iw * amap.shape[0] / amap.shape[1]))
            if y + mh < height:
                img[y:y + mh, x0:x0 + iw] = cv2.resize(amap, (iw, mh), interpolation=cv2.INTER_AREA)
            y += mh + int(12 * s)
        else:
            text("(G1_NAV=static: planning on the true geometry)", y, 0.38, PANEL_DIM)
            y += int(24 * s)

        # live numbers
        lines = []
        if cmap is not None:
            new_cells, _ = cmap.changed_vs_prior()
            lines.append(f"LiDAR scans {cmap.scans}   depth frames {cmap.depth_frames}")
            lines.append(f"obstacles seen (not in plan) {new_cells} cells")
        est = getattr(robot, "estimator", None)
        replans = len(getattr(robot, "nav_log", []))
        loc = f"localisation error {est.error()['xy_m'] * 100:.1f} cm   " if est is not None else ""
        lines.append(f"{loc}re-plans {replans}")
        locked = sorted(getattr(robot.world, "locked_doors", ()) or ())
        if locked:
            lines.append("locked doors known: " + ", ".join(d.replace("_", "-") for d in locked))
        lines.append(f"holding: {(robot.holding or '-').replace('_', ' ')}")
        for ln in lines:
            if y + 14 * s > height - 22 * s:
                break
            text(ln, y)
            y += int(19 * s)
        cv2.putText(img, "grey = floor plan  black = seen  blue = route  red = robot", (x0, int(height - 10 * s)),
                    FONT, 0.33 * s, PANEL_DIM, th, cv2.LINE_AA)
        return img


class LightRecorder(Recorder):
    """Follow camera on the left, SidePanel on the right, caption in a bar across the top of the
    main view. Shadows/reflections are off in the main view (with the house's 6 lights they
    cost ~3.5x per frame on software GL)."""

    def __call__(self, robot) -> None:
        self.k += 1
        if self.k % self.every:
            return
        import mujoco
        W, Ht = self.size
        P = max(240, W // 4)  # side panel width
        MW = W - P
        if self.renderer is None:
            self.renderer = mujoco.Renderer(robot.model, Ht, MW)
            self.panel = SidePanel()
            self.cam.distance, self.cam.elevation = 3.6, -35
        target = robot.wbc.base_pos + np.array([0, 0, 0.1])
        self.cam.lookat[:] = 0.92 * self.cam.lookat + 0.08 * target if self.k > 2 else target
        self.renderer.update_scene(robot.data, self.cam)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        frame = np.empty((Ht, W, 3), np.uint8)
        frame[:, :MW] = self.renderer.render()
        frame[:, MW:] = self.panel.render(robot, P, Ht)
        if self.caption:
            cv2.rectangle(frame, (0, 0), (MW, 34), (20, 20, 20), -1)
            cv2.putText(frame, self.caption, (12, 23), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("house_demo.mp4"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="interactive MuJoCo viewer (needs GLX), no video")
    ap.add_argument("--speed", type=float, default=1.0, help="with --viewer: playback speed (1 = real time)")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default="tidy", help="Jack's layouts: tidy, laundry_day, door_shut, moving_house, long_way_round")
    ap.add_argument("--with-cup", action="store_true", help="after the bottle, also fetch the cup")
    args = ap.parse_args()

    t0 = time.time()
    env = G1House(seed=args.seed)
    info = env.reset(args.layout, "kitchen_drawer", seed=args.seed)
    viewer = None
    if args.viewer:
        import mujoco
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("pelvis").id
        viewer.cam.distance, viewer.cam.elevation = 3.5, -30
        clock = {"t0": None}

        def sync(_robot=None):
            """Show the current state, paced to (speed x) real time."""
            if not viewer.is_running():
                raise KeyboardInterrupt
            now, sim = time.time(), env.data.time
            if clock["t0"] is None:
                clock["t0"] = (now, sim)
            ahead = (sim - clock["t0"][1]) / args.speed - (now - clock["t0"][0])
            if ahead > 0:
                time.sleep(ahead)
            viewer.sync()
        rec = None
    else:
        sync = None
        rec = None if args.no_video else LightRecorder(args.out, fps=args.fps, size=tuple(args.size))
    robot = AssistiveRobot(world=env, on_frame=sync if viewer is not None else rec)
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
        model = capture_sim_drawer(robot, KITCHEN_TOP, video_frames=frames,
                                   on_step=sync if viewer is not None else None)
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
        r = robot.open_articulated(model, KITCHEN_TOP.anchor_marker, f"grasp_{KITCHEN_TOP.body}",
                                   part_markers=(KITCHEN_TOP.front_marker,))
        results["open"] = {**r, "drawer_now_m": round(env.drawer_open_m(KITCHEN_TOP), 3)}
        say(f"Opened the drawer {env.drawer_open_m(KITCHEN_TOP) * 100:.0f} cm")
        seen = robot.look()
        results["look_after_open"] = sorted(seen["items"])
        if "water_bottle" in seen["items"]:
            say("The bottle is in the drawer - picking it from above")
            r = robot.pick("water_bottle", from_above=True)
            results["pick_bottle"] = r
            if r["ok"]:
                say("Closing the drawer with the other hand")
                results["close"] = {**robot.close_opened(), "drawer_now_m": round(env.drawer_open_m(KITCHEN_TOP), 3),
                                    "holding_ok": robot.holding_ok()}
                say("Bringing it over")
                results["go_person"] = robot.go_to("person")
                results["bottle_before_handover"] = np.round(env.data.body("waterbottle").xpos, 3).tolist()
                results["hand_over"] = robot.hand_over()
                robot.wait(1.0)
                results["bottle_after_handover"] = np.round(env.data.body("waterbottle").xpos, 3).tolist()
                say("Here you go")
            else:
                say(f"Pick failed: {r.get('error')}")
        else:
            say("Can't see the bottle in the drawer")
            robot.save_view("debug_drawer_view.png")

    if args.with_cup:
        say("'Could I have my cup?' - walking to the kitchen counter")
        results["go_counter"] = robot.go_to("kitchen_counter")
        results["bottle_after_leaving"] = np.round(env.data.body("waterbottle").xpos, 3).tolist()
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
    if viewer is not None:
        print("Done - close the viewer window (or Ctrl+C) to exit.")
        try:
            while viewer.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        import sys
        sys.stdout.flush()
        os._exit(0)  # viewer.close() can hang on an X error
    d = env.data
    for item, body in env.item_body.items():
        p = d.body(body).xpos
        results.setdefault("final_item_pos", {})[item] = np.round(p, 3).tolist()
    results["fallen"] = robot.wbc.fallen()
    if robot.estimator is not None:  # self-localization vs simulator truth (evaluation only)
        results["localization_error_now"] = robot.estimator.error()
        results["lidar_scans"] = dict(robot.estimator.stats)
    results["sim_s"] = round(d.time, 1)
    results["wall_s"] = round(time.time() - t0, 1)
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
