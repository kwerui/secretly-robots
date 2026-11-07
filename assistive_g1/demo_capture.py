"""Record ONE carer demonstration of opening the cupboard and fit its articulation model.

Two sources, same maths:

  1. Real world (the pitch demo): a webcam/phone video of the carer opening a real
     cupboard that has ArUco 10 on the door front and ArUco 11 fixed on the cupboard
     body/counter. Needs camera intrinsics (calibrate once, see README).

        python demo_capture.py video --source 0 --calib calib.npz --out cupboard_model.json

  2. Simulation (for testing the pipeline): a room camera watches the sim door being
     swung open by the "carer" (the hinge is driven kinematically).

        python demo_capture.py sim --out cupboard_model.json

Either way the robot only receives cupboard_model.json: hinge axis, opening angle and
grasp point, all relative to the anchor marker.
"""

from __future__ import annotations

import os
import sys

# Offscreen rendering (robot camera, video) through EGL: uses the GPU without X/GLX. The
# MuJoCo --viewer window still needs GLX. Must run before anything imports mujoco.
# Override with MUJOCO_GL=glfw/osmesa.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import cv2
import numpy as np

from articulation import ArticulationModel, fit_articulation, relative_pose
from scene import DOOR_MARKER, ANCHOR_MARKER, DOOR_MARKER_SIZE, MARKER_DICT

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def detect_in_camera(gray: np.ndarray, detector, K: np.ndarray, dist: np.ndarray, sizes: dict[int, float]):
    """Marker poses in the (OpenCV) camera frame: {id: (R, t)}."""
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, mid in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid not in sizes:
            continue
        s = sizes[mid] / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            out[mid] = (cv2.Rodrigues(rvec)[0], tvec.flatten())
    return out


def fit_from_tracks(tracks: list[tuple[np.ndarray, np.ndarray]], grasp_in_part: np.ndarray | None) -> ArticulationModel:
    rots = np.array([r for r, _ in tracks])
    pos = np.array([p for _, p in tracks])
    return fit_articulation(rots, pos, grasp_in_part)


# ----------------------------------------------------------------------------- real video
def capture_video(source, calib: Path, out: Path, grasp_in_part: np.ndarray, show: bool = True) -> ArticulationModel:
    data = np.load(calib)
    K, dist = data["K"], data["dist"]
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)
    sizes = {DOOR_MARKER: DOOR_MARKER_SIZE, ANCHOR_MARKER: DOOR_MARKER_SIZE}
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    tracks = []
    print("Recording: open the cupboard once, then press q (or let the video end).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect_in_camera(gray, detector, K, dist, sizes)
        if DOOR_MARKER in det and ANCHOR_MARKER in det:
            (Ra, ta), (Rd, td) = det[ANCHOR_MARKER], det[DOOR_MARKER]
            tracks.append(relative_pose(Ra, ta, Rd, td))
        if show:
            for mid, (R, t) in det.items():
                cv2.drawFrameAxes(frame, K, dist, cv2.Rodrigues(R)[0], t, 0.03)
            cv2.putText(frame, f"samples: {len(tracks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("carer demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    model = fit_from_tracks(tracks, grasp_in_part)
    model.save(out)
    return model


# ----------------------------------------------------------------------------- simulation
def capture_sim(out: Path, open_angle: float = 1.4, seconds: float = 3.0, video: Path | None = None) -> ArticulationModel:
    import mujoco

    from perception import HeadCamera
    from skills import AssistiveRobot

    robot = AssistiveRobot()
    m, d = robot.model, robot.data
    cam = HeadCamera(m, d, cam="room_cam", width=640, height=480)
    hinge = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
    qadr = m.jnt_qposadr[hinge]
    handle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "door_handle_site")
    writer = None
    if video is not None:
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))

    tracks, grasp_in_part = [], None
    n = int(seconds / 0.04)
    for k in range(n + 10):
        angle = open_angle * min(1.0, k / n) ** 1.0
        d.qpos[qadr] = angle  # the carer's hand moves the door
        d.qvel[m.jnt_dofadr[hinge]] = 0.0
        robot.wbc.step(20)
        mujoco.mj_forward(m, d)
        rgb = cam.render()
        det = cam.detect(rgb)
        if DOOR_MARKER in det and ANCHOR_MARKER in det:
            A, D = det[ANCHOR_MARKER], det[DOOR_MARKER]
            tracks.append(relative_pose(A.rotation, A.position, D.rotation, D.position))
            if grasp_in_part is None:
                # Where the carer's hand is on the door, in the door-marker frame. With a real
                # video this comes from hand tracking (e.g. MediaPipe wrist) at grasp time.
                grasp_in_part = D.rotation.T @ (d.site_xpos[handle] - D.position)
        if writer is not None:
            writer.write(cv2.cvtColor(cam.annotate(rgb, det), cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()
    model = fit_from_tracks(tracks, grasp_in_part)
    model.save(out)
    return model


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("video")
    v.add_argument("--source", default="0", help="camera index or video file")
    v.add_argument("--calib", type=Path, required=True, help="npz with K and dist (see calibrate_camera.py)")
    v.add_argument("--grasp", type=float, nargs=3, required=True,
                   help="handle position in the door-marker frame, metres (x right, y up as printed, z out)")
    v.add_argument("--out", type=Path, default=Path("cupboard_model.json"))
    s = sub.add_parser("sim")
    s.add_argument("--out", type=Path, default=Path("cupboard_model.json"))
    s.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()
    if args.mode == "video":
        model = capture_video(args.source, args.calib, args.out, np.array(args.grasp))
    else:
        model = capture_sim(args.out, video=args.video)
    print(model)


if __name__ == "__main__":
    main()
