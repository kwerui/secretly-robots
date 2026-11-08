"""Head-camera rendering and ArUco detection -> marker poses in the world frame.

The robot only knows where things are through this module: it renders the head camera,
detects markers with OpenCV and lifts them to world coordinates using the camera pose.
Ground truth from the simulator is used only by tests, never by the skills.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from scene import ITEMS, MARKER_DICT, DOOR_MARKER, ANCHOR_MARKER, DOOR_MARKER_SIZE

MARKER_SIZES = {it.marker_id: it.marker_size for it in ITEMS.values()}
MARKER_SIZES[DOOR_MARKER] = DOOR_MARKER_SIZE
MARKER_SIZES[ANCHOR_MARKER] = DOOR_MARKER_SIZE
ID_TO_ITEM = {it.marker_id: name for name, it in ITEMS.items()}

# OpenCV camera frame (x right, y down, z forward) -> MuJoCo camera frame (x right, y up, z back)
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass
class MarkerPose:
    marker_id: int
    position: np.ndarray  # world, metres (marker centre)
    rotation: np.ndarray  # world, 3x3; column 2 is the marker's outward normal
    distance: float  # from the camera, metres


class HeadCamera:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cam: str = "head_cam",
                 width: int = 960, height: int = 720):
        # (the video recorder reuses this renderer for its inset)
        self.model, self.data = model, data
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        f = (height / 2) / np.tan(fovy / 2)
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros(5)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)
        self.last_rgb: np.ndarray | None = None

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def detect(self, rgb: np.ndarray | None = None) -> dict[int, MarkerPose]:
        if rgb is None:
            rgb = self.render()
        self.last_rgb = rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids = self._detect_raw(gray)
        out: dict[int, MarkerPose] = {}
        if ids is None:
            return out
        cam_pos = self.data.cam_xpos[self.cam_id].copy()
        cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in MARKER_SIZES:
                continue
            s = MARKER_SIZES[mid] / 2
            obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), self.K, self.dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            r_cv, _ = cv2.Rodrigues(rvec)
            p_mj = CV_TO_MJ @ tvec.flatten()
            r_mj = CV_TO_MJ @ r_cv
            out[mid] = MarkerPose(mid, cam_pos + cam_rot @ p_mj, cam_rot @ r_mj, float(np.linalg.norm(tvec)))
        return out

    def _detect_raw(self, gray: np.ndarray):
        """Detect markers; markers missed in the plain image (small or in shadow, e.g. inside a
        cupboard) get a second pass on a 2x upscaled, contrast-equalised (CLAHE) image."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        corners, ids = list(corners), ([] if ids is None else [int(i) for i in ids.flatten()])
        if len(set(ids) & set(MARKER_SIZES)) < len(MARKER_SIZES):
            eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            big = cv2.resize(eq, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            c2, i2, _ = self.detector.detectMarkers(big)
            if i2 is not None:
                for c, mid in zip(c2, i2.flatten()):
                    if int(mid) not in ids:
                        corners.append((c / 2.0 - 0.25).astype(np.float32))  # undo the 2x resize (pixel centres)
                        ids.append(int(mid))
        if not ids:
            return [], None
        return corners, np.array(ids).reshape(-1, 1)

    def annotate(self, rgb: np.ndarray, detections: dict[int, MarkerPose]) -> np.ndarray:
        img = np.ascontiguousarray(rgb.copy())
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids = self._detect_raw(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                label = ID_TO_ITEM.get(int(mid), {DOOR_MARKER: "door", ANCHOR_MARKER: "anchor"}.get(int(mid), "?"))
                x, y = c.reshape(4, 2)[0].astype(int)
                cv2.putText(img, label, (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return img
