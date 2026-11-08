"""RGB-D Head-camera rendering, depth-fused ArUco detection, and 3D point clouds.

The robot only knows where things are through its onboard sensors:
  * Renders both RGB and metric Depth (`enable_depth_rendering()`) from `head_cam`.
  * Refines monocular ArUco `solvePnP` depth using the metric depth buffer.
  * Updates the robot's localizer when known anchor markers are seen (`landmark_callback`).
  * Lifts detections into world coordinates using the localizer's estimated camera pose
    (`pose_provider`, computed via forward kinematics from estimated base pose + waist
    encoders), eliminating any dependency on ground-truth `data.cam_xpos`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import mujoco
import numpy as np

from scene import ITEMS, MARKER_DICT, DOOR_MARKER, ANCHOR_MARKER, DOOR_MARKER_SIZE

MARKER_SIZES = {it.marker_id: it.marker_size for it in ITEMS.values()}
MARKER_SIZES[DOOR_MARKER] = DOOR_MARKER_SIZE
MARKER_SIZES[ANCHOR_MARKER] = DOOR_MARKER_SIZE
# Support HouseEnv marker IDs (bottle 0, doors 10..17, drawers 20..29, anchors 30..34, user 40, person 45)
for _mid, _sz in (
    [(0, 0.05)]
    + [(m, 0.09) for m in range(10, 18)]
    + [(m, 0.084) for m in range(20, 30)]
    + [(m, 0.08) for m in range(30, 35)]
    + [(40, 0.12), (45, 0.09)]
):
    MARKER_SIZES.setdefault(_mid, _sz)

ID_TO_ITEM = {it.marker_id: name for name, it in ITEMS.items()}

# OpenCV camera frame (x right, y down, z forward) -> MuJoCo camera frame (x right, y up, z back)
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass
class MarkerPose:
    marker_id: int
    position: np.ndarray      # world (estimated), metres (marker centre)
    rotation: np.ndarray      # world (estimated), 3x3; column 2 is the marker's outward normal
    distance: float           # from the camera, metres
    cam_position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in MuJoCo camera frame
    cam_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))    # in MuJoCo camera frame
    depth_verified: bool = False


class HeadCamera:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cam: str = "head_cam",
        width: int = 960,
        height: int = 720,
        pose_provider: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
        landmark_callback: Callable[[dict[int, tuple[np.ndarray, np.ndarray]]], None] | None = None,
    ):
        # (the video recorder reuses this renderer for its inset)
        self.model, self.data = model, data
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        self.width, self.height = width, height
        self.pose_provider = pose_provider
        self.landmark_callback = landmark_callback
        self.renderer = mujoco.Renderer(model, height, width)
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        f = (height / 2) / np.tan(fovy / 2)
        self.fx = self.fy = float(f)
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.K = np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros(5)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)
        self.last_rgb: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None

    def _camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (cam_pos, cam_rot) from estimated state provider if wired, else MjData."""
        if self.pose_provider is not None:
            return self.pose_provider()
        return self.data.cam_xpos[self.cam_id].copy(), self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def render(self) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def render_depth(self) -> np.ndarray:
        """Render metric depth image (H, W) in metres along the camera optical axis."""
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data, camera=self.cam_id)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        # In MuJoCo >= 3.0, Renderer.render() with depth enabled returns metric distance in metres
        # (or normalized [0, 1] in older builds; handle both transparently).
        if depth.max() <= 1.0 and self.model.vis.map.zfar > 1.0:
            extent = float(self.model.stat.extent)
            near = float(self.model.vis.map.znear) * extent
            far = float(self.model.vis.map.zfar) * extent
            denom = 1.0 - depth * (1.0 - near / far)
            depth = near / np.maximum(denom, 1e-6)
        self.last_depth = depth
        return depth

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """Render both RGB (H, W, 3) and metric Depth (H, W) in metres."""
        rgb = self.render()
        depth = self.render_depth()
        self.last_rgb = rgb
        self.last_depth = depth
        return rgb, depth

    def _sample_marker_depth(self, corners_2d: np.ndarray, depth_img: np.ndarray) -> float | None:
        """Sample median metric depth inside the inner 50% of a 4-corner marker quad."""
        pts = corners_2d.reshape(4, 2)
        center = pts.mean(axis=0)
        inner = center + 0.5 * (pts - center)
        x0 = max(0, int(np.floor(inner[:, 0].min())))
        x1 = min(self.width - 1, int(np.ceil(inner[:, 0].max())))
        y0 = max(0, int(np.floor(inner[:, 1].min())))
        y1 = min(self.height - 1, int(np.ceil(inner[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            u = int(np.clip(round(center[0]), 0, self.width - 1))
            v = int(np.clip(round(center[1]), 0, self.height - 1))
            d_val = float(depth_img[v, u])
            return d_val if (0.08 < d_val < 6.0 and np.isfinite(d_val)) else None
        patch = depth_img[y0 : y1 + 1, x0 : x1 + 1]
        valid = patch[(patch > 0.08) & (patch < 6.0) & np.isfinite(patch)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def detect(self, rgb: np.ndarray | None = None, depth: np.ndarray | None = None) -> dict[int, MarkerPose]:
        if rgb is None:
            rgb, depth = self.render_rgbd()
        elif depth is None:
            depth = self.last_depth
        self.last_rgb = rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids = self._detect_raw(gray)
        out: dict[int, MarkerPose] = {}
        if ids is None:
            return out

        cam_dets: dict[int, tuple[np.ndarray, np.ndarray, float, bool]] = {}
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in MARKER_SIZES:
                continue
            s = MARKER_SIZES[mid] / 2
            obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                obj,
                c.reshape(4, 2).astype(np.float64),
                self.K,
                self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            t_cv = tvec.flatten()
            depth_verified = False
            if depth is not None and t_cv[2] > 0.08:
                z_d = self._sample_marker_depth(c, depth)
                # Fuse RGB-D metric depth with solvePnP z when unoccluded (within 6 cm gate)
                if z_d is not None and abs(z_d - t_cv[2]) < 0.06:
                    z_fused = 0.5 * z_d + 0.5 * float(t_cv[2])
                    t_cv = t_cv * (z_fused / float(t_cv[2]))
                    depth_verified = True

            r_cv, _ = cv2.Rodrigues(rvec)
            p_mj = CV_TO_MJ @ t_cv
            r_mj = CV_TO_MJ @ r_cv
            cam_dets[mid] = (p_mj, r_mj, float(np.linalg.norm(t_cv)), depth_verified)

        # Let the localizer fuse any static/movable anchor markers BEFORE querying camera world pose
        if self.landmark_callback is not None and cam_dets:
            self.landmark_callback({mid: (p_mj, r_mj) for mid, (p_mj, r_mj, _, _) in cam_dets.items()})

        cam_pos, cam_rot = self._camera_pose()
        for mid, (p_mj, r_mj, dist_m, depth_ok) in cam_dets.items():
            out[mid] = MarkerPose(
                marker_id=mid,
                position=cam_pos + cam_rot @ p_mj,
                rotation=cam_rot @ r_mj,
                distance=dist_m,
                cam_position=p_mj,
                cam_rotation=r_mj,
                depth_verified=depth_ok,
            )
        return out

    def point_cloud_world(
        self,
        depth: np.ndarray | None = None,
        stride: int = 8,
        max_dist: float = 2.5,
        min_dist: float = 0.15,
    ) -> np.ndarray:
        """Deproject the metric depth map into (N, 3) 3D points in the estimated world frame."""
        if depth is None:
            depth = self.last_depth if self.last_depth is not None else self.render_depth()
        vs, us = np.mgrid[0 : self.height : stride, 0 : self.width : stride]
        z = depth[vs, us]
        valid = (z >= min_dist) & (z <= max_dist) & np.isfinite(z)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float64)
        u_v = us[valid].astype(np.float64)
        v_v = vs[valid].astype(np.float64)
        z_v = z[valid].astype(np.float64)
        x_cv = (u_v - self.cx) * z_v / self.fx
        y_cv = (v_v - self.cy) * z_v / self.fy
        pts_cv = np.column_stack([x_cv, y_cv, z_v])
        pts_mj = pts_cv @ CV_TO_MJ.T
        cam_pos, cam_rot = self._camera_pose()
        return pts_mj @ cam_rot.T + cam_pos

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
