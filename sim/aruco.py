"""ArUco marker generation, registry, and detection for secretly-robots simulation.

Uses OpenCV's DICT_4X4_50 standard. Pre-generates crisp PNG assets with quiet
margins for placement on doors, drawers, target items, delivery stations, and dynamic agents.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
ARUCO_DIR = ASSETS_DIR / "aruco"

# Standard semantic registry for the assistive robot fetch challenge
class MarkerInfo(TypedDict):
    name: str
    kind: str
    desc: str

ARUCO_REGISTRY: dict[int, MarkerInfo] = {
    0: {"name": "target_bottle", "kind": "target", "desc": "Water bottle target item"},
    1: {"name": "target_medicine", "kind": "target", "desc": "Medicine bottle secondary target"},
    10: {"name": "door_living_kitchen_a", "kind": "door_push", "desc": "Living room push plate to kitchen"},
    11: {"name": "door_living_kitchen_b", "kind": "door_push", "desc": "Kitchen push plate to living room"},
    12: {"name": "door_bedroom_study_a", "kind": "door_push", "desc": "Bedroom push plate to study"},
    13: {"name": "door_bedroom_study_b", "kind": "door_push", "desc": "Study push plate to bedroom"},
    14: {"name": "door_living_bedroom_a", "kind": "door_push", "desc": "Living room push plate to bedroom"},
    15: {"name": "door_living_bedroom_b", "kind": "door_push", "desc": "Bedroom push plate to living room"},
    16: {"name": "door_kitchen_study_a", "kind": "door_push", "desc": "Kitchen push plate to study"},
    17: {"name": "door_kitchen_study_b", "kind": "door_push", "desc": "Study push plate to kitchen"},
    20: {"name": "drawer_dresser_0", "kind": "drawer", "desc": "Bedroom dresser bottom drawer"},
    21: {"name": "drawer_dresser_1", "kind": "drawer", "desc": "Bedroom dresser drawer 1"},
    22: {"name": "drawer_dresser_2", "kind": "drawer_handle", "desc": "Bedroom dresser drawer 2"},
    23: {"name": "drawer_dresser_3", "kind": "drawer_handle", "desc": "Bedroom dresser top drawer"},
    24: {"name": "drawer_kitchen_0", "kind": "drawer_handle", "desc": "Kitchen unit bottom drawer"},
    25: {"name": "drawer_kitchen_1", "kind": "drawer_handle", "desc": "Kitchen unit middle drawer"},
    26: {"name": "drawer_kitchen_2", "kind": "drawer_handle", "desc": "Kitchen unit top drawer"},
    27: {"name": "drawer_filing_0", "kind": "drawer_handle", "desc": "Study filing cabinet bottom drawer"},
    28: {"name": "drawer_filing_1", "kind": "drawer_handle", "desc": "Study filing cabinet middle drawer"},
    29: {"name": "drawer_filing_2", "kind": "drawer_handle", "desc": "Study filing cabinet top drawer"},
    30: {"name": "anchor_coffee_table", "kind": "anchor", "desc": "Living room coffee table anchor"},
    31: {"name": "anchor_dining_table", "kind": "anchor", "desc": "Kitchen dining table anchor"},
    32: {"name": "anchor_kitchen_counter", "kind": "anchor", "desc": "Kitchen counter anchor"},
    33: {"name": "anchor_bedside_table", "kind": "anchor", "desc": "Bedroom bedside table anchor"},
    34: {"name": "anchor_study_desk", "kind": "anchor", "desc": "Study desk anchor"},
    40: {"name": "user_station", "kind": "delivery_target", "desc": "Seated user delivery tray station"},
    45: {"name": "dynamic_person", "kind": "agent_tag", "desc": "Dynamic walking person tracking marker"},
}


def generate_marker_image(marker_id: int, size: int = 256, quiet_zone: int = 24) -> Image.Image:
    """Generate a high-contrast ArUco marker image with a white quiet margin.

    Returns a PIL Image in RGB format.
    """
    if HAS_OPENCV:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_size = size - 2 * quiet_zone
        marker_mat = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        bordered = cv2.copyMakeBorder(
            marker_mat, quiet_zone, quiet_zone, quiet_zone, quiet_zone,
            cv2.BORDER_CONSTANT, value=255
        )
        return Image.fromarray(bordered).convert("RGB")
    else:
        # Fallback pure-Python generator for environments where cv2 is unavailable
        # Renders a high-contrast 4x4 matrix with black boundary
        rng = np.random.default_rng(marker_id * 31337 + 7)
        bits = rng.integers(0, 2, size=(4, 4), dtype=np.uint8)
        # 6x6 grid: 1-cell black border + 4x4 data
        grid = np.zeros((6, 6), dtype=np.uint8)
        grid[1:5, 1:5] = bits * 255
        img = Image.fromarray(grid, mode="L").resize((size - 2 * quiet_zone, size - 2 * quiet_zone), Image.NEAREST)
        canvas = Image.new("L", (size, size), 255)
        canvas.paste(img, (quiet_zone, quiet_zone))
        return canvas.convert("RGB")


def ensure_aruco_assets(out_dir: Path | None = None) -> list[Path]:
    """Pre-generate all required ArUco marker images into the assets directory."""
    target_dir = Path(out_dir) if out_dir else ARUCO_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mid in ARUCO_REGISTRY:
        path = target_dir / f"marker_{mid}.png"
        img = generate_marker_image(mid, size=256, quiet_zone=24)
        img.save(path)
        paths.append(path)
    return paths


def detect_markers(image_rgb: np.ndarray) -> list[dict]:
    """Detect ArUco markers in an RGB image.

    Returns a list of dicts with:
      - id: int
      - name: str
      - kind: str
      - center: tuple[float, float] in image pixel coordinates
      - corners: list of (x, y) coordinates
    """
    if not HAS_OPENCV:
        return []

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []

    results = []
    flat_ids = ids.flatten()
    for i, mid in enumerate(flat_ids):
        c = corners[i][0]  # shape (4, 2)
        cx = float(np.mean(c[:, 0]))
        cy = float(np.mean(c[:, 1]))
        meta = ARUCO_REGISTRY.get(int(mid), {"name": f"marker_{mid}", "kind": "unknown", "desc": ""})
        results.append({
            "id": int(mid),
            "name": meta["name"],
            "kind": meta["kind"],
            "desc": meta["desc"],
            "center": (cx, cy),
            "corners": [(float(pt[0]), float(pt[1])) for pt in c],
        })
    return results


if __name__ == "__main__":
    generated = ensure_aruco_assets()
    print(f"Generated {len(generated)} ArUco marker assets in {ARUCO_DIR}")
