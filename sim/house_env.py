"""Simulated house for the fetch-robot project (MuJoCo).

An 8 x 6 m house with four rooms (living room, kitchen, bedroom, study) joined
by four doorways that form a loop. Walls and furniture are realistic architectural geoms.

Features & Difficulty Variables:
- Mode 1 ('static'): Baseline house layout, nominal furniture, static seated user with limited mobility.
- Mode 2 ('shuffled'): Furniture positions shuffled within rooms and target items placed on surfaces or inside drawers.
- Mode 3 ('dynamic'): A dynamic person walks into the house, approaches an item, and relocates it, testing obstacle avoidance and adaptive replanning.
- Doors are push-only with wide stainless steel push plates and ArUco markers on both faces (no turn handles).
- Drawers have long bar handles and ArUco markers, with full support for target items stored inside drawers.
- Visually revamped with photorealistic procedural textures (parquet, ceramic tile, woven carpet, marble, wood, fabric, brushed metal).
- Detailed human models: seated person with limited mobility and dynamic walking human.
- First-person robot head camera (`cam_robot_head`) and room cameras.

CLI:
    python house_env.py --check                 # verify every layout x target
    python house_env.py --snapshot              # PNGs into sim/snapshots/
    python house_env.py --view                  # interactive viewer (use run_sim.sh on macOS)
    python house_env.py --view --mode dynamic   # test dynamic walking person relocating bottle
    python house_env.py --view --mode shuffled  # test shuffled furniture & drawer targets
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

try:
    from .aruco import ARUCO_DIR, ARUCO_REGISTRY, ensure_aruco_assets, HAS_OPENCV
    from .dynamic_agent import DynamicAgentController, TrajectoryWaypoint
    from .textures import TEXTURES_DIR, ensure_texture_assets
except ImportError:
    from aruco import ARUCO_DIR, ARUCO_REGISTRY, ensure_aruco_assets, HAS_OPENCV
    from dynamic_agent import DynamicAgentController, TrajectoryWaypoint
    from textures import TEXTURES_DIR, ensure_texture_assets

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"

# Ensure assets are present
if not (ARUCO_DIR.exists() and any(ARUCO_DIR.iterdir())):
    ensure_aruco_assets(ARUCO_DIR)
if not (TEXTURES_DIR.exists() and any(TEXTURES_DIR.iterdir())):
    ensure_texture_assets(TEXTURES_DIR)

# =============================================================================
# CONFIG
# =============================================================================


class Mode(str, Enum):
    STATIC = "static"        # Level 1: Standard house, static layout
    SHUFFLED = "shuffled"    # Level 2: Shuffled furniture & targets (including drawers)
    DYNAMIC = "dynamic"      # Level 3: Dynamic person walks in and relocates an item


# --- Humanoid Robot Tunables --------------------------------------------------
ROBOT_R = 0.28   # Humanoid turning radius
ROBOT_H = 1.65   # Humanoid standing height
REACH = 0.85     # Humanoid arm reach radius
ROBOT_START = (-1.8, -1.0, 0.0)  # Delivery pose in front of user in living room
HEAD_CAM_Z = 1.55  # height of cam_robot_head, which follows the robot's root body

# --- House Geometry -----------------------------------------------------------
WALL_T = 0.10
WALL_H = 1.20   # Dollhouse wall height for optimal camera visibility
DOOR_W = 1.00   # Doorway clear width
DOOR_H = 1.00   # Door leaf height
GRID_RES = 0.05

WALLS = {
    "outer_s": (-4.0, -3.0, 4.0, -3.0),
    "outer_n": (-4.0, 3.0, 4.0, 3.0),
    "outer_w": (-4.0, -3.0, -4.0, 3.0),
    "outer_e": (4.0, -3.0, 4.0, 3.0),
    "inner_v_s": (0.0, -3.0, 0.0, 0.0),  # living | kitchen
    "inner_v_n": (0.0, 0.0, 0.0, 3.0),   # bedroom | study
    "inner_h_w": (-4.0, 0.0, 0.0, 0.0),  # living | bedroom
    "inner_h_e": (0.0, 0.0, 4.0, 0.0),   # kitchen | study
}

DOORWAYS = {
    "living_kitchen": dict(wall="inner_v_s", center=(0.0, -1.5), hinge="end", marker_a=10, marker_b=11),
    "bedroom_study": dict(wall="inner_v_n", center=(0.0, 1.5), hinge="end", marker_a=12, marker_b=13),
    "living_bedroom": dict(wall="inner_h_w", center=(-2.0, 0.0), hinge="start", marker_a=14, marker_b=15),
    "kitchen_study": dict(wall="inner_h_e", center=(2.0, 0.0), hinge="end", marker_a=16, marker_b=17),
}
DOOR_SWING_DEG = 90.0
DOOR_DAMPING = 1.8
DOOR_FRICTION = 0.35
DOOR_MASS = 12.0

ROOMS = {
    "living": (-4.0, -3.0, 0.0, 0.0, "parquet"),
    "kitchen": (0.0, -3.0, 4.0, 0.0, "tile"),
    "bedroom": (-4.0, 0.0, 0.0, 3.0, "parquet"),
    "study": (0.0, 0.0, 4.0, 3.0, "carpet"),
}

# Furniture definitions
_WOOD = (0.55, 0.38, 0.22)
_LIGHT_WOOD = (0.76, 0.60, 0.42)
_WHITE = (0.92, 0.92, 0.90)

FURNITURE = {
    # living room
    "sofa": dict(kind="sofa", pos=(-3.55, -1.5), size=(0.80, 2.00), h=0.45, rgb=(0.35, 0.45, 0.60)),
    "coffee_table": dict(kind="table", pos=(-2.70, -1.5), size=(0.50, 0.90), h=0.40, rgb=_LIGHT_WOOD, movable=True),
    "tv_unit": dict(kind="box", pos=(-1.80, -2.775), size=(1.20, 0.35), h=0.50, rgb=(0.25, 0.25, 0.28)),
    # kitchen
    "counter": dict(kind="counter", pos=(2.25, -2.65), size=(2.10, 0.60), h=0.90, rgb=_WHITE),
    "fridge": dict(kind="fridge", pos=(3.62, -2.64), size=(0.65, 0.62), h=1.80, rgb=(0.85, 0.87, 0.90)),
    "kitchen_table": dict(kind="dining", pos=(3.30, -1.00), size=(0.80, 0.60), h=0.75, rgb=_WOOD, movable=True),
    # bedroom
    "bed": dict(kind="bed", pos=(-2.95, 2.25), size=(2.00, 1.40), h=0.50, rgb=(0.90, 0.85, 0.80)),
    "bedside_table": dict(kind="box", pos=(-3.70, 1.10), size=(0.40, 0.40), h=0.55, rgb=_LIGHT_WOOD, movable=True),
    # study
    "desk": dict(kind="desk", pos=(3.60, 1.50), size=(0.70, 1.40), h=0.75, rgb=_LIGHT_WOOD, movable=True),
    "bookshelf": dict(kind="bookshelf", pos=(2.00, 2.80), size=(1.40, 0.30), h=1.10, rgb=_WOOD),
}

# Chests of drawers with long bar handles and ArUco markers
DRAWER_UNITS = {
    # tall enough (1.10 m) that the 21 cm bottle stands upright in every drawer
    "dresser": dict(pos=(-0.90, 2.725), facing=-90, size=(1.00, 0.45), h=1.10, n=4, rgb=_WOOD, marker_start=20),
    "kitchen_drawers": dict(pos=(0.90, -2.65), facing=90, size=(0.60, 0.60), h=0.90, n=3, rgb=_WHITE, marker_start=24),
    "filing_cabinet": dict(pos=(3.725, 2.60), facing=180, size=(0.60, 0.45), h=1.00, n=3, rgb=(0.45, 0.50, 0.55), marker_start=27),
}
DRAWER_EXTENSION = 0.80
HANDLE_FRACTION = 0.75
HANDLE_R = 0.012
HANDLE_STANDOFF = 0.045
DRAWER_DAMPING = 25.0  # N s/m: a 10 N pull opens at ~0.4 m/s and a bottle inside stays upright
DRAWER_FRICTION = 0.5

# Surface target locations
SURFACE_TARGETS = {
    "tv_unit": (-1.50, -2.78, "tv_unit"),
    "kitchen_counter": (1.80, -2.62, "counter"),
    "kitchen_table": (3.10, -0.90, "kitchen_table"),
    "bedside_table": (-3.70, 1.10, "bedside_table"),
    "study_desk": (3.50, 1.20, "desk"),
    "coffee_table": (-2.70, -1.50, "coffee_table"),
}

# Targets located inside drawers (drawer_name, drawer_index)
DRAWER_TARGETS = {
    "dresser_drawer": (-0.90, 2.725, "dresser", 3),
    "kitchen_drawer": (0.90, -2.65, "kitchen_drawers", 2),
    "filing_drawer": (3.725, 2.60, "filing_cabinet", 2),
}

ALL_TARGETS = {**SURFACE_TARGETS, **DRAWER_TARGETS}
TARGETS = SURFACE_TARGETS  # preserves backward compatibility with existing tests

# Clutter obstacles
_BROWN = (0.70, 0.55, 0.35)
OBSTACLE_TYPES = {
    "box_small": dict(rgb=_BROWN, geoms=[("box", (0.20, 0.15, 0.15), (0, 0, 0.15))]),
    "box_large": dict(rgb=(0.62, 0.48, 0.30), geoms=[("box", (0.30, 0.25, 0.25), (0, 0, 0.25))]),
    "laundry_basket": dict(rgb=(0.85, 0.85, 0.75), geoms=[("cyl", (0.25, 0.20), (0, 0, 0.20))]),
    "clothes_pile": dict(rgb=(0.80, 0.30, 0.35), geoms=[("box", (0.30, 0.22, 0.06), (0, 0, 0.06))]),
    "bag": dict(rgb=(0.20, 0.30, 0.25), geoms=[("box", (0.20, 0.12, 0.18), (0, 0, 0.18))]),
    "chair": dict(
        rgb=(0.45, 0.30, 0.20),
        geoms=[
            ("box", (0.22, 0.22, 0.225), (0, 0, 0.225)),
            ("box", (0.03, 0.22, 0.25), (-0.19, 0, 0.70)),
        ],
    ),
}

LAYOUTS = {
    "tidy": dict(obstacles=[]),
    "laundry_day": dict(
        obstacles=[
            ("laundry_basket", -1.00, 0.80, 0),
            ("clothes_pile", -1.10, -2.10, 20),
            ("clothes_pile", 1.10, 1.00, -30),
            ("clothes_pile", -1.55, 1.45, 10),
            ("bag", 0.80, -0.50, 45),
            ("chair", 2.30, -1.80, 0),
        ],
    ),
    "door_shut": dict(
        doors_locked=["living_kitchen"],
        obstacles=[
            ("chair", 1.40, -1.30, 0),
            ("box_small", -0.80, -2.30, 0),
        ],
    ),
    "moving_house": dict(
        doors_closed=["bedroom_study"],
        obstacles=[
            ("box_large", -0.80, -0.60, 10),
            ("box_small", -0.70, -2.40, 0),
            ("box_large", 1.70, -1.75, -15),
            ("box_small", 3.20, -0.60, 30),
            ("box_large", 0.90, 2.40, 0),
            ("box_small", 3.00, 0.50, 15),
            ("box_large", -1.35, 1.25, 0),
            ("box_small", -3.30, 0.50, 5),
        ],
    ),
    "long_way_round": dict(
        doors_locked=["living_bedroom"],
        obstacles=[
            ("laundry_basket", 1.00, -0.60, 0),
            ("clothes_pile", 3.20, -1.00, 60),
            ("bag", 2.80, 0.70, 0),
            ("chair", 1.30, 2.20, 180),
        ],
    ),
}

JITTER_XY = 0.15
JITTER_YAW = 10.0
# Shuffled mode (Level 2): every FURNITURE item with movable=True is moved by up to this
# much on each reset (and the bottle with it). A move that hits a wall, other furniture,
# a door swing or a drawer, or leaves the bottle unreachable, is re-drawn.
SHUFFLE_XY = 0.40
SHUFFLE_YAW = 20.0
SHUFFLE_TRIES = 30  # draws per piece before it stays at its nominal pose
MAX_RESET_TRIES = 50

# Dynamic person (Mode 3). Their route is planned on the occupancy grid after the
# layout is placed, so they walk round furniture, clutter, open door leaves and the robot.
PERSON_R = 0.32          # footprint radius incl. swinging arms, carried bottle and a margin
PERSON_H = 1.80          # anything lower than this blocks them
PERSON_SPEED = 0.9       # walking speed, m/s
PERSON_TURN_S = 0.5      # time to turn on the spot
PERSON_HANDLE_S = 2.0    # time spent picking up / putting down
PERSON_REACH = 0.80      # stands within this of the item to pick it up / put it down
# Where they come in from: one point per room; they use one not in the bottle's room
PERSON_ENTRIES = [(2.70, 1.00), (-2.30, 1.00), (2.50, -0.45), (-0.90, -1.20)]
# Where they put the bottle: the first of these that isn't the bottle's start and both
# the person and the robot can reach
PERSON_DROPS = ["kitchen_table", "kitchen_counter", "study_desk", "coffee_table", "tv_unit", "bedside_table"]
PARK_Z = -5.0

# Bottle specifications
BOTTLE_R = 0.033
BOTTLE_H = 0.21
BOTTLE_MASS = 0.5

_THEIR_BODY_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0])
_THEIR_SITE_Q = np.array([-0.5, 0.5, -0.5, 0.5])


def yaw_quat(deg: float) -> np.ndarray:
    a = math.radians(deg) / 2.0
    return np.array([math.cos(a), 0.0, 0.0, math.sin(a)])


def lookat_quat(pos, target) -> np.ndarray:
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    mat = np.column_stack([x, y, -f]).flatten()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _bottle_site_quat() -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, _THEIR_BODY_Q, _THEIR_SITE_Q)
    return q


HINGE_INSET = 0.03
DOOR_LEAF_L = DOOR_W - HINGE_INSET - 0.02
DOOR_LEAF_T = 0.04


def door_frame(name: str):
    dw = DOORWAYS[name]
    x0, y0, x1, y1 = WALLS[dw["wall"]]
    u = np.array([x1 - x0, y1 - y0], float)
    u /= np.linalg.norm(u)
    c = np.array(dw["center"], float)
    sgn = 1.0 if dw["hinge"] == "end" else -1.0
    hinge = c + sgn * u * (DOOR_W / 2 - HINGE_INSET)
    d = -sgn * u
    return hinge, math.degrees(math.atan2(d[1], d[0]))


def _wall_segments():
    segs = []
    for wname, (x0, y0, x1, y1) in WALLS.items():
        p0, p1 = np.array([x0, y0]), np.array([x1, y1])
        length = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / length
        cuts = []
        for dw in DOORWAYS.values():
            if dw["wall"] == wname:
                t = float(np.dot(np.array(dw["center"]) - p0, u))
                cuts.append((t - DOOR_W / 2, t + DOOR_W / 2))
        cuts.sort()
        start, end = -WALL_T / 2, length + WALL_T / 2
        if wname.startswith("inner"):
            start, end = WALL_T / 2, length - WALL_T / 2
        pieces, cur = [], start
        for a, b in cuts:
            pieces.append((cur, a))
            cur = b
        pieces.append((cur, end))
        yaw = math.degrees(math.atan2(u[1], u[0]))
        for i, (a, b) in enumerate(pieces):
            if b - a < 1e-3:
                continue
            mid = p0 + u * (a + b) / 2
            segs.append((f"wall_{wname}_{i}", mid[0], mid[1], (b - a) / 2, yaw))
    return segs


def _obstacle_pool():
    pool = {}
    for lay in LAYOUTS.values():
        counts = {}
        for typ, *_ in lay["obstacles"]:
            counts[typ] = counts.get(typ, 0) + 1
        for typ, c in counts.items():
            pool[typ] = max(pool.get(typ, 0), c)
    return pool


# =============================================================================
# Scene XML Generation
# =============================================================================


def build_xml() -> str:
    L = []
    a = L.append
    a('<mujoco model="secretly_robots_house">')
    a(f'  <compiler assetdir="{ASSETS_DIR}" angle="radian" autolimits="true"/>')
    a('  <option timestep="0.002" integrator="implicitfast"/>')
    a('  <statistic center="0 0 0.5" extent="5"/>')
    a("  <visual>")
    a('    <global azimuth="120" elevation="-50" offwidth="1600" offheight="1200"/>')
    a('    <headlight ambient="0.30 0.30 0.30" diffuse="0.35 0.35 0.35" specular="0.10 0.10 0.10"/>')
    a('    <quality shadowsize="4096"/>')
    a("  </visual>")
    a("  <asset>")
    a('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.88 0.92 0.96" rgb2="0.65 0.75 0.85" width="256" height="256"/>')
    a('    <texture name="ground" type="2d" builtin="checker" rgb1="0.40 0.44 0.42" rgb2="0.36 0.40 0.38" width="256" height="256"/>')
    a('    <material name="ground" texture="ground" texrepeat="20 20"/>')

    # Procedural texture assets
    a('    <texture name="tex_parquet" type="2d" file="textures/floor_parquet.png"/>')
    a('    <texture name="tex_tile" type="2d" file="textures/floor_tile.png"/>')
    a('    <texture name="tex_carpet" type="2d" file="textures/floor_carpet.png"/>')
    a('    <texture name="tex_wall" type="2d" file="textures/wall_plaster.png"/>')
    a('    <texture name="tex_wood" type="2d" file="textures/wood_grain.png"/>')
    a('    <texture name="tex_marble" type="2d" file="textures/marble_counter.png"/>')
    a('    <texture name="tex_fabric" type="2d" file="textures/fabric_sofa.png"/>')
    a('    <texture name="tex_metal" type="2d" file="textures/metal_brushed.png"/>')
    a('    <texture name="tex_rug" type="2d" file="textures/rug_geometric.png"/>')

    # Materials
    a('    <material name="mat_parquet" texture="tex_parquet" texrepeat="6 5" roughness="0.3" reflectance="0.15"/>')
    a('    <material name="mat_tile" texture="tex_tile" texrepeat="6 5" roughness="0.2" reflectance="0.25"/>')
    a('    <material name="mat_carpet" texture="tex_carpet" texrepeat="8 6" roughness="0.8" reflectance="0.05"/>')
    a('    <material name="mat_wall" texture="tex_wall" texrepeat="4 2" roughness="0.6" reflectance="0.1"/>')
    a('    <material name="mat_wood" texture="tex_wood" texrepeat="2 2" roughness="0.4" reflectance="0.15"/>')
    a('    <material name="mat_marble" texture="tex_marble" texrepeat="2 2" roughness="0.15" reflectance="0.35"/>')
    a('    <material name="mat_fabric" texture="tex_fabric" texrepeat="4 4" roughness="0.7" reflectance="0.05"/>')
    a('    <material name="mat_metal" texture="tex_metal" texrepeat="2 2" roughness="0.25" reflectance="0.5"/>')
    a('    <material name="mat_rug" texture="tex_rug" texrepeat="1 1" roughness="0.6" reflectance="0.05"/>')
    a('    <material name="door" texture="tex_wood" texrepeat="1 2" roughness="0.35" reflectance="0.15"/>')

    # ArUco marker textures & materials
    for mid in ARUCO_REGISTRY:
        a(f'    <texture name="tex_marker_{mid}" type="2d" file="aruco/marker_{mid}.png"/>')
        a(f'    <material name="mat_marker_{mid}" texture="tex_marker_{mid}" texuniform="false" texrepeat="1 1" roughness="0.2"/>')
    a("  </asset>")

    a("  <worldbody>")
    # Lighting: Sun, sky ambient, and warm recessed room spot lights
    a('    <light name="sun" pos="3 -4 8" dir="-0.3 0.4 -1" directional="true" castshadow="true" diffuse="0.55 0.52 0.48" specular="0.10 0.10 0.10"/>')
    a('    <light name="fill" pos="-3 3 7" dir="0.3 -0.3 -1" directional="true" castshadow="false" diffuse="0.25 0.26 0.28" specular="0 0 0"/>')
    a('    <light name="light_living" pos="-2.0 -1.5 2.5" dir="0 0 -1" directional="false" diffuse="0.45 0.42 0.38" specular="0.05 0.05 0.05" castshadow="false"/>')
    a('    <light name="light_kitchen" pos="2.0 -1.5 2.5" dir="0 0 -1" directional="false" diffuse="0.48 0.46 0.44" specular="0.05 0.05 0.05" castshadow="false"/>')
    a('    <light name="light_bedroom" pos="-2.0 1.5 2.5" dir="0 0 -1" directional="false" diffuse="0.42 0.40 0.38" specular="0.05 0.05 0.05" castshadow="false"/>')
    a('    <light name="light_study" pos="2.0 1.5 2.5" dir="0 0 -1" directional="false" diffuse="0.44 0.43 0.40" specular="0.05 0.05 0.05" castshadow="false"/>')

    a('    <geom name="floor" type="plane" size="14 14 0.05" material="ground"/>')

    # Room floors with realistic materials
    for rname, (x0, y0, x1, y1, mat_type) in ROOMS.items():
        mat_name = f"mat_{mat_type}"
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy = (x1 - x0) / 2, (y1 - y0) / 2
        a(
            f'    <geom name="floor_{rname}" type="box" pos="{_fmt([cx, cy, 0.0005])}" '
            f'size="{_fmt([sx, sy, 0.0005])}" material="{mat_name}" contype="0" conaffinity="0"/>'
        )

    # Living room area rug
    a('    <geom name="living_rug" type="box" pos="-2.70 -1.50 0.001" size="1.20 1.50 0.0005" material="mat_rug" contype="0" conaffinity="0"/>')

    # Walls with plaster texture
    for name, cx, cy, hl, yaw in _wall_segments():
        a(
            f'    <geom name="{name}" type="box" pos="{_fmt([cx, cy, WALL_H / 2])}" quat="{_fmt(yaw_quat(yaw))}" '
            f'size="{_fmt([hl, WALL_T / 2, WALL_H / 2])}" material="mat_wall"/>'
        )
        # Baseboard skirt
        a(
            f'    <geom name="{name}_skirt" type="box" pos="{_fmt([cx, cy, 0.04])}" quat="{_fmt(yaw_quat(yaw))}" '
            f'size="{_fmt([hl, WALL_T / 2 + 0.008, 0.04])}" material="mat_wood" contype="0" conaffinity="0"/>'
        )

    # Furniture
    _render_furniture(a)

    # Realistic Seated User on the sofa (person with limited mobility)
    _render_seated_user(a)

    # Dynamic Walking Person (mocap body, animated in Mode 3)
    _render_dynamic_person(a)

    # Doors: Push-only with wide stainless steel push plates & ArUco markers on both faces
    _render_push_doors(a)

    # Chests of drawers with long bar handles and ArUco markers
    for uname, u in DRAWER_UNITS.items():
        _drawer_unit_xml(a, uname, u)

    # Clutter obstacles (mocap, parked)
    for typ, count in _obstacle_pool().items():
        spec = OBSTACLE_TYPES[typ]
        for i in range(count):
            a(f'    <body name="obs_{typ}_{i}" mocap="true" pos="0 0 {PARK_Z}">')
            for j, g in enumerate(spec["geoms"]):
                kind, size, pos = g
                gtype = "box" if kind == "box" else "cylinder"
                a(f'      <geom name="obs_{typ}_{i}_{j}" type="{gtype}" size="{_fmt(size)}" pos="{_fmt(pos)}" rgba="{_fmt(spec["rgb"])} 1"/>')
            a("    </body>")

    # Target water bottle with ArUco Marker #0 badge
    _render_waterbottle(a)

    # Cameras
    a('    <camera name="overview" pos="0 0 8.2" quat="1 0 0 0" fovy="45"/>')
    # Dedicated First-Person Humanoid Robot Head Camera looking forward from eye height
    # It sits on a mocap mount that HouseEnv moves to the robot's root body every step
    # (x, y and heading; height HEAD_CAM_Z). Without a robot it stays at ROBOT_START.
    rx, ry, ryaw = ROBOT_START
    head_q = lookat_quat([0, 0, 0], [1.5, 0, 0.60 - HEAD_CAM_Z])
    a(f'    <body name="robot_head_mount" mocap="true" pos="{_fmt([rx, ry, HEAD_CAM_Z])}" quat="{_fmt(yaw_quat(ryaw))}">')
    a(f'      <camera name="cam_robot_head" quat="{_fmt(head_q)}" fovy="65"/>')
    a("    </body>")

    for rname, (x0, y0, x1, y1, _) in ROOMS.items():
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        px = x0 + 0.3 if cx < 0 else x1 - 0.3
        py = y0 + 0.3 if cy < 0 else y1 - 0.3
        q = lookat_quat([px, py, 3.0], [cx, cy, 0.3])
        a(f'    <camera name="cam_{rname}" pos="{_fmt([px, py, 3.0])}" quat="{_fmt(q)}" fovy="70"/>')

    a("  </worldbody>")

    # Door locks (equality constraints, activated per layout)
    a("  <equality>")
    for dname in DOORWAYS:
        a(f'    <joint name="lock_{dname}" joint1="door_{dname}_hinge" polycoef="0 1 0 0 0" active="false"/>')
    a("  </equality>")
    a("</mujoco>")
    return "\n".join(L)


def _render_furniture(a):
    for name, f in FURNITURE.items():
        (x, y), (sx, sy), h, rgb = f["pos"], f["size"], f["h"], f["rgb"]
        col = f'rgba="{_fmt(rgb)} 1"'
        if f.get("movable"):
            # own body, so shuffled mode can move it with model.body_pos / body_quat
            a(f'    <body name="furn_{name}" pos="{_fmt([x, y, 0])}">')
            x, y = 0.0, 0.0

        if f["kind"] == "sofa":
            # Realistic sofa with comfortable fabric cushions, armrests, and throw pillows
            back = 0.18
            seat_h = h
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x + back / 2, y, seat_h / 2])}" size="{_fmt([(sx - back) / 2, sy / 2, seat_h / 2])}" material="mat_fabric"/>')
            a(f'    <geom name="{name}_back" type="box" pos="{_fmt([x - sx / 2 + back / 2, y, 0.45])}" size="{_fmt([back / 2, sy / 2, 0.40])}" material="mat_fabric"/>')
            # Armrests
            arm_w = 0.12
            for i, side_y in enumerate((-sy / 2 + arm_w / 2, sy / 2 - arm_w / 2)):
                a(f'    <geom name="{name}_arm{i}" type="box" pos="{_fmt([x, y + side_y, seat_h + 0.08])}" size="{_fmt([sx / 2, arm_w / 2, 0.10])}" material="mat_fabric"/>')

        elif f["kind"] == "table":
            top = 0.03
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h - top / 2])}" size="{_fmt([sx / 2, sy / 2, top / 2])}" material="mat_wood"/>')
            lx, ly, lh = sx / 2 - 0.05, sy / 2 - 0.05, (h - top) / 2
            for i, (px, py) in enumerate([(lx, ly), (-lx, ly), (lx, -ly), (-lx, -ly)]):
                a(f'    <geom name="{name}_leg{i}" type="box" pos="{_fmt([x + px, y + py, lh])}" size="{_fmt([0.025, 0.025, lh])}" material="mat_wood"/>')
            # Anchor ArUco marker on table corner
            a(f'    <geom name="{name}_marker" type="box" pos="{_fmt([x - sx / 2 + 0.06, y + sy / 2 - 0.06, h + 0.001])}" size="0.04 0.04 0.001" material="mat_marker_30" contype="0" conaffinity="0"/>')

        elif f["kind"] == "dining":
            top = 0.035
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h - top / 2])}" size="{_fmt([sx / 2, sy / 2, top / 2])}" material="mat_wood"/>')
            lx, ly, lh = sx / 2 - 0.05, sy / 2 - 0.05, (h - top) / 2
            for i, (px, py) in enumerate([(lx, ly), (-lx, ly), (lx, -ly), (-lx, -ly)]):
                a(f'    <geom name="{name}_leg{i}" type="box" pos="{_fmt([x + px, y + py, lh])}" size="{_fmt([0.025, 0.025, lh])}" material="mat_wood"/>')
            a(f'    <geom name="{name}_marker" type="box" pos="{_fmt([x + sx / 2 - 0.06, y - sy / 2 + 0.06, h + 0.001])}" size="0.04 0.04 0.001" material="mat_marker_31" contype="0" conaffinity="0"/>')

        elif f["kind"] == "counter":
            # Kitchen counter with polished marble countertop and sink basin
            top_h = 0.04
            base_h = h - top_h
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, base_h / 2])}" size="{_fmt([sx / 2, sy / 2, base_h / 2])}" {col}/>')
            a(f'    <geom name="{name}_top" type="box" pos="{_fmt([x, y, h - top_h / 2])}" size="{_fmt([sx / 2, sy / 2, top_h / 2])}" material="mat_marble"/>')
            # Faucet
            a(f'    <geom name="{name}_faucet" type="cylinder" pos="{_fmt([x - 0.40, y - 0.15, h + 0.10])}" size="0.015 0.10" material="mat_metal" contype="0" conaffinity="0"/>')
            a(f'    <geom name="{name}_marker" type="box" pos="{_fmt([x - sx / 2 + 0.08, y + 0.15, h + 0.001])}" size="0.04 0.04 0.001" material="mat_marker_32" contype="0" conaffinity="0"/>')

        elif f["kind"] == "fridge":
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h / 2])}" size="{_fmt([sx / 2, sy / 2, h / 2])}" material="mat_metal"/>')
            # French door handles
            for i, hx in enumerate((-0.03, 0.03)):
                a(f'    <geom name="{name}_handle{i}" type="capsule" fromto="{_fmt([x + hx, y - sy / 2 - 0.03, 0.85, x + hx, y - sy / 2 - 0.03, 1.35])}" size="0.012" material="mat_metal" contype="0" conaffinity="0"/>')

        elif f["kind"] == "bed":
            head_h, head_t = 0.90, 0.08
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h / 2])}" size="{_fmt([sx / 2, sy / 2, h / 2])}" material="mat_wood"/>')
            # Mattress and folded duvet
            a(f'    <geom name="{name}_mattress" type="box" pos="{_fmt([x, y, h + 0.10])}" size="{_fmt([sx / 2 - 0.04, sy / 2 - 0.04, 0.10])}" rgba="0.94 0.94 0.96 1"/>')
            a(f'    <geom name="{name}_duvet" type="box" pos="{_fmt([x + 0.20, y, h + 0.15])}" size="{_fmt([sx / 2 - 0.24, sy / 2 - 0.05, 0.06])}" rgba="0.65 0.72 0.82 1"/>')
            # Pillows against headboard (west side)
            a(f'    <geom name="{name}_pillow0" type="box" pos="{_fmt([x - sx / 2 + 0.25, y - 0.35, h + 0.22])}" size="0.18 0.25 0.05" rgba="0.95 0.95 0.95 1"/>')
            a(f'    <geom name="{name}_pillow1" type="box" pos="{_fmt([x - sx / 2 + 0.25, y + 0.35, h + 0.22])}" size="0.18 0.25 0.05" rgba="0.95 0.95 0.95 1"/>')
            # Wooden headboard against west wall
            a(f'    <geom name="{name}_headboard" type="box" pos="{_fmt([x - sx / 2 + head_t / 2, y, head_h / 2])}" size="{_fmt([head_t / 2, sy / 2, head_h / 2])}" material="mat_wood"/>')

        elif f["kind"] == "desk":
            top = 0.035
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h - top / 2])}" size="{_fmt([sx / 2, sy / 2, top / 2])}" material="mat_wood"/>')
            lx, ly, lh = sx / 2 - 0.05, sy / 2 - 0.05, (h - top) / 2
            for i, (px, py) in enumerate([(lx, ly), (-lx, ly), (lx, -ly), (-lx, -ly)]):
                a(f'    <geom name="{name}_leg{i}" type="box" pos="{_fmt([x + px, y + py, lh])}" size="{_fmt([0.025, 0.025, lh])}" material="mat_metal"/>')
            # Laptop on desk
            a(f'    <geom name="{name}_laptop_base" type="box" pos="{_fmt([x, y - 0.20, h + 0.008])}" size="0.12 0.16 0.006" material="mat_metal" contype="0" conaffinity="0"/>')
            a(f'    <geom name="{name}_laptop_screen" type="box" pos="{_fmt([x + 0.08, y - 0.20, h + 0.10])}" size="0.005 0.16 0.09" material="mat_metal" contype="0" conaffinity="0"/>')
            a(f'    <geom name="{name}_marker" type="box" pos="{_fmt([x + sx / 2 - 0.06, y + sy / 2 - 0.06, h + 0.001])}" size="0.04 0.04 0.001" material="mat_marker_34" contype="0" conaffinity="0"/>')

        elif f["kind"] == "bookshelf":
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h / 2])}" size="{_fmt([sx / 2, sy / 2, h / 2])}" material="mat_wood"/>')
            # Shelves with books
            for s_idx, sz in enumerate((0.35, 0.70, 1.00)):
                for b_idx in range(6):
                    bx = x - sx / 2 + 0.15 + b_idx * 0.20
                    book_col = ["0.7 0.2 0.2", "0.2 0.4 0.7", "0.2 0.6 0.3", "0.8 0.6 0.2", "0.5 0.3 0.6", "0.3 0.3 0.3"][b_idx % 6]
                    a(f'    <geom name="{name}_book_{s_idx}_{b_idx}" type="box" pos="{_fmt([bx, y, sz])}" size="0.03 0.10 0.10" rgba="{book_col} 1" contype="0" conaffinity="0"/>')

        elif f["kind"] == "box":
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h / 2])}" size="{_fmt([sx / 2, sy / 2, h / 2])}" {col}/>')
            if name == "bedside_table":
                # Bedside lamp with warm shade
                a(f'    <geom name="{name}_lamp_base" type="cylinder" pos="{_fmt([x, y, h + 0.02])}" size="0.06 0.015" material="mat_metal" contype="0" conaffinity="0"/>')
                a(f'    <geom name="{name}_lamp_stem" type="cylinder" pos="{_fmt([x, y, h + 0.12])}" size="0.012 0.10" material="mat_metal" contype="0" conaffinity="0"/>')
                a(f'    <geom name="{name}_lamp_shade" type="cylinder" pos="{_fmt([x, y, h + 0.25])}" size="0.08 0.06" rgba="0.95 0.92 0.82 1" contype="0" conaffinity="0"/>')
                a(f'    <geom name="{name}_marker" type="box" pos="{_fmt([x + sx / 2 - 0.05, y - sy / 2 + 0.05, h + 0.001])}" size="0.035 0.035 0.001" material="mat_marker_33" contype="0" conaffinity="0"/>')
            elif name == "tv_unit":
                # OLED flatscreen TV
                a(f'    <geom name="tv_stand" type="box" pos="{_fmt([x, y, h + 0.02])}" size="0.18 0.08 0.01" material="mat_metal" contype="0" conaffinity="0"/>')
                a(f'    <geom name="tv_screen" type="box" pos="{_fmt([x, y, h + 0.32])}" size="0.55 0.02 0.30" rgba="0.05 0.05 0.06 1" contype="0" conaffinity="0"/>')

        if f.get("movable"):
            a("    </body>")


def _render_seated_user(a):
    """Render the seated person with limited mobility on the living room sofa."""
    sx, sy = FURNITURE["sofa"]["pos"]
    seat = FURNITURE["sofa"]["h"]
    ux = sx - 0.12

    skin = "0.87 0.72 0.60 1"
    hair = "0.32 0.26 0.22 1"
    sweater = "0.22 0.38 0.55 1"   # warm slate-blue knit
    collar = "0.92 0.92 0.94 1"
    trousers = "0.26 0.28 0.34 1"  # dark comfortable slacks
    shoes = "0.18 0.18 0.20 1"
    blanket = "0.68 0.58 0.48 1"

    a(f'    <body name="user" pos="{_fmt([ux, sy, seat])}">')
    # Head & realistic facial features
    a(f'      <geom name="user_head" type="sphere" pos="0 0 0.70" size="0.10" rgba="{skin}"/>')
    a(f'      <geom name="user_hair" type="capsule" fromto="-0.02 0 0.73 0.02 0 0.73" size="0.095" rgba="{hair}"/>')
    a(f'      <geom name="user_nose" type="box" pos="0.095 0 0.69" size="0.015 0.012 0.02" rgba="{skin}"/>')
    a(f'      <geom name="user_neck" type="cylinder" pos="0 0 0.58" size="0.055 0.04" rgba="{skin}"/>')

    # Torso with comfortable knitted sweater
    a(f'      <geom name="user_torso" type="capsule" fromto="0 0 0.18 0 0 0.52" size="0.15" rgba="{sweater}"/>')
    a(f'      <geom name="user_collar" type="cylinder" pos="0 0 0.55" size="0.08 0.015" rgba="{collar}"/>')

    # Shoulders and arms resting on lap / armrests
    a(f'      <geom name="user_shoulder_l" type="sphere" pos="0 -0.19 0.48" size="0.07" rgba="{sweater}"/>')
    a(f'      <geom name="user_shoulder_r" type="sphere" pos="0 0.19 0.48" size="0.07" rgba="{sweater}"/>')
    a(f'      <geom name="user_arm_l" type="capsule" fromto="0 -0.19 0.48 0.18 -0.18 0.22" size="0.05" rgba="{sweater}"/>')
    a(f'      <geom name="user_arm_r" type="capsule" fromto="0 0.19 0.48 0.18 0.18 0.22" size="0.05" rgba="{sweater}"/>')
    a(f'      <geom name="user_forearm_l" type="capsule" fromto="0.18 -0.18 0.22 0.32 -0.12 0.15" size="0.045" rgba="{sweater}"/>')
    a(f'      <geom name="user_forearm_r" type="capsule" fromto="0.18 0.18 0.22 0.32 0.12 0.15" size="0.045" rgba="{sweater}"/>')
    a(f'      <geom name="user_hand_l" type="sphere" pos="0.34 -0.12 0.15" size="0.04" rgba="{skin}"/>')
    a(f'      <geom name="user_hand_r" type="sphere" pos="0.34 0.12 0.15" size="0.04" rgba="{skin}"/>')

    # Seated thighs and shins
    a(f'      <geom name="user_thighs" type="box" pos="0.22 0 0.08" size="0.22 0.17 0.07" rgba="{trousers}"/>')
    a(f'      <geom name="user_shins" type="box" pos="0.42 0 -0.20" size="0.06 0.15 0.22" rgba="{trousers}"/>')
    a(f'      <geom name="user_shoes" type="box" pos="0.45 0 -0.42" size="0.10 0.14 0.03" rgba="{shoes}"/>')

    # Warm lap blanket
    a(f'      <geom name="user_blanket" type="box" pos="0.24 0 0.16" size="0.18 0.20 0.02" rgba="{blanket}"/>')

    # Side table delivery tray station with ArUco Marker #40
    a(f'      <geom name="user_tray_table" type="box" pos="0.38 -0.36 -0.10" size="0.16 0.16 0.25" material="mat_wood"/>')
    a(f'      <geom name="user_delivery_tray" type="box" pos="0.38 -0.36 0.16" size="0.15 0.15 0.015" material="mat_metal"/>')
    a(f'      <geom name="user_tray_marker" type="box" pos="0.38 -0.36 0.18" size="0.06 0.06 0.001" material="mat_marker_40" contype="0" conaffinity="0"/>')

    # Delivery handover target site
    a('      <site name="user_hand" pos="0.38 -0.36 0.25" size="0.03" rgba="1 0.8 0 1" group="1"/>')
    a("    </body>")


def _render_dynamic_person(a):
    """Render the dynamic walking adult human (mocap body, height 1.75m)."""
    skin = "0.86 0.70 0.58 1"
    jacket = "0.20 0.24 0.30 1"
    hair = "0.20 0.18 0.15 1"
    jeans = "0.22 0.28 0.40 1"
    sneakers = "0.90 0.90 0.92 1"

    a(f'    <body name="person_dynamic" mocap="true" pos="0 0 {PARK_Z}">')
    # Head & neck
    a(f'      <geom name="pd_head" type="sphere" pos="0 0 1.62" size="0.10" rgba="{skin}"/>')
    a(f'      <geom name="pd_hair" type="capsule" fromto="-0.02 0 1.65 0.02 0 1.65" size="0.095" rgba="{hair}"/>')
    a(f'      <geom name="pd_neck" type="cylinder" pos="0 0 1.50" size="0.055 0.04" rgba="{skin}"/>')

    # Torso and jacket
    a(f'      <geom name="pd_torso" type="capsule" fromto="0 0 1.05 0 0 1.45" size="0.16" rgba="{jacket}"/>')
    # ArUco tracking badge (Marker #45) on chest
    a('      <geom name="pd_marker" type="box" pos="0.165 0 1.35" size="0.001 0.045 0.045" material="mat_marker_45" contype="0" conaffinity="0"/>')

    # Shoulders and arms
    a(f'      <geom name="pd_arm_l" type="capsule" fromto="0 -0.22 1.40 0 -0.22 1.05" size="0.05" rgba="{jacket}"/>')
    a(f'      <geom name="pd_arm_r" type="capsule" fromto="0 0.22 1.40 0.10 0.22 1.05" size="0.05" rgba="{jacket}"/>')
    a(f'      <geom name="pd_hand_l" type="sphere" pos="0 -0.22 0.98" size="0.04" rgba="{skin}"/>')
    a(f'      <geom name="pd_hand_r" type="sphere" pos="0.15 0.22 0.98" size="0.04" rgba="{skin}"/>')

    # Pelvis and legs
    a(f'      <geom name="pd_pelvis" type="box" pos="0 0 0.98" size="0.14 0.16 0.08" rgba="{jeans}"/>')
    a(f'      <geom name="pd_thigh_l" type="capsule" fromto="0 -0.09 0.92 0 -0.09 0.52" size="0.07" rgba="{jeans}"/>')
    a(f'      <geom name="pd_thigh_r" type="capsule" fromto="0 0.09 0.92 0 0.09 0.52" size="0.07" rgba="{jeans}"/>')
    a(f'      <geom name="pd_shin_l" type="capsule" fromto="0 -0.09 0.50 0 -0.09 0.10" size="0.06" rgba="{jeans}"/>')
    a(f'      <geom name="pd_shin_r" type="capsule" fromto="0 0.09 0.50 0 0.09 0.10" size="0.06" rgba="{jeans}"/>')
    a(f'      <geom name="pd_shoe_l" type="box" pos="0.04 -0.09 0.04" size="0.10 0.055 0.04" rgba="{sneakers}"/>')
    a(f'      <geom name="pd_shoe_r" type="box" pos="0.04 0.09 0.04" size="0.10 0.055 0.04" rgba="{sneakers}"/>')
    a("    </body>")


def _render_push_doors(a):
    """Render push-only doors with wide brushed-steel push plates and ArUco markers."""
    sw = math.radians(DOOR_SWING_DEG)
    leaf_vol = DOOR_LEAF_L * DOOR_LEAF_T * DOOR_H

    for dname, dw in DOORWAYS.items():
        hinge, yaw = door_frame(dname)
        lx = 0.01 + DOOR_LEAF_L / 2
        mid_a = dw["marker_a"]
        mid_b = dw["marker_b"]

        a(f'    <body name="door_{dname}" pos="{_fmt([hinge[0], hinge[1], 0])}" quat="{_fmt(yaw_quat(yaw))}">')
        a(f'      <joint name="door_{dname}_hinge" type="hinge" axis="0 0 1" range="{-sw:.6g} {sw:.6g}" '
          f'damping="{DOOR_DAMPING}" frictionloss="{DOOR_FRICTION}" armature="0.05"/>')
        a(f'      <geom name="door_{dname}" type="box" pos="{_fmt([lx, 0, DOOR_H / 2 + 0.01])}" '
          f'size="{_fmt([DOOR_LEAF_L / 2, DOOR_LEAF_T / 2, DOOR_H / 2])}" material="door" density="{DOOR_MASS / leaf_vol:.6g}"/>')

        # Push-only plates on BOTH faces (no turn handles/latches)
        plate_x = DOOR_LEAF_L - 0.12
        plate_z = 0.60
        plate_w = 0.07
        plate_h = 0.16

        # Side A push plate + ArUco marker
        a(f'      <geom name="door_{dname}_plate_a" type="box" pos="{_fmt([plate_x, -(DOOR_LEAF_T / 2 + 0.003), plate_z])}" '
          f'size="{_fmt([plate_w, 0.003, plate_h])}" material="mat_metal" contype="0" conaffinity="0"/>')
        a(f'      <geom name="door_{dname}_marker_a" type="box" pos="{_fmt([plate_x, -(DOOR_LEAF_T / 2 + 0.0065), plate_z])}" '
          f'size="0.045 0.0005 0.045" material="mat_marker_{mid_a}" contype="0" conaffinity="0"/>')
        a(f'      <site name="door_{dname}_push_a" pos="{_fmt([plate_x, -(DOOR_LEAF_T / 2 + 0.005), plate_z])}" size="0.02" rgba="0 0.8 1 0" group="3"/>')

        # Side B push plate + ArUco marker
        a(f'      <geom name="door_{dname}_plate_b" type="box" pos="{_fmt([plate_x, DOOR_LEAF_T / 2 + 0.003, plate_z])}" '
          f'size="{_fmt([plate_w, 0.003, plate_h])}" material="mat_metal" contype="0" conaffinity="0"/>')
        a(f'      <geom name="door_{dname}_marker_b" type="box" pos="{_fmt([plate_x, DOOR_LEAF_T / 2 + 0.0065, plate_z])}" '
          f'size="0.045 0.0005 0.045" material="mat_marker_{mid_b}" contype="0" conaffinity="0"/>')
        a(f'      <site name="door_{dname}_push_b" pos="{_fmt([plate_x, DOOR_LEAF_T / 2 + 0.005, plate_z])}" size="0.02" rgba="0 0.8 1 0" group="3"/>')

        a("    </body>")


def _render_waterbottle(a):
    """Render the target water bottle with ArUco Marker #0."""
    sq = _bottle_site_quat()
    r, h = BOTTLE_R, BOTTLE_H
    body_h = h * 0.78
    a('    <body name="waterbottle" pos="0 0 1">')
    a('      <freejoint name="waterbottle_joint"/>')
    a(f'      <geom name="waterbottle" type="cylinder" size="{_fmt([r, body_h / 2])}" pos="{_fmt([0, 0, body_h / 2])}" '
      f'mass="{BOTTLE_MASS * 0.9:.4g}" rgba="0.18 0.48 0.95 0.88" friction="1 0.01 0.001" condim="4"/>')
    a(f'      <geom name="waterbottle_neck" type="cylinder" size="{_fmt([r * 0.55, (h - body_h) / 2 * 0.7])}" '
      f'pos="{_fmt([0, 0, body_h + (h - body_h) * 0.35])}" mass="{BOTTLE_MASS * 0.07:.4g}" rgba="0.18 0.48 0.95 0.88"/>')
    a(f'      <geom name="waterbottle_cap" type="cylinder" size="{_fmt([r * 0.45, (h - body_h) * 0.15])}" '
      f'pos="{_fmt([0, 0, h - (h - body_h) * 0.15])}" mass="{BOTTLE_MASS * 0.03:.4g}" rgba="0.10 0.20 0.65 1"/>')
    # ArUco Marker #0 badge on the bottle (facing front toward robot)
    a(f'      <geom name="waterbottle_marker" type="box" pos="{_fmt([0, -(r + 0.001), body_h / 2])}" size="0.025 0.0005 0.025" '
      'material="mat_marker_0" contype="0" conaffinity="0"/>')
    a(f'      <site name="waterbottle_site" pos="{_fmt([0, 0, h / 2])}" quat="{_fmt(sq)}" size="0.01" rgba="1 0 0 1" group="1"/>')
    a("    </body>")


def _drawer_unit_xml(a, uname: str, u: dict):
    """Append XML for one chest of drawers with long bar handles and ArUco markers."""
    (x, y), (w, d), h, n, rgb = u["pos"], u["size"], u["h"], u["n"], u["rgb"]
    marker_start = u.get("marker_start", 20)
    t, plinth, ft = 0.018, 0.06, 0.02
    col = f'rgba="{_fmt(rgb)} 1"'
    front_rgb = tuple(min(1.0, c * 1.08 + 0.03) for c in rgb)
    fcol = f'rgba="{_fmt(front_rgb)} 1"'
    cx = (-d / 2 + d / 2 - ft) / 2
    cd = d - ft

    a(f'    <body name="chest_{uname}" pos="{_fmt([x, y, 0])}" quat="{_fmt(yaw_quat(u["facing"]))}">')
    a(f'      <geom name="chest_{uname}_top" type="box" pos="{_fmt([0, 0, h - t / 2])}" size="{_fmt([d / 2, w / 2, t / 2])}" {col}/>')
    for i, sy in enumerate((-1, 1)):
        a(f'      <geom name="chest_{uname}_side{i}" type="box" pos="{_fmt([cx, sy * (w / 2 - t / 2), (h - t) / 2])}" '
          f'size="{_fmt([cd / 2, t / 2, (h - t) / 2])}" {col}/>')
    a(f'      <geom name="chest_{uname}_back" type="box" pos="{_fmt([-d / 2 + t / 2, 0, (h - t) / 2])}" '
      f'size="{_fmt([t / 2, w / 2 - t, (h - t) / 2])}" {col}/>')
    a(f'      <geom name="chest_{uname}_plinth" type="box" pos="{_fmt([cx - 0.015, 0, (plinth + t) / 2])}" '
      f'size="{_fmt([cd / 2 - 0.015, w / 2 - t, (plinth + t) / 2])}" {col}/>')

    z0, z1 = plinth + t, h - t
    s = (z1 - z0) / n
    di = d - t - ft - 0.02
    wi = w - 2 * t - 0.012
    ht = min(0.7 * s, s - 0.03)
    xb = d / 2 - ft - di / 2
    hl = HANDLE_FRACTION * w
    hx = d / 2 + HANDLE_STANDOFF
    ext = DRAWER_EXTENSION * di

    for i in range(n):
        name = f"drawer_{uname}_{i}"
        mid = marker_start + i
        zc = z0 + s * (i + 0.5)
        zb = -s / 2 + 0.015
        a(f'      <body name="{name}" pos="{_fmt([0, 0, zc])}">')
        a(f'        <joint name="{name}_slide" type="slide" axis="1 0 0" range="0 {ext:.6g}" '
          f'damping="{DRAWER_DAMPING}" frictionloss="{DRAWER_FRICTION}" armature="0.05"/>')
        a(f'        <geom name="{name}_front" type="box" pos="{_fmt([d / 2 - ft / 2, 0, 0])}" '
          f'size="{_fmt([ft / 2, w / 2 - 0.003, s / 2 - 0.003])}" {fcol} density="500"/>')
        a(f'        <geom name="{name}_floor" type="box" pos="{_fmt([xb, 0, zb])}" size="{_fmt([di / 2, wi / 2, 0.005])}" {col} density="500"/>')
        for j, sy in enumerate((-1, 1)):
            a(f'        <geom name="{name}_wall{j}" type="box" pos="{_fmt([xb, sy * (wi / 2 - 0.005), zb + ht / 2])}" '
              f'size="{_fmt([di / 2, 0.005, ht / 2])}" {col} density="500"/>')
        a(f'        <geom name="{name}_rear" type="box" pos="{_fmt([d / 2 - ft - di + 0.005, 0, zb + ht / 2])}" '
          f'size="{_fmt([0.005, wi / 2, ht / 2])}" {col} density="500"/>')

        # Long bar handle with brushed stainless steel finish
        a(f'        <geom name="{name}_handle" type="capsule" fromto="{_fmt([hx, -hl / 2, 0, hx, hl / 2, 0])}" size="{HANDLE_R}" material="mat_metal"/>')
        for j, sy in enumerate((-1, 1)):
            a(f'        <geom name="{name}_post{j}" type="box" pos="{_fmt([d / 2 + HANDLE_STANDOFF / 2, sy * (hl / 2 - 0.03), 0])}" '
              f'size="{_fmt([HANDLE_STANDOFF / 2, 0.008, 0.008])}" material="mat_metal"/>')
        a(f'        <site name="{name}_handle" pos="{_fmt([hx, 0, 0])}" size="0.015" rgba="1 0.5 0 1" group="1"/>')

        # ArUco marker on drawer front (comfortably above the bar handle)
        mz = min(0.045, s * 0.25)
        a(f'        <geom name="{name}_marker" type="box" pos="{_fmt([d / 2 + 0.002, 0, mz])}" size="{_fmt([0.001, 0.042, 0.042])}" '
          f'material="mat_marker_{mid}" contype="0" conaffinity="0"/>')
        a("      </body>")
    a("    </body>")


# =============================================================================
# Runtime Environment
# =============================================================================


@dataclass
class CheckResult:
    solvable: bool
    start_free: bool
    path_len: float
    grid: np.ndarray


class HouseEnv:
    """Compiled house model + reset logic supporting 3 testing modes.

    Modes:
      - 'static': Baseline static environment
      - 'shuffled': Shuffled furniture and target item locations (including drawers)
      - 'dynamic': Dynamic walking person enters and relocates target item
    """

    def __init__(self, robot_xml: str | None = None, robot_joint: str | None = None, seed: int | None = None):
        self.xml = build_xml()
        spec = mujoco.MjSpec.from_string(self.xml)
        self.robot_root = None
        if robot_xml:
            self._attach_robot(spec, robot_xml)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attaching a spec with an empty prefix")
            self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if robot_xml:
            self.robot_root = self.model.body(self._robot_root_name).id
        self.rng = np.random.default_rng(seed)

        m = self.model
        self.bottle_body = m.body("waterbottle").id
        self.bottle_qadr = m.jnt_qposadr[m.joint("waterbottle_joint").id]
        self.bottle_dadr = m.jnt_dofadr[m.joint("waterbottle_joint").id]
        self.door_jnt = {d: m.joint(f"door_{d}_hinge").id for d in DOORWAYS}
        self.door_eq = {d: m.equality(f"lock_{d}").id for d in DOORWAYS}
        self.drawer_jnt = [j for j in range(m.njnt) if m.joint(j).name.startswith("drawer_")]
        self.obs_mocap = {}
        for typ, count in _obstacle_pool().items():
            self.obs_mocap[typ] = [m.body(f"obs_{typ}_{i}").mocapid[0] for i in range(count)]

        # Dynamic person mocap
        self.dynamic_person_id = m.body("person_dynamic").mocapid[0] if m.body("person_dynamic").mocapid[0] >= 0 else None
        self.dynamic_controller: DynamicAgentController | None = None
        self.mode = Mode.STATIC

        # Geom categorization
        mocap_bodies = {i for i in range(m.nbody) if m.body_mocapid[i] >= 0}
        robot_bodies = self._subtree(self.robot_root) if self.robot_root is not None else set()
        self.mocap_geoms, self.static_geoms, self.drawer_geoms = [], [], []
        self.person_geoms = []
        self.door_geoms = {dn: [] for dn in DOORWAYS}
        door_body = {m.body(f"door_{dn}").id: dn for dn in DOORWAYS}
        skip = {"floor", "waterbottle", "waterbottle_neck", "waterbottle_cap", "waterbottle_marker"}

        for g in range(m.ngeom):
            name = m.geom(g).name
            b = m.geom_bodyid[g]
            if name in skip or name.startswith("floor_") or b in robot_bodies or b == self.bottle_body:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            if b == m.body("person_dynamic").id:
                self.person_geoms.append(g)
            elif b in mocap_bodies:
                self.mocap_geoms.append(g)
            elif b in door_body:
                self.door_geoms[door_body[b]].append(g)
            elif m.body(b).name.startswith("drawer_"):
                self.drawer_geoms.append(g)
            else:
                self.static_geoms.append(g)

        self.all_door_geoms = [g for gs in self.door_geoms.values() for g in gs]

        # Movable furniture (shuffled in Mode 2) and the robot head camera mount
        self.furn_body = {n: m.body(f"furn_{n}").id for n, f in FURNITURE.items() if f.get("movable")}
        self.furn_geoms = {n: [g for g in self.static_geoms if m.geom_bodyid[g] == b] for n, b in self.furn_body.items()}
        self.furn_pose = {n: (*FURNITURE[n]["pos"], 0.0) for n in self.furn_body}
        self.head_mount = m.body("robot_head_mount").mocapid[0]
        self.locked_doors: set = set()
        self.door_range = self._fit_door_ranges()

        # Robot root joint
        self.robot_qadr = None
        if self.robot_root is not None:
            jid = m.joint(robot_joint).id if robot_joint else self._find_free_joint(self.robot_root)
            self.robot_qadr = m.jnt_qposadr[jid] if jid is not None else None

        nx = int(round(8.0 / GRID_RES)) + 1
        ny = int(round(6.0 / GRID_RES)) + 1
        self.gx = np.linspace(-4.0, 4.0, nx)
        self.gy = np.linspace(-3.0, 3.0, ny)
        self.GX, self.GY = np.meshgrid(self.gx, self.gy)

        self.layout = None
        self.target = None
        self.bottle_pos = None
        self.target_container = None
        self.info = {}
        self.sim_time = 0.0

    # ---- Doors / Drawers ------------------------------------------------------
    def _overlaps(self, geoms_a, geoms_b, tol=1e-4):
        m, d = self.model, self.data
        for g in geoms_a:
            for h in geoms_b:
                if m.geom_bodyid[h] == m.geom_bodyid[g]:
                    continue
                if np.linalg.norm(d.geom_xpos[g, :2] - d.geom_xpos[h, :2]) > m.geom_rbound[g] + m.geom_rbound[h]:
                    continue
                if mujoco.mj_geomDistance(m, d, g, h, 0.05, None) < -tol:
                    return (m.geom(g).name, m.geom(h).name)
        return None

    def _fit_door_ranges(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.set_drawers(1.0)
        ranges = {}
        others = self.static_geoms + self.drawer_geoms
        for dn, j in self.door_jnt.items():
            qa = m.jnt_qposadr[j]
            lim = []
            for sign in (-1, 1):
                ok = 0
                for deg in range(1, int(DOOR_SWING_DEG) + 1):
                    d.qpos[qa] = sign * math.radians(deg)
                    mujoco.mj_kinematics(m, d)
                    if self._overlaps(self.door_geoms[dn], others + [g for g in self.all_door_geoms if g not in self.door_geoms[dn]]):
                        break
                    ok = deg
                lim.append(math.radians(max(ok - 2, 0)) if ok < DOOR_SWING_DEG else math.radians(DOOR_SWING_DEG))
            d.qpos[qa] = 0.0
            ranges[dn] = (-lim[0], lim[1])
            m.jnt_range[j] = ranges[dn]
        mujoco.mj_resetData(m, d)
        return ranges

    def set_drawers(self, frac: float):
        m = self.model
        for j in self.drawer_jnt:
            self.data.qpos[m.jnt_qposadr[j]] = frac * m.jnt_range[j, 1]
            self.data.qvel[m.jnt_dofadr[j]] = 0.0

    def door_angle(self, name: str) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.door_jnt[name]]])

    def set_door(self, name: str, angle: float):
        j = self.door_jnt[name]
        self.data.qpos[self.model.jnt_qposadr[j]] = angle
        self.data.qvel[self.model.jnt_dofadr[j]] = 0.0

    def randomise_doors(self):
        lay = LAYOUTS[self.layout]
        for dn in DOORWAYS:
            if dn in self.locked_doors or dn in lay.get("doors_closed", []):
                self.set_door(dn, 0.0)
            else:
                lo, hi = self.door_range[dn]
                self.set_door(dn, float(self.rng.uniform(lo, hi)))

    def sweeps_clear(self, steps: int = 13):
        m, d = self.model, self.data
        active = self.active_mocap_geoms()
        if any(p != (*FURNITURE[n]["pos"], 0.0) for n, p in self.furn_pose.items()):
            active = active + [g for gs in self.furn_geoms.values() for g in gs]
        return self._sweep_clear_of(active, steps)

    def _sweep_clear_of(self, active, steps: int = 13) -> bool:
        """True if no unlocked door, swung through its range, and no fully pulled-out
        drawer touches any of `active`."""
        m, d = self.model, self.data
        if not active:
            return True
        saved = d.qpos.copy()
        try:
            for dn in DOORWAYS:
                if dn in self.locked_doors:
                    continue
                lo, hi = self.door_range[dn]
                for ang in np.linspace(lo, hi, steps):
                    self.set_door(dn, ang)
                    mujoco.mj_kinematics(m, d)
                    if self._overlaps(self.door_geoms[dn], active):
                        return False
                d.qpos[:] = saved
            self.set_drawers(1.0)
            mujoco.mj_kinematics(m, d)
            if self._overlaps(self.drawer_geoms, active):
                return False
            return True
        finally:
            d.qpos[:] = saved
            mujoco.mj_kinematics(m, d)

    # ---- Robot Merging --------------------------------------------------------
    def _attach_robot(self, spec: mujoco.MjSpec, robot_xml: str):
        robot = mujoco.MjSpec.from_file(str(Path(robot_xml).resolve()))
        for g in list(robot.worldbody.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                robot.delete(g)
        for lt in list(robot.worldbody.lights):
            robot.delete(lt)
        roots = list(robot.worldbody.bodies)
        if not roots:
            raise ValueError(f"{robot_xml} has no bodies under <worldbody>")
        x, y, yaw = ROBOT_START
        frame = spec.worldbody.add_frame(pos=[x, y, 0.0], quat=yaw_quat(yaw))
        spec.attach(robot, frame=frame, prefix="")
        self._robot_root_name = roots[0].name

    def _subtree(self, root: int) -> set:
        m = self.model
        out = {root}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = m.body_parentid[p]
        return out

    def _find_free_joint(self, root):
        m = self.model
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] in self._subtree(root):
                return j
        return None

    # ---- Reset ----------------------------------------------------------------
    def reset(
        self,
        layout: str | None = None,
        target: str | None = None,
        seed: int | None = None,
        jitter: bool = True,
        mode: Mode | str = Mode.STATIC,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.mode = Mode(mode)
        layout = layout or ("moving_house" if self.mode == Mode.SHUFFLED else str(self.rng.choice(list(LAYOUTS))))

        target_pool = ALL_TARGETS if self.mode == Mode.SHUFFLED else TARGETS
        target = target or str(self.rng.choice(list(target_pool)))

        if layout not in LAYOUTS:
            raise KeyError(f"unknown layout {layout!r}; choose from {list(LAYOUTS)}")
        if target not in ALL_TARGETS:
            raise KeyError(f"unknown target {target!r}; choose from {list(ALL_TARGETS)}")

        self.layout, self.target = layout, target
        self.sim_time = 0.0

        mujoco.mj_resetData(self.model, self.data)
        if self.robot_root is not None and self.model.nkey > 0:
            key = self.model.key("home").id if _has_key(self.model, "home") else 0
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)

        self._place_robot()
        self._place_bottle(target)

        lay = LAYOUTS[layout]
        self.locked_doors = set(lay.get("doors_locked", []))
        for dn, eq in self.door_eq.items():
            self.data.eq_active[eq] = dn in self.locked_doors
        self.randomise_doors()

        # The person stays parked while the layout is placed; their route is planned after
        self.dynamic_controller = None
        self.person_drop = None
        self._bottle_put_down = False
        if self.dynamic_person_id is not None:
            self.data.mocap_pos[self.dynamic_person_id] = [0.0, 0.0, PARK_Z]
            self.data.mocap_quat[self.dynamic_person_id] = [1, 0, 0, 0]
        want_person = self.mode == Mode.DYNAMIC and self.dynamic_person_id is not None

        res, why = None, ""
        for attempt in range(MAX_RESET_TRIES):
            use_jitter = jitter and attempt < MAX_RESET_TRIES - 1
            self._place_furniture(shuffle=self.mode == Mode.SHUFFLED and use_jitter)
            self._place_bottle(target)
            self._place_layout(layout, use_jitter)
            mujoco.mj_forward(self.model, self.data)
            pen = self.penetrations()
            if pen:
                why = f"overlap {pen[0][:2]}"
                continue
            if not self.sweeps_clear():
                why = "an obstacle is in a door swing or in front of a drawer"
                continue
            res = self.check()
            if not res.solvable:
                why = "bottle unreachable"
                continue
            if self.mode == Mode.SHUFFLED and not self.doorways_reachable(res.grid):
                why = "furniture cuts off a doorway"
                continue
            if want_person:
                route = self.plan_person_route()
                if route is None:
                    why = "no clear route for the dynamic person"
                    self.randomise_doors()
                    continue
            break
        else:
            raise RuntimeError(f"layout {layout!r} + target {target!r} is not valid even without jitter: {why}")

        if want_person:
            self.dynamic_controller = DynamicAgentController(
                waypoints=route,
                item_initial_pos=tuple(self.bottle_pos),
                item_target_pos=tuple(self.person_drop_pos),
                pick_time=self.person_pick_time,
                place_time=self.person_place_time,
            )
            st0 = self.dynamic_controller.state_at(0.0)
            self.data.mocap_pos[self.dynamic_person_id] = st0["pos"]
            self.data.mocap_quat[self.dynamic_person_id] = st0["quat"]
            mujoco.mj_forward(self.model, self.data)

        self._follow_robot_with_camera()
        mujoco.mj_forward(self.model, self.data)

        self.info = dict(
            layout=layout,
            target=target,
            mode=self.mode.value,
            attempts=attempt + 1,
            path_len=res.path_len,
            container=self.target_container,
            doors={dn: round(math.degrees(self.door_angle(dn))) for dn in DOORWAYS},
            locked=sorted(self.locked_doors),
            furniture={n: tuple(round(v, 3) for v in p) for n, p in self.furn_pose.items()},
        )
        if self.dynamic_controller is not None:
            self.info["person_drop"] = self.person_drop
            self.info["person_route_s"] = round(self.dynamic_controller.total_duration, 2)
        return self.info

    def _place_robot(self):
        if self.robot_qadr is None:
            return
        x, y, yaw = ROBOT_START
        q = self.data.qpos
        a = self.robot_qadr
        q[a : a + 2] = [x, y]
        q[a + 3 : a + 7] = yaw_quat(yaw)

    def _place_bottle(self, target: str):
        target_info = ALL_TARGETS[target]
        if len(target_info) == 3:
            # Surface target: (x, y, furniture_name)
            x, y, z = self.surface_xyz(target)
            self.target_container = None
        else:
            # Drawer target: (x, y, drawer_unit_name, drawer_index)
            cx, cy, uname, d_idx = target_info
            u = DRAWER_UNITS[uname]
            plinth, t = 0.06, 0.018
            z0 = plinth + t
            s = (u["h"] - t - z0) / u["n"]
            # Center of the drawer tray in world z
            z = z0 + s * (d_idx + 0.5) - s / 2 + 0.02
            facing_rad = math.radians(u["facing"])
            # In drawer frame, tray center is slightly recessed from front
            tray_offset = u["size"][1] * 0.15
            x = cx + tray_offset * math.cos(facing_rad)
            y = cy + tray_offset * math.sin(facing_rad)
            self.target_container = f"drawer_{uname}_{d_idx}"

        a = self.bottle_qadr
        self.data.qpos[a : a + 7] = [x, y, z, 1, 0, 0, 0]
        self.data.qvel[self.bottle_dadr : self.bottle_dadr + 6] = 0
        self.bottle_pos = np.array([x, y, z])

    def surface_xyz(self, target: str) -> np.ndarray:
        """World position of a surface target, following its furniture if it has been moved."""
        x, y, furn = SURFACE_TARGETS[target]
        z = FURNITURE[furn]["h"] + 0.001
        if furn in self.furn_pose:
            fx0, fy0 = FURNITURE[furn]["pos"]
            fx, fy, fyaw = self.furn_pose[furn]
            c, s_ = math.cos(math.radians(fyaw)), math.sin(math.radians(fyaw))
            lx, ly = x - fx0, y - fy0
            x, y = fx + c * lx - s_ * ly, fy + s_ * lx + c * ly
        return np.array([x, y, z])

    def _place_furniture(self, shuffle: bool):
        """Put movable furniture at its nominal pose, or (shuffle) at a random offset."""
        m, d = self.model, self.data
        for n, b in self.furn_body.items():
            self._set_furniture(n, *FURNITURE[n]["pos"], 0.0)
        if not shuffle:
            return
        # One piece at a time: re-draw a piece until it is clear of walls, the other
        # furniture, drawers and door swings (keeping it at nominal if no draw works)
        furn = {g for gs in self.furn_geoms.values() for g in gs}
        for n in self.furn_body:
            fixed = [g for g in self.static_geoms if g not in self.furn_geoms[n]] + self.drawer_geoms + self.all_door_geoms
            x0, y0 = FURNITURE[n]["pos"]
            for _ in range(SHUFFLE_TRIES):
                self._set_furniture(n, x0 + self.rng.uniform(-SHUFFLE_XY, SHUFFLE_XY),
                                    y0 + self.rng.uniform(-SHUFFLE_XY, SHUFFLE_XY),
                                    self.rng.uniform(-SHUFFLE_YAW, SHUFFLE_YAW))
                mujoco.mj_kinematics(m, d)
                if not self._overlaps(self.furn_geoms[n], fixed) and self._sweep_clear_of(self.furn_geoms[n]):
                    break
            else:
                self._set_furniture(n, x0, y0, 0.0)
        mujoco.mj_kinematics(m, d)

    def _set_furniture(self, n, x, y, yaw):
        self.furn_pose[n] = (x, y, yaw)
        self.model.body_pos[self.furn_body[n]] = [x, y, 0.0]
        self.model.body_quat[self.furn_body[n]] = yaw_quat(yaw)

    def doorways_reachable(self, grid) -> bool:
        """Every unlocked doorway can be reached from the robot start (furniture hasn't
        been shuffled into a doorway and cut a room off)."""
        si, sj = self._cell(ROBOT_START[0], ROBOT_START[1])
        seen = np.zeros(grid.shape, dtype=bool)
        seen[si, sj] = True
        q = deque([(si, sj)])
        ny, nx = grid.shape
        while q:
            i, j = q.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < ny and 0 <= b < nx and not grid[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    q.append((a, b))
        return all(seen[self._cell(*dw["center"])] for dn, dw in DOORWAYS.items() if dn not in self.locked_doors)

    def _place_layout(self, layout: str, jitter: bool):
        lay = LAYOUTS[layout]
        d = self.data
        used = {t: 0 for t in self.obs_mocap}
        for typ, x, y, yaw in lay["obstacles"]:
            mid = self.obs_mocap[typ][used[typ]]
            used[typ] += 1
            if jitter:
                x += self.rng.uniform(-JITTER_XY, JITTER_XY)
                y += self.rng.uniform(-JITTER_XY, JITTER_XY)
                yaw += self.rng.uniform(-JITTER_YAW, JITTER_YAW)
            d.mocap_pos[mid] = [x, y, 0.0]
            d.mocap_quat[mid] = yaw_quat(yaw)
        for typ, ids in self.obs_mocap.items():
            for mid in ids[used[typ] :]:
                d.mocap_pos[mid] = [0.0, 0.0, PARK_Z]
                d.mocap_quat[mid] = [1, 0, 0, 0]

    # ---- Validity Checks ------------------------------------------------------
    def active_mocap_geoms(self):
        return [g for g in self.mocap_geoms if self.data.geom_xpos[g, 2] > PARK_Z / 2]

    def penetrations(self, tol: float = 1e-4):
        m, d = self.model, self.data
        active = self.active_mocap_geoms()
        others = self.static_geoms + self.drawer_geoms + self.all_door_geoms + [self.bottle_geom_id()]
        bad = []
        for i, g in enumerate(active):
            cands = others + active[i + 1 :]
            for h in cands:
                if m.geom_bodyid[h] == m.geom_bodyid[g]:
                    continue
                if np.linalg.norm(d.geom_xpos[g, :2] - d.geom_xpos[h, :2]) > m.geom_rbound[g] + m.geom_rbound[h]:
                    continue
                dist = mujoco.mj_geomDistance(m, d, g, h, 0.05, None)
                if dist < -tol:
                    bad.append((m.geom(g).name, m.geom(h).name, dist))
        if any(p != (*FURNITURE[n]["pos"], 0.0) for n, p in self.furn_pose.items()):
            furn = [g for gs in self.furn_geoms.values() for g in gs]
            fixed = [g for g in self.static_geoms if g not in furn] + self.drawer_geoms + self.all_door_geoms
            for i, g in enumerate(furn):
                for h in fixed + furn[i + 1 :]:
                    if m.geom_bodyid[h] == m.geom_bodyid[g]:
                        continue
                    if np.linalg.norm(d.geom_xpos[g, :2] - d.geom_xpos[h, :2]) > m.geom_rbound[g] + m.geom_rbound[h]:
                        continue
                    dist = mujoco.mj_geomDistance(m, d, g, h, 0.05, None)
                    if dist < -tol:
                        bad.append((m.geom(g).name, m.geom(h).name, dist))
        return bad

    def bottle_geom_id(self):
        return self.model.geom("waterbottle").id

    def occupancy(self) -> np.ndarray:
        """Cells the robot centre can't occupy: static, drawers, locked doors, clutter."""
        locked = [g for dn in self.locked_doors for g in self.door_geoms[dn]]
        geoms = self.static_geoms + self.drawer_geoms + locked + self.active_mocap_geoms()
        return self._clearance(geoms, ROBOT_H) <= ROBOT_R

    def _clearance(self, geoms, max_z: float) -> np.ndarray:
        """Per grid cell, 2D distance to the nearest of `geoms` that starts below max_z."""
        m, d = self.model, self.data
        clear = np.full(self.GX.shape, np.inf)
        for g in geoms:
            pos, mat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            size = m.geom_size[g]
            typ = m.geom_type[g]

            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                half_z = np.abs(mat[2]) @ size
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                half_z = size[1] + (size[0] if typ == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            else:
                half_z = m.geom_rbound[g]

            if pos[2] - half_z > max_z:
                continue

            dx, dy = self.GX - pos[0], self.GY - pos[1]
            if typ == mujoco.mjtGeom.mjGEOM_BOX and abs(mat[2, 2]) > 0.99:
                c, s = mat[0, 0], mat[1, 0]
                lx, ly = c * dx + s * dy, -s * dx + c * dy
                ox = np.maximum(np.abs(lx) - size[0], 0)
                oy = np.maximum(np.abs(ly) - size[1], 0)
                dist = np.hypot(ox, oy)
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE) and abs(mat[2, 2]) > 0.99:
                dist = np.hypot(dx, dy) - size[0]
            elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE and abs(mat[2, 2]) < 0.01:
                ax = mat[:2, 2]
                t = np.clip(dx * ax[0] + dy * ax[1], -size[1], size[1])
                dist = np.hypot(dx - t * ax[0], dy - t * ax[1]) - size[0]
            else:
                dist = np.hypot(dx, dy) - m.geom_rbound[g]

            np.minimum(clear, dist, out=clear)
        return clear

    # ---- Dynamic person route -------------------------------------------------
    def person_occupancy(self) -> np.ndarray:
        """Cells the person's centre can't occupy: everything solid, every door leaf at its
        current angle (they walk round open doors rather than through them) and the robot."""
        geoms = self.static_geoms + self.drawer_geoms + self.all_door_geoms + self.active_mocap_geoms()
        blocked = self._clearance(geoms, PERSON_H) <= PERSON_R
        if self.robot_qadr is not None:
            rx, ry = self.data.qpos[self.robot_qadr : self.robot_qadr + 2]
        else:
            rx, ry = ROBOT_START[:2]
        blocked |= np.hypot(self.GX - rx, self.GY - ry) <= ROBOT_R + PERSON_R
        return blocked

    def _bfs(self, grid, start_cell, goal_mask):
        """8-connected BFS on free cells. Returns the cell path to the nearest goal cell, or None."""
        ny, nx = grid.shape
        if grid[start_cell]:
            return None
        parent = {start_cell: None}
        q = deque([start_cell])
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while q:
            c = q.popleft()
            if goal_mask[c]:
                path = []
                while c is not None:
                    path.append(c)
                    c = parent[c]
                return path[::-1]
            i, j = c
            for di, dj in nbrs:
                a, b = i + di, j + dj
                if not (0 <= a < ny and 0 <= b < nx) or grid[a, b] or (a, b) in parent:
                    continue
                if di and dj and (grid[i, b] or grid[a, j]):  # no corner cutting
                    continue
                parent[(a, b)] = c
                q.append((a, b))
        return None

    def _line_free(self, grid, p, q) -> bool:
        n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / (GRID_RES / 3)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
            i = int(np.clip(round((y - self.gy[0]) / GRID_RES), 0, grid.shape[0] - 1))
            j = int(np.clip(round((x - self.gx[0]) / GRID_RES), 0, grid.shape[1] - 1))
            if grid[i, j]:
                return False
        return True

    def _plan_xy(self, grid, start_xy, goal_mask):
        """Shortest grid path from start_xy into goal_mask, shortened to straight runs that
        stay on free cells. Returns a list of (x, y) points, or None."""
        cells = self._bfs(grid, self._cell(*start_xy), goal_mask)
        if cells is None:
            return None
        pts = [(float(self.gx[j]), float(self.gy[i])) for i, j in cells]
        pts[0] = tuple(start_xy)
        out = [pts[0]]
        k = 0
        while k < len(pts) - 1:
            nxt = len(pts) - 1
            while nxt > k + 1 and not self._line_free(grid, pts[k], pts[nxt]):
                nxt -= 1
            out.append(pts[nxt])
            k = nxt
        return out

    def _near(self, grid, xy, r):
        return (~grid) & (np.hypot(self.GX - xy[0], self.GY - xy[1]) <= r)

    def _surface_pos(self, target: str) -> np.ndarray:
        return self.surface_xyz(target)

    def plan_person_route(self) -> list[TrajectoryWaypoint] | None:
        """Route for the dynamic person: walk in, pick up the bottle, carry it to another
        surface, put it down, walk back out. If half-open doors are in the way, the person
        has pushed the unlocked ones fully open first. None if there is still no route."""
        item = self.bottle_pos[:2]

        def room(xy):
            return (xy[0] > 0, xy[1] > 0)

        away = [PERSON_ENTRIES[k] for k in self.rng.permutation(len(PERSON_ENTRIES)) if room(PERSON_ENTRIES[k]) != room(item)]
        home = [e for e in PERSON_ENTRIES if room(e) == room(item)]
        for entries in (away, home):
            for open_doors in (False, True):
                route = self._plan_with_doors(entries, open_doors)
                if route is not None:
                    return route
        return None

    def _plan_with_doors(self, entries, open_doors: bool):
        if not open_doors:
            return self._plan_person_route(entries)
        saved = {dn: self.door_angle(dn) for dn in DOORWAYS}
        for dn in DOORWAYS:
            if dn in self.locked_doors:
                continue
            lo, hi = self.door_range[dn]
            same_side, other_side = (hi, lo) if saved[dn] >= 0 else (lo, hi)
            self.set_door(dn, same_side if abs(same_side) > 0.5 else other_side)
        mujoco.mj_kinematics(self.model, self.data)
        route = self._plan_person_route(entries) if self.sweeps_clear() else None
        if route is None:
            for dn, q in saved.items():
                self.set_door(dn, q)
            mujoco.mj_kinematics(self.model, self.data)
        return route

    def _plan_person_route(self, entries) -> list[TrajectoryWaypoint] | None:
        grid = self.person_occupancy()
        item = self.bottle_pos[:2]
        drops = [t for t in PERSON_DROPS if t != self.target]
        for entry in entries:
            # nudge the entry point onto the nearest free cell
            if grid[self._cell(*entry)]:
                free = np.argwhere(~grid & (np.hypot(self.GX - entry[0], self.GY - entry[1]) < 0.5))
                if not len(free):
                    continue
                i, j = min(free, key=lambda c: np.hypot(self.gx[c[1]] - entry[0], self.gy[c[0]] - entry[1]))
                entry = (float(self.gx[j]), float(self.gy[i]))
            leg_in = self._plan_xy(grid, entry, self._near(grid, item, PERSON_REACH))
            if leg_in is None:
                continue
            for drop in drops:
                drop_pos = self._surface_pos(drop)
                if not self._robot_can_reach(drop_pos):
                    continue
                leg_carry = self._plan_xy(grid, leg_in[-1], self._near(grid, drop_pos[:2], PERSON_REACH))
                if leg_carry is None:
                    continue
                leg_out = self._plan_xy(grid, leg_carry[-1], self._near(grid, entry, GRID_RES))
                if leg_out is None:
                    continue
                self.person_drop = drop
                self.person_drop_pos = drop_pos
                return self._timeline(leg_in, item, leg_carry, drop_pos[:2], leg_out)
        return None

    def _robot_can_reach(self, pos) -> bool:
        saved = self.bottle_pos
        self.bottle_pos = np.asarray(pos)
        try:
            return self.check().solvable
        finally:
            self.bottle_pos = saved

    def _timeline(self, leg_in, item_xy, leg_carry, drop_xy, leg_out):
        wps: list[TrajectoryWaypoint] = []
        t = 0.0

        def heading(p, q):
            return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))

        def add(pos, yaw, action, dt=0.0):
            nonlocal t
            t += dt
            wps.append(TrajectoryWaypoint(round(t, 4), (pos[0], pos[1], 0.0), yaw, action))

        def walk(leg, action):
            for p, q in zip(leg[:-1], leg[1:]):
                yaw = heading(p, q)
                if abs((yaw - wps[-1].yaw_deg + 180) % 360 - 180) > 1:
                    add(p, yaw, action, PERSON_TURN_S)  # turn on the spot
                add(q, yaw, action, math.dist(p, q) / PERSON_SPEED)

        def face(xy, action):
            here = wps[-1].pos
            add(here, heading(here, xy), action, PERSON_TURN_S)

        first = leg_in[1] if len(leg_in) > 1 else item_xy
        add(leg_in[0], heading(leg_in[0], first), "walk")
        walk(leg_in, "walk")
        face(item_xy, "reach_pick")
        add(wps[-1].pos, wps[-1].yaw_deg, "reach_pick", PERSON_HANDLE_S)
        self.person_pick_time = t - PERSON_HANDLE_S / 2
        walk(leg_carry, "carry")
        face(drop_xy, "reach_place")
        add(wps[-1].pos, wps[-1].yaw_deg, "reach_place", PERSON_HANDLE_S)
        self.person_place_time = t - PERSON_HANDLE_S / 2
        walk(leg_out, "walk_away")
        add(wps[-1].pos, wps[-1].yaw_deg, "idle", 1.0)
        return wps

    def _cell(self, x, y):
        i = int(round((y - self.gy[0]) / GRID_RES))
        j = int(round((x - self.gx[0]) / GRID_RES))
        return i, j

    def check(self) -> CheckResult:
        grid = self.occupancy()
        ny, nx = grid.shape
        si, sj = self._cell(ROBOT_START[0], ROBOT_START[1])
        if grid[si, sj]:
            return CheckResult(False, False, math.inf, grid)

        bx, by = self.bottle_pos[:2]
        goal = (~grid) & (np.hypot(self.GX - bx, self.GY - by) <= REACH)
        dist = np.full(grid.shape, -1, dtype=np.int32)
        dist[si, sj] = 0
        q = deque([(si, sj)])
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while q:
            i, j = q.popleft()
            if goal[i, j]:
                return CheckResult(True, True, dist[i, j] * GRID_RES, grid)
            for di, dj in nbrs:
                a, b = i + di, j + dj
                if 0 <= a < ny and 0 <= b < nx and not grid[a, b] and dist[a, b] < 0:
                    dist[a, b] = dist[i, j] + 1
                    q.append((a, b))
        return CheckResult(False, True, math.inf, grid)

    # ---- Stepping & Dynamic Trajectory ----------------------------------------
    def step(self, n: int = 1):
        dt = float(self.model.opt.timestep)
        for _ in range(n):
            if self.mode == Mode.DYNAMIC and self.dynamic_controller is not None:
                self.sim_time += dt
                dyn_state = self.dynamic_controller.state_at(self.sim_time)
                # Update dynamic person mocap
                self.data.mocap_pos[self.dynamic_person_id] = dyn_state["pos"]
                self.data.mocap_quat[self.dynamic_person_id] = dyn_state["quat"]
                # Bottle follows the hand while carried, is set down once, then is free
                # again (physics owns it, so the robot can pick it up)
                put_down = dyn_state["relocated"] and not self._bottle_put_down
                if put_down:
                    self._bottle_put_down = True
                if dyn_state["carrying"] or put_down:
                    ipos = dyn_state["item_pos"]
                    a = self.bottle_qadr
                    self.data.qpos[a : a + 3] = ipos
                    self.data.qvel[self.bottle_dadr : self.bottle_dadr + 6] = 0
                    self.bottle_pos = np.array(ipos)

            mujoco.mj_step(self.model, self.data)
            self._follow_robot_with_camera()

    def _follow_robot_with_camera(self):
        """Move the cam_robot_head mount to the robot root's x, y and heading."""
        if self.robot_root is None:
            x, y, yaw = ROBOT_START
            q = yaw_quat(yaw)
        else:
            x, y = self.data.xpos[self.robot_root, :2]
            r = self.data.xmat[self.robot_root].reshape(3, 3)
            q = yaw_quat(math.degrees(math.atan2(r[1, 0], r[0, 0])))
        self.data.mocap_pos[self.head_mount] = [x, y, HEAD_CAM_Z]
        self.data.mocap_quat[self.head_mount] = q

    def bottle_xyz(self) -> np.ndarray:
        return self.data.xpos[self.bottle_body].copy()

    @property
    def dynamic_state(self) -> dict:
        if self.mode != Mode.DYNAMIC or self.dynamic_controller is None:
            return {"active": False, "person_too_close": False, "item_relocated": False}
        state = self.dynamic_controller.state_at(self.sim_time)
        robot_pos = ROBOT_START[:2] if self.robot_qadr is None else self.data.qpos[self.robot_qadr : self.robot_qadr + 2]
        too_close, dist = self.dynamic_controller.check_robot_clearance(robot_pos, self.sim_time, clearance_threshold=1.10)
        return {
            "active": True,
            "sim_time": self.sim_time,
            "person_pos": tuple(state["pos"]),
            "action": state["action"],
            "carrying": state["carrying"],
            "item_relocated": state["relocated"],
            "person_too_close": too_close,
            "distance_to_robot": dist,
        }


def _has_key(model, name):
    try:
        model.key(name)
        return True
    except KeyError:
        return False


# =============================================================================
# CLI & Interactive Viewer
# =============================================================================


def cmd_check(env: HouseEnv, seeds: int = 5) -> bool:
    ok = True
    print("door swing (deg): " + ", ".join(
        f"{dn} {math.degrees(lo):.0f}..{math.degrees(hi):+.0f}" for dn, (lo, hi) in env.door_range.items()))
    print(f"{'layout':<16}{'target':<18}{'path m':>8}{'tries':>7}  status")
    for lay in LAYOUTS:
        for tgt in TARGETS:
            try:
                lens = []
                for s in range(seeds):
                    info = env.reset(lay, tgt, seed=s)
                    lens.append(info["path_len"])
                    pen = env.penetrations()
                    assert not pen, pen
                info = env.reset(lay, tgt, jitter=False)
                status = "ok"
                print(f"{lay:<16}{tgt:<18}{info['path_len']:>8.2f}{info['attempts']:>7}  {status}")
            except Exception as e:
                ok = False
                print(f"{lay:<16}{tgt:<18}{'-':>8}{'-':>7}  FAIL: {e}")
    print("ALL OK" if ok else "SOME COMBINATIONS FAILED")
    return ok


def cmd_snapshot(env: HouseEnv, outdir: Path):
    from PIL import Image

    outdir.mkdir(parents=True, exist_ok=True)
    tgts = list(TARGETS)
    cams = ["overview", "cam_robot_head"] + [f"cam_{n}" for n in ROOMS]

    with mujoco.Renderer(env.model, height=900, width=1200) as r:
        for k, lay in enumerate(LAYOUTS):
            tgt = tgts[k % len(tgts)]
            env.reset(lay, tgt, seed=0)
            env.step(500)
            for cam in cams:
                r.update_scene(env.data, camera=cam)
                Image.fromarray(r.render()).save(outdir / f"{lay}__{cam}.png")
            occupancy_image(env).save(outdir / f"{lay}__grid.png")
            print(f"saved {lay} (bottle on {tgt})")

        # 3/4 perspective view of the house
        env.reset("tidy", "kitchen_table", seed=0)
        env.step(500)
        env.set_drawers(0.6)
        mujoco.mj_forward(env.model, env.data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 10.5, 120, -50
        r.update_scene(env.data, camera=cam)
        Image.fromarray(r.render()).save(outdir / "tidy__perspective.png")

        # Dynamic mode snapshot: walking person in room
        env.reset("tidy", "kitchen_counter", mode=Mode.DYNAMIC, seed=0)
        env.step(3500)  # person carrying item
        r.update_scene(env.data, camera="cam_kitchen")
        Image.fromarray(r.render()).save(outdir / "dynamic__person_relocating.png")
        print(f"saved dynamic relocation snapshot to {outdir}")

    print(f"snapshots saved in {outdir}")


def occupancy_image(env: HouseEnv, scale: int = 6):
    from PIL import Image

    grid = env.occupancy()
    img = np.where(grid[..., None], [90, 90, 90], [245, 245, 245]).astype(np.uint8)
    bx, by = env.bottle_pos[:2]
    goal = (~grid) & (np.hypot(env.GX - bx, env.GY - by) <= REACH)
    img[goal] = [150, 200, 255]
    for (x, y), col, rad in [((ROBOT_START[0], ROBOT_START[1]), [40, 180, 60], 4), ((bx, by), [20, 60, 200], 2)]:
        i, j = env._cell(x, y)
        img[max(i - rad, 0) : i + rad + 1, max(j - rad, 0) : j + rad + 1] = col
    im = Image.fromarray(img[::-1])
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def cmd_view(env: HouseEnv, layout=None, target=None, seed=None, mode=Mode.STATIC):
    import mujoco.viewer

    lays, tgts = list(LAYOUTS), list(ALL_TARGETS)
    modes = [Mode.STATIC, Mode.SHUFFLED, Mode.DYNAMIC]
    current_mode = Mode(mode)
    state = {"pending": None, "mode": current_mode}

    def show():
        i = env.info
        print(f"mode={state['mode'].value:<10} layout={i['layout']:<14} target={i['target']:<16} path={i['path_len']:.2f} m")
        print("  doors (deg): " + ", ".join(f"{k} {v:+d}" + (" LOCKED" if k in i["locked"] else "") for k, v in i["doors"].items()))
        if state["mode"] == Mode.DYNAMIC:
            ds = env.dynamic_state
            print(f"  dynamic agent: action={ds.get('action')} carrying={ds.get('carrying')} relocated={ds.get('item_relocated')}")

    env.reset(layout, target, seed=seed, mode=current_mode)
    show()

    def on_key(key):
        # SPACE: random reset | L: next layout | T: next target | O: doors | M: cycle mode (1: static, 2: shuffled, 3: dynamic)
        if key == 32:
            state["pending"] = ("random",)
        elif key == ord("L"):
            state["pending"] = ("layout",)
        elif key == ord("T"):
            state["pending"] = ("target",)
        elif key == ord("O"):
            state["pending"] = ("doors",)
        elif key == ord("M") or key == ord("1"):
            state["pending"] = ("mode", Mode.STATIC)
        elif key == ord("2"):
            state["pending"] = ("mode", Mode.SHUFFLED)
        elif key == ord("3"):
            state["pending"] = ("mode", Mode.DYNAMIC)

    print("keys: SPACE random reset | L layout | T target (surfaces + drawers) | O doors | 1/2/3 modes")
    print("      1: Static House | 2: Shuffled Furniture & Drawers | 3: Dynamic Walking Person")
    print("push a door or pull a drawer: double-click it, then Ctrl + right-drag")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        v.cam.lookat[:] = [0, 0, 0.3]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 10.5, 120, -55
        while v.is_running():
            t0 = time.time()
            p, state["pending"] = state["pending"], None
            if p:
                with v.lock():
                    if p[0] == "random":
                        env.reset(mode=state["mode"])
                    elif p[0] == "layout":
                        env.reset(lays[(lays.index(env.layout) + 1) % len(lays)], env.target, mode=state["mode"])
                    elif p[0] == "target":
                        env.reset(env.layout, tgts[(tgts.index(env.target) + 1) % len(tgts)], mode=state["mode"])
                    elif p[0] == "doors":
                        env.randomise_doors()
                        mujoco.mj_forward(env.model, env.data)
                        env.info["doors"] = {dn: round(math.degrees(env.door_angle(dn))) for dn in DOORWAYS}
                    elif p[0] == "mode":
                        state["mode"] = p[1]
                        env.reset(mode=state["mode"])
                    show()
            with v.lock():
                env.step(int(round(1 / 60 / env.model.opt.timestep)))
            v.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify every layout x target is valid")
    ap.add_argument("--snapshot", action="store_true", help="render PNGs to sim/snapshots/")
    ap.add_argument("--view", action="store_true", help="open the interactive viewer")
    ap.add_argument("--layout", choices=list(LAYOUTS))
    ap.add_argument("--target", choices=list(ALL_TARGETS))
    ap.add_argument("--mode", choices=["static", "shuffled", "dynamic"], default="static", help="test difficulty mode")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--robot", help="robot MJCF to attach at ROBOT_START")
    ap.add_argument("--robot-joint", help="name of the robot's root free joint (auto-detected if omitted)")
    ap.add_argument("--write-xml", action="store_true", help="write the house scene to sim/house.xml")
    args = ap.parse_args()

    env = HouseEnv(robot_xml=args.robot, robot_joint=args.robot_joint, seed=args.seed)
    if args.write_xml:
        (HERE / "house.xml").write_text(env.xml)
        print(f"wrote {HERE / 'house.xml'}")
    ok = True
    if args.check:
        ok = cmd_check(env)
    if args.snapshot:
        cmd_snapshot(env, HERE / "snapshots")
    if args.view or not (args.check or args.snapshot or args.write_xml):
        cmd_view(env, args.layout, args.target, args.seed, mode=args.mode)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
