"""Simulated house for the fetch-robot project (MuJoCo).

An 8 x 6 m house with four rooms (living room, kitchen, bedroom, study) joined
by four doorways that form a loop. Walls and furniture are static geoms. Clutter
obstacles and doors are mocap bodies that `HouseEnv.reset()` moves around, so
the model is compiled once and every episode just repositions things. Unused
obstacles are parked under the floor. A water bottle (free body) is placed on
one of several surfaces, and a seated user sits on the living-room sofa.

Every reset is checked: no obstacle may penetrate a wall or furniture, and an
occupancy grid + BFS confirms the robot can drive from its start pose to within
arm's reach of the bottle (and therefore back again).

Everything you are likely to tweak is in the CONFIG section below.

CLI
    python house_env.py --check                 # verify every layout x target
    python house_env.py --snapshot              # PNGs into sim/snapshots/
    python house_env.py --view                  # interactive viewer (use run_sim.sh on macOS)
    python house_env.py --view --layout door_shut --target study_desk --seed 3
    python house_env.py --view --robot ../trash-collection-isaaclab/deploy/mujoco/models/aliengo_z1.xml
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent

# =============================================================================
# CONFIG
# Coordinates in metres. The house spans x in [-4, 4], y in [-3, 3], floor at
# z = 0. Rooms: living (SW), kitchen (SE), bedroom (NW), study (NE).
# =============================================================================

# --- Robot tunables (adjust once the robot's real dimensions are known) -------
ROBOT_R = 0.30  # footprint radius used to inflate obstacles (Aliengo+Z1 body is ~0.65 x 0.30 m;
                # its circumscribed radius is ~0.38, so 0.30 assumes it turns in place carefully)
ROBOT_H = 0.80  # anything whose bottom is above this height doesn't block the robot
REACH = 0.80  # robot centre must get within this xy distance of the bottle
ROBOT_START = (-1.8, -1.0, 0.0)  # x, y, yaw(deg): in front of the user, living room

# --- House geometry -----------------------------------------------------------
WALL_T = 0.10  # wall thickness
WALL_H = 1.20  # low "dollhouse" walls so angled cameras can see in
DOOR_W = 1.00  # doorway clear width
DOOR_H = 1.00  # door leaf height
GRID_RES = 0.05  # occupancy grid cell size

# Wall centre-lines (x0, y0, x1, y1). Doorways cut gaps into these.
WALLS = {
    "outer_s": (-4.0, -3.0, 4.0, -3.0),
    "outer_n": (-4.0, 3.0, 4.0, 3.0),
    "outer_w": (-4.0, -3.0, -4.0, 3.0),
    "outer_e": (4.0, -3.0, 4.0, 3.0),
    "inner_v_s": (0.0, -3.0, 0.0, 0.0),  # living | kitchen
    "inner_v_n": (0.0, 0.0, 0.0, 3.0),  # bedroom | study
    "inner_h_w": (-4.0, 0.0, 0.0, 0.0),  # living | bedroom
    "inner_h_e": (0.0, 0.0, 4.0, 0.0),  # kitchen | study
}

# Doorways: which wall, where along it (centre), and how the door leaf hangs.
#   hinge: "start" or "end" jamb (relative to the wall's x0,y0 -> x1,y1 direction)
#   swing: +1 / -1, which side of the wall the open leaf swings into (left / right
#          of the wall direction)
DOORWAYS = {
    "living_kitchen": dict(wall="inner_v_s", center=(0.0, -1.5), hinge="end", swing=-1),
    "bedroom_study": dict(wall="inner_v_n", center=(0.0, 1.5), hinge="end", swing=-1),
    "living_bedroom": dict(wall="inner_h_w", center=(-2.0, 0.0), hinge="start", swing=-1),
    "kitchen_study": dict(wall="inner_h_e", center=(2.0, 0.0), hinge="end", swing=+1),
}

ROOMS = {  # x0, y0, x1, y1, floor colour
    "living": (-4.0, -3.0, 0.0, 0.0, (0.80, 0.70, 0.55)),
    "kitchen": (0.0, -3.0, 4.0, 0.0, (0.82, 0.82, 0.80)),
    "bedroom": (-4.0, 0.0, 0.0, 3.0, (0.70, 0.76, 0.85)),
    "study": (0.0, 0.0, 4.0, 3.0, (0.72, 0.80, 0.70)),
}

# Fixed furniture. Each item: centre (x, y), footprint size (sx, sy), height h,
# colour, and kind:
#   "box"   solid block on the floor
#   "table" top slab + four legs (robot still treats the whole footprint as blocked)
#   "sofa"  seat + backrest (backrest on the -x side)
_WOOD = (0.55, 0.38, 0.22)
_LIGHT_WOOD = (0.76, 0.60, 0.42)
_WHITE = (0.92, 0.92, 0.90)
FURNITURE = {
    # living room
    "sofa": dict(kind="sofa", pos=(-3.55, -1.5), size=(0.80, 2.00), h=0.45, rgb=(0.35, 0.45, 0.60)),
    "coffee_table": dict(kind="table", pos=(-2.70, -1.5), size=(0.50, 0.90), h=0.40, rgb=_LIGHT_WOOD),
    "tv_unit": dict(kind="box", pos=(-1.80, -2.775), size=(1.20, 0.35), h=0.50, rgb=(0.25, 0.25, 0.28)),
    # kitchen
    "counter": dict(kind="box", pos=(1.95, -2.65), size=(2.70, 0.60), h=0.90, rgb=_WHITE),
    "fridge": dict(kind="box", pos=(3.62, -2.64), size=(0.65, 0.62), h=1.80, rgb=(0.85, 0.87, 0.90)),
    "kitchen_table": dict(kind="table", pos=(3.30, -1.00), size=(0.80, 0.60), h=0.75, rgb=_WOOD),
    # bedroom
    "bed": dict(kind="box", pos=(-2.95, 2.25), size=(2.00, 1.40), h=0.50, rgb=(0.90, 0.85, 0.80)),
    "bedside_table": dict(kind="box", pos=(-3.70, 1.10), size=(0.40, 0.40), h=0.55, rgb=_LIGHT_WOOD),
    "wardrobe": dict(kind="box", pos=(-0.90, 2.65), size=(1.20, 0.60), h=1.10, rgb=_WOOD),
    # study
    "desk": dict(kind="table", pos=(3.60, 1.50), size=(0.70, 1.40), h=0.75, rgb=_LIGHT_WOOD),
    "bookshelf": dict(kind="box", pos=(2.00, 2.80), size=(1.40, 0.30), h=1.10, rgb=_WOOD),
}

# Where the bottle can be: name -> (x, y, furniture it stands on).
# z is taken from that furniture's height.
TARGETS = {
    "tv_unit": (-1.50, -2.78, "tv_unit"),
    "kitchen_counter": (1.20, -2.62, "counter"),
    "kitchen_table": (3.10, -0.90, "kitchen_table"),
    "bedside_table": (-3.70, 1.10, "bedside_table"),
    "study_desk": (3.50, 1.20, "desk"),
}

# Clutter obstacle types. Each is a list of primitive geoms in the body frame:
#   ("box", (hx, hy, hz), (px, py, pz))  or  ("cyl", (r, hz), (px, py, pz))
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
            ("box", (0.22, 0.22, 0.225), (0, 0, 0.225)),  # seat block
            ("box", (0.03, 0.22, 0.25), (-0.19, 0, 0.70)),  # backrest
        ],
    ),
}

# Layouts: which doors are shut + clutter as (type, x, y, yaw_deg).
LAYOUTS = {
    "tidy": dict(doors_shut=[], obstacles=[]),
    "laundry_day": dict(
        doors_shut=[],
        obstacles=[
            ("laundry_basket", -1.00, 0.80, 0),
            ("clothes_pile", -1.10, -2.10, 20),
            ("clothes_pile", 1.10, 1.00, -30),
            ("clothes_pile", -2.80, 0.60, 10),
            ("bag", 0.80, -0.50, 45),
            ("chair", 2.40, -0.40, -90),
        ],
    ),
    "door_shut": dict(
        doors_shut=["living_kitchen"],
        obstacles=[
            ("chair", 1.40, -1.30, 0),
            ("box_small", -0.80, -2.30, 0),
        ],
    ),
    "moving_house": dict(
        doors_shut=[],
        obstacles=[
            ("box_large", -0.80, -0.60, 10),
            ("box_small", -0.70, -2.40, 0),
            ("box_large", 0.90, -1.90, -15),
            ("box_small", 3.20, -0.60, 30),
            ("box_large", 0.90, 2.40, 0),
            ("box_small", 3.00, 0.50, 15),
            ("box_large", -1.20, 1.40, 0),
            ("box_small", -3.30, 0.50, 5),
        ],
    ),
    # The longest detour: the bedroom door is shut and the kitchen is cluttered,
    # so reaching the bedroom means living -> kitchen -> study -> bedroom.
    "long_way_round": dict(
        doors_shut=["living_bedroom"],
        obstacles=[
            ("laundry_basket", 1.00, -0.60, 0),
            ("clothes_pile", 3.20, -1.00, 60),
            ("bag", 2.80, 0.70, 0),
            ("chair", 1.00, 1.60, 180),
        ],
    ),
}

JITTER_XY = 0.15  # +/- metres applied to every obstacle on reset
JITTER_YAW = 10.0  # +/- degrees
MAX_RESET_TRIES = 50  # jitter attempts before falling back to the nominal layout

PARK_Z = -5.0  # unused mocap bodies are parked here, under the floor

# Bottle: a 500 ml water bottle as primitives (body frame origin at its base).
BOTTLE_R = 0.033
BOTTLE_H = 0.21
BOTTLE_MASS = 0.5
# Grasp-frame site orientation, matching the robot repo's scene (their bottle mesh
# body is rotated by (0.7071, 0.7071, 0, 0) and the site by (-0.5, 0.5, -0.5, 0.5)
# inside it). Computed once below so this z-up bottle yields the same world frame.
_THEIR_BODY_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0])
_THEIR_SITE_Q = np.array([-0.5, 0.5, -0.5, 0.5])

# =============================================================================
# Geometry helpers
# =============================================================================


def yaw_quat(deg: float) -> np.ndarray:
    a = math.radians(deg) / 2.0
    return np.array([math.cos(a), 0.0, 0.0, math.sin(a)])


def lookat_quat(pos, target) -> np.ndarray:
    """Quaternion for a MuJoCo camera at `pos` looking at `target` (z-up world)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    mat = np.column_stack([x, y, -f]).flatten()  # camera looks along its -z
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _bottle_site_quat() -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, _THEIR_BODY_Q, _THEIR_SITE_Q)
    return q


def door_poses(name: str):
    """Return (shut_pos, shut_quat, open_pos, open_quat) for a door leaf centre."""
    dw = DOORWAYS[name]
    x0, y0, x1, y1 = WALLS[dw["wall"]]
    u = np.array([x1 - x0, y1 - y0], float)
    u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0]])  # left of wall direction
    c = np.array(dw["center"], float)
    sgn = 1.0 if dw["hinge"] == "end" else -1.0
    hinge = c + sgn * u * (DOOR_W / 2)
    wall_yaw = math.degrees(math.atan2(u[1], u[0]))
    z = DOOR_H / 2 + 0.01
    shut_pos = np.array([c[0], c[1], z])
    open_c = hinge + dw["swing"] * n * (DOOR_W / 2) - sgn * u * 0.03
    open_pos = np.array([open_c[0], open_c[1], z])
    return shut_pos, yaw_quat(wall_yaw), open_pos, yaw_quat(wall_yaw + 90.0)


def _wall_segments():
    """Split wall centre-lines at doorways -> list of (name, cx, cy, half_len, yaw)."""
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
        # extend outer walls by half thickness so corners close
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
    """How many mocap bodies of each obstacle type we need (max over layouts)."""
    pool = {}
    for lay in LAYOUTS.values():
        counts = {}
        for typ, *_ in lay["obstacles"]:
            counts[typ] = counts.get(typ, 0) + 1
        for typ, c in counts.items():
            pool[typ] = max(pool.get(typ, 0), c)
    return pool


# =============================================================================
# Scene XML
# =============================================================================


def build_xml() -> str:
    L = []
    a = L.append
    a('<mujoco model="secretly_robots_house">')
    a('  <compiler angle="radian" autolimits="true"/>')
    a('  <option timestep="0.002" integrator="implicitfast"/>')
    a('  <statistic center="0 0 0.5" extent="5"/>')
    a("  <visual>")
    a('    <global azimuth="120" elevation="-50" offwidth="1600" offheight="1200"/>')
    a('    <headlight ambient="0.15 0.15 0.15" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>')
    a('    <quality shadowsize="4096"/>')
    a("  </visual>")
    a("  <asset>")
    a('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.85 0.9 0.95" rgb2="0.6 0.7 0.8" width="256" height="256"/>')
    a('    <texture name="ground" type="2d" builtin="checker" rgb1="0.35 0.42 0.32" rgb2="0.32 0.39 0.29" width="256" height="256"/>')
    a('    <material name="ground" texture="ground" texrepeat="20 20"/>')
    a('    <material name="wall" rgba="0.62 0.60 0.58 1"/>')
    a('    <material name="door" rgba="0.60 0.42 0.26 1"/>')
    a("  </asset>")
    a("  <worldbody>")
    a('    <light name="sun" pos="2 -3 8" dir="-0.2 0.3 -1" directional="true" castshadow="true" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>')
    a('    <light name="fill" pos="-2 2 6" dir="0.2 -0.2 -1" directional="true" castshadow="false" diffuse="0.2 0.2 0.2" specular="0 0 0"/>')
    a('    <geom name="floor" type="plane" size="12 12 0.05" material="ground"/>')

    # room floors: thin visual-only tiles
    for rname, (x0, y0, x1, y1, rgb) in ROOMS.items():
        a(
            f'    <geom name="floor_{rname}" type="box" pos="{_fmt([(x0 + x1) / 2, (y0 + y1) / 2, 0.0005])}" '
            f'size="{_fmt([(x1 - x0) / 2, (y1 - y0) / 2, 0.0005])}" rgba="{_fmt(rgb)} 1" '
            f'contype="0" conaffinity="0"/>'
        )

    # walls
    for name, cx, cy, hl, yaw in _wall_segments():
        a(
            f'    <geom name="{name}" type="box" pos="{_fmt([cx, cy, WALL_H / 2])}" quat="{_fmt(yaw_quat(yaw))}" '
            f'size="{_fmt([hl, WALL_T / 2, WALL_H / 2])}" material="wall"/>'
        )

    # furniture
    for name, f in FURNITURE.items():
        (x, y), (sx, sy), h, rgb = f["pos"], f["size"], f["h"], f["rgb"]
        col = f'rgba="{_fmt(rgb)} 1"'
        if f["kind"] == "box":
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h / 2])}" size="{_fmt([sx / 2, sy / 2, h / 2])}" {col}/>')
        elif f["kind"] == "table":
            top = 0.03
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x, y, h - top / 2])}" size="{_fmt([sx / 2, sy / 2, top / 2])}" {col}/>')
            lx, ly, lh = sx / 2 - 0.05, sy / 2 - 0.05, (h - top) / 2
            for i, (px, py) in enumerate([(lx, ly), (-lx, ly), (lx, -ly), (-lx, -ly)]):
                a(f'    <geom name="{name}_leg{i}" type="box" pos="{_fmt([x + px, y + py, lh])}" size="{_fmt([0.025, 0.025, lh])}" {col}/>')
        elif f["kind"] == "sofa":
            back = 0.18
            a(f'    <geom name="{name}" type="box" pos="{_fmt([x + back / 2, y, h / 2])}" size="{_fmt([(sx - back) / 2, sy / 2, h / 2])}" {col}/>')
            a(f'    <geom name="{name}_back" type="box" pos="{_fmt([x - sx / 2 + back / 2, y, 0.425])}" size="{_fmt([back / 2, sy / 2, 0.425])}" {col}/>')
        else:
            raise ValueError(f"unknown furniture kind {f['kind']!r}")

    # seated user on the sofa (static)
    sx, sy = FURNITURE["sofa"]["pos"]
    seat = FURNITURE["sofa"]["h"]
    ux = sx - 0.12
    skin, shirt, trousers = "0.87 0.70 0.58 1", "0.75 0.35 0.25 1", "0.20 0.22 0.35 1"
    a(f'    <body name="user" pos="{_fmt([ux, sy, seat])}">')
    a(f'      <geom name="user_torso" type="capsule" fromto="0 0 0.12 0 0 0.52" size="0.16" rgba="{shirt}"/>')
    a(f'      <geom name="user_head" type="sphere" pos="0 0 0.70" size="0.11" rgba="{skin}"/>')
    a(f'      <geom name="user_thighs" type="box" pos="0.22 0 0.08" size="0.22 0.17 0.07" rgba="{trousers}"/>')
    a(f'      <geom name="user_shins" type="box" pos="0.42 0 -0.20" size="0.06 0.15 0.22" rgba="{trousers}"/>')
    a('      <site name="user_hand" pos="0.30 -0.25 0.25" size="0.03" rgba="1 0.8 0 1"/>')
    a("    </body>")

    # doors (mocap, start open)
    for dname in DOORWAYS:
        _, _, op, oq = door_poses(dname)
        a(f'    <body name="door_{dname}" mocap="true" pos="{_fmt(op)}" quat="{_fmt(oq)}">')
        a(f'      <geom name="door_{dname}" type="box" size="{_fmt([DOOR_W / 2 - 0.02, 0.02, DOOR_H / 2])}" material="door"/>')
        a("    </body>")

    # clutter obstacles (mocap, parked)
    for typ, count in _obstacle_pool().items():
        spec = OBSTACLE_TYPES[typ]
        for i in range(count):
            a(f'    <body name="obs_{typ}_{i}" mocap="true" pos="0 0 {PARK_Z}">')
            for j, g in enumerate(spec["geoms"]):
                kind, size, pos = g
                gtype = "box" if kind == "box" else "cylinder"
                a(f'      <geom name="obs_{typ}_{i}_{j}" type="{gtype}" size="{_fmt(size)}" pos="{_fmt(pos)}" rgba="{_fmt(spec["rgb"])} 1"/>')
            a("    </body>")

    # bottle (free body; origin at its base)
    sq = _bottle_site_quat()
    r, h = BOTTLE_R, BOTTLE_H
    body_h = h * 0.78
    a('    <body name="waterbottle" pos="0 0 1">')
    a('      <freejoint name="waterbottle_joint"/>')
    a(f'      <geom name="waterbottle" type="cylinder" size="{_fmt([r, body_h / 2])}" pos="{_fmt([0, 0, body_h / 2])}" '
      f'mass="{BOTTLE_MASS * 0.9:.4g}" rgba="0.15 0.45 0.95 0.85" friction="1 0.01 0.001" condim="4"/>')
    a(f'      <geom name="waterbottle_neck" type="cylinder" size="{_fmt([r * 0.55, (h - body_h) / 2 * 0.7])}" '
      f'pos="{_fmt([0, 0, body_h + (h - body_h) * 0.35])}" mass="{BOTTLE_MASS * 0.07:.4g}" rgba="0.15 0.45 0.95 0.85"/>')
    a(f'      <geom name="waterbottle_cap" type="cylinder" size="{_fmt([r * 0.45, (h - body_h) * 0.15])}" '
      f'pos="{_fmt([0, 0, h - (h - body_h) * 0.15])}" mass="{BOTTLE_MASS * 0.03:.4g}" rgba="0.05 0.15 0.6 1"/>')
    a(f'      <site name="waterbottle_site" pos="{_fmt([0, 0, h / 2])}" quat="{_fmt(sq)}" size="0.01" rgba="1 0 0 1" group="1"/>')
    a("    </body>")

    # cameras
    a('    <camera name="overview" pos="0 0 8.2" quat="1 0 0 0" fovy="45"/>')
    for rname, (x0, y0, x1, y1, _) in ROOMS.items():
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # put the camera above the outer corner of the room, looking at its centre
        px = x0 + 0.3 if cx < 0 else x1 - 0.3
        py = y0 + 0.3 if cy < 0 else y1 - 0.3
        q = lookat_quat([px, py, 3.0], [cx, cy, 0.3])
        a(f'    <camera name="cam_{rname}" pos="{_fmt([px, py, 3.0])}" quat="{_fmt(q)}" fovy="70"/>')
    a("  </worldbody>")
    a("</mujoco>")
    return "\n".join(L)


# =============================================================================
# Runtime environment
# =============================================================================


@dataclass
class CheckResult:
    solvable: bool
    start_free: bool
    path_len: float  # metres along the grid path (inf if unsolvable)
    grid: np.ndarray  # bool occupancy (True = blocked), shape (ny, nx)


class HouseEnv:
    """Compiled house model + reset logic.

    Attributes after reset(): layout, target, bottle_pos (3,), info dict.
    """

    def __init__(self, robot_xml: str | None = None, robot_joint: str | None = None, seed: int | None = None):
        self.xml = build_xml()
        spec = mujoco.MjSpec.from_string(self.xml)
        self.robot_root = None
        if robot_xml:
            self._attach_robot(spec, robot_xml)
        with warnings.catch_warnings():  # empty prefix keeps the robot's own names (trunk, ...)
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
        self.door_mocap = {d: m.body(f"door_{d}").mocapid[0] for d in DOORWAYS}
        self.obs_mocap = {}  # type -> list of mocap ids
        for typ, count in _obstacle_pool().items():
            self.obs_mocap[typ] = [m.body(f"obs_{typ}_{i}").mocapid[0] for i in range(count)]

        # geom sets
        mocap_bodies = {i for i in range(m.nbody) if m.body_mocapid[i] >= 0}
        robot_bodies = self._subtree(self.robot_root) if self.robot_root is not None else set()
        self.mocap_geoms, self.static_geoms = [], []
        skip = {"floor", "waterbottle", "waterbottle_neck", "waterbottle_cap"}
        for g in range(m.ngeom):
            name = m.geom(g).name
            b = m.geom_bodyid[g]
            if name in skip or name.startswith("floor_") or b in robot_bodies or b == self.bottle_body:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            (self.mocap_geoms if b in mocap_bodies else self.static_geoms).append(g)

        # robot root joint
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
        self.info = {}

    # ---- robot merging --------------------------------------------------------
    def _attach_robot(self, spec: mujoco.MjSpec, robot_xml: str):
        robot = mujoco.MjSpec.from_file(str(Path(robot_xml).resolve()))
        # drop the robot file's own floor / lights so the house's are used
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

    # ---- reset ----------------------------------------------------------------
    def reset(self, layout: str | None = None, target: str | None = None, seed: int | None = None, jitter: bool = True):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        layout = layout or str(self.rng.choice(list(LAYOUTS)))
        target = target or str(self.rng.choice(list(TARGETS)))
        if layout not in LAYOUTS:
            raise KeyError(f"unknown layout {layout!r}; choose from {list(LAYOUTS)}")
        if target not in TARGETS:
            raise KeyError(f"unknown target {target!r}; choose from {list(TARGETS)}")
        self.layout, self.target = layout, target

        mujoco.mj_resetData(self.model, self.data)
        if self.robot_root is not None and self.model.nkey > 0:
            key = self.model.key("home").id if _has_key(self.model, "home") else 0
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self._place_robot()
        self._place_bottle(target)

        res = None
        for attempt in range(MAX_RESET_TRIES):
            use_jitter = jitter and attempt < MAX_RESET_TRIES - 1
            self._place_layout(layout, use_jitter)
            mujoco.mj_forward(self.model, self.data)
            if self.penetrations():
                continue
            res = self.check()
            if res.solvable:
                break
        else:
            raise RuntimeError(f"layout {layout!r} + target {target!r} is not valid even without jitter")

        self.info = dict(layout=layout, target=target, attempts=attempt + 1, path_len=res.path_len)
        return self.info

    def _place_robot(self):
        if self.robot_qadr is None:
            return
        x, y, yaw = ROBOT_START
        q = self.data.qpos
        a = self.robot_qadr
        q[a : a + 2] = [x, y]
        # keep the robot's own default height; rotate by start yaw
        q[a + 3 : a + 7] = yaw_quat(yaw)

    def _place_bottle(self, target: str):
        x, y, furn = TARGETS[target]
        z = FURNITURE[furn]["h"] + 0.001
        a = self.bottle_qadr
        self.data.qpos[a : a + 7] = [x, y, z, 1, 0, 0, 0]
        self.data.qvel[self.bottle_dadr : self.bottle_dadr + 6] = 0
        self.bottle_pos = np.array([x, y, z])

    def _place_layout(self, layout: str, jitter: bool):
        lay = LAYOUTS[layout]
        d = self.data
        for dname, mid in self.door_mocap.items():
            sp, sq, op, oq = door_poses(dname)
            shut = dname in lay["doors_shut"]
            d.mocap_pos[mid] = sp if shut else op
            d.mocap_quat[mid] = sq if shut else oq
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

    # ---- validity checks ------------------------------------------------------
    def active_mocap_geoms(self):
        return [g for g in self.mocap_geoms if self.data.geom_xpos[g, 2] > PARK_Z / 2]

    def penetrations(self, tol: float = 1e-4):
        """Pairs (geomA, geomB, dist) where an active mocap geom overlaps a wall,
        furniture, the user, another obstacle, or the bottle."""
        m, d = self.model, self.data
        active = self.active_mocap_geoms()
        others = self.static_geoms + [self.bottle_geom_id()]
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
        return bad

    def bottle_geom_id(self):
        return self.model.geom("waterbottle").id

    def occupancy(self) -> np.ndarray:
        """Bool grid (ny, nx): True where the robot centre can't be."""
        m, d = self.model, self.data
        blocked = np.zeros_like(self.GX, dtype=bool)
        for g in self.static_geoms + self.active_mocap_geoms():
            pos, mat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            size = m.geom_size[g]
            typ = m.geom_type[g]
            # vertical extent of the geom in world z
            if typ == mujoco.mjtGeom.mjGEOM_BOX:
                half_z = np.abs(mat[2]) @ size
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                half_z = size[1] + (size[0] if typ == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            else:
                half_z = m.geom_rbound[g]
            if pos[2] - half_z > ROBOT_H:
                continue
            dx, dy = self.GX - pos[0], self.GY - pos[1]
            if typ == mujoco.mjtGeom.mjGEOM_BOX and abs(mat[2, 2]) > 0.99:
                c, s = mat[0, 0], mat[1, 0]  # body x axis in world xy
                lx, ly = c * dx + s * dy, -s * dx + c * dy
                ox = np.maximum(np.abs(lx) - size[0], 0)
                oy = np.maximum(np.abs(ly) - size[1], 0)
                dist = np.hypot(ox, oy)
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE) and abs(mat[2, 2]) > 0.99:
                dist = np.hypot(dx, dy) - size[0]
            else:  # fall back to bounding sphere
                dist = np.hypot(dx, dy) - m.geom_rbound[g]
            blocked |= dist <= ROBOT_R
        return blocked

    def _cell(self, x, y):
        i = int(round((y - self.gy[0]) / GRID_RES))
        j = int(round((x - self.gx[0]) / GRID_RES))
        return i, j

    def check(self) -> CheckResult:
        """Can the robot get from ROBOT_START to within REACH of the bottle?"""
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

    # ---- convenience ----------------------------------------------------------
    def step(self, n: int = 1):
        mujoco.mj_step(self.model, self.data, nstep=n)

    def bottle_xyz(self) -> np.ndarray:
        return self.data.xpos[self.bottle_body].copy()


def _has_key(model, name):
    try:
        model.key(name)
        return True
    except KeyError:
        return False


# =============================================================================
# CLI
# =============================================================================


def cmd_check(env: HouseEnv, seeds: int = 5) -> bool:
    ok = True
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
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"{lay:<16}{tgt:<18}{'-':>8}{'-':>7}  FAIL: {e}")
    print("ALL OK" if ok else "SOME COMBINATIONS FAILED")
    return ok


def cmd_snapshot(env: HouseEnv, outdir: Path):
    from PIL import Image

    outdir.mkdir(parents=True, exist_ok=True)
    tgts = list(TARGETS)
    with mujoco.Renderer(env.model, height=900, width=1200) as r:
        for k, lay in enumerate(LAYOUTS):
            tgt = tgts[k % len(tgts)]
            env.reset(lay, tgt, seed=0)
            env.step(500)  # let the bottle settle
            for cam in ["overview"] + [f"cam_{n}" for n in ROOMS]:
                r.update_scene(env.data, camera=cam)
                Image.fromarray(r.render()).save(outdir / f"{lay}__{cam}.png")
            occupancy_image(env).save(outdir / f"{lay}__grid.png")
            print(f"saved {lay} (bottle on {tgt})")
        # free-camera 3/4 view of the tidy house
        env.reset("tidy", "kitchen_table", seed=0)
        env.step(500)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 10.5, 120, -50
        r.update_scene(env.data, camera=cam)
        Image.fromarray(r.render()).save(outdir / "tidy__perspective.png")
    print(f"snapshots in {outdir}")


def occupancy_image(env: HouseEnv, scale: int = 6):
    """Top-down picture of the planner's view: grey = blocked (inflated by ROBOT_R),
    white = free, green = start, light blue = cells within REACH of the bottle,
    dark blue = bottle."""
    from PIL import Image

    grid = env.occupancy()
    img = np.where(grid[..., None], [90, 90, 90], [245, 245, 245]).astype(np.uint8)
    bx, by = env.bottle_pos[:2]
    goal = (~grid) & (np.hypot(env.GX - bx, env.GY - by) <= REACH)
    img[goal] = [150, 200, 255]
    for (x, y), col, rad in [((ROBOT_START[0], ROBOT_START[1]), [40, 180, 60], 4), ((bx, by), [20, 60, 200], 2)]:
        i, j = env._cell(x, y)
        img[max(i - rad, 0) : i + rad + 1, max(j - rad, 0) : j + rad + 1] = col
    im = Image.fromarray(img[::-1])  # row 0 = y max at the top, like the overview camera
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def cmd_view(env: HouseEnv, layout=None, target=None, seed=None):
    import mujoco.viewer

    lays, tgts = list(LAYOUTS), list(TARGETS)
    state = {"pending": None}

    def show():
        i = env.info
        print(f"layout={i['layout']:<15} target={i['target']:<16} path={i['path_len']:.2f} m  (tries {i['attempts']})")

    env.reset(layout, target, seed=seed)
    show()

    def on_key(key):
        # SPACE: random layout + target. L / T: cycle layout / target. O: toggle all doors.
        if key == 32:
            state["pending"] = ("random",)
        elif key == ord("L"):
            state["pending"] = ("layout",)
        elif key == ord("T"):
            state["pending"] = ("target",)
        elif key == ord("O"):
            state["pending"] = ("doors",)

    print("keys: SPACE random reset | L next layout | T next target | O toggle doors")
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        v.cam.lookat[:] = [0, 0, 0.3]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 10.5, 120, -55
        doors_flipped = False
        while v.is_running():
            t0 = time.time()
            p, state["pending"] = state["pending"], None
            if p:
                with v.lock():
                    if p[0] == "random":
                        env.reset()
                    elif p[0] == "layout":
                        env.reset(lays[(lays.index(env.layout) + 1) % len(lays)], env.target)
                    elif p[0] == "target":
                        env.reset(env.layout, tgts[(tgts.index(env.target) + 1) % len(tgts)])
                    elif p[0] == "doors":
                        doors_flipped = not doors_flipped
                        for dname, mid in env.door_mocap.items():
                            sp, sq, op, oq = door_poses(dname)
                            shut = (dname in LAYOUTS[env.layout]["doors_shut"]) != doors_flipped
                            env.data.mocap_pos[mid] = sp if shut else op
                            env.data.mocap_quat[mid] = sq if shut else oq
                        print("doors " + ("flipped" if doors_flipped else "back to layout"))
                    if p[0] != "doors":
                        doors_flipped = False
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
    ap.add_argument("--target", choices=list(TARGETS))
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
        cmd_view(env, args.layout, args.target, args.seed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
