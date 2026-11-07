"""Simulated house for the fetch-robot project (MuJoCo).

An 8 x 6 m house with four rooms (living room, kitchen, bedroom, study) joined
by four doorways that form a loop. Walls and furniture are static geoms.

- Doors are hinged leaves that swing up to 90 degrees either way, so the robot
  can push them open from either side. Each reset gives them a random starting
  angle. A layout can lock a door shut (it then can't be pushed).
- Chests of drawers have sliding drawers with long bar handles.
- Clutter obstacles are mocap bodies that `HouseEnv.reset()` moves around, so the
  model is compiled once. Unused obstacles are parked under the floor.
- A water bottle (free body) stands on one of several surfaces, and a seated
  user sits on the living-room sofa.

Every reset is checked: no obstacle may penetrate a wall, furniture, door or
drawer; no obstacle may sit in a door's swing or in front of a drawer; and an
occupancy grid + BFS confirms the robot can drive from its start pose to within
arm's reach of the bottle (and therefore back again). Unlocked doors count as
passable, since the robot can push them.

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

# Doorways: which wall, where along it (centre), and which jamb the door hangs on
# ("start" or "end", relative to the wall's x0,y0 -> x1,y1 direction). Every door
# swings both ways (push from either side), up to DOOR_SWING_DEG.
DOORWAYS = {
    "living_kitchen": dict(wall="inner_v_s", center=(0.0, -1.5), hinge="end"),
    "bedroom_study": dict(wall="inner_v_n", center=(0.0, 1.5), hinge="end"),
    "living_bedroom": dict(wall="inner_h_w", center=(-2.0, 0.0), hinge="start"),
    "kitchen_study": dict(wall="inner_h_e", center=(2.0, 0.0), hinge="end"),
}
DOOR_SWING_DEG = 90.0  # max swing each way; trimmed automatically if a leaf would hit furniture
DOOR_DAMPING = 1.5  # N m s/rad: how sluggish the door is when pushed
DOOR_FRICTION = 0.3  # N m: static friction, so a door stays where it's left
DOOR_MASS = 12.0  # kg

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
    "counter": dict(kind="box", pos=(2.25, -2.65), size=(2.10, 0.60), h=0.90, rgb=_WHITE),
    "fridge": dict(kind="box", pos=(3.62, -2.64), size=(0.65, 0.62), h=1.80, rgb=(0.85, 0.87, 0.90)),
    "kitchen_table": dict(kind="table", pos=(3.30, -1.00), size=(0.80, 0.60), h=0.75, rgb=_WOOD),
    # bedroom
    "bed": dict(kind="box", pos=(-2.95, 2.25), size=(2.00, 1.40), h=0.50, rgb=(0.90, 0.85, 0.80)),
    "bedside_table": dict(kind="box", pos=(-3.70, 1.10), size=(0.40, 0.40), h=0.55, rgb=_LIGHT_WOOD),
    # study
    "desk": dict(kind="table", pos=(3.60, 1.50), size=(0.70, 1.40), h=0.75, rgb=_LIGHT_WOOD),
    "bookshelf": dict(kind="box", pos=(2.00, 2.80), size=(1.40, 0.30), h=1.10, rgb=_WOOD),
}

# Chests of drawers. pos = footprint centre, facing = direction (deg) the drawers
# pull out towards, size = (width, depth), h = height, n = number of drawers.
# Each drawer slides out up to DRAWER_EXTENSION of its depth and has a long bar
# handle (HANDLE_FRACTION of the chest width) with a site "<drawer>_handle" at
# its centre for grasping.
DRAWER_UNITS = {
    "dresser": dict(pos=(-0.90, 2.725), facing=-90, size=(1.00, 0.45), h=0.85, n=4, rgb=_WOOD),
    "kitchen_drawers": dict(pos=(0.90, -2.65), facing=90, size=(0.60, 0.60), h=0.90, n=3, rgb=_WHITE),
    "filing_cabinet": dict(pos=(3.725, 2.60), facing=180, size=(0.60, 0.45), h=1.00, n=3, rgb=(0.45, 0.50, 0.55)),
}
DRAWER_EXTENSION = 0.80  # fraction of drawer depth it can slide out
HANDLE_FRACTION = 0.75  # handle length as a fraction of chest width
HANDLE_R = 0.012  # handle bar radius
HANDLE_STANDOFF = 0.045  # gap between drawer front and handle bar centre
DRAWER_DAMPING = 4.0  # N s/m
DRAWER_FRICTION = 0.5  # N

# Where the bottle can be: name -> (x, y, furniture it stands on).
# z is taken from that furniture's height.
TARGETS = {
    "tv_unit": (-1.50, -2.78, "tv_unit"),
    "kitchen_counter": (1.80, -2.62, "counter"),
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

# Layouts: clutter as (type, x, y, yaw_deg), plus optional door settings:
#   doors_locked: shut and locked (can't be pushed, blocks the planner)
#   doors_closed: start shut but can still be pushed open
# Any other door starts at a random angle within its swing.
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
    # The longest detour: the bedroom door is locked and the kitchen is cluttered,
    # so reaching the bedroom means living -> kitchen -> study -> bedroom.
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


HINGE_INSET = 0.03  # hinge axis sits this far inside the doorway from the jamb
DOOR_LEAF_L = DOOR_W - HINGE_INSET - 0.02  # leaves a 1 cm gap at each end
DOOR_LEAF_T = 0.04  # leaf thickness


def door_frame(name: str):
    """Return (hinge_xy, yaw_deg): the hinge axis position and the direction the
    shut leaf points from the hinge (across the doorway)."""
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

    # doors: hinged leaves that swing both ways (push from either side)
    sw = math.radians(DOOR_SWING_DEG)
    leaf_vol = DOOR_LEAF_L * DOOR_LEAF_T * DOOR_H
    for dname in DOORWAYS:
        hinge, yaw = door_frame(dname)
        lx = 0.01 + DOOR_LEAF_L / 2
        a(f'    <body name="door_{dname}" pos="{_fmt([hinge[0], hinge[1], 0])}" quat="{_fmt(yaw_quat(yaw))}">')
        a(f'      <joint name="door_{dname}_hinge" type="hinge" axis="0 0 1" range="{-sw:.6g} {sw:.6g}" '
          f'damping="{DOOR_DAMPING}" frictionloss="{DOOR_FRICTION}" armature="0.05"/>')
        a(f'      <geom name="door_{dname}" type="box" pos="{_fmt([lx, 0, DOOR_H / 2 + 0.01])}" '
          f'size="{_fmt([DOOR_LEAF_L / 2, DOOR_LEAF_T / 2, DOOR_H / 2])}" material="door" density="{DOOR_MASS / leaf_vol:.6g}"/>')
        for side in (-1, 1):  # push plates on both faces (visual only)
            a(f'      <geom name="door_{dname}_plate{"ab"[side > 0]}" type="box" '
              f'pos="{_fmt([DOOR_LEAF_L - 0.08, side * (DOOR_LEAF_T / 2 + 0.002), 0.6])}" size="0.05 0.002 0.12" '
              f'rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0" mass="0"/>')
        a("    </body>")

    # chests of drawers: a static carcass body with sliding drawers inside
    for uname, u in DRAWER_UNITS.items():
        _drawer_unit_xml(a, uname, u)

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
    # door locks (activated per layout by HouseEnv.reset)
    a("  <equality>")
    for dname in DOORWAYS:
        a(f'    <joint name="lock_{dname}" joint1="door_{dname}_hinge" polycoef="0 1 0 0 0" active="false"/>')
    a("  </equality>")
    a("</mujoco>")
    return "\n".join(L)


def _drawer_unit_xml(a, uname: str, u: dict):
    """Append XML for one chest of drawers. Local frame: +x = front (pull direction)."""
    (x, y), (w, d), h, n, rgb = u["pos"], u["size"], u["h"], u["n"], u["rgb"]
    t, plinth, ft = 0.018, 0.06, 0.02  # panel thickness, kick plinth height, drawer-front thickness
    col = f'rgba="{_fmt(rgb)} 1"'
    front_rgb = tuple(min(1.0, c * 1.08 + 0.03) for c in rgb)
    fcol = f'rgba="{_fmt(front_rgb)} 1"'
    metal = 'rgba="0.78 0.78 0.80 1"'
    cx = (-d / 2 + d / 2 - ft) / 2  # carcass sits behind the drawer fronts
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
    s = (z1 - z0) / n  # slot height
    di = d - t - ft - 0.02  # tray depth
    wi = w - 2 * t - 0.012  # tray width
    ht = min(0.7 * s, s - 0.03)  # tray wall height
    xb = d / 2 - ft - di / 2  # tray centre (x)
    hl = HANDLE_FRACTION * w
    hx = d / 2 + HANDLE_STANDOFF
    ext = DRAWER_EXTENSION * di
    for i in range(n):
        name = f"drawer_{uname}_{i}"
        zc = z0 + s * (i + 0.5)
        zb = -s / 2 + 0.015  # tray floor (relative)
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
        # long bar handle on two standoffs
        a(f'        <geom name="{name}_handle" type="capsule" fromto="{_fmt([hx, -hl / 2, 0, hx, hl / 2, 0])}" size="{HANDLE_R}" {metal}/>')
        for j, sy in enumerate((-1, 1)):
            a(f'        <geom name="{name}_post{j}" type="box" pos="{_fmt([d / 2 + HANDLE_STANDOFF / 2, sy * (hl / 2 - 0.03), 0])}" '
              f'size="{_fmt([HANDLE_STANDOFF / 2, 0.008, 0.008])}" {metal}/>')
        a(f'        <site name="{name}_handle" pos="{_fmt([hx, 0, 0])}" size="0.015" rgba="1 0.5 0 1" group="1"/>')
        a("      </body>")
    a("    </body>")


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
        self.door_jnt = {d: m.joint(f"door_{d}_hinge").id for d in DOORWAYS}
        self.door_eq = {d: m.equality(f"lock_{d}").id for d in DOORWAYS}
        self.drawer_jnt = [j for j in range(m.njnt) if m.joint(j).name.startswith("drawer_")]
        self.obs_mocap = {}  # type -> list of mocap ids
        for typ, count in _obstacle_pool().items():
            self.obs_mocap[typ] = [m.body(f"obs_{typ}_{i}").mocapid[0] for i in range(count)]

        # geom sets
        mocap_bodies = {i for i in range(m.nbody) if m.body_mocapid[i] >= 0}
        robot_bodies = self._subtree(self.robot_root) if self.robot_root is not None else set()
        self.mocap_geoms, self.static_geoms, self.drawer_geoms = [], [], []
        self.door_geoms = {dn: [] for dn in DOORWAYS}
        door_body = {m.body(f"door_{dn}").id: dn for dn in DOORWAYS}
        skip = {"floor", "waterbottle", "waterbottle_neck", "waterbottle_cap"}
        for g in range(m.ngeom):
            name = m.geom(g).name
            b = m.geom_bodyid[g]
            if name in skip or name.startswith("floor_") or b in robot_bodies or b == self.bottle_body:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            if b in mocap_bodies:
                self.mocap_geoms.append(g)
            elif b in door_body:
                self.door_geoms[door_body[b]].append(g)
            elif m.body(b).name.startswith("drawer_"):
                self.drawer_geoms.append(g)
            else:
                self.static_geoms.append(g)
        self.all_door_geoms = [g for gs in self.door_geoms.values() for g in gs]
        self.locked_doors: set = set()
        self.door_range = self._fit_door_ranges()

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

    # ---- doors / drawers ------------------------------------------------------
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
        """Trim each door's swing so the leaf never hits walls, furniture or a fully
        open drawer. Returns {door: (lo_rad, hi_rad)} and writes it to jnt_range."""
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
        """Open every drawer to `frac` of its travel (0 = shut, 1 = fully out)."""
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
        """Give every unlocked door that isn't forced closed a random angle."""
        lay = LAYOUTS[self.layout]
        for dn in DOORWAYS:
            if dn in self.locked_doors or dn in lay.get("doors_closed", []):
                self.set_door(dn, 0.0)
            else:
                lo, hi = self.door_range[dn]
                self.set_door(dn, float(self.rng.uniform(lo, hi)))

    def sweeps_clear(self, steps: int = 13):
        """True if no active obstacle sits anywhere in an unlocked door's swing or
        in front of a drawer (so doors can be pushed both ways and drawers pulled
        fully out). Leaves qpos untouched."""
        m, d = self.model, self.data
        active = self.active_mocap_geoms()
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

        lay = LAYOUTS[layout]
        self.locked_doors = set(lay.get("doors_locked", []))
        for dn, eq in self.door_eq.items():
            self.data.eq_active[eq] = dn in self.locked_doors
        self.randomise_doors()

        res, why = None, ""
        for attempt in range(MAX_RESET_TRIES):
            use_jitter = jitter and attempt < MAX_RESET_TRIES - 1
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
            if res.solvable:
                break
            why = "bottle unreachable"
        else:
            raise RuntimeError(f"layout {layout!r} + target {target!r} is not valid even without jitter: {why}")

        self.info = dict(
            layout=layout, target=target, attempts=attempt + 1, path_len=res.path_len,
            doors={dn: round(math.degrees(self.door_angle(dn))) for dn in DOORWAYS},
            locked=sorted(self.locked_doors),
        )
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
        furniture, a drawer, a door (at its current angle), the user, another
        obstacle, or the bottle."""
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
        return bad

    def bottle_geom_id(self):
        return self.model.geom("waterbottle").id

    def occupancy(self) -> np.ndarray:
        """Bool grid (ny, nx): True where the robot centre can't be."""
        m, d = self.model, self.data
        blocked = np.zeros_like(self.GX, dtype=bool)
        locked = [g for dn in self.locked_doors for g in self.door_geoms[dn]]
        for g in self.static_geoms + self.drawer_geoms + locked + self.active_mocap_geoms():
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
            elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE and abs(mat[2, 2]) < 0.01:  # horizontal bar
                ax = mat[:2, 2]
                t = np.clip(dx * ax[0] + dy * ax[1], -size[1], size[1])
                dist = np.hypot(dx - t * ax[0], dy - t * ax[1]) - size[0]
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
        env.set_drawers(0.6)  # show the drawers pulled out
        mujoco.mj_forward(env.model, env.data)
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
        print("  doors (deg): " + ", ".join(f"{k} {v:+d}" + (" LOCKED" if k in i["locked"] else "") for k, v in i["doors"].items()))

    env.reset(layout, target, seed=seed)
    show()

    def on_key(key):
        # SPACE: random layout + target. L / T: cycle layout / target. O: new random door angles.
        if key == 32:
            state["pending"] = ("random",)
        elif key == ord("L"):
            state["pending"] = ("layout",)
        elif key == ord("T"):
            state["pending"] = ("target",)
        elif key == ord("O"):
            state["pending"] = ("doors",)

    print("keys: SPACE random reset | L next layout | T next target | O re-randomise doors")
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
                        env.reset()
                    elif p[0] == "layout":
                        env.reset(lays[(lays.index(env.layout) + 1) % len(lays)], env.target)
                    elif p[0] == "target":
                        env.reset(env.layout, tgts[(tgts.index(env.target) + 1) % len(tgts)])
                    elif p[0] == "doors":
                        env.randomise_doors()
                        mujoco.mj_forward(env.model, env.data)
                        env.info["doors"] = {dn: round(math.degrees(env.door_angle(dn))) for dn in DOORWAYS}
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
