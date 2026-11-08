"""Home scene for the assistive G1: furniture, ArUco-tagged items, a hinged cupboard.

Everything is built with MjSpec on top of the GR00T G1 model, so the layout is plain
Python you can edit. Coordinates are metres in the world frame; the robot starts at
the origin facing +x.

Collisions: everything collides with everything (robot, furniture, door, items), except
  * torso vs the two shoulder links that sit against its hull at the joint, and
  * the grasping (right) hand vs items and the door: the hand is a rigid block, so the palm
    has to be allowed inside what it grasps. The forearm and the rest of the arm still collide.
Grasping is a weld constraint switched on when the palm reaches the item (see skills.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from g1_wbc import ASSET_DIR

MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_DIR = ASSET_DIR / "markers"


@dataclass(frozen=True)
class Station:
    """Where the robot stands to work at a piece of furniture."""

    name: str
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Item:
    name: str
    marker_id: int
    size: tuple[float, float, float]  # box half-sizes
    rgba: tuple[float, float, float, float]
    marker_size: float = 0.05  # edge length of the black square, metres


# Marker IDs (DICT_4X4_50). Jack's house (sim/aruco.py) owns 0, 1, 10-17, 20-34, 40 and 45,
# so ours live in the gaps: items 2-4, cupboard 46/47, furniture anchors 35-39.
ITEMS = {
    "water_bottle": Item("water_bottle", 2, (0.03, 0.03, 0.10), (0.2, 0.5, 0.9, 1)),
    "cup": Item("cup", 3, (0.035, 0.035, 0.05), (0.9, 0.9, 0.85, 1)),
    "pill_box": Item("pill_box", 4, (0.035, 0.03, 0.03), (0.9, 0.3, 0.3, 1)),
}

RIGHT_HAND_BODIES = ["right_wrist_yaw_link"] + [f"right_hand_{f}_link" for f in
                                                ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1",
                                                 "middle_0", "middle_1")]

DOOR_MARKER = 46  # on the cupboard door front
ANCHOR_MARKER = 47  # on the cupboard body, fixed
DOOR_MARKER_SIZE = 0.06

STATIONS = {
    "home": Station("home", 0.0, 0.0, 0.0),
    "table": Station("table", 1.42, 0.0, 0.0),
    "shelf": Station("shelf", 0.0, 1.62, np.pi / 2),
    "cupboard": Station("cupboard", 0.25, -1.55, -np.pi / 2),  # left of the handle: right-arm pull
    "person": Station("person", -0.75, 0.83, 3 * np.pi / 4),
}

# Furniture surfaces (centre x, y, top height, half extents x, y)
TABLE = dict(pos=(2.0, 0.0), top=0.75, half=(0.4, 0.5))
SHELF = dict(pos=(0.0, 2.15), top=0.85, half=(0.5, 0.2))
COUNTER = dict(pos=(0.0, -2.15), top=0.80, half=(0.5, 0.25))
CUPBOARD = dict(width=0.34, height=0.30, depth=0.30)  # box sitting on the counter
SIDE_TABLE = dict(pos=(-0.95, 1.2), top=0.62, half=(0.21, 0.21))  # next to the armchair

# Default item placements: (surface name, x, y). The location memory decides where to look.
DEFAULT_PLACEMENT = {
    "water_bottle": ("table", 1.72, -0.12),
    "cup": ("cupboard", 0.0, -2.12),
    "pill_box": ("shelf", 0.12, 2.0),
}


def marker_png(marker_id: int, px: int = 240, border_frac: float = 0.25) -> Path:
    """Write an ArUco marker with a white quiet zone and return its path."""
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKER_DIR / f"aruco_{marker_id}.png"
    if not path.exists():
        dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
        inner = int(px / (1 + 2 * border_frac))
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, inner)
        pad = (px - inner) // 2
        img = cv2.copyMakeBorder(img, pad, px - inner - pad, pad, px - inner - pad, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return path


def _add_marker_material(spec: mujoco.MjSpec, marker_id: int) -> str:
    name = f"aruco_{marker_id}"
    tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(marker_png(marker_id)))
    del tex
    mat = spec.add_material(name=name)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    return name


def _sticker(body, spec, marker_id: int, size: float, pos, quat=(1, 0, 0, 0), name=None):
    """Thin plate carrying a marker. Its local +z face shows the marker.

    The plate is 1.5x the marker so the white quiet zone is included.
    """
    mat = _add_marker_material(spec, marker_id)
    half = size * 1.5 / 2
    body.add_geom(
        name=name or f"marker_{marker_id}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, 0.001],
        pos=list(pos),
        quat=list(quat),
        material=mat,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def _box(body, size, pos, rgba, contype=3, conaffinity=3, name=None):
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(size),
        pos=list(pos),
        rgba=list(rgba),
        contype=contype,
        conaffinity=conaffinity,
    )


def _table(world, name, pos, top, half, rgba=(0.55, 0.4, 0.25, 1)):
    body = world.add_body(name=name, pos=[pos[0], pos[1], 0])
    _box(body, (half[0], half[1], 0.02), (0, 0, top - 0.02), rgba)
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(body, (0.025, 0.025, (top - 0.04) / 2),
                 (sx * (half[0] - 0.05), sy * (half[1] - 0.05), (top - 0.04) / 2), rgba)
    return body


def add_robot_extras(spec: mujoco.MjSpec) -> None:
    """What every scene adds to the bare GR00T G1: head camera, palm sites, self-collision excludes.

    Shared by build_scene() and the house (house.py) so both worlds give the skills the same robot.
    """
    # Robot head camera (approximately the G1's RealSense).
    torso = spec.body("torso_link")
    pitch = 0.65  # ~37 deg down: sees the floor ~0.6 m ahead up to the top of the cupboard
    torso.add_camera(name="head_cam", pos=[0.08, 0.0, 0.45], fovy=70,
                     xyaxes=[0, -1, 0, np.sin(pitch), 0, np.cos(pitch)])
    # Wrist roll and wrist yaw are two links apart and their meshes touch at the joint.
    for side in ("left", "right"):
        spec.add_exclude(bodyname1=f"{side}_wrist_roll_link", bodyname2=f"{side}_wrist_yaw_link")
    # Arms collide with the torso. Only the two shoulder links that sit against the torso's
    # collision hull at the joint are excluded (0.8-2 cm apart even in the home pose).
    for side in ("left", "right"):
        for link in ("shoulder_pitch", "shoulder_roll"):
            spec.add_exclude(bodyname1="torso_link", bodyname2=f"{side}_{link}_link")
    # Palm site used as the grasp point (fingers extend along +x of the wrist yaw link).
    for side in ("left", "right"):
        spec.body(f"{side}_wrist_yaw_link").add_site(name=f"{side}_palm", pos=[0.10, 0.0, 0.0], size=[0.012])


def build_scene(placement: dict[str, tuple[str, float, float]] | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    placement = placement or DEFAULT_PLACEMENT
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "g1" / "g1.xml"))
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    world = spec.worldbody

    # Floor and light
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.78, 0.72], rgb2=[0.74, 0.70, 0.64], width=300, height=300)
    floor_mat = spec.add_material(name="floor", texrepeat=[8, 8], texuniform=True, specular=0.05, shininess=0.1,
                                  reflectance=0.0)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    # The GR00T model ships its own floor; restyle it and make it collide with items too.
    floor = spec.geom("floor")
    floor.material = "floor"
    floor.contype, floor.conaffinity = 3, 3
    # (the model already has a directional light; add two fill lights)
    world.add_light(pos=[2.0, 0, 2.6], dir=[0, 0, -1], diffuse=[0.2, 0.2, 0.2], specular=[0, 0, 0], cutoff=70)
    world.add_light(pos=[0, 0, 2.6], dir=[0, 0, -1], diffuse=[0.2, 0.2, 0.2], specular=[0, 0, 0], cutoff=70)

    add_robot_extras(spec)
    # A fixed room camera (a home camera / the carer's phone on a shelf) that watches the cupboard.
    world.add_camera(name="room_cam", pos=[-0.75, -1.25, 1.45],
                     xyaxes=_look_at_xyaxes([-0.75, -1.25, 1.45], [0.0, -2.05, 0.95]), fovy=60)

    # Furniture
    _table(world, "table", TABLE["pos"], TABLE["top"], TABLE["half"])
    _table(world, "shelf", SHELF["pos"], SHELF["top"], SHELF["half"], rgba=(0.45, 0.45, 0.5, 1))
    _table(world, "counter", COUNTER["pos"], COUNTER["top"], COUNTER["half"], rgba=(0.85, 0.85, 0.85, 1))
    _table(world, "side_table", SIDE_TABLE["pos"], SIDE_TABLE["top"], SIDE_TABLE["half"], rgba=(0.5, 0.35, 0.25, 1))
    armchair = world.add_body(name="armchair", pos=[-1.5, 1.5, 0])
    _box(armchair, (0.35, 0.35, 0.22), (0, 0, 0.22), (0.35, 0.45, 0.35, 1))
    _box(armchair, (0.08, 0.35, 0.4), (-0.3, 0, 0.6), (0.35, 0.45, 0.35, 1))
    person = world.add_body(name="person", pos=[-1.5, 1.5, 0])
    person.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.1, 0, 0.55, -0.12, 0, 0.95], size=[0.14],
                    rgba=[0.6, 0.5, 0.75, 1], contype=0, conaffinity=0)
    person.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[-0.1, 0, 1.15], size=[0.1],
                    rgba=[0.9, 0.75, 0.65, 1], contype=0, conaffinity=0)

    # Cupboard on the counter: open front faces +y (towards the robot's station).
    cb = CUPBOARD
    cx, cy = COUNTER["pos"]
    z0 = COUNTER["top"]
    cup_body = world.add_body(name="cupboard", pos=[cx, cy, z0])
    wall = (0.75, 0.75, 0.8, 1)
    w, h, dpt = cb["width"] / 2, cb["height"], cb["depth"] / 2
    _box(cup_body, (w, dpt, 0.01), (0, 0, h), wall)  # roof
    _box(cup_body, (0.01, dpt, h / 2), (-w, 0, h / 2), wall)  # left wall
    _box(cup_body, (0.01, dpt, h / 2), (w, 0, h / 2), wall)  # right wall
    _box(cup_body, (w, 0.01, h / 2), (0, -dpt, h / 2), wall)  # back wall
    q_front = _quat_facing(np.array([0, 1, 0]))
    # Anchor marker (rigid with the cupboard) on a stand on the roof, tilted 45 deg towards the
    # room: above the door, so the opening door never hides it from the room camera, and in
    # view of the robot's head camera at the station.
    q_tilt = _quat_facing(np.array([0, 1.0, 1.0]))
    anchor_pos = np.array([w - 0.07, dpt - 0.05, h + 0.045])
    cup_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.055, 0.004],
                      pos=list(anchor_pos - 0.005 * np.array([0, 0.707, 0.707])),
                      quat=q_tilt, rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
    _box(cup_body, (0.04, 0.01, 0.03), (w - 0.07, dpt - 0.08, h + 0.03), (0.75, 0.75, 0.8, 1))  # stand
    _sticker(cup_body, spec, ANCHOR_MARKER, DOOR_MARKER_SIZE, anchor_pos, q_tilt)

    # Door: hinged on the cupboard's -x edge (the robot's RIGHT when it faces the cupboard),
    # handle near the +x edge. Positive angle swings it open towards the robot and to its
    # right, so a right-handed pull never crosses the body and the open door ends up beside
    # the robot's right shoulder, clear of the opening.
    door = cup_body.add_body(name="cupboard_door", pos=[-w, dpt + 0.012, 0])
    door.add_joint(name="door_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                   range=[0, 1.9], damping=0.3, frictionloss=0.05)
    # panel clears the counter by 5 mm so it swings freely
    _box(door, (w - 0.003, 0.01, h / 2 - 0.008), (w, 0, h / 2), (0.6, 0.45, 0.3, 1), name="door_panel")
    hx = 2 * w - 0.05
    door.add_geom(name="door_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.012],
                  fromto=[hx, 0.04, h / 2 - 0.06, hx, 0.04, h / 2 + 0.06],
                  rgba=[0.2, 0.2, 0.2, 1], contype=3, conaffinity=3)
    for dz in (-0.05, 0.05):
        _box(door, (0.008, 0.02, 0.008), (hx, 0.02, h / 2 + dz), (0.2, 0.2, 0.2, 1), contype=0, conaffinity=0)
    door.add_site(name="door_handle_site", pos=[hx, 0.04, h / 2])
    _sticker(door, spec, DOOR_MARKER, DOOR_MARKER_SIZE, (w - 0.02, 0.0105, h / 2 + 0.06), q_front)

    # Items (free bodies) with a marker on top
    surfaces = {"table": TABLE["top"], "shelf": SHELF["top"], "cupboard": z0 + 0.0, "counter": z0, "floor": 0.0,
                "side_table": SIDE_TABLE["top"]}
    for name, (surface, x, y) in placement.items():
        it = ITEMS[name]
        body = world.add_body(name=name, pos=[x, y, surfaces[surface] + it.size[2] + 0.002])
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(it.size), rgba=list(it.rgba),
                      mass=0.15, contype=3, conaffinity=3, friction=[1.0, 0.01, 0.001])
        _sticker(body, spec, it.marker_id, it.marker_size, (0, 0, it.size[2] + 0.001))
        # Weld used for grasping (inactive until the hand closes on the item)
        spec.add_equality(name=f"grasp_{name}", type=mujoco.mjtEq.mjEQ_WELD, name1="right_wrist_yaw_link",
                          name2=name, objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
    spec.add_equality(name="grasp_door", type=mujoco.mjtEq.mjEQ_WELD, name1="right_wrist_yaw_link",
                      name2="cupboard_door", objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
    # The grasping hand may overlap what it grasps (rigid hand): exclude hand vs items and door.
    for target in list(placement) + ["cupboard_door"]:
        for hb in RIGHT_HAND_BODIES:
            spec.add_exclude(bodyname1=hb, bodyname2=target)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _look_at_xyaxes(eye, target) -> list[float]:
    """xyaxes for a MuJoCo camera at `eye` looking at `target` with world-up roughly up."""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x, *y]


def _quat_facing(normal: np.ndarray) -> list[float]:
    """Quaternion rotating local +z onto `normal` (for stickers on vertical faces)."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(z, n)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return [1, 0, 0, 0] if n[2] > 0 else [0, 1, 0, 0]
    angle = np.arctan2(s, z @ n)
    axis /= s
    return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]
