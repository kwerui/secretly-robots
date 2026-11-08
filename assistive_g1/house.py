"""The G1 in Jack's house (sim/house_env.py), with everything our skills need added on top.

G1House subclasses HouseEnv and hooks its `_attach_robot`, which runs on Jack's MjSpec
*before* he compiles. So his file is untouched: the house, doors, drawers, modes, grid and
cameras are exactly his, and we only add:

  * the robot extras every scene gets (scene.add_robot_extras): torso head camera, palm
    sites, self-collision excludes,
  * our ArUco items (cup, pill box) and a top marker on his water bottle,
  * a fixed anchor marker on each chest of drawers (his drawer fronts already carry markers
    20-29, which become the moving "part" marker of the articulation demo),
  * position-only grasp welds for the drawer fronts (the handle grip) and the matching
    hand/drawer exclusions; items are NOT welded or excluded: the Dex3 fingers (hand.py, added
    by add_robot_extras) pinch them physically,
  * a room camera that watches the kitchen drawers for the carer demonstration.

Physics: the G1's own Euler integrator, not the house's implicitfast (the GR00T policies
were run against the G1 file's options).

Navigation: Jack's occupancy grid (robot radius ROBOT_R, unlocked doors are passable: the
robot pushes them) and his BFS + line-of-sight planner, followed by our Navigator.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))  # repo root, so `sim` imports

from sim import house_env as H  # noqa: E402
from sim.house_env import HouseEnv, DRAWER_UNITS, FURNITURE, ROBOT_START  # noqa: E402

from g1_wbc import ASSET_DIR  # noqa: E402
from hand import item_contact_params  # noqa: E402
from scene import (ITEMS, Item, Station, RIGHT_HAND_BODIES, DOOR_MARKER_SIZE, add_robot_extras,  # noqa: E402
                   grasp_solver_options, _sticker, _box, _look_at_xyaxes, _quat_facing)

G1_XML = ASSET_DIR / "g1" / "g1.xml"

# Jack's drawer-front markers: 0.084 m plate, 256 px image with a 24 px quiet zone
JACK_DRAWER_MARKER = 0.084 * (256 - 2 * 24) / 256
# Our fixed anchors on his chests of drawers (IDs outside his registry)
CHEST_ANCHORS = {"kitchen_drawers": 35, "dresser": 36, "filing_cabinet": 37}
ANCHOR_SIZE = DOOR_MARKER_SIZE

# Items in the house. The water bottle is Jack's (body "waterbottle", cylinder r 3.3 cm,
# 21 cm tall, his marker 0 on the side); we add our marker 2 on its cap so the same top-marker
# pick works. Cup and pill box are ours.
HOUSE_ITEMS = {
    "water_bottle": Item("water_bottle", ITEMS["water_bottle"].marker_id, (H.BOTTLE_R, H.BOTTLE_R, H.BOTTLE_H / 2),
                         (0.18, 0.48, 0.95, 1)),
    "cup": ITEMS["cup"],
    "pill_box": ITEMS["pill_box"],
}
ITEM_BODY = {"water_bottle": "waterbottle", "cup": "cup", "pill_box": "pill_box"}

# Coffee table moved towards the bedroom wall (and clear of the living-bedroom door swing) so the
# robot can stand in front of the user's tray
COFFEE_TABLE_POS = (-2.80, -1.15)

# Where the user's delivery tray is (Jack's seated user, tray geom "user_delivery_tray")
_ux, _uy = FURNITURE["sofa"]["pos"][0] - 0.12, FURNITURE["sofa"]["pos"][1]
_seat = FURNITURE["sofa"]["h"]
DELIVERY = dict(pos=(_ux + 0.38, _uy - 0.36), top=_seat + 0.16 + 0.015, half=(0.15, 0.15))


@dataclass(frozen=True)
class Drawer:
    unit: str
    index: int
    body: str  # drawer body (moves)
    joint: str
    handle_site: str
    front_marker: int  # Jack's marker on the drawer front
    anchor_marker: int  # ours, fixed on the chest


def drawer(unit: str, index: int) -> Drawer:
    u = DRAWER_UNITS[unit]
    name = f"drawer_{unit}_{index}"
    return Drawer(unit, index, name, f"{name}_slide", f"{name}_handle", u["marker_start"] + index, CHEST_ANCHORS[unit])


KITCHEN_TOP = drawer("kitchen_drawers", 2)


def _station_facing(target_xy, yaw, standoff, lateral=0.15) -> Station:
    """Pelvis pose for working on `target_xy`: facing `yaw`, `standoff` behind it, the target in
    line with the right shoulder (pelvis `lateral` to its left)."""
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    p = np.asarray(target_xy) - standoff * fwd + lateral * left
    return p, yaw


def default_stations() -> dict[str, Station]:
    out = {}

    def add(name, target_xy, yaw, standoff, lateral=0.15):
        p, y = _station_facing(target_xy, yaw, standoff, lateral)
        out[name] = Station(name, float(p[0]), float(p[1]), float(y))

    out["home"] = Station("home", ROBOT_START[0], ROBOT_START[1], math.radians(ROBOT_START[2]))
    # kitchen counter: Jack's bottle target spot, facing the back wall (-y)
    add("kitchen_counter", H.SURFACE_TARGETS["kitchen_counter"][:2], -math.pi / 2, 0.75)
    # kitchen drawers: in front of the top drawer handle, facing -y
    u = DRAWER_UNITS["kitchen_drawers"]
    handle_y = u["pos"][1] + u["size"][1] / 2 + H.HANDLE_STANDOFF
    add("kitchen_drawers", (u["pos"][0], handle_y), -math.pi / 2, 0.45)
    # kitchen table: approach from the living-room side (-x), facing +x
    kt = FURNITURE["kitchen_table"]
    add("kitchen_table", (kt["pos"][0] - kt["size"][0] / 2 + 0.15, kt["pos"][1]), 0.0, 0.55)
    # the user: in front of the delivery tray, facing the sofa (-x)
    add("person", DELIVERY["pos"], math.pi, 0.42)
    return out


class G1House(HouseEnv):
    def __init__(self, seed: int | None = 0, placement: dict | None = None):
        # Jack's bottle goes where his reset() puts it (default target: the top kitchen drawer,
        # where it stands up tall enough to be seen over the drawer front); ours on surfaces.
        self.placement = placement or {"cup": ("surface", "kitchen_counter"), "pill_box": ("surface", "kitchen_table")}
        super().__init__(robot_xml=str(G1_XML), seed=seed)
        m = self.model
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        grasp_solver_options(m.opt)  # elliptic cones, impratio, Newton: the pinch holds by friction
        # Grasp contacts on Jack's bottle (his file is untouched: set on the compiled model) and
        # on our items (set at build time in _attach_robot).
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name == "waterbottle" and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_condim[g] = 4
                m.geom_friction[g] = [1.2, 0.02, 0.001]
                m.geom_solref[g] = [0.01, 1.0]
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1280)
        m.vis.global_.offheight = max(m.vis.global_.offheight, 720)
        # Collision bits: house geoms 1/1 -> 3/3, so a held item (2/2) still hits furniture but
        # not the robot (G1 geoms stay 1/1).
        robot = self._subtree(self.robot_root)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in robot:
                continue
            if m.geom_contype[g] == 1 and m.geom_conaffinity[g] == 1:
                m.geom_contype[g] = m.geom_conaffinity[g] = 3
        self.items = HOUSE_ITEMS
        self.item_body = ITEM_BODY
        self.stations = default_stations()
        self.delivery = DELIVERY
        self.marker_sizes = {d: JACK_DRAWER_MARKER for u in DRAWER_UNITS.values()
                             for d in range(u["marker_start"], u["marker_start"] + u["n"])}
        self.marker_sizes.update({mid: ANCHOR_SIZE for mid in CHEST_ANCHORS.values()})
        self.marker_labels = {mid: f"{u}_anchor" for u, mid in CHEST_ANCHORS.items()}
        for un, u in DRAWER_UNITS.items():
            for i in range(u["n"]):
                self.marker_labels[u["marker_start"] + i] = f"{un}_{i}"
        self.graspable = [f"drawer_{un}_{i}" for un, u in DRAWER_UNITS.items() for i in range(u["n"])]

    # ------------------------------------------------------------------ build hook
    def _attach_robot(self, spec: mujoco.MjSpec, robot_xml: str):
        super()._attach_robot(spec, robot_xml)
        add_robot_extras(spec)
        world = spec.worldbody

        # Marker on the bottle's cap (ours, ID 2); Jack's side badge (ID 0) stays.
        bottle = spec.body("waterbottle")
        _sticker(bottle, spec, ITEMS["water_bottle"].marker_id, ITEMS["water_bottle"].marker_size,
                 (0, 0, H.BOTTLE_H + 0.001), name="waterbottle_top_marker")
        # Our free items (placed in reset())
        for name in ("cup", "pill_box"):
            it = ITEMS[name]
            body = world.add_body(name=name, pos=[0, 0, -3.0 - 0.3 * len(name)])
            body.add_freejoint(name=f"{name}_free")
            g = body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(it.size),
                              rgba=list(it.rgba), mass=0.15, contype=3, conaffinity=3)
            item_contact_params(g)
            _sticker(body, spec, it.marker_id, it.marker_size, (0, 0, it.size[2] + 0.001))

        # Anchor marker on each chest: on a small stand on the top, near the front-right corner,
        # tilted 45 deg towards the room (like the cupboard anchor), so the robot sees it from the
        # station and the room camera sees it during the demo. Chest frame: +x = front.
        for unit, mid in CHEST_ANCHORS.items():
            u = DRAWER_UNITS[unit]
            chest = spec.body(f"chest_{unit}")
            d, w, h = u["size"][1], u["size"][0], u["h"]
            q_tilt = _quat_facing(np.array([1.0, 0.0, 1.0]))
            pos = np.array([d / 2 - 0.06, -(w / 2 - 0.09), h + 0.05])
            _box(chest, (0.01, 0.04, 0.03), (d / 2 - 0.09, -(w / 2 - 0.09), h + 0.03), (0.75, 0.75, 0.8, 1),
                 contype=0, conaffinity=0)
            chest.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.055, 0.004],
                           pos=list(pos - 0.005 * np.array([0.707, 0, 0.707])), quat=q_tilt,
                           rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
            _sticker(chest, spec, mid, ANCHOR_SIZE, pos, q_tilt, name=f"chest_{unit}_anchor")

        # Handle-grip welds for the drawer fronts (inactive until open_articulated switches one
        # on) and the matching hand/drawer exclusions. Items get neither: the fingers hold them.
        targets = []
        for un, u in DRAWER_UNITS.items():
            for i in range(u["n"]):
                b = f"drawer_{un}_{i}"
                targets.append(b)
                spec.add_equality(name=f"grasp_{b}", type=mujoco.mjtEq.mjEQ_WELD, name1="right_wrist_yaw_link",
                                  name2=b, objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
        for t in targets:
            for hb in RIGHT_HAND_BODIES:
                spec.add_exclude(bodyname1=hb, bodyname2=t)

        # The carer-demo camera: on the kitchen wall shelf, looking at the drawer unit
        u = DRAWER_UNITS["kitchen_drawers"]
        eye = [u["pos"][0] + 0.9, u["pos"][1] + 1.3, 1.45]
        look = [u["pos"][0], u["pos"][1] + 0.2, 0.75]
        world.add_camera(name="drawer_cam", pos=eye, xyaxes=_look_at_xyaxes(eye, look), fovy=55)

    # ------------------------------------------------------------------ reset
    def reset(self, layout: str = "tidy", target: str = "kitchen_drawer", seed: int | None = None,
              jitter: bool = False, mode: str = "static"):
        info = super().reset(layout, target, seed=seed, jitter=jitter, mode=mode)
        if mode == "static":
            self._set_furniture("coffee_table", *COFFEE_TABLE_POS, 0.0)
        self.set_drawers(0.0)
        mujoco.mj_kinematics(self.model, self.data)
        for name, (kind, where) in self.placement.items():
            self.place_item(name, kind, where)
        mujoco.mj_forward(self.model, self.data)
        return info

    def place_item(self, name: str, kind: str, where) -> None:
        it = self.items[name]
        m, d = self.model, self.data
        if kind == "surface":
            x, y, z = self.surface_xyz(where)
            z += it.size[2] + 0.002
        elif kind == "drawer":
            dr: Drawer = where
            b = m.body(dr.body).id
            R = d.xmat[b].reshape(3, 3)
            floor = d.geom_xpos[m.geom(f"{dr.body}_floor").id]
            # 6 cm behind the inside of the drawer front, centred: close to the opening, so the
            # robot reaches it once the drawer is out
            u = DRAWER_UNITS[dr.unit]
            front_in = u["size"][1] / 2 - 0.02
            local = np.array([front_in - 0.07, 0.0, 0.0])
            x, y, _ = d.xpos[b] + R @ local
            z = floor[2] + m.geom_size[m.geom(f"{dr.body}_floor").id][2] + it.size[2] + 0.002
        else:
            x, y, z = where
        j = m.joint(f"{name}_free").id
        a = m.jnt_qposadr[j]
        d.qpos[a:a + 7] = [x, y, z, 1, 0, 0, 0]
        d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0

    # ------------------------------------------------------------------ navigation
    def plan_path(self, start_xy, goal_xy, goal_radius: float = 0.25) -> list[tuple[float, float]] | None:
        """Jack's planner on his robot occupancy grid: list of (x, y) from start into the free
        cells within `goal_radius` of goal_xy (the final approach to an exact station pose is
        done by the caller, since stations sit closer to furniture than the grid's inflation)."""
        # Prefer paths with more clearance than Jack's ROBOT_R (his doorway band is only
        # +-0.22 m for a 0.28 m radius, and the walking G1 sways ~0.1 m), falling back to his.
        locked = [g for dn in self.locked_doors for g in self.door_geoms[dn]]
        clear = self._clearance(self.static_geoms + self.drawer_geoms + locked + self.active_mocap_geoms(), H.ROBOT_H)
        for r in (H.ROBOT_R + 0.12, H.ROBOT_R + 0.07, H.ROBOT_R):
            path = self._plan_on(clear <= r, start_xy, goal_xy, goal_radius)
            if path is not None:
                return self._through_doorways(path)
        return None

    def _through_doorways(self, path, lead: float = 0.65):
        """Make every doorway crossing square: a waypoint `lead` before the doorway centre and
        one `lead` after it, on the line through the centre perpendicular to the wall. The robot
        then meets a swung leaf straight on and pushes it round instead of being deflected
        into the jamb (the leaf swings both ways)."""
        out = [path[0]]
        for p, q in zip(path[:-1], path[1:]):
            p, q = np.asarray(p, float), np.asarray(q, float)
            for dn, dw in H.DOORWAYS.items():
                x0, y0, x1, y1 = H.WALLS[dw["wall"]]
                c = np.asarray(dw["center"], float)
                u = np.array([x1 - x0, y1 - y0], float)
                u /= np.linalg.norm(u)
                nrm = np.array([-u[1], u[0]])
                sp, sq = (p - c) @ nrm, (q - c) @ nrm
                if sp * sq >= 0:
                    continue  # doesn't cross this wall line
                t = sp / (sp - sq)
                hit = p + t * (q - p)
                if abs((hit - c) @ u) > H.DOOR_W / 2:
                    continue  # crosses the wall elsewhere (not through this doorway)
                side = np.sign(sp)
                # drop points already inside the approach zone on this side (they'd make the
                # robot double back), then pre/post on the centre line
                while len(out) > 1 and 0 < side * ((np.asarray(out[-1]) - c) @ nrm) < lead \
                        and abs((np.asarray(out[-1]) - c) @ u) < lead + H.DOOR_W:
                    out.pop()
                out += [tuple(c + side * lead * nrm), tuple(c - side * lead * nrm)]
                if 0 < -side * sq < lead and abs((q - c) @ u) < lead + H.DOOR_W:
                    q = None  # inside the exit zone: the post waypoint replaces it
                break
            if q is not None:
                out.append(tuple(q))
        if tuple(path[-1]) != out[-1]:
            out.append(tuple(path[-1]))
        return [(float(x), float(y)) for x, y in out]

    def _plan_on(self, grid, start_xy, goal_xy, goal_radius):
        start = tuple(start_xy)
        i, j = self._cell(*start)
        if grid[i, j]:
            # standing close to furniture (e.g. at a station): start from the nearest free cell
            free = np.argwhere(~grid)
            k = np.argmin(np.hypot(self.gx[free[:, 1]] - start[0], self.gy[free[:, 0]] - start[1]))
            start = (float(self.gx[free[k, 1]]), float(self.gy[free[k, 0]]))
        goal_mask = self._near(grid, goal_xy, goal_radius)
        if not goal_mask.any():
            goal_mask = self._near(grid, goal_xy, goal_radius + 0.2)
        path = self._plan_xy(grid, start, goal_mask)
        if path is not None and start != tuple(start_xy):
            path = [tuple(start_xy)] + path
        return path

    def drawer_open_m(self, dr: Drawer) -> float:
        j = self.model.joint(dr.joint).id
        return float(self.data.qpos[self.model.jnt_qposadr[j]])
