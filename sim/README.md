# House simulation (MuJoCo)

The house the fetch robot works in. It's an 8 × 6 m flat with four rooms (living room, kitchen, bedroom, study) connected by four doorways in a loop. A user sits on the living-room sofa, and a blue water bottle stands on one of five surfaces. Clutter and doors move on every reset.

Everything lives in this folder and is driven by one file, [`house_env.py`](house_env.py).

## Run it

```bash
cd sim
./setup.sh      # makes .venv, installs mujoco/numpy/pillow/pytest, checks every layout, renders snapshots
./run_sim.sh    # interactive viewer (uses mjpython on macOS, which the viewer needs)
```

Viewer options and keys:

```bash
./run_sim.sh --layout door_shut --target study_desk --seed 3
```

| Key   | Action                                       |
|-------|----------------------------------------------|
| SPACE | new random layout + bottle position          |
| L     | next layout (same bottle position)           |
| T     | next bottle position (same layout)           |
| O     | flip every door (open ↔ shut)                |

Other commands (run them from `sim/` with `.venv/bin/python`):

```bash
.venv/bin/python house_env.py --check       # table of every layout × target: solvable? path length?
.venv/bin/python house_env.py --snapshot    # PNGs in sim/snapshots/ (overview, 4 room cams, occupancy grid)
.venv/bin/python house_env.py --write-xml   # dump the scene to sim/house.xml
.venv/bin/python -m pytest -q               # tests
```

## Layouts

| Layout           | Doors shut       | Clutter                                       | What it tests                                  |
|------------------|------------------|-----------------------------------------------|------------------------------------------------|
| `tidy`           | none             | none                                          | baseline                                       |
| `laundry_day`    | none             | laundry basket, 3 clothes piles, bag, chair   | narrow gaps, small obstacles                   |
| `door_shut`      | living ↔ kitchen | chair, box                                    | must go round via bedroom + study              |
| `moving_house`   | none             | 8 cardboard boxes in every room               | heavy clutter, some doorways semi-blocked      |
| `long_way_round` | living ↔ bedroom | basket, clothes, bag, chair                   | longest detour (9.25 m to the bedside table)   |

On each reset every obstacle is jittered by ±15 cm and ±10°. A jittered layout is only kept if no obstacle penetrates a wall, furniture or another obstacle, and the robot can still drive from its start pose to within reach of the bottle. Otherwise the jitter is resampled, and after 50 failed tries the un-jittered layout is used (the tests check that one is always valid).

Bottle positions (`TARGETS`): `tv_unit`, `kitchen_counter`, `kitchen_table`, `bedside_table`, `study_desk`.

## Changing things

All the settings are plain dicts and constants at the top of `house_env.py`, in the `CONFIG` section.

**Move or add clutter.** Edit a layout in `LAYOUTS`. Each obstacle is `(type, x, y, yaw_deg)`, and the types come from `OBSTACLE_TYPES`. Coordinates are in metres: the house spans x ∈ [-4, 4] and y ∈ [-3, 3]. The living room is at bottom-left (−x, −y), the kitchen bottom-right, the bedroom top-left and the study top-right. The robot starts at `ROBOT_START = (-1.8, -1.0)`.

**Add a layout.** Add a new key to `LAYOUTS`:

```python
"chaos": dict(
    doors_shut=["kitchen_study"],
    obstacles=[("box_large", 1.0, -0.5, 30), ("bag", -1.0, 1.0, 0)],
),
```

Then run `--check`. It lists any combination that's blocked or overlapping, and `--snapshot` writes a `<layout>__grid.png` showing what the planner sees: grey is blocked, green is the start, blue is the bottle and the area within reach of it.

**Add a bottle position.** Add `"name": (x, y, "furniture_name")` to `TARGETS`. The z height comes from that furniture's `h`, so the point must lie on its top.

**New obstacle shape.** Add an entry to `OBSTACLE_TYPES` made of `("box", (hx, hy, hz), (px, py, pz))` / `("cyl", (r, hz), (px, py, pz))` geoms. Sizes are half-extents and the body origin is on the floor.

**Furniture / walls / doors.** These are set in `FURNITURE`, `WALLS` and `DOORWAYS`. Doorways cut gaps into walls automatically. `hinge` picks the jamb, and `swing` picks the room the open leaf swings into.

### Tunables

| Name                       | Default | Meaning                                                           |
|----------------------------|---------|-------------------------------------------------------------------|
| `ROBOT_R`                  | 0.30 m  | robot footprint radius used to inflate obstacles in the checker  |
| `ROBOT_H`                  | 0.80 m  | geoms whose underside is above this don't block the robot         |
| `REACH`                    | 0.80 m  | robot centre must get this close (xy) to the bottle               |
| `ROBOT_START`              | (-1.8, -1.0, 0°) | start / delivery pose in front of the user               |
| `DOOR_W`                   | 1.00 m  | doorway width                                                     |
| `WALL_H`                   | 1.20 m  | wall height (low so the angled cameras can see in)                |
| `JITTER_XY`, `JITTER_YAW`  | 0.15 m, 10° | per-reset randomisation                                       |
| `GRID_RES`                 | 0.05 m  | occupancy grid cell size                                          |

The Aliengo + Z1 body is about 0.65 × 0.30 m, so its circumscribed radius is about 0.38 m. `ROBOT_R = 0.30` assumes it turns carefully in doorways. Raise it to 0.38 for a conservative check, and widen doorways or move furniture if `--check` then fails.

## Using it from code

```python
from house_env import HouseEnv
env = HouseEnv(seed=0)                        # or HouseEnv(robot_xml=".../aliengo_z1.xml")
info = env.reset()                            # random layout + target; or reset("door_shut", "study_desk")
env.step(500)
env.bottle_xyz(), env.data.site("waterbottle_site").xpos
env.check().grid                              # bool occupancy grid (True = blocked), useful for a planner
```

Names follow the robot repo's MuJoCo scene: body `waterbottle` with site `waterbottle_site`, whose world orientation matches theirs. The user's hand-over point is the site `user_hand`. The cameras are `overview` (top-down) and `cam_living`, `cam_kitchen`, `cam_bedroom`, `cam_study`.

## Adding the robot

```bash
./run_sim.sh --robot ../trash-collection-isaaclab/deploy/mujoco/models/aliengo_z1.xml
```

The robot MJCF is attached with `MjSpec.attach`, so mesh paths resolve relative to the robot file and its names (`trunk`, keyframes `home`/`down`) are kept unchanged. Any floor plane or light in the robot file is dropped. The root free joint is auto-detected (override it with `--robot-joint`). The robot is placed at `ROBOT_START`, at the height from its `home` keyframe. No controller is attached here; that's the robot team's code.

**Open question for the team:** the walking policy was trained in Isaac Lab. The robot repo already ships a MuJoCo sim-to-sim runner (`deploy/play_mujoco.py`, MuJoCo 3.7), so it should transfer, but nobody has run it in this house yet. That repo pins MuJoCo 3.7. This folder works with any MuJoCo ≥ 3.3 and was tested on 4.0.
