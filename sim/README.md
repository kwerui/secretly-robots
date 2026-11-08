# House simulation (MuJoCo)

The house the assistive fetch robot works in. It's an 8 × 6 m flat with four rooms (living room, kitchen, bedroom, study) connected by four doorways in a loop. A user with limited mobility sits on the living-room sofa with a dedicated delivery station, and a target water bottle is placed on furniture surfaces or inside drawers.

- **Three Testing Modes / Variables**:
  - `static` (Level 1): Baseline static house layout, nominal furniture, static seated user.
  - `shuffled` (Level 2): The coffee table, kitchen table, bedside table and desk are moved (up to ±0.4 m, ±20°) on every reset, and target items are placed on tabletops, counters, or inside closed drawers.
  - `dynamic` (Level 3): A dynamic person walks into the house, approaches an item, and relocates it, testing obstacle avoidance and adaptive replanning.
- **Doors are push-only**: Hinged leaves with wide stainless steel push plates on both faces and ArUco markers. No knobs or turn latches; the robot pushes through from either side.
- **Chests of drawers**: Sliding drawers with long bar handles (spanning ~75% of chest width) for easy grasping, ArUco markers on each drawer front, and full support for target items stored inside drawer trays.
- **Visual Revamp**: Photorealistic procedural textures (oak parquet flooring, ceramic slate tiles, woven loop carpet, wall plaster, marble countertops, natural wood grain, sofa fabric, brushed stainless steel, and Scandinavian geometric rug) with warm recessed room downlights.
- **Realistic People**: Detailed anatomical models for the seated person with limited mobility and the dynamic walking adult human.
- **ArUco Markers**: Standard OpenCV `DICT_4X4_50` markers on the target bottle, door push plates, drawer fronts, and user delivery station.
- **Cameras**: Full dollhouse `overview`, four room cameras (`cam_living`, `cam_kitchen`, `cam_bedroom`, `cam_study`), and a first-person robot head camera (`cam_robot_head`). The head camera follows the attached robot's root body every `step()` (its x, y and heading, at height `HEAD_CAM_Z`). Without a robot it stays at `ROBOT_START`.

Everything lives in this folder and is driven by [`house_env.py`](house_env.py).

## Run it

```bash
cd sim
./setup.sh      # makes .venv, installs mujoco/numpy/pillow/pytest/opencv, checks every layout, renders snapshots
./run_sim.sh    # interactive viewer (uses mjpython on macOS, which the viewer needs)
```

Viewer options and modes:

```bash
./run_sim.sh --mode static                       # Level 1: Static house
./run_sim.sh --mode shuffled                     # Level 2: Shuffled furniture & drawer targets
./run_sim.sh --mode dynamic                      # Level 3: Dynamic human walking and moving bottle
./run_sim.sh --layout door_shut --target study_desk --seed 3
```

| Key   | Action                                                   |
|-------|----------------------------------------------------------|
| SPACE | new random layout + bottle position                      |
| 1     | switch to Mode 1 (Static House)                          |
| 2     | switch to Mode 2 (Shuffled Furniture & Drawers)          |
| 3     | switch to Mode 3 (Dynamic Walking Person Relocating Item)|
| L     | next layout (same bottle position)                       |
| T     | next target position (surfaces + drawer targets)         |
| O     | new random angles for the unlocked doors                 |
| V     | person on the sofa asks for water (plays a voice clip)   |

## Person's voice (Gemini text-to-speech)

Press **V** in the viewer and the person on the sofa says one of eight phrases that all mean "get me water" ("I am thirsty.", "I need water.", "I'm parched.", ...). The clip plays through your speakers, a caption appears at the bottom of the viewer, and the line is recorded in `env.info["last_utterance"]` with its sim time. The list is `PHRASES` in [`person_voice.py`](person_voice.py).

The clips are made once with Gemini text-to-speech and saved in `assets/voice/`, so the sim needs no network or key to play them. To make or remake them:

```bash
.venv/bin/pip install -r requirements-voice.txt
export GEMINI_API_KEY='your-key'                 # shell only, never in a file
.venv/bin/python person_voice.py --generate      # only makes missing clips; --force remakes all
.venv/bin/python person_voice.py --say           # play one from the terminal
.venv/bin/python person_voice.py                 # list which clips exist
```

The model defaults to `gemini-3.8-flash-tts` with the prebuilt voice `Gacrux`. Override them with `--model`/`--voice` or `GEMINI_TTS_MODEL`/`GEMINI_TTS_VOICE`. This is the *person's* voice only. The robot's listening runs onboard and has to work from the audio, not from the phrase text.

To push a door or pull a drawer yourself: double-click it, then hold Ctrl and right-drag.

Other commands (run them from `sim/` with `.venv/bin/python`):

```bash
.venv/bin/python house_env.py --check       # table of every layout × target: solvable? path length?
.venv/bin/python house_env.py --snapshot    # PNGs in sim/snapshots/ (overview, room cams, robot head cam, dynamic)
.venv/bin/python house_env.py --write-xml   # dump the scene to sim/house.xml
.venv/bin/python -m pytest -q               # tests
```

## Challenge Levels / Test Variables

| Mode       | Level | Description | What it tests |
|------------|-------|-------------|---------------|
| `static`   | 1     | Nominal furniture, static seated user, push doors, closed drawers | Baseline fetch task |
| `shuffled` | 2     | Movable furniture (`movable=True` in `FURNITURE`) shifted and rotated each reset, with the bottle on it; jittered clutter; bottle can be placed on surfaces or **inside drawers** | Adapting to relocated furniture, finding items in drawers, pulling long handles |

Each piece is re-drawn until it's clear of walls, other furniture, drawers and door swings. The whole layout is then checked like clutter: the bottle must be reachable and no doorway may be cut off. `info["furniture"]` gives each piece's `(x, y, yaw_deg)`. The dresser is a 1.10 m tallboy so the bottle stands upright in every drawer. Pull drawers steadily (about 10 N): the bottle stays upright, but a yank that slams the drawer into its end stop knocks it over.
| `dynamic`  | 3     | A person walks in from another room, picks up the water bottle, carries it to another surface and walks back out | Obstacle avoidance (`person_too_close`), dynamic replanning when item moves mid-run |

The person's route is planned on the occupancy grid at every reset, after the clutter is placed, so they walk round furniture, clutter, open door leaves and the robot instead of through them. If half-open doors cut off every route, they push the unlocked ones fully open first. The bottle goes to the first surface in `PERSON_DROPS` that isn't where it started and that both the person and the robot can reach (`info["person_drop"]`). Once it's put down it's an ordinary free body again, so the robot can pick it up. Tunables: `PERSON_R`, `PERSON_SPEED`, `PERSON_REACH`, `PERSON_ENTRIES`, `PERSON_DROPS`.

## ArUco Marker Registry

Pre-generated into `sim/assets/aruco/` using standard OpenCV `DICT_4X4_50`:

| Marker ID | Name | Entity |
|-----------|------|--------|
| **0** | `target_bottle` | Water bottle target item badge |
| **1** | `target_medicine` | Secondary medicine target item |
| **10 / 11** | `door_living_kitchen_a/b` | Living ↔ Kitchen push plates (Side A / Side B) |
| **12 / 13** | `door_bedroom_study_a/b` | Bedroom ↔ Study push plates (Side A / Side B) |
| **14 / 15** | `door_living_bedroom_a/b`| Living ↔ Bedroom push plates (Side A / Side B) |
| **16 / 17** | `door_kitchen_study_a/b` | Kitchen ↔ Study push plates (Side A / Side B) |
| **20–23** | `drawer_dresser_0..3` | Bedroom dresser sliding drawers (bottom to top) |
| **24–26** | `drawer_kitchen_0..2` | Kitchen drawer unit sliding drawers (bottom to top) |
| **27–29** | `drawer_filing_0..2` | Study filing cabinet sliding drawers (bottom to top) |
| **30–34** | `anchor_*` | Table / counter / desk location anchors |
| **40** | `user_station` | Seated user delivery tray station |
| **45** | `dynamic_person` | Dynamic walking human tracking badge |

Perception helper:
```python
from sim.aruco import detect_markers
dets = detect_markers(rgb_image)  # returns [{'id': int, 'name': str, 'kind': str, 'center': (x,y), ...}]
```

## Layouts

| Layout           | Doors locked     | Clutter                                       | What it tests                                  |
|------------------|------------------|-----------------------------------------------|------------------------------------------------|
| `tidy`           | none             | none                                          | baseline                                       |
| `laundry_day`    | none             | laundry basket, 3 clothes piles, bag, chair   | narrow gaps, small obstacles                   |
| `door_shut`      | living ↔ kitchen | chair, box                                    | must go round via bedroom + study              |
| `moving_house`   | none             | 8 cardboard boxes in every room               | heavy clutter, some doorways semi-blocked      |
| `long_way_round` | living ↔ bedroom | basket, clothes, bag, chair                   | longest detour (9.10 m to the bedside table)   |

`moving_house` also starts the bedroom ↔ study door closed. That door isn't locked, so the robot can push it open.

## Target Locations (`ALL_TARGETS`)

- **Surfaces**: `tv_unit`, `kitchen_counter`, `kitchen_table`, `bedside_table`, `study_desk`, `coffee_table`
- **Drawers**: `dresser_drawer` (top dresser drawer), `kitchen_drawer` (top kitchen drawer), `filing_drawer` (top filing drawer)

### Tunables

| Name                       | Default | Meaning                                                           |
|----------------------------|---------|-------------------------------------------------------------------|
| `ROBOT_R`                  | 0.28 m  | humanoid robot footprint radius used to inflate obstacles in grid |
| `ROBOT_H`                  | 1.65 m  | humanoid standing height                                          |
| `REACH`                    | 0.85 m  | humanoid robot arm reach envelope                                 |
| `ROBOT_START`              | (-1.8, -1.0, 0°) | start / delivery pose in front of user in living room     |
| `DOOR_W`                   | 1.00 m  | doorway width                                                     |
| `WALL_H`                   | 1.20 m  | wall height (low so angled cameras can see in)                    |
| `JITTER_XY`, `JITTER_YAW`  | 0.15 m, 10° | per-reset randomisation                                       |
| `DOOR_SWING_DEG`           | 90°     | max door swing each way (push-only)                               |
| `DOOR_DAMPING`, `DOOR_FRICTION`, `DOOR_MASS` | 1.8, 0.35, 12 kg | push door inertia and friction stay   |
| `DRAWER_EXTENSION`         | 0.8     | fraction of its depth a drawer can slide out                      |
| `HANDLE_FRACTION`, `HANDLE_R`, `HANDLE_STANDOFF` | 0.75, 12 mm, 45 mm | long bar handle length, radius, standoff |
| `DRAWER_DAMPING`, `DRAWER_FRICTION` | 25 N·s/m, 0.5 N | sliding resistance of drawers (10 N pull ≈ 0.4 m/s) |
| `SHUFFLE_XY`, `SHUFFLE_YAW`, `SHUFFLE_TRIES` | 0.40 m, 20°, 30 | shuffled-mode furniture moves, and draws per piece |
| `HEAD_CAM_Z`               | 1.55 m  | height of `cam_robot_head`                                        |
| `GRID_RES`                 | 0.05 m  | occupancy grid cell size                                          |

## Using it from code

```python
from sim.house_env import HouseEnv, Mode
env = HouseEnv(seed=0)

# Mode 1: Static baseline
info = env.reset(mode=Mode.STATIC)

# Mode 2: Shuffled furniture and drawer targets
info = env.reset("tidy", "kitchen_drawer", mode=Mode.SHUFFLED)
env.bottle_xyz()                              # bottle is located inside drawer tray
env.data.site("drawer_kitchen_drawers_2_handle").xpos # drawer handle site for grasping

# Mode 3: Dynamic human walking and moving bottle
env.reset("tidy", "kitchen_counter", mode=Mode.DYNAMIC)
env.step(4000)                                # person walks in and carries bottle
print(env.dynamic_state)                      # {'active': True, 'carrying': True, 'person_too_close': False, ...}
```
