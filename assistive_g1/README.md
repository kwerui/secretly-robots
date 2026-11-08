# assistive_g1: a Unitree G1 that learns from the carer

A humanoid (Unitree G1, MuJoCo) fetches things for someone with limited mobility. It
learns two things by watching the carer:

- **Where things live.** It keeps a location memory, a probability for each item over
  stations, built from the carer putting things away.
- **How to open things.** From one demonstration it fits an articulation model (hinge
  axis, opening angle, grasp point), all relative to an ArUco anchor on the furniture.

Walking and balance come from NVIDIA's GR00T decoupled whole-body-control policy. The arm
uses our own damped-least-squares IK for the precise approach and a sampling-based MPC (MPPI,
`mppi.py`) for the transport motions, perception is ArUco on the robot's head camera, and
**items are grasped physically with the Dex3 fingers** (`hand.py`): the finger joints are
enabled, position-controlled with the hardware torque limits, and the item is held by friction
between the finger pads and the item. No weld or other constraint is involved in the item path
(see "Physical grasp" below; the cupboard door handle is still a position-only weld).

## Quick start

```bash
pip install -r requirements.txt
python fetch_assets.py            # G1 model + 49 meshes + 2 GR00T ONNX policies (~35 MB)
python run_demo.py --watch        # live window (OpenCV + EGL, no GLX needed), also writes demo.mp4
python run_demo.py                # headless, recorded to demo.mp4
python run_demo.py --viewer       # interactive MuJoCo viewer (needs working GLX, see below)
python run_demo.py --fps 12.5 --size 800 450   # lighter video if you have no GPU
python eval_grasp.py --trials 6 --jitter 0.03  # repeated-trial numbers for the pitch (no video)
ARM_MPPI=0 python run_demo.py                  # IK-only transport motions (faster to simulate)
```

`run_demo.py` records the carer's cupboard demo in sim the first time and saves
`cupboard_model.json`. To use a real demonstration instead, see "Real carer demo".

### Rendering / OpenGL

All offscreen rendering (robot camera, video, `--watch`) uses EGL (`MUJOCO_GL=egl`, set
automatically), so it needs no X/GLX. Only `--viewer` opens a GLFW window, which needs a
working GLX in your Python env. Inside anaconda, GLX fails (`GLX: No GLXFBConfigs
returned` / `gladLoadGL error`) because conda's old `libstdc++` shadows the newer system
one that Mesa needs. `viewer_env.py` handles this: with `--viewer` from a conda Python,
the script restarts itself once with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.
New scripts that open the viewer should call `relaunch_for_viewer()` before importing mujoco.

## Why not the stock LeRobot G1 stack

We researched the LeRobot G1 integration (`lerobot/robots/unitree_g1`) and the
GR00T-WholeBodyControl repo. The gaps for us:

| Need | LeRobot G1 | Here |
|---|---|---|
| Walk from code | Only via fake joystick keys (`remote.lx/ly/rx`) in the action dict | `set_velocity(vx, vy, wz)` |
| Crouch | Buttons R1/R2 nudge the height 1 mm per tick, clipped to 0.50–1.00 m | `set_height(h)`; the policy accepts 0.2–0.74 m (GR00T's own teleop clips there). Tested down to 0.33 m pelvis height |
| Lean to reach | Not exposed | `set_torso_rpy(r, p, y)`; GR00T's obs has a torso orientation command |
| Grasp API | None; dex1/dex3 is only a model choice | Physical Dex3 pinch with contact verification (`pick`), position-only weld for the door handle |
| Arm IK | `G1_29_ArmIK` (pinocchio + CasADi from conda-forge, URDF from HF) | Plain-numpy DLS IK on the live MuJoCo model, collision-checked |
| Custom scene | Hub env (`lerobot/unitree-g1-mujoco`), DDS + ZMQ, `unitree_sdk2py` (CycloneDDS build) | One process, scene built with `MjSpec` in `scene.py` |

We kept the trained part (the GR00T Balance/Walk ONNX policies, the same files LeRobot
downloads) and dropped the plumbing. The observation layout, scales and gains are copied
from `decoupled_wbc/control/policy/g1_gear_wbc_policy.py`, and they match LeRobot's
`gr00t_locomotion.py`.

## Files

| File | What |
|---|---|
| `g1_wbc.py` | G1 + GR00T policy loop at 50 Hz (legs + waist), arm PD + gravity feed-forward. Velocity, height and torso commands |
| `hand.py` | Dex3 finger joints + position actuators (added at build time; the GR00T model ships them commented out), pad contact parameters, open/close postures, contact queries |
| `mppi.py` | MPPI (sampling-based MPC) for the arm's transport motions; `MotionLog` smoothness metrics |
| `eval_grasp.py` | N jittered trials of the two-request scenario: success, pinch force, slip, upright-at-the-end, arm smoothness |
| `scene.py` | Home scene: table, shelf, counter + hinged cupboard, armchair + side table, tagged items, head camera, room camera |
| `perception.py` | Head-camera render, ArUco detect, marker pose → world (6–9 mm error in tests) |
| `ik.py` | 7-DoF arm IK (position + axis alignment), collision check |
| `skills.py` | `go_to`, `look`, `pick`, `open_cupboard`, `hand_over`: the robot API, returning JSON-able dicts |
| `articulation.py` | One-shot articulation fit: plane + circle (revolute) or line (prismatic), model selection by residual |
| `demo_capture.py` | Carer demo → `cupboard_model.json`, from a real video or from the sim room camera |
| `memory.py` | Location memory (recency-weighted counts, negative evidence while searching) |
| `tasks.py` | Deterministic fetch planner (search ranked stations, open cupboard if needed, pick, hand over) |
| `agent.py` | LLM tool-use agent over the same skills (Anthropic SDK). Untested against the API |
| `run_demo.py` | Scenario + MP4 recorder (follow cam + head-cam inset + captions) |
| `calibrate_camera.py` | ChArUco calibration for the real demo camera |

## Real carer demo (for the pitch)

1. Print ArUco 4x4_50 **id 46** (door front) and **id 47** (fixed on the cupboard body or
   counter, upright). Size: 6 cm black square, or change `DOOR_MARKER_SIZE`.
   `scene.marker_png(46)` writes the images to `assets/markers/`.
2. `python calibrate_camera.py board`, print it, then
   `python calibrate_camera.py calib --square <m>`.
3. `python demo_capture.py video --source 0 --calib calib.npz --grasp gx gy gz`.
   The carer opens the door once in view of the camera. `--grasp` (required) is the
   handle position in the door-marker frame (x right, y up as printed, z out of the
   marker, metres); measure it with a ruler. A MediaPipe wrist track at the moment the
   door starts moving would supply this automatically: good teammate task.
4. `python run_demo.py --model cupboard_model.json` runs the robot with the real
   demonstration. The sim cupboard's hinge is on the robot's right (handle on its left,
   door swings out to the robot's right), so demo the real cupboard the same way round,
   or mirror `scene.py`. If you change the cupboard geometry, delete
   `cupboard_model.json` so the sim demo is re-recorded.

## Physical grasp (how `pick` works now)

1. The hand opens (thumb swung back, fingers straight) and comes in along the robot's heading
   so the item sits between the finger pads and the thumb, at the **grasp site** (`right_grasp`,
   the measured pinch equilibrium of a 6-7 cm item in the hand frame). All item moves use this
   site as the IK target; the palm site is kept for the door handle.
2. The fingers close under position control for 0.6 s; the actuators saturate at the hardware
   torque limits (1.4 N*m per finger joint, 2.45 N*m thumb abduction), so the squeeze is "as hard
   as the hand can", not a pose.
3. The pinch is **verified** before anything else happens: thumb AND at least one finger in
   contact, on opposite sides, with at least 4 N of total normal force. Then the item is lifted
   5 cm and the hold is verified again (contacts still opposing, item moved < 3 cm in the hand
   frame). Only then does the arm retract and the robot step back and carry.
4. Placing is **by touch**: the hand lowers until the item makes contact with the table, the
   fingers open gradually while the grasp site is servoed in place, and the hand withdraws
   sideways first, then up, then back, so neither the finger roots nor the wrist knock the item
   (or its neighbour) over.
5. Every grasp leaves evidence in `robot.grasp_log` (pads in contact, normal force, finger
   angles, item pose in the hand before/after the lift, slip). `pick()` returns it too.

Contact model: elliptic friction cones, `impratio` 10, Newton solver, implicit-fast integration,
finger pads with `condim` 4 (torsional friction), friction 1.5, slightly soft `solref` and
`priority` 1 (MuJoCo's documented recipe against grasp slip). No `noslip` iterations: they would
hide slip instead of modelling it.

### Measured (`python eval_grasp.py --trials 6 --jitter 0.03`)

Two-request scenario, item start positions jittered by up to +-3 cm, 6 trials:

| | fetched | delivered upright on the side table |
|---|---|---|
| water bottle (table) | 5/6 | 5/6 |
| cup (closed cupboard, opened first) | 6/6 | 6/6 |

11 of 11 pinches that formed held through the lift, the carry and the walk: 41-45 N mean total
normal force (30-78), 0.5-0.7 mm mean slip after the lift (max 1.3 mm). The one failure is a
reach failure from the stance the robot ended up in (`could not reach ... residual 0.42 m`),
reported by the skill, not a dropped item. Perception-error tolerance of the pinch, measured by
biasing the detected item pose: about -2/+3 cm along the pinch axis and +-3 cm along the
approach (the open hand has ~2 cm of finger clearance on one side of the item).

### MPPI for the transport motions

`mppi.py` is a STORM-style MPPI: joint-velocity samples over a 0.32 s horizon, rollouts scored
on the live MuJoCo kinematics (moving-reference tracking, hand-axis alignment, arm clearance via
signed distances at horizon knots, joint limits, posture, velocity and acceleration), sampling
centred on a one-step IK nominal, exponentially weighted update, plan filtered along time,
re-planned every 2 ticks. It drives the carry, the hand-over approach and the withdrawal; the
precise approach to a grasp or a placement stays on the per-tick damped-least-squares IK.

A/B on the same scenario (commanded joint targets, mean of 6 trials): acceleration RMS 138 ->
115 rad/s^2 (-17 %), peak 7300 -> 6080 (-17 %), jerk RMS 12350 -> 10170 (-18 %), same 11/12
task success, +40 % wall time (pure-Python rollouts, ~20 ms per plan on one laptop core).
`ARM_MPPI=0` switches it off for faster iteration.

### In Jack's house (`house_demo.py`)

Same hand, same skills, Jack's `HouseEnv` (`house.py`): the robot opens the top kitchen
drawer the way the carer did, **pinches Jack's bottle (a 6.6 cm cylinder) out of the drawer
from above** and carries it ~4 m across the house to the tray by the sofa, upright
(`house_demo.py --no-video`: pick 42 N, slip after lift 9 mm, 28 mm over the walk, delivered
upright). The drawer pick is the hard case for this arm: the open drawer front is at elbow
height, so the hand comes down pitched ~35 deg with the pinch axis horizontal and the planner
is now scored on orientation residual too; the item's own axis (read from its pose in the hand,
a stand-in for a wrist camera) is kept vertical during the carry and the placement. With the
weld, this pick did not work at all (planner score 0.133). Known: the 30 cm tray fits one
item; a second placement is reported as "crowded" and can disturb the first.

### What the grasp does NOT do (yet)

- **Floor pick is not reliable with the physical pinch.** From the deep squat the arm cannot
  bring the pinch low enough with the pinch axis horizontal (the fingertips reach 11 cm past the
  pinch); the weld version "worked" only because it ignored hand orientation. `_pick_low` now
  re-detects the item by leaning forward and tries 45-deg-pitched finger candidates, and it
  reports failure cleanly when the pinch does not form. Not part of the demo scenario.
- The door is still opened with a position-only weld on the bar handle (the fingers do close
  around it). Making that physical is the same recipe as the items.
- No tactile sensing beyond MuJoCo's contact forces, no regrasp strategy after a failed pinch
  (the skill opens, retreats and reports).

## What is tested (in this repo's sim)

- The GR00T policy stands, walks (0.4 m/s command → ~0.38 m/s), turns, squats to a 0.33 m
  pelvis height and leans, without falling.
- Fetch the bottle from the table to the side table, upright, with the physical pinch (grasp
  site error 4-5 mm, 31-45 N pinch, < 1 mm slip through the walk).
- Floor pick: see "What the grasp does NOT do" above - the weld-era result (2 mm top-down
  grasp from a 0.27 m squat) does not carry over to a real pinch.
- From the sim carer demo (door swung to 1.75 rad): learned angle 1.64 rad, fit
  residual 4.3 mm, 97 samples.
- The robot opens the cupboard along the learned arc to ~1.45-1.6 rad (83-92°), walking
  backwards while it pulls so the door never comes into its body, steps in (standing a little
  further left if the open leaf blocks the first stance), pinches the cup (6/6 trials) and
  places it on the side table.
- Arm collisions: across the full scenario 0% of control ticks have any arm penetration
  over 1 cm (worst 0.8 cm). Before collision avoidance: 21.6% of ticks, worst 12 cm.
- Full scenario (`run_demo.py`): the bottle is searched for at the shelf first (most
  likely per memory), not found, then found on the table and delivered. The cup is
  fetched from the closed cupboard. Both end upright on the side table. That's ~140 s of
  sim time, ~20 s wall-clock without rendering (2-core CPU).

## Known shortcuts (say them if asked)

- Localisation uses the simulator's base pose; the stations are fixed.
- Items are grasped physically (fingers, friction). The door grip is still a position-only
  weld (torquescale 0) on the bar handle, like a loose grip, with the fingers closed around it.
- Collisions are on for everything. The arm IK keeps 3 cm clearance from the torso, legs,
  furniture and the door (`mj_geomDistance` rows in `ik.py`), and every grasp is planned
  collision-free first. Excluded pairs: torso vs the two shoulder links (they sit against
  the torso hull at the joint), wrist roll vs wrist yaw, the hand vs the door (weld grip),
  and hand-internal pairs (finger links vs palm; MuJoCo only skips parent-child contacts).
  Items are NOT excluded from anything: the fingers hold them.
- The cabinet hinge has 0.5 N*m of friction so the open leaf stays where the robot leaves it
  (with the original 0.05 it drifted shut when the robot's elbow brushed it).
- The finger position gain (4 N*m/rad) and pad contact parameters are tuned in sim, not
  identified from the hardware; the torque limits are the hardware's.
- The hand-over places the item on the side table next to the armchair, at a free spot
  (seen by the camera, plus where the robot put things before).
