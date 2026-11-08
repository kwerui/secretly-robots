# assistive_g1: a Unitree G1 that learns from the carer

A humanoid (Unitree G1, MuJoCo) fetches things for someone with limited mobility. It
learns two things by watching the carer:

- **Where things live.** It keeps a location memory, a probability for each item over
  stations, built from the carer putting things away.
- **How to open things.** From one demonstration it fits an articulation model (hinge
  axis, opening angle, grasp point), all relative to an ArUco anchor on the furniture.

Walking and balance come from NVIDIA's GR00T decoupled whole-body-control policy. The arm
uses our own damped-least-squares IK, and perception is ArUco on the robot's head camera.

## Quick start

```bash
pip install -r requirements.txt
python fetch_assets.py            # G1 model + 49 meshes + 2 GR00T ONNX policies (~35 MB)
python run_demo.py --watch        # live window (OpenCV + EGL, no GLX needed), also writes demo.mp4
python run_demo.py                # headless, recorded to demo.mp4
python run_demo.py --viewer       # interactive MuJoCo viewer (needs working GLX, see below)
python run_demo.py --fps 12.5 --size 800 450   # lighter video if you have no GPU
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
| Grasp API | None; dex1/dex3 is only a model choice | Weld-on-contact grasp (rigid or position-only) |
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

## What is tested (in this repo's sim)

- The GR00T policy stands, walks (0.4 m/s command → ~0.38 m/s), turns, squats to a 0.33 m
  pelvis height and leans, without falling.
- Fetch the bottle from the table to the side table, upright (grasp error 3 mm).
- Floor pick: a bottle on the floor is picked from a deep squat (pelvis ~0.27 m), reaching
  down beside the right knee with a top-down grasp (2 mm), then delivered. Lowest
  reachable grasp point is ~0.11 m above the floor, so a cup (10 cm tall) and the pill
  box (6 cm) are too low with the GR00T policy.
- From the sim carer demo (door swung to 1.75 rad): learned angle 1.64 rad, fit
  residual 4.3 mm, 97 samples.
- The robot opens the cupboard along the learned arc to ~1.55 rad (89°), walking
  backwards while it pulls so the door never comes into its body, steps in, picks the cup
  (2.6 cm) and places it on the side table.
- Arm collisions: across the full scenario 0% of control ticks have any arm penetration
  over 1 cm (worst 0.8 cm). Before collision avoidance: 21.6% of ticks, worst 12 cm.
- Full scenario (`run_demo.py`): the bottle is searched for at the shelf first (most
  likely per memory), not found, then found on the table and delivered. The cup is
  fetched from the closed cupboard. Both end upright on the side table. That's ~140 s of
  sim time, ~20 s wall-clock without rendering (2-core CPU).

## Known shortcuts (say them if asked)

- Localisation uses the simulator's base pose; the stations are fixed.
- Grasping is a weld switched on when the palm reaches the item (the hand mesh is
  rigid). The door grip is position-only (torquescale 0), like a loose grip on a bar
  handle.
- Collisions are on for everything. The arm IK keeps 3 cm clearance from the torso, legs,
  furniture and the door (`mj_geomDistance` rows in `ik.py`), and every grasp is planned
  collision-free first. Excluded pairs: torso vs the two shoulder links (they sit against
  the torso hull at the joint), wrist roll vs wrist yaw, and the grasping hand vs items
  and the door (the hand is a rigid block, so the palm has to be allowed inside what it
  grasps; the forearm still collides).
- A held item collides with furniture and other items but not with the robot; robot
  contact comes back once the arm has retracted after setting it down.
- The hand-over places the item on the side table next to the armchair, at a free spot
  (seen by the camera, plus where the robot put things before).
