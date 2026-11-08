# Overnight handover: G1 integration (Eren, 7–8 Nov 2026)

> **For coding agents:** this file sums up everything Eren changed overnight on branch
> `G1-Integration`. Read it before you touch `assistive_g1/`, `brain/safety.py`, `brain/intent.py`
> or `sim/`. More detail lives in `docs/G1_BRAIN_INTEGRATION.md` (brain ↔ robot) and
> `docs/G1_LOCALIZATION_AND_DRAWER.md` (localization and arm fixes). Where those docs disagree with
> this file, **this file is newer** (see "Stale docs" below).

---

## 1. TL;DR

- **Robot:** Unitree G1 humanoid in MuJoCo. Walking and balance come from NVIDIA's GR00T WBC ONNX
  policies, run in a single MuJoCo process (no LeRobot DDS/ZMQ stack). The BinWalker quadruped is
  dropped.
- **World:** Jack's `sim/house_env.py` (`HouseEnv`: 4 rooms, push doors, drawers). We subclass it in
  `assistive_g1/house.py`, and **`sim/` is not modified.**
- **Brain:** Kristina's `brain/` (planner + `safety.validate`, Gemini optional) now drives the
  robot through `assistive_g1/brain_bridge.py`. **`brain/` has two small edits** (section 6).
- **One-shot learning:** the carer opens a drawer or cupboard once while a camera watches. The robot
  fits a hinge or slide from ArUco marker tracks and then opens *and closes* it along that model.
- **Location memory:** a recency-weighted belief over stations for each item decides where to
  search first, and failed looks lower a station's belief during the search.
- **Localization (latest commit):** the robot no longer reads its pose from `data.qpos`. It uses an
  IMU, leg odometry, a 360° LiDAR scan-matched against the static house map, and an RGB-D head
  camera.
- **End-to-end status:** "I'm thirsty" → search counter (not there) → open kitchen drawer → see
  bottle → pick from above → push drawer shut with the left hand → deliver to the user's tray by the
  sofa. Then "Could you bring me my cup?" → counter → deliver. This works headless in sim on the
  localization estimate, with no falls.

---

## 2. Branches and commits

| Ref | Commit | Meaning |
|---|---|---|
| `G1-Integration` (local **and** `origin`, in sync) | `7220036` | All of Eren's work. **Use this branch.** |
| `Gemini-G1-Localization` | `9543915` | Another agent's localization attempt ("checkpoint – not working"). Only `sensors.py` was taken from it (with a bug fix). **Do not merge it.** |
| `origin/grasp-physical` | — | Dex3 finger pinch + MPPI transport. Used for **ideas only**, not merged. |
| `origin/main` | `85c24d7` | Has moved on since our last merge of main into `G1-Integration`. Run `git log G1-Integration..origin/main` before merging either way. |

`G1-Integration` commit history (London time):

| Time | Commit | What |
|---|---|---|
| Sat 20:51 | `a5d3b9d` | G1 + GR00T policy, cupboard opening, one-room scene |
| Sat 23:40 | `56deda5` | PD go-to-pose controller (`nav.py`) on top of GR00T velocity commands |
| Sat 23:51 | `96abe6e` | Merge `origin/main` (Jack's `sim/`, Kristina's `brain/`) |
| Sun 00:55 | `2a426c3` | Arm collision avoidance, mesh collisions on, better IK |
| Sun 02:24 | `b48dff4` | G1 in Jack's house (`house.py`, `house_demo.py`) |
| Sun 03:15 | `e7523a8` | Split localization work off to its own branch |
| Sun 03:49 | `2978957` | Closing cupboards/drawers (left hand) |
| Sun 04:29 | `92ae05d` | IK improvements |
| Sun 05:23 | `786f4b2` | Brain integration (`brain_bridge.py`, `house_brain_demo.py`) |
| Sun 08:23 | `7220036` | Self-localization (`state_estimation.py`, `sensors.py`), arm/drawer fixes, depth camera |

Run `git status` first in case something is uncommitted locally.

---

## 3. Architecture

```
person's request ("I'm thirsty")
   │
   ▼
brain.engine.Brain.start_task ──► FetchTask(item="water_bottle", destination="sofa")
   │
   ▼  loop: brain.integration.take_turn(brain, bridge, bridge)
   │    obs      = RobotBridge.latest_observation()        # what the ROBOT knows (no object ground truth)
   │    decision = brain.next_action(obs)                   # rules planner, or Gemini proposal
   │    brain.safety.validate(decision, task, obs)          # vets every decision
   │    feedback = RobotBridge.execute_high_level(decision) # runs G1 skills
   │    brain.record_feedback(feedback)                     # repeated failures → ask_for_help
   ▼
assistive_g1/skills.py  AssistiveRobot: go_to, look, pick, open_articulated, close_opened, hand_over
   │
   ├── nav.py              PD go-to-pose → (vx, vy, wz) for GR00T; Jack's occupancy grid + BFS planner
   ├── ik.py               7-DoF damped-least-squares arm IK with collision rows (mj_geomDistance)
   ├── perception.py       head-camera ArUco (RGB-D refined)
   ├── articulation.py     one-shot hinge/slide fit
   ├── memory.py           location memory
   ├── state_estimation.py IMU + leg odometry + LiDAR scan-matching → base pose
   └── g1_wbc.py           GR00T Balance/Walk ONNX at 50 Hz, arm PD + gravity feed-forward
   │
   ▼
assistive_g1/house.py  G1House(HouseEnv)   ← Jack's sim/house_env.py, untouched
```

---

## 4. New and changed files in `assistive_g1/`

| File | Status | What it does |
|---|---|---|
| `g1_wbc.py` | new | GR00T policy loop. API: `set_velocity(vx, vy, wz)`, `set_height(h)` (policy accepts 0.2–0.74 m; tested to 0.33 m pelvis), `set_torso_rpy`. Uses the **estimated** pose for `base_pos / base_yaw / base_vel_body / fallen()`. `true_base_*` is simulator truth, **for evaluation and setup only**. |
| `scene.py` | new | Original one-room scene (table, shelf, counter + hinged cupboard, armchair). Still used by `run_demo.py`. Also `add_robot_extras` (head camera on `torso_link`, palm sites, self-collision excludes). |
| `house.py` | new | `G1House` subclasses Jack's `HouseEnv` and hooks `_attach_robot` (runs before his compile). Adds robot extras, our items, chest anchor markers, grasp welds, `drawer_cam` for the carer demo, stations, the user's tray as the delivery surface, and path planning on his grid. |
| `house_demo.py` | new | Scripted scenario in the house (no brain): carer demo → fetch bottle from the drawer → close → deliver; `--with-cup` adds the cup. |
| `house_brain_demo.py` | new | **The main demo.** Kristina's brain + G1 in the house. |
| `brain_bridge.py` | new | `RobotBridge` implements both `PerceptionBridge` and `ControllerBridge` from `brain/integration.py`. |
| `memory.py` | new | `LocationMemory`: `observe_placement`, `observe_search`, `ranked`, `belief`, save/load. Decay 0.85/day, prior 0.2 per station; a failed look multiplies that station by 0.05 for the episode. |
| `articulation.py` | new | One-shot fit: plane + circle (revolute) vs line (prismatic); the lower residual wins. The robot generates its own path along the fit and does not replay the recording. |
| `demo_capture.py` | new | Carer demo → `cupboard_model.json` / `drawer_model.json`, from the sim camera (`capture_sim_drawer`) or a real video (`demo_capture.py video`). |
| `skills.py` | new | Robot API returning JSON-able dicts (see section 5). |
| `ik.py` | new | DLS IK for either arm, 3 cm clearance rows to torso, legs, furniture and doors. One planner (`_plan`) + one executor (`_reach`) since the latest commit. |
| `nav.py` | new | PD go-to-pose with acceleration limits, turn-then-walk, clean stop (GR00T steps in place for any small command, so the controller sends either ≥ `MIN_CMD` or exactly 0). |
| `perception.py` | new | ArUco from the head camera; depth refinement, with a PnP fallback if the two disagree by > 10 cm. Error is ~2–4 mm at the drawers. |
| `state_estimation.py` | new (08:23) | Self-localization, see section 8. |
| `sensors.py` | new (08:23) | IMU/AHRS, leg odometry, LiDAR. Copied from `Gemini-G1-Localization`, with **one bug fix**: `camera_pose_for_base` needed `mj_camlight` (it returned zeros). |
| `tasks.py` | new | Deterministic fetch for the one-room scene (pre-brain). |
| `agent.py` | new, **superseded** | Anthropic tool-use agent. Replaced by Kristina's Gemini brain; don't extend it. |
| `run_demo.py` | new | One-room scenario + MP4 recorder. |
| `fetch_assets.py` | new | Downloads the G1 model, 49 meshes and 2 GR00T ONNX policies (~35 MB) from GitHub's media host; no git-lfs or HF login needed. |
| `viewer_env.py` | new | Fixes the conda GLX problem for `--viewer` (relaunches with the system `libstdc++`). |
| `calibrate_camera.py` | new | ChArUco calibration for the real carer camera. |
| `cupboard_model.json`, `drawer_model.json` | generated | Learned articulation models. **Delete `drawer_model.json` once** if it predates the depth camera (the old fit was 38–39 cm; the new one is 36 cm against 36 true, 1.4 mm residual). |
| `arm_probe.py`, `try_floor_pick.py` | dev scripts | Not part of the demo. |

---

## 5. Brain ↔ robot contract (`brain_bridge.py`)

**Areas** = station names: `kitchen_counter`, `kitchen_drawers`, `kitchen_table`. `sofa` = the
user's station (internally `"person"`). Any other `navigate` target is rejected by
`safety.validate` ("Unknown action target").

| Brain action | What the G1 does |
|---|---|
| `search(item)` | Visits stations in location-memory order, skipping ones already searched this task. At a container station it opens the drawer with the learned model and looks again. Stops when the head camera sees the item, and fails only after every station has been checked. Calls `memory.observe_search` after each look. |
| `navigate(area)` | `go_to(station)` on Jack's grid (pushes unlocked doors by walking through them), then looks. |
| `open_drawer` | `open_articulated` at the current station with the carer-demo model. Squares up first. |
| `grasp(item)` | `pick` with the right hand, **from above if the item was seen inside a container**. On failure: arm home, step back 0.2 m, re-approach, one retry. If something is open, `close_opened()` with the **left hand** while still holding the item. |
| `deliver(sofa)` | `hand_over()` onto the user's tray. |
| `stop` | Zero velocity, wait 0.5 s. |

**The `Observation` the brain sees** (built from the robot's own knowledge, **not** simulator
object ground truth, unlike `brain/mujoco_observation.py`):

- `visible_objects`: the last head-camera sighting of each item (`container="drawer"` if seen inside
  one). A sighting is deleted when the robot looks at that area again and the item is gone.
- `robot_area`, `drawer_open` (this robot has the container at this station open), `holding_item`,
  `delivered_item`.
- `person_too_close`: from Jack's dynamic person (dynamic mode only).
- `sensors_ok = not fallen`.
- `scene_note`: a plain-language context string for Gemini: skills, areas, containers, and the
  **memory ranking for the current item**, minus stations already searched.

Decisions and feedback are logged to `assistive_g1/logs/house_brain.jsonl`; `logs/` is gitignored.

---

## 6. Edits to teammates' code

**`sim/` (Jack): no changes.** Everything is done through the `G1House` subclass.

**`brain/` (Kristina): two edits, both marked `G1 integration, 8 Nov` in the code.**
1. `brain/safety.py`, the `open_drawer` check: it used to require `robot_area == "cupboard"`. Now
   the robot must be at an area where the item was seen inside a container, because the house has
   several containers.
2. `brain/intent.py`, `RulesIntentParser`: "thirsty"/"drink" maps to `water_bottle` without a fetch
   verb. "Could I have my cup?" still needs a verb offline ("Could you bring me my cup?"); Gemini
   handles free-form requests.

No other files in `brain/` changed. The Gemini adapters (`gemini_optional.py`, `gemini_client.py`)
are used as they are.

**World changes that exist only inside `G1House` (Jack's defaults are unchanged):**
- The coffee table moves to (-2.80, -1.15) so the robot can stand at the user's tray.
- The water bottle goes in the top kitchen drawer (Jack's own `kitchen_drawer` target, moved to
  10 cm behind the drawer front).
- The cup goes on the kitchen counter.
- The G1's Euler integrator is used (the house defaults to implicitfast, but GR00T was run
  against Euler).
- Our head camera sits on `torso_link`, not Jack's `cam_robot_head` (a placeholder at a fixed
  1.55 m).
- Since 08:23 there are no left-hand/drawer contact excludes, because the push-to-close needs that
  contact.

---

## 7. ArUco marker registry (all `DICT_4X4_50`)

| IDs | Owner | Use |
|---|---|---|
| 0, 1, 10–17, 20–34, 40, 45 | Jack | His registry. 20–29 are drawer fronts and serve as our **moving** articulation markers. |
| 2 | ours | Water bottle (on the cap of Jack's bottle; his marker 0 is on its side) |
| 3 | ours | Cup |
| 4 | ours | Pill box (deliberately out of brain scope: no medication handling) |
| 35, 36, 37 | ours | Fixed anchors on the chests: kitchen drawers, dresser, filing cabinet |
| 46, 47, 48 | ours | Cupboard door front, cupboard anchor, inside of the cupboard door |

---

## 8. Self-localization (latest commit `7220036`)

The robot's decisions use only the estimated `pos`, `yaw` and `vel_body`.

| Sensor | Rate | Use |
|---|---|---|
| Pelvis IMU (gyro + accel) | 500 Hz | roll/pitch/gravity, yaw rate. Also feeds GR00T's angular velocity and gravity inputs. |
| Leg odometry (29 encoders + IMU tilt, stance foot) | 50 Hz | pelvis height, body velocity → dead reckoning |
| 360° LiDAR, 8 slices × 180 rays (`mj_multiRay`) | 10 Hz | Gauss-Newton scan-to-map (trimmed + Huber) against a 2.5 cm distance field of the static house, built once at reset |
| Head RGB-D | per look | depth-refined marker positions; camera pose from the *estimated* base + forward kinematics |

- **Fusion:** a complementary filter. Each accepted scan pulls the pose 50% of the way. A scan is
  accepted if > 40% of points are inliers, RMS < 3 cm and the jump < 0.3 m. Open drawers, doors,
  people and items are trimmed as outliers.
- **Measured:** 1–2 cm and < 0.3° error at stations, 6–7 cm worst-case while walking, every scan
  accepted.
- **A/B switch:** `G1_LOCALIZE=0` runs on simulator truth instead.
- **Still read from the simulator:** the simulated sensors themselves, the start (dock) pose at
  reset, arm gravity feed-forward (`qfrc_bias`), and `holding_ok()`.
- **Not done:** a live dynamic occupancy map (the planner still uses Jack's static grid), and
  `person_too_close` from the LiDAR (points not in the map that are < 0.5 m away).

---

## 9. Arm and drawer fixes (latest commit)

1. **Plans were not executed:** the arm re-solved from its current pose and drifted. Now one planner
   and one executor: a joint-space swing to the planned pre-pose, a straight approach with the null
   space pulled to the planned angles, and a joint-space finish if the approach stalls.
2. **No overhead poses:** comfort joint limits; plans with the elbow more than 5 cm above the
   shoulder are rejected (20 cm for top-down grasps). Env: `G1_ELBOW_MAX`, `G1_ABOVE_ELBOW_MAX`.
3. **Drawer pick:** the hand comes in 12 cm above the item (was 30 cm), with fingers pitched 20–50°
   forward, and lifts 32 cm so the bottle clears the drawer front.
4. **Held item stays upright:** the constraint is now "the item axis stays vertical", not "the
   finger direction stays fixed".
5. **Drawer close = push** with the left fingertips 8 cm above the bar, paced by the feet. The robot
   backs off before stepping sideways to the close stance (the bottle snagged on the handle).
6. **Open from any stance:** if the handle can't be planned, the robot steps in to 0.36 / 0.31 m,
   looks again and re-plans (the brain run can arrive off-station).

---

## 10. How to run

```bash
cd assistive_g1
pip install -r requirements.txt          # mujoco, onnxruntime, opencv-contrib-python, numpy
python fetch_assets.py                   # once, ~35 MB

# Main demo: brain + robot
python house_brain_demo.py --offline                 # rules parser + deterministic planner
python house_brain_demo.py                           # Gemini (needs GEMINI_API_KEY + google-genai,
                                                     #   pip install -r ../requirements-gemini.txt);
                                                     #   prints a WARNING and falls back to rules if missing
python house_brain_demo.py --viewer --reuse-demo     # watch it; reuse drawer_model.json
python house_brain_demo.py --request "I'm thirsty" --request "Could you bring me my cup?"

# Scripted (no brain)
python house_demo.py --viewer --with-cup
MUJOCO_GL=egl python house_demo.py --no-video        # numbers only, fast
G1_LOCALIZE=0 python house_demo.py --no-video        # simulator pose instead of the estimate
ASSISTIVE_DEBUG=1 python house_demo.py --no-video    # reach/close internals

# Original one-room scene
python run_demo.py --watch
```

- **Gemini key:** set `export GEMINI_API_KEY=...` in the shell. Never commit it. `.env` is
  gitignored, and right now the repo's `.env` holds no key.
- **Rendering:** video rendering in the house needs shadows off (0.9 → 0.25 s/frame with software
  GL). Prefer headless `--no-video` runs while developing.
- **Summary output:** the demos end with a summary: item positions, whether the drawer is open,
  `fallen`, the localization error (estimate vs truth) and LiDAR stats.

Expected `house_brain_demo` flow for "I'm thirsty":
1. Memory ranks the counter first (~57%): walk there, look: not there.
2. Next the drawers (~33%): open, bottle seen, grasp from above, left hand pushes the drawer shut.
3. Navigate to the sofa, deliver, complete.

"Could you bring me my cup?" goes straight to the counter, because the head camera saw the cup there
during the first search.

---

## 11. Results so far (sim)

| Test | Result |
|---|---|
| Drawer one-shot fit (depth camera) | prismatic, 36 cm vs 36 cm true, 1.4 mm residual |
| Cupboard one-shot fit | hinge within ~1.6 cm, axis within ~5°, 1.37 rad vs 1.40 true, 4 mm residual; robot opened it to 1.39–1.41 rad |
| `house_demo --with-cup` on the localization estimate | bottle (drawer) and cup (counter) both delivered upright, drawer shut |
| `laundry_day` seed 3, `door_shut` layout (long way round) | drawer shut (`door_shut` after one retry) |
| Falls | none |
| Arm penetration (one-room scenario) | 0% of ticks over 1 cm (was 21.6%, worst 12 cm, before collision avoidance) |

---

## 12. Shortcuts to admit if judges ask

- Grasps are **welds switched on at contact** (position-only for handles); there are no finger
  physics.
- Held items don't collide with the robot.
- Torso–arm collisions are excluded only for the two shoulder links against the torso hull.
- The carer demo in sim drives the drawer joint directly (the "carer's hand").
- **Location memory is seeded with a synthetic "week of watching the carer"** in
  `house_brain_demo.learned_memory()`. It is not learned live from a demo yet (see open items).
- Stations are fixed to nominal furniture, so Jack's `shuffled` mode and the `dynamic` person are
  not supported for fetching.
- Room doors are pushed open by walking, not with a hand.
- The sensors are simulated from the true state (that's how any sim works), and the start pose is a
  known dock pose.

---

## 13. Open items, in priority order

1. **Teach-once → location memory (agreed with Kristina, not done).** `brain/teach_once.py` still
   stores a fixed `item → destination` `FetchTask`. Agreed change: when the carer puts an item
   somewhere, call `LocationMemory.observe_placement(item, station)` so that place becomes a
   candidate location. This would also replace the synthetic seeding in `learned_memory()`.
   `memory.nearest_station()` exists but still uses the one-room scene's surfaces; it needs a
   house version.
2. **Gemini context:** the memory ranking already goes into `scene_note`. Check whether
   `GeminiPlanner.propose` actually uses it to choose *where* to search, and pass the station list
   explicitly if not.
3. **Item enum mismatch:** the Gemini intent schema has `water_bottle`, `cup`, `book`; the house has
   `water_bottle`, `cup`, `pill_box`. Leave the pill box out (scope: no medication). Either add a
   book to the house or drop it from the enum.
4. **Real carer video for the pitch:** print markers (46/47 + 48 for the cupboard; 35 + a drawer
   front marker for a drawer), run `calibrate_camera.py`, then
   `demo_capture.py video --source 0 --calib calib.npz --grasp gx gy gz`. Show it side by side with
   the sim twin.
5. **Speech input** for the person's request; currently `--request` strings.
6. **Dashboard:** show the location-memory probabilities and the fitted hinge/slide live.
7. **Merge with `origin/main`** (it has new commits since our merge). `sim/` is subclassed, not
   edited, so conflicts should mostly be in `brain/safety.py`, `brain/intent.py` and docs.
8. **Cleanup:** `robot_core/`, `scenes/smoke.xml` and `assistive_g1/agent.py` are dead code for the
   demo.

---

## 14. Stale docs (don't trust these lines)

- `README.md` (repo root), "Status": says there is no camera detection, no robot control and no
  grasping. All of these now exist on `G1-Integration`.
- `docs/G1_BRAIN_INTEGRATION.md`: says "the robot's base pose still comes from the simulator (no
  localisation yet)". Outdated since `7220036`.
- `docs/G1_LOCALIZATION_AND_DRAWER.md`: calls the changes "uncommitted". They are committed in
  `7220036` and pushed.
- `assistive_g1/brain_bridge.py` docstring and the `_context()` string sent to Gemini still say
  "base pose from the simulator". This one-line fix also changes what Gemini is told.
- `assistive_g1/README.md` describes mainly the one-room scene. For the house, read this file.
- `memory.py` defaults (`SEARCHABLE = ["table", "shelf", "cupboard"]`, `nearest_station`) are for
  the one-room scene. The house demo passes its own station list.
