# Secretly Robots — offline-first AI brain starter

**AgentVerse 2026 · Robotics track · 7–8 November 2026**

**Mission:** A simulated household assistant helps an older adult with limited mobility by fetching a light object. The current house places a water bottle on named furniture surfaces (for example `tv_unit`, `kitchen_counter` or `study_desk`); a future robot must use fresh observations to adapt when it moves.

> **Status — integrated planning scaffold, not yet an autonomous fetching robot.** This package has an offline intent parser, high-level state-aware task planner, validation, event logs, a real MuJoCo ground-truth observation adapter and tests. It **does not yet** have camera detection, actual voice recognition, articulated grasping, robot control, or genuine one-shot visual imitation learning. Those require teammates' robot and perception implementation.

## Quick start (Mac / Python 3.14.3 already works for your local MuJoCo)

Open a terminal from this project root (a cloned copy or extracted folder):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m brain.demo
python scripts/smoke_test.py
python -m pytest -q tests/test_house_env_observation.py
```

- `pytest`: verifies the brain and existing scaffold rules.
- `brain.demo`: **hand-authored observations** showcasing decisions and re-planning; *does not move a robot*. Event log in `logs/brain_demo.jsonl`.
- `smoke_test`: **real MuJoCo physics** for a simple example scene and state logging; *does not run the AI brain on a robot*.
- `test_house_env_observation.py`: reads the teammate-owned `sim.HouseEnv`'s genuine MuJoCo state, converts ground truth into the brain contract, and proves a moved bottle changes the next decision. It is **not** a camera-perception test.

To open the minimal MuJoCo scene:

```bash
python -m mujoco.viewer --mjcf=scenes/smoke.xml
```

You can use the virtual environment you've already created in your original neutral starter, rather than rebuilding it: copy the `brain/`, `docs/` and `tests/test_brain.py` files into that project and run the commands there. **Don't overwrite teammates' files without agreeing on the repository structure.**

## Architecture in 30 seconds

```text
User: "Bring me the bottle"
             |
  Intent understanding [offline rules / optional local Gemma 4 / optional Gemini Argon]
             |
    FetchTask(item_id="water_bottle", destination="sofa")
             |
  MuJoCo -> Observation adapter (ground truth) OR teammates' camera perception
             |
   Local planner selects NEXT action based on current world state
       \ optional Gemini Argon proposal only when task-relevant state changes
             |
  Deterministic Python guard checks whether action is permitted
             |
   Decision --> teammates' movement/grasp/controller functions
             |
  ActionFeedback --> updated observation --> replan
```

The team architecture is one Python simulation coordinator connecting MuJoCo, perception, AI planning and modular controllers. **You own the high-level AI brain.** Teammates own the simulated robot, camera/perception, arm motion, physical control and verified result signals. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/INTEGRATION_CHECKLIST.md`](docs/INTEGRATION_CHECKLIST.md).

### MuJoCo ground-truth adapter (current integration baseline)

`brain.mujoco_observation.HouseEnvObservationAdapter` reads the existing
`sim.HouseEnv`'s `model`, `data`, `bottle_xyz()` and named furniture targets;
it does not load a controller or issue motor commands. With no robot attached,
it explicitly reports `robot_area="unknown"` rather than inventing a pose.
When the robot team attaches its model, supply its root-body name if automatic
detection is not appropriate:

```python
from brain.mujoco_observation import HouseEnvObservationAdapter, HouseObservationConfig
from sim.house_env import HouseEnv

env = HouseEnv(seed=0)
env.reset("tidy", "kitchen_counter")
perception = HouseEnvObservationAdapter(env)
observation = perception.latest_observation()
```

For an attached robot whose root cannot be auto-detected, use
`HouseObservationConfig(robot_body="trunk")`.

The adapter sets `scene_note` to identify these values as **MuJoCo HouseEnv
ground truth, not camera perception**. It exposes `last_changes` for meaningful
named bottle-location changes while ignoring physics ticks and small pose
jitter. The current house has room doors and furniture surfaces, not a cupboard
drawer; `drawer_open` is therefore always `False`. Replace or supplement it
with a real sensor bridge only when the perception team supplies
confidence-calibrated detections.

## How the brain makes decisions

```python
from brain.engine import Brain
from brain.contracts import Observation, ObjectSighting

brain = Brain(log_file="logs/brain_events.jsonl")
brain.start_task("Please bring my water bottle")

# This is a manually created example, not a camera result.
obs = Observation(
    timestamp_s=0.0,
    robot_area="unknown",
    visible_objects=(ObjectSighting("water_bottle", "kitchen_counter", None, .95),),
    drawer_open=False,
)
print(brain.next_action(obs))  # Decision(action='navigate', target='kitchen_counter', ...)
```

Then the simulation team executes the controller action, reads new state, and asks for the **next** decision. A moved bottle changes the next target. If the person is too close or sensors become unavailable, the decision is `stop`; repeated failures escalate to human assistance. The deterministic checks are guardrails, **not full collision or real-world safety assurance**.

### Optional genuine local AI (no Wi-Fi at inference time)

1. Install [Ollama](https://ollama.com/download).
2. Download a small on-device model while internet is available:

   ```bash
   ollama pull gemma4:e2b
   ```

3. Ensure the Ollama local server is running (usually the Ollama app, or `ollama serve`). Test:

   ```bash
   python -m scripts.local_brain
   ```

This sends a **real** structured-output request to your local model, using only `http://127.0.0.1:11434`. The trained model's weights must already be downloaded. Memory needs and latency depend on your laptop; the E2B model download can be several GB. The robot's basic high-level planner and safety guard are **still Python**, not the LLM. If Ollama is missing, the example reports that fact rather than faking a successful call.

### Optional Google DeepMind prize integration (cloud, not required for offline base)

Gemini is intentionally opt-in. Install the official SDK separately, set the key only in your shell, then choose the adapter explicitly in application code:

```bash
python -m pip install -r requirements-gemini.txt
export GEMINI_API_KEY='your-key'  # never place a real key in a repository file
python -m scripts.gemini_smoke_test --live
```

```python
from brain.engine import Brain
from brain.gemini_optional import GeminiIntentParser, GeminiPlanner

brain = Brain(parser=GeminiIntentParser(), gemini_planner=GeminiPlanner())
brain.start_task("Bring the water bottle from the kitchen counter to the person on the sofa")
```

The adapter uses `gemini-4-argon` and the official `google-genai` structured JSON schema API. It creates a checked `FetchTask` from the request and can propose one checked next `Decision` from an interpreted `Observation`; it never parses free-form explanations. The cloud proposal is attempted once for each task-relevant state change, not every physics step. The existing local planner remains the default and is used if the API is unavailable, times out, returns invalid JSON, or proposes an action rejected by local safety validation. `python -m scripts.gemini_smoke_test --live` is the deliberate real-API check and may incur API usage; normal tests use mocked SDK responses and never need a key or network access.

## What is already implemented vs. not

| Feature | Status |
| --- | --- |
| MuJoCo installation / minimal physics stepping | ✅ Implemented and previously user-tested |
| HouseEnv ground-truth → AI `Observation` adapter | ✅ Implemented and integration-tested with real `HouseEnv` state; explicitly not camera perception |
| Validated high-level fetch-task decisions | ✅ Implemented in this scaffold; unit-tested |
| Changes to observed bottle location alter next action | ✅ Integration-tested from changed MuJoCo state |
| Human-near / lost-sensor stop and retry limit | ✅ High-level decision logic; real emergency braking must be wired |
| JSONL decision and feedback logging | ✅ Implemented |
| Deterministic Python planner | ✅ Working offline fallback |
| Gemma 4 local inference through Ollama | 🟡 Optional adapter; not yet confirmed live-tested |
| Gemini 4 Argon cloud option | ✅ Implemented and successfully live-tested; still optional and locally validated |
| Simulated cameras/object-recognition/robot control | ⏳ Teammates' integration needed |
| Autonomous grasping and delivery | ⏳ Teammates' integration needed |
| One-shot learned manipulation policy from video | ❌ Not implemented; `teach_once.py` stores **symbolic** task goals only |
| Actual WebRTC call | ❌ Not implemented; browser mic/camera permission test exists in `web/` |

## File guide

```text
brain/
  contracts.py        # typed observation, task, decision and feedback messages
  intent.py           # deterministic parser + optional local Gemma/Ollama parser
  planner.py          # choose next high-level action from fresh observations
  safety.py           # deterministic command validation
  engine.py           # orchestration, local fallback, retries and JSONL logs
  integration.py      # interface teammates implement for actual MuJoCo
  mujoco_observation.py # HouseEnv ground truth -> brain Observation adapter
  teach_once.py       # small symbolic goal memory; not motor imitation
  hearing.py          # robot's ears, onboard: sim/mic audio -> speech segments (python -m scripts.hearing_demo)
  gemini_client.py    # official google-genai structured-output boundary
  gemini_optional.py  # checked Gemini task/next-decision adapters
  demo.py             # hand-authored explanatory example
robot_core/            # original neutral MuJoCo smoke-test scaffold
scenes/smoke.xml       # real minimal physics test scene
sim/house_env.py        # teammate-owned house environment used by the adapter
scripts/               # smoke test and local inference checks
tests/                 # automatic regression tests
docs/ROADMAP.md        # TWO-DAY execution plan, roles, demo and plan B
docs/ARCHITECTURE.md   # boundaries, ownership and honest capabilities
docs/INTEGRATION_CHECKLIST.md
docs/PITCH_EVIDENCE.md # facts, sources and pitch framing
```

## Timeline and deliverables

Read [`docs/ROADMAP.md`](docs/ROADMAP.md) **before building more features**. The key priority is to make the robot genuinely perceive → decide → act → verify and to change the environment mid-run. That maps to the highest scoring judging categories (25% perception, 25% adaptation). The robotics slides require a one-page technical summary, a one-minute uncut simulation video, and a five-minute pitch; confirm final upload instructions with organisers.

## Accessibility, safety and scope

The intended users are older adults and people with mobility limitations. The prototype is **simulation-only** and is not a medical device. It must not diagnose conditions, administer medicines, autonomously call emergency services or buy groceries. If an item is not confidently detected, stop or request help; never claim it was delivered without verified feedback. Keep any future camera processing local where possible, and do not save personal video by default.

## Background and research

For grounded facts about ageing and adult social care staffing, see [`docs/PITCH_EVIDENCE.md`](docs/PITCH_EVIDENCE.md). Three useful external sources:

- [WHO, Ageing and health](https://www.who.int/news-room/fact-sheets/detail/ageing-and-health)
- [Skills for Care, England 2025/26 workforce update](https://www.skillsforcare.org.uk/news-and-events/news/adult-social-care-vacancy-rate-falls-to-lowest-level-in-a-decade-as-workforce-grows)
- [McKinsey Health Institute / FII, Economic Case for Healthy Longevity (2025)](https://fii-institute.org/wp-content/uploads/2025/11/THEECO2.pdf)

## Repo and submission hygiene

Commit work on a dedicated branch. The project `kwerui/secretly-robots` was private at last check; plan a **public submission copy** when the organisers require it. Remove secrets and personally identifying data from files and commit history before publication. Keep the original project folder and robot code safe; this package is delivered separately for review, not pushed automatically.
