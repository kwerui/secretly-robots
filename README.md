# Secretly Robots — offline-first AI brain starter

**AgentVerse 2026 · Robotics track · 7–8 November 2026**

**Mission:** A simulated household assistant helps an older adult with limited mobility by fetching a light object. The hero demo: a person on a sofa asks for a water bottle; the robot discovers it in a cupboard drawer, opens the drawer, picks up the bottle and brings it back. During a run, the object can be moved, and the robot should use fresh observations to adapt.

> **Status — engineering scaffold, not a functioning autonomous fetching robot.** This package has an offline intent parser, high-level state-aware task planner, validation, event logs, a minimal real MuJoCo physics scene and tests. It **does not yet** have camera detection, actual voice recognition, articulated grasping, drawer motor control, live sensor-to-brain integration, or genuine one-shot visual imitation learning. Those require teammates' robot and perception implementation.

## Quick start (Mac / Python 3.14.3 already works for your local MuJoCo)

Open a terminal from this project root (a cloned copy or extracted folder):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m brain.demo
python scripts/smoke_test.py
```

- `pytest`: verifies the brain and existing scaffold rules.
- `brain.demo`: **hand-authored observations** showcasing decisions and re-planning; *does not move a robot*. Event log in `logs/brain_demo.jsonl`.
- `smoke_test`: **real MuJoCo physics** for a simple example scene and state logging; *does not run the AI brain on a robot*.

To open the minimal MuJoCo scene:

```bash
python -m mujoco.viewer --mjcf=scenes/smoke.xml
```

You can use the virtual environment you've already created in your original neutral starter, rather than rebuilding it: copy the `brain/`, `docs/` and `tests/test_brain.py` files into that project and run the commands there. **Don't overwrite teammates' files without agreeing on the repository structure.**

## Architecture in 30 seconds

```text
User: "Bring me the bottle"
             |
  Intent understanding [offline rules / OPTIONAL local Gemma 4]
             |
    FetchTask(item_id="water_bottle", destination="sofa")
             |
  Fresh sensor-derived observation  <--- MuJoCo + teammates' perception code
             |
   Task planner selects NEXT action based on current world state
             |
  Deterministic Python guard checks whether action is permitted
             |
   Command --> teammates' movement/grasp/controller functions
             |
  MuJoCo performs movement; perception checks actual outcome
             |
              Feedback -> replan
```

**You own the high-level AI brain.** Teammates own the simulated robot, camera/perception, arm motion, physical control and verified result signals. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/INTEGRATION_CHECKLIST.md`](docs/INTEGRATION_CHECKLIST.md).

## How the brain makes decisions

```python
from brain.engine import Brain
from brain.contracts import Observation, ObjectSighting

brain = Brain(log_file="logs/brain_events.jsonl")
brain.start_task("Please bring my water bottle")

# Until connected to camera/depth perception, this is a MANUALLY CREATED sample.
obs = Observation(
    timestamp_s=0.0,
    robot_area="cupboard",
    visible_objects=(ObjectSighting("water_bottle", "cupboard", "drawer", .95),),
    drawer_open=False,
)
print(brain.next_action(obs))  # Decision(action='open_drawer', ...)
```

Then the simulation team drives the drawer's actuator, reads new sensor state, and asks for the **next** decision. A moved bottle changes the next target. If the person is too close or sensors become unavailable, the decision is `stop`; repeated failures escalate to human assistance. The deterministic checks are guardrails, **not full collision or real-world safety assurance**.

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

Use the opt-in `brain/gemini_optional.py` adapter only if you decide to demonstrate genuine Gemini 4 Argon API usage. This requires `pip install "google-genai>=2.3.0"`, `GEMINI_API_KEY` as an environment variable and network access; its call path has **not been live-tested with credentials**. The primary offline loop must keep working if Gemini cannot be reached. Never commit credentials or disguise rules as Gemini output. For the Best Use of Gemini prize, add a meaningful, verified multimodal cloud call *if the team has time*.

## What is already implemented vs. not

| Feature | Status |
| --- | --- |
| MuJoCo installation / minimal physics stepping | ✅ Implemented and previously user-tested |
| Validated high-level fetch-task decisions | ✅ Implemented in this scaffold; unit-tested |
| Changes to observed bottle location alter next action | ✅ Unit-tested on synthetic observations |
| Human-near / lost-sensor stop and retry limit | ✅ High-level decision logic; real emergency braking must be wired |
| JSONL decision and feedback logging | ✅ Implemented |
| Local model interpreting requests through Ollama | 🟡 Adapter implemented; requires Ollama/model; not live-tested here |
| Google Gemini cloud option | 🟡 Optional adapter; requires API credentials; not live-tested here |
| Simulated cameras/object-recognition/robot control | ⏳ Teammates' integration needed |
| Autonomous drawer opening and grasping | ⏳ Teammates' integration needed |
| One-shot learned manipulation policy from video | ❌ Not implemented; `teach_once.py` stores **symbolic** task goals only |
| Actual WebRTC call | ❌ Not implemented; browser mic/camera permission test exists in `web/` |

## File guide

```text
brain/
  contracts.py        # typed observation, task, decision and feedback messages
  intent.py           # deterministic parser + optional local Gemma/Ollama parser
  planner.py          # choose next high-level action from fresh observations
  safety.py           # deterministic command validation
  engine.py           # orchestration, retries, JSONL logs
  integration.py      # interface teammates implement for actual MuJoCo
  teach_once.py       # small symbolic goal memory; not motor imitation
  gemini_optional.py  # opt-in cloud Gemini interpreter
  demo.py             # hand-authored explanatory example
robot_core/            # original neutral MuJoCo smoke-test scaffold
scenes/smoke.xml       # real minimal physics test scene
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
