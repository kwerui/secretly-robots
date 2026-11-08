# Secretly Robots — perception-driven adaptive-memory robot prototype

**AgentVerse 2026 · Robotics track · 7–8 November 2026**

**Mission:** A Unitree G1 humanoid in MuJoCo helps a person with limited mobility retrieve a small household item. It uses head-camera perception, a recency-weighted location memory and fresh observations to search, manipulate a drawer, and deliver to a tray by the sofa.

> **Status — simulation research prototype.** The G1 integration supports camera-derived item sightings, location-memory search, learned drawer motion, locomotion, grasping and delivery in MuJoCo. The current relocation benchmark verified a physical delivery in all three recorded episodes, but only **1/3** met the full-task criterion because drawer closing failed in episodes 1–2. It is not a real-world robot, medical device, or reliable autonomous household product.

## Judge-facing benchmark snapshot

The latest three-episode relocation run is deliberately reported as separate checks rather than a single inflated success number:

| Measure | Result | Interpretation |
| --- | --- | --- |
| Verified physical deliveries | **3/3** | The item reached the simulated delivery tray and passed the simulator end-state check. |
| Full-task successes | **1/3** | A full success also requires the drawer to be closed; episodes 1–2 failed that check. |
| Adaptive-memory behavior | **Observed in episode 3** | After memory reload, the robot searched the counter first before continuing its search. |

See [`docs/G1_EVALUATION.md`](docs/G1_EVALUATION.md) for the definitions, evidence format and limitations. The benchmark is simulation-only; it does not establish real-world safety or reliability.

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

## Voice: the person asks, the robot listens (onboard)

Press **V** in the sim and the person on the sofa asks for water (clips made with Gemini text-to-speech, see `sim/README.md`). The robot hears it from its head camera position, transcribes it with Whisper on the laptop's CPU, and turns the text into a command state. A live window shows each transcript, how long speech-to-text took and the command it produced. Nothing in the robot's listening path uses the network.

```bash
sim/.venv/bin/pip install -r requirements-hearing.txt -r requirements-stt.txt
sim/.venv/bin/python -m brain.speech_to_text --download    # once, ~145 MB into models/ (gitignored)
sim/.venv/bin/mjpython -m scripts.voice_robot              # sim viewer + window; press V
sim/.venv/bin/python -m scripts.voice_robot --mic          # talk to it through the laptop mic instead
```

**Hints get a question first.** A request that names the drink ("I need water", "Can you get me something to drink?") sets the fetch command straight away. A hint that doesn't ("My mouth is so dry", "I'm parched", "I am thirsty") makes the robot ask *"Would you like some water?"* in its own Gemini voice (clips in `assets/robot_voice/`, made once with `python -m brain.robot_voice --generate`, played locally). The command state is `confirming` until the person answers: yes starts the fetch, no or 10 s of silence ends as `declined`. In the sim, press **Y** or **N** to answer; on the mic, just say it. While the robot talks, the mic ignores what it hears so it doesn't answer itself.

The window is at http://127.0.0.1:8765 and opens by itself. Each run is also logged to `logs/voice_events.jsonl`; replay one with `python -m brain.voice_window logs/voice_events.jsonl`.

### The G1 voice demo (the full sequence)

The Unitree G1 stands in the house, balancing on its GR00T policy. The person on the sofa speaks; the robot hears, interprets, asks if it needs to, and goes into **fetch mode**. Everything the robot does runs on the laptop: no network, no API key.

**One-time setup** (from the repo root, using the sim's virtualenv):

```bash
sim/.venv/bin/pip install -r requirements-hearing.txt -r requirements-stt.txt
sim/.venv/bin/python -m brain.speech_to_text --download        # Whisper base.en, ~145 MB into models/
(cd assistive_g1 && ../sim/.venv/bin/python fetch_assets.py)   # G1 model + GR00T policies, ~35 MB
# Optional, for free-form speech (see "Gemma" below):
brew install ollama && brew services start ollama && ollama pull gemma4:e4b   # ~9.5 GB
```

The voice clips are already in the repo (`sim/assets/voice/` for the person, `assets/robot_voice/` for the robot), so the demo needs no Gemini key.

**Run it:**

```bash
cd assistive_g1
../sim/.venv/bin/python voice_demo.py --list          # the 25 scenarios
../sim/.venv/bin/mjpython voice_demo.py --scenario 19  # pick one
```

On macOS the viewer must run under `mjpython`; on Linux use `../sim/.venv/bin/python`. The MuJoCo viewer opens with the G1 at the sofa, and the speech-to-text window opens in your browser at http://127.0.0.1:8765.

| Key | What happens |
|---|---|
| **V** | The person says one of eight water lines. Some are requests ("I need water", "Can you get me something to drink?"), some are hints ("My mouth is so dry", "I'm parched", "I am thirsty"). |
| **C** | The person asks for their cup ("Could you bring me my cup?", "I'd like my cup, please.", "Can I have my mug?"). |
| **Y** / **N** | The person answers the robot's question ("Yes, please." / "No, thank you."). |
| **F** | Toggle what fetch mode does: carry the fetch out (default) or only plan it. |
| **X** | Cancel and go back to listening. |

**Scenarios: showing adaptation.** There are 25 fixed scenarios, `5 × placement + setup`, all run through with the real G1 (`validate_scenarios.py`):

| Placement | Bottle (spawn odds) | Cup (spawn odds) |
|---|---|---|
| 0 | kitchen counter (30%) | kitchen counter (35%) |
| 1 | kitchen drawer (25%) | kitchen table (25%) |
| 2 | kitchen table (20%) | study desk (15%) |
| 3 | study desk (15%) | coffee table (15%) |
| 4 | coffee table (10%) | TV unit (10%) |

| Setup | Floor |
|---|---|
| 0 | tidy |
| 1 | light clutter (laundry) |
| 2 | heavy clutter (moving boxes) |
| 3 | living–kitchen door locked |
| 4 | living–kitchen door locked + clutter |

So scenario 19 is placement 3 (bottle on the study desk) with setup 4 (locked door + clutter). Each item has five spots with set probabilities (`assistive_g1/scenario.py`, `SPAWNS`), and the robot's location memory starts from the same odds. What adapting looks like:
- **Where things are:** it searches the likeliest place first. The markers over each place show its beliefs: they shrink and turn red as places are ruled out, and turn green where it finds the item.
- **Locked door:** it doesn't know which door is locked. It plans through the shortest doorway, finds the door shut when it gets there, says *"That door's locked. I'll go round the other way."*, marks it (red bar) and replans through the bedroom and study.
- **Clutter:** it plans and walks round it; the route being walked is drawn on the floor.

`--random --seed N` draws a new house from the odds instead (`--bottle`, `--cup`, `--lock-door`, `--clutter` fix parts of it).

**What you see:**

1. **Listening.** Grey light above the G1's head; top-left of the viewer says LISTENING.
2. **The person speaks.** A caption at the bottom; the G1's head-camera microphone hears it (about 2 m away, quieter with distance), and Whisper transcribes it in about 0.3 s. The window adds a card with the transcript, timing, loudness and distance.
3. **A hint** makes the robot say *"Would you like some water?"* in its own voice. The light turns amber and the viewer says ASKING. The question lapses after 10 s with no answer.
4. **A request, or a yes,** puts it in **FETCH MODE**. The light turns green and the robot says *"Okay, I'll get you some water."* (or *"…your cup"*). The brain now has the task `water_bottle → sofa`, and the robot carries it out with the G1 team's `RobotBridge`: search (opening the drawer with the learned motion if needed), grasp, walk to the sofa, put it on the tray, *"Here's your water."* (light turns blue). Each step is captioned and listed in the window with the robot's current beliefs. If it can't find the item it says so and asks for help (light turns red). With **F** (or `--plan-only`) it only shows the first step.
5. **A no** gets *"Okay. Just let me know if you need anything."* and it goes back to listening.

Other options: `--plan-only`, `--speed 2` (sim faster than real time while fetching), `--no-browser`, `--no-gemma`. The grasp uses the G1 team's fallback (`G1_CHEAT_GRASP=fallback`): it tries the real grasp first, and abstracts the contact if that fails.

**Test phrasing by talking to it.** `sim/.venv/bin/python -m scripts.voice_robot --mic` listens to the laptop mic. Each thing you say prints one line, FETCH / HINT / DECLINED / NOT A REQUEST, with which interpreter decided and how long it took. The robot answers out loud, and Ctrl+C saves a summary to `logs/phrase_test.csv`. Use headphones, or wait for it to finish talking. The rules handle the bottle and the cup ("Could I have my cup, please?", "I need my mug", "I want my bottle"), and Gemma handles freer phrasing if it's running.

**The pipeline** (all onboard):

```text
person's clip (Gemini TTS, pre-made) -> G1 head_cam microphone (brain/hearing.py: distance falloff, room noise, voice activity)
  -> Whisper base.en on CPU (brain/speech_to_text.py)
  -> interpreter (brain/command.py): rules first, instant; local Gemma 4 only for phrasing the rules don't know
  -> request / hint / yes / no / other -> CommandState: confirming -> understood | declined
  -> Brain.set_task(...) + planner's first step shown (assistive_g1/voice_demo.py)
```

**Gemma (optional).** The rules cover every scripted phrase and Y/N, so the demo works without Gemma. Gemma 4 (`gemma4:e4b` on the local Ollama server, `127.0.0.1` only) handles anything else, e.g. "I could do with something cold, it's warm in here" → hint, "pop over and grab my book" → request, "bring me my pills" → refused (no medication). Live-tested: about 0.2–0.45 s per answer once loaded.
- **The first load takes about 2 minutes** (9.5 GB of weights). The demo starts on the rules straight away and switches Gemma on when it's ready; the window's status line says when. Start the demo a couple of minutes before you present.
- It then stays in memory (about 9.4 GB) for 2 hours after its last use. On a 16 GB Mac that's a lot: free it with `ollama stop gemma4:e4b`, and stop the server with `brew services stop ollama`.
- Some Ollama builds (the MLX engine) answer `501` to schema-constrained output. The client then asks for JSON in the prompt instead and validates every field, so this needs no action.

**Troubleshooting**
- *Window opens but never shows the house:* make sure you're on the latest branch (an early version deadlocked the viewer).
- *Exit code 143 when you close the viewer:* normal under `mjpython`.
- *"Gemma 4 unavailable":* Ollama isn't running (`brew services start ollama`) or the model isn't pulled.
- *No sound:* the clips play with `afplay` (macOS) or `paplay`/`aplay` (Linux); the captions still work without audio.
- Logs: `logs/voice_events.jsonl` (everything the window shows) and `logs/brain_events.jsonl`. Replay a run with `sim/.venv/bin/python -m brain.voice_window logs/voice_events.jsonl`.

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

Gemma 4 runs on the laptop through [Ollama](https://ollama.com/download), using only `http://127.0.0.1:11434`:

```bash
brew install ollama && brew services start ollama
ollama pull gemma4:e4b            # ~9.5 GB, once, while online
python -m scripts.local_brain     # a real local request
```

The voice demo uses it for speech the rules don't recognise (see "Gemma" above): live-tested, about 0.3 s per answer once loaded, about 2 minutes for the first load. The planner and safety guard are **still Python**, not the LLM, and every model answer is validated. If Ollama is missing, the code says so rather than faking a result, and the rules carry on.

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
| Gemma 4 local inference through Ollama | ✅ Live-tested (`gemma4:e4b`, ~0.3 s per answer once loaded); optional, rules fallback |
| Voice: person speaks (Gemini TTS clips), robot hears, Whisper STT, interprets, asks, fetch mode | ✅ Onboard; headless tests drive the real G1, clips and Whisper (`assistive_g1/voice_demo.py`) |
| Gemini 4 Argon cloud option | ✅ Implemented and successfully live-tested; still optional and locally validated |
| G1 head-camera perception, locomotion, manipulation and delivery | ✅ Integrated in MuJoCo; benchmarked with the limitations reported above |
| Full-task drawer close + delivery reliability | 🟡 1/3 in the latest relocation benchmark; drawer closing failed in two otherwise delivered episodes |
| Location-memory teaching from a live carer demonstration | 🟡 Memory can be saved/reloaded; the current benchmark uses reloaded saved memory rather than a live teaching flow |
| One-shot learned manipulation policy from arbitrary video | ❌ Not implemented; drawer motion uses a fitted articulation model in the G1 simulation path |
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
  speech_to_text.py   # onboard Whisper (base.en, CPU): speech segment -> transcript
  voice_pipeline.py   # heard speech -> transcript -> command state, off the sim thread
  voice_window.py     # live speech-to-text window at http://127.0.0.1:8765
  command.py          # onboard interpreter (rules, then Gemma) + CommandListener: confirm, understood, declined
  robot_voice.py      # robot's spoken lines (Gemini TTS clips in assets/robot_voice/, played locally)
  gemini_client.py    # official google-genai structured-output boundary
  gemini_optional.py  # checked Gemini task/next-decision adapters
  demo.py             # hand-authored explanatory example
robot_core/            # original neutral MuJoCo smoke-test scaffold
scenes/smoke.xml       # real minimal physics test scene
sim/house_env.py        # teammate-owned house environment used by the adapter
sim/person_voice.py     # the sofa person's voice lines (Gemini TTS clips in sim/assets/voice/)
assistive_g1/voice_demo.py # the G1 voice demo: V speak, Y/N answer, fetch mode
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
