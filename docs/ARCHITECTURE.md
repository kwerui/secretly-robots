# Architecture and ownership

## What runs where

```text
                  LOCAL COMPUTER (no network required for basic operation)
User request (text, voice transcript later)
           |
      Intent parser
    /             \
 Rules offline    Gemma 4 via LOCAL Ollama (optional; pre-downloaded weights)
    \             /
         FetchTask(item_id, destination)
                   |
MuJoCo camera/range/contact -> perception team's detector -> Observation
                   |                                    |
                   +--> Python high-level planner <-----+
                                  |
                      deterministic validation
                                  |
                       high-level decision
                                  |
                  robotics team's action controller
                                  |
             MuJoCo dynamics + fresh sensor readings
                                  |
                    verified feedback / JSONL log
                                  |
                              REPEAT

Optional CLOUD Gemini 4 Argon: opt-in intent interpretation only;
never required for the base demo and never silently used as fallback.
```

## Division of responsibilities

| Layer | Proposed owner | Contract |
| --- | --- | --- |
| AI brain / task interpretation / plan + replanning | You | `Brain.start_task(text)` and `Brain.next_action(observation)` |
| Scene and robot (quadruped + arm if feasible) | Robot team | Valid MuJoCo MJCF, actuators, sensors |
| Perception (camera/depth/object detection) | Perception teammate | `PerceptionBridge.latest_observation()` |
| Locomotion, drawer opening and grasping | Robot team | `ControllerBridge.execute_high_level(decision)` |
| Integration / testing / demo | Shared | Real action feedback, tests, one-take video |

## Example state and action contract

**Observation from teammates:**

```python
Observation(
    timestamp_s=10.0, robot_area="cupboard",
    visible_objects=(ObjectSighting("water_bottle", "cupboard", "drawer", 0.92),),
    drawer_open=False, sensors_ok=True, person_too_close=False,
)
```

**Your AI brain returns:**

```python
Decision(action="open_drawer", target="drawer", reason="Object is inside a closed drawer")
```

Only the team's controller translates `open_drawer` into joint control and a grasp attempt. The AI brain MUST NOT set torques or lie about the grasp outcome. At the end, the controller/perception pipeline reports updated observations and verified action feedback. Replanning uses that new evidence.

## Offline / cloud modes

- **Rules (default):** offline and instant, but limited to predefined commands; this is NOT a learned or generative AI model.
- **Gemma 4 via Ollama:** a REAL, fully local text-model call for request interpretation (after downloading weights). It does not do navigation or perceive images in the current scaffold. Failure is explicit; no fake responses.
- **Gemini 4 Argon:** optional cloud API adapter for contest demonstration or comparison. Requires Wi-Fi and a key; not part of the offline base. Most of the system works if it is unavailable.

Local model memory, size and latency depend on hardware. Measure them on the intended demo device; do not claim unsupported throughput or inference timings.

## Learning from one demonstration

The `teach_once.py` component currently stores the *symbolic task goal* (which item goes to which destination) from a verified demonstration record. This is not yet learning an arm-grasp policy from a video or from one teleoperation trajectory. Honest language for judging: “We store and generalise the demonstrated task goal; low-level skills are pre-implemented.”

## Safety boundaries

- `person_too_close` or missing sensors -> `stop`.
- Unknown object / repeated failure -> ask human to clarify or help.
- Sensor confidence below threshold -> search/inspect, not grasp.
- Do not dispense medication, make health decisions, detect emergency incidents, or initiate purchases in the MVP.
- Human controls override all AI output; a real collision-aware robot-specific controller is still necessary.
- No raw camera data persisted by default. JSONL logs include only interpreted observations, descriptions and actions; audit logs before publishing if real user data is ever introduced.
