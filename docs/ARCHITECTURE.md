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
MuJoCo -> ground-truth adapter (current) -> Observation
       \-> camera/range/contact -> perception detector -> Observation (future)
                                                   |
                         Python simulation coordinator
                                                   |
                           Python high-level planner
                                  |
                      deterministic validation
                                  |
                       high-level decision
                                  |
                  robotics team's action controller
                                  |
             ActionFeedback + updated MuJoCo observation
                                  |
                    verified feedback / JSONL log
                                  |
                              REPEAT

Optional CLOUD Gemini 4 Argon: opt-in request interpretation and one current
high-level decision proposal per task-relevant observation change. It never
controls motors and is never required for the base demo.
```

The agreed coordinator dataflow is:

```text
MuJoCo -> Observation -> AI Brain -> Decision -> Controller -> ActionFeedback -> updated Observation
```

The current adapter reads the teammate-owned `sim.HouseEnv` simulator ground
truth from its `MjModel`, `MjData` and bottle pose, and marks this in
`Observation.scene_note`. That makes it a useful integration baseline, but it
is not a claim that a camera detected the bottle. The eventual perception bridge
must derive the same contract from configured sensors and report calibrated
confidence. If no robot has been attached, the adapter reports
`robot_area="unknown"`; it never invents a robot position.


## Example state and action contract

**Observation from teammates:**

```python
Observation(
    timestamp_s=10.0, robot_area="unknown",
    visible_objects=(ObjectSighting("water_bottle", "kitchen_counter", None, 1.0),),
    drawer_open=False, sensors_ok=True, person_too_close=False,
    scene_note="MuJoCo HouseEnv ground truth; not camera perception. ...",
)
```

**Your AI brain returns:**

```python
Decision(action="navigate", target="kitchen_counter", reason="Move to freshly observed object location")
```

Only the team's controller translates high-level navigation and grasp requests into joint control. The AI brain MUST NOT set torques or lie about the grasp outcome. The current house has room doors, not a cupboard drawer. At the end, the controller/perception pipeline reports updated observations and verified action feedback. Replanning uses that new evidence.

## Offline / cloud modes

- **Rules (default):** offline and instant, but limited to predefined commands; this is NOT a learned or generative AI model.
- **Deterministic Python planner:** working offline fallback for every high-level decision.
- **Gemma 4 via Ollama:** optional fully local request interpretation after downloading weights; not yet confirmed live-tested. It does not do navigation or perceive images in this scaffold.
- **Gemini 4 Argon:** implemented, successfully live-tested optional `google-genai` structured-output adapter. It converts a request into a checked `FetchTask`, or proposes one checked next `Decision` from the latest interpreted observation. The engine calls it only after a task-relevant state change, never merely because a physics tick advanced. Local schema checks and `safety.validate` independently reject invalid or unsafe proposals, then retain the deterministic planner's decision. It requires Wi-Fi and `GEMINI_API_KEY`; it is not part of the offline base.

## Gemini boundary and failure handling

`brain/gemini_client.py` is the only module that imports the optional official SDK and it requests `application/json` with an explicit response JSON schema, following the [Google Gen AI SDK structured-output API](https://googleapis.github.io/python-genai/#json-response-schema). `brain/gemini_optional.py` validates exact keys, permitted task fields and existing action names before a `Decision` can cross into the engine. `brain/engine.py` validates every selected decision again against fresh observation with `brain/safety.py`.

If the SDK is not installed, no key is configured, the request times out, the internet is unavailable, JSON is malformed, a field is invalid, or local safety rejects a proposed target, the engine uses the local deterministic plan. If a cloud intent parser fails, `Brain` falls back to the offline rules parser. API keys are read only from `GEMINI_API_KEY` at runtime and are never logged.

The planner operates on `Observation`. The current MuJoCo adapter supplies explicitly labelled simulation ground truth; future teammates' perception supplies interpreted sensor data. Neither claim means camera detection, raw-image reasoning, robot navigation or grasping has been implemented. `ControllerBridge.execute_high_level()` remains the interface that robot-control code must implement.

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
