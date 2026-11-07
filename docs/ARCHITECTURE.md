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

Optional CLOUD Gemini 4 Argon: opt-in request interpretation and one current
high-level decision proposal per task-relevant observation change. It never
controls motors and is never required for the base demo.
```


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
- **Gemini 4 Argon:** optional `google-genai` structured-output adapter. It converts a request into a checked `FetchTask`, or proposes one checked next `Decision` from the latest interpreted observation. The engine calls it only after a task-relevant state change, never merely because a physics tick advanced. Local schema checks and `safety.validate` independently reject invalid or unsafe proposals, then retain the deterministic planner's decision. It requires Wi-Fi and `GEMINI_API_KEY`; it is not part of the offline base.

## Gemini boundary and failure handling

`brain/gemini_client.py` is the only module that imports the optional official SDK and it requests `application/json` with an explicit response JSON schema, following the [Google Gen AI SDK structured-output API](https://googleapis.github.io/python-genai/#json-response-schema). `brain/gemini_optional.py` validates exact keys, permitted task fields and existing action names before a `Decision` can cross into the engine. `brain/engine.py` validates every selected decision again against fresh observation with `brain/safety.py`.

If the SDK is not installed, no key is configured, the request times out, the internet is unavailable, JSON is malformed, a field is invalid, or local safety rejects a proposed target, the engine uses the local deterministic plan. If a cloud intent parser fails, `Brain` falls back to the offline rules parser. API keys are read only from `GEMINI_API_KEY` at runtime and are never logged.

The planner operates on `Observation`, which is interpreted sensor data supplied by teammates; it does not claim camera detection, raw-image reasoning, robot navigation, drawer actuation or grasping has been implemented. `ControllerBridge.execute_high_level()` remains the interface that robot-control code must implement.

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
