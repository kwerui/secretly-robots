# G1 reproducible baseline evaluation

Phase 1 evaluates the existing tidy-house bottle-fetch demo. It does not change locomotion, IK, manipulation, perception, or `sim/`.

Run one offline baseline from the repository root:

```bash
cd assistive_g1
python house_brain_demo.py --offline --request "I'm thirsty" \
  --results-dir ../evaluation_results/single
```

Run repeated baseline trials:

```bash
cd assistive_g1
python benchmark_g1.py --trials 3 --output ../evaluation_results/baseline
```

Run the controlled drawer-first comparison without changing the default memory ranking:

```bash
cd assistive_g1
python house_brain_demo.py --offline --drawer-first --reuse-demo --request "I'm thirsty" \
  --results-dir ../evaluation_results/drawer_first_v3
```

Both commands need the normal G1 prerequisites first (`pip install -r requirements.txt` and `python fetch_assets.py`). The batch runner uses `--offline`, so it makes no paid cloud/API calls.

## What is recorded

Each task episode writes `<experiment-id>.json`; the batch root also has `results.csv` and `summary.md`. Records include the Git branch and commit, planner mode, layout, seed, actual bottle target/container/position, reset configuration (including doors), requested item, action-stage outcomes, retry count, timing, failure details, decision provenance, and non-sensitive cloud proposal events.

`success`, `failure`, `not_attempted`, and `unknown` are distinct. Action fields (`navigation`, `drawer_opening`, `grasp`, `drawer_closing`, and `delivery`) describe only actions that were actually executed. `physical_delivery`, `drawer_closed_in_sim`, `grasp_attachment_released`, and `item_remained_on_tray` are separate simulator end-state checks. A tray check can therefore fail while `delivery` remains `not_attempted`; it never rewrites the action history. Each result also preserves the first failed high-level action, raw per-station navigation attempts, and before/after telemetry for each drawer opening (estimated base pose, arm commands/palm position, and actual drawer displacement). A failed drawer opening remains distinct from the enclosing `search` failure.

Historical result files are immutable evidence. The initial baseline used schema version 1 and conflated the tray end-state with `delivery`; it is retained unchanged. Schema version 2 is also retained unchanged; it exposed the navigation aggregation and no-grasp verification issues corrected in schema version 3. Its JSONL contains enough evidence to identify the first failed high-level action as `search(water_bottle)`, but not the individual station attempt, because that telemetry did not exist yet.

The current tidy baseline is deliberately a repeatability check, not a scenario sweep: `G1House.reset()` defaults to `jitter=False`, and `house_brain_demo.py` explicitly calls `reset(layout, "kitchen_drawer", seed=seed)`. Changing `--seed` alone does not claim a different bottle-placement scenario; every record includes the actual configuration.

## Independent simulator-truth verification

After planning/control completes, `MujocoTruthReader` reads simulator state outside `RobotBridge.latest_observation()`. The AI brain still receives only camera-derived sightings, robot state, and other normal bridge observations.

For the requested item, the evaluator checks:

- tray containment from its MuJoCo body origin after hand-over and again after `--settle-s` simulated seconds (default 1.0);
- that the matching grasp weld is inactive after settling;
- kitchen top-drawer travel is at most 0.02 m (closed threshold);
- the WBC fall signal.

Tray containment requires the item's origin to be within the tray's 0.15 m half-extents with a 0.01 m edge inset, and its height to be from 0.04 m below to 0.35 m above the tray top. The threshold accommodates the bottle's body origin being at its base and is documented rather than hidden in a skill result.

These are useful simulator ground-truth checks, but are not a real-world guarantee: a body-origin-in-tray check does not prove stable support contacts or correct orientation, and `fallen()` is the existing WBC fall detector rather than an independently derived contact criterion. The evaluator reports `unknown` when truth is unavailable rather than inferring success.

## Decision provenance and privacy

`deterministic_planner`, `gemini_argon`, and `fallback_after_cloud_rejection_or_failure` are distinct provenance values. The brain JSONL log now records accepted and rejected cloud plans. It records the validated action/reason or an error reason only; it never records an API key or the raw cloud request. Offline runs record the deterministic planner and do not claim Gemini use.
