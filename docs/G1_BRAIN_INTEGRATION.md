# G1 ↔ brain integration (8 Nov, branch `G1-Integration`)

For Kristina (and her agent): what changed so `brain/` drives the Unitree G1 in Jack's house, and what to look at next.

## How it fits together

```
request ──► brain.Brain.start_task ──► FetchTask(item, "sofa")
                 │
   loop: brain/integration.take_turn(brain, bridge, bridge)
                 │  obs = bridge.latest_observation()        (assistive_g1/brain_bridge.py)
                 │  decision = brain.next_action(obs)          (planner rules, or Gemini if enabled)
                 │  brain/safety.validate(decision, task, obs) (unchanged role: vets every decision)
                 │  feedback = bridge.execute_high_level(decision)  ──► our G1 skills (assistive_g1/skills.py)
                 └► brain.record_feedback(feedback)   (repeated failures → ask_for_help)
```

`assistive_g1/brain_bridge.py` → `RobotBridge` implements both `PerceptionBridge` and `ControllerBridge`:

| Brain action | What the G1 does |
|---|---|
| `search(item)` | Visits stations in **location-memory** order (`assistive_g1/memory.py`: recency-weighted beliefs from watching the carer). At a container station (the kitchen drawers) it opens the drawer with the model learned from the carer's one demonstration, then looks again. Stops when the head camera sees the item. Fails only after every likely place has been checked. |
| `navigate(area)` | Walks there on Jack's occupancy grid (pushes unlocked doors). Areas are station names: `kitchen_counter`, `kitchen_drawers`, `kitchen_table`; `sofa` is the user's station. |
| `open_drawer` | Opens the container at the current station along the learned slide/arc. |
| `grasp(item)` | Picks it with the right hand (from above if it was seen inside a drawer). If something is open, **closes it with the left hand** while still holding the item. |
| `deliver(sofa)` | Sets the item on the user's tray beside the sofa. |
| `stop` | Stands still. |

**Observation** (no simulator ground truth for objects, unlike `brain/mujoco_observation.py`):
- `visible_objects`: where the robot's **head camera** last saw each item (ArUco), with `container="drawer"` for sightings inside a drawer. A sighting is dropped when the robot looks at that place again and the item is gone.
- `robot_area`: the station the robot is at.
- `drawer_open`: the robot has the drawer at this station open.
- `holding_item`, `delivered_item`: from the robot's own grasp/hand-over results.
- `person_too_close`: Jack's dynamic-person state (dynamic mode only).
- `sensors_ok`: false if the robot has fallen.
- The robot's base pose is estimated from simulated IMU, leg odometry, LiDAR and RGB-D;
  simulator truth is retained only for evaluation/setup, not as the navigation input.

## Changes to `brain/` (small, both marked "G1 integration, 8 Nov" in the code)
1. **`brain/safety.py`, `open_drawer` check.** It used to require `robot_area == "cupboard"`. Jack's house has several containers, so now the robot must be at an area where the item was seen inside a container.
2. **`brain/intent.py`, `RulesIntentParser`.** "thirsty" or "drink" now maps to `water_bottle` without needing a fetch verb, so the offline baseline understands "I'm thirsty". Requests like "Could I have my cup?" still need a verb offline ("Could you bring me my cup?"); Gemini handles the free-form ones.

Nothing else in `brain/` changed. The Gemini adapters are used as they are.

## Run
```bash
cd assistive_g1
python house_brain_demo.py                 # default: Gemini parses + proposes each step (needs GEMINI_API_KEY
                                           #   and google-genai); safety still vets; prints a WARNING and runs
                                           #   the rules if either is missing
python house_brain_demo.py --offline       # rules parser + deterministic planner only
python house_brain_demo.py --viewer        # watch it
python house_brain_demo.py --reuse-demo    # skip the carer demo, load drawer_model.json
```
Decisions and feedback are logged to `logs/house_brain.jsonl`.

Expected flow for "I'm thirsty":
- Memory says the counter (57%) → walk there, look: not there.
- Next the drawers (33%) → open → bottle seen → grasp from above → close the drawer with the left hand.
- Navigate to the sofa → deliver → complete.

"Could you bring me my cup?" goes straight to the counter, because the head camera saw the cup there during the first search.

## Open items for the brain work
- **Teach-once (agreed change).** `brain/teach_once.py` should add the place where the carer put an item as a new candidate location in the robot's location memory (`LocationMemory.observe_placement(item, station)`), not store a fixed item → destination.
- **Gemini prompt context.** `GeminiPlanner.propose` only sees the `Observation`. Passing it the memory ranking (`memory.ranked(item)`) and the list of stations would let it propose where to search, not just *that* it should search.
- **Item names.** The Gemini intent schema enum is `water_bottle`, `cup`, `book`; the house has `water_bottle`, `cup`, `pill_box`. The pill box was left out on purpose: the scope says no medication handling.
- **Areas.** If Gemini proposes `navigate`, the target must be one of the area names above or `sofa`, otherwise `safety.validate` rejects it ("Unknown action target").
