# G1 manipulation: stance selection, arm path planning, batch testing

Uncommitted changes on the G1-Integration working tree (Sun 8 Nov, morning).

## What was wrong

A batch of headless runs (`batch_eval.py`) and a pick/close probe (`probe_pick.py`) showed the
same pattern behind most failures: the arm IK was fine, but **the robot stood in the wrong
place**, and when the straight arm swing was blocked, **the arm got stuck on its own torso**.

* Fixed standoffs (40 cm from the cup, 42 cm behind the drawer handle) don't fit every case.
  At the counter the 40 cm stance puts the hips and knees 5-6 cm *inside* the counter's
  cabinet, so the legs push against it and the base ends up wherever it gets pushed. That is
  why runs differ between machines.
* The drawer close only lined up if the walk arrived within 6 cm of the stance. With the
  drawer opened further than learned (it often ends at 42-43 cm), the handle was 44 cm to the
  right of the pelvis and the left arm couldn't reach it ("handle not reachable with the left
  hand"). This was the most common failure: 3 of the 4 runs that finished in the baseline batch.
* When the straight joint-space swing to the pre-grasp hit something, the fallback was a
  Cartesian move, which pinned the thumb against the torso (joint tracking error 2.5 rad).
* Leaning the torso (0.3 rad) while the arm reaches 50 cm forward makes the balance policy
  shuffle backwards: 36-47 cm of drift during one reach.

## What changed (`skills.py`, `ik.py`)

1. **Stance selection (where to stand), before walking.** This is the inverse-reachability
   idea humanoid planners use (stance selection from reachability maps, e.g. Burget &
   Bennewitz, ICRA 2015), done online: candidate stances (distance, side offset, heading,
   posture) are tried in the **IK's scratch copy of the model**, with the pelvis moved there and
   the legs/waist in a posture measured at start-up (`_calibrate_postures`, ~8 s of
   stretching). The search rejects stances where the legs or torso would intersect furniture,
   scores the arm's reach (error, collisions, raised elbow, joint-limit margin), prefers small
   moves, standing upright and reaches under 45 cm, and checks the winner still works from
   3 cm off in each direction (the feet never land exactly). Then the robot walks there once.
   Used by: side pick, top-down drawer pick, drawer close (left arm). `G1_STANCE_SEARCH=0` turns
   it off (the old fixed standoffs) for A/B.
2. **Step correction.** If a reach ends a few cm short (< 15 cm), the robot steps by the miss,
   looks again, re-plans and tries once more (pick, top pick, close). If no plan is found from
   where it ended up, it searches the stance again from there (`_restance`).
3. **Arm path planning.** If the straight swing to the planned pre-grasp would collide, an
   RRT-Connect planner (Kuffner & LaValle 2000) in the arm's 7-D joint space finds a way around
   the body and furniture, with shortcut smoothing, using the same contact check as the plans.
   It backs out first if the arm starts touching something.
4. **Push close accepts contact.** If the fingertips stop on the drawer front before the
   planned point (touch detected from contacts), the push starts from there instead of failing.
5. **Smaller fixes:** the side pick skips the far pre-grasp when it isn't cleanly reachable;
   `hand_over` refuses to let go if the robot isn't at the tray (it used to drop the bottle on
   the kitchen floor when the walk failed); the brain log prints each decision *before* its
   result (Gemini's "not holding anything" was the reason for the grasp, written before it).

## Testing

```bash
cd assistive_g1
python probe_pick.py --item water_bottle --close --opened 0.42   # 8 start offsets: drawer pick + push shut
python probe_pick.py --item cup                                  # 8 start offsets: counter side grasp
python batch_eval.py --seeds 0-3 --layouts tidy laundry_day -j 3 --tag after
python batch_eval.py --compare before after
G1_STANCE_SEARCH=0 python batch_eval.py ... --tag nosearch      # A/B on the same machine
```

Each sim process uses ~2.5 GB of RAM, so pick `-j` by memory as well as cores.

Results so far (this machine, headless):

| test | before | after |
|---|---|---|
| drawer pick + close, 8 start offsets (`probe_pick --close --opened 0.42`) | 1/8 | 6/8 (before the last two fixes) |
| full scripted demo, tidy + laundry_day seeds 0-3 (runs that finished) | 1/4 full; close 1/4 | see `logs/batch/` |

## Still shortcuts

The grasp is still a weld switched on at contact, `holding_ok()` still reads the item pose
from the sim, and the arm plans use the robot's own model of the furniture (the sim geometry),
not a live map.
