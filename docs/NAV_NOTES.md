# Perceived navigation: the robot maps what is in its way

Until this branch the planner read the house straight out of MuJoCo: `house.plan_path` built
Jack's occupancy grid from the true geoms every time it planned, so a chair that someone had
moved was either walked into (if it was moved *into* the route while the robot walked) or,
more quietly, never a problem because the planner had privileged knowledge. Now the clearance
map is built from the robot's own sensors and a floor plan, and the route is re-planned while
walking when a new obstacle cuts it.

`assistive_g1/costmap.py` (map + sensors), `house.plan_path` (planner on the map),
`skills.go_to` / `_walk_path` (re-planning), `nav_test.py` (evidence), `house_demo.py` (map inset
in the video). `G1_NAV=static` restores the old behaviour.

## What the robot is given, and what it has to see

| | source | justification |
|---|---|---|
| walls, doorways, where each door hangs | floor plan (seeded once at start from the scene as it is when the robot is switched on) | a home robot is given the plan of the flat; walls don't move |
| fixed furniture: counter, fridge, sofa, bed, bookshelf, chests of drawers, the seated user | same seed | doesn't move either (the localiser already matches its LiDAR against exactly this) |
| movable furniture (coffee table, kitchen table, desk, bedside table), clutter (chairs, boxes, laundry), people | **not given** – LiDAR + depth camera | this is what gets moved |
| locked doors | layout (as before) | the carer tells the robot which doors are locked |

The prior is soft: walls are near-permanent, fixed furniture takes ~9 clear scans through it to
be erased, so if a "fixed" piece is moved the map follows after a second or two.

## Sensors

* **3D LiDAR** (`HeadLidar`): Livox-style scanner on the torso (the G1 carries a Mid-360):
  120 azimuths × 16 elevations from −60° to +6° (1920 beams), 10 Hz, 8 m, `mj_multiRay` with the
  robot's own geoms and a carried item excluded. Ray-cast from the *true* sensor pose (physics),
  placed in the world with the *estimated* pelvis pose (state_estimation.py), like the
  localiser's own scanner. ~2 ms per scan.
* **Head RGB-D** (`HeadCamera.render_depth`): the depth image is back-projected and its points
  above the floor are added as hits, at every `look()` and once a second while walking. Dense
  where the robot looks: fills table tops and small things in front.
* Points below `FLOOR_Z + 1.5 cm/m` are the floor (attitude error at range); points within
  0.22 m (LiDAR) / 0.5 m (depth, which can't skip the robot's own hands) of the base are ignored.

## The map

Log-odds occupancy in a coarse voxel grid: Jack's 5 cm xy cells × 6 height bands over
0.05–1.65 m (6 × 121 × 161 floats). Every beam carves free space in the voxels it passes
through and marks its hit voxel. The bands matter: a beam that passes *under* a table top
clears only that band, the band with the table top (seen by another beam) stays occupied. The
planner's 2D map is "any band occupied"; clearance is its Euclidean distance transform
(`scipy.ndimage`), inflated by the robot radius in `plan_path`'s three tiers
(ROBOT_R + 0.12 / + 0.07 / + 0).

Two things a plain log-odds map gets wrong, and what is done about them:

* **A LiDAR sees the near face of a chair, not the chair.** The robot would plan right behind
  the face as if the chair were a sheet of paper. Hits on things that are not in the prior
  also give the 15 cm behind them along the beam a weak "occupied" vote (capped), so the
  object gets a volume; a beam that later passes through that volume clears it again.
* **Door leaves are not walls.** The floor plan says where each unlocked door hangs. In one
  frame the leaf's hits all lie at one bearing from its hinge over a run of radius (a thin
  plank), while a chair in the swing spreads over bearings. Per door the busiest 3° bearing
  bin in the swing half-plane (the wall it hangs on is just as radial, the other way) is the
  leaf; its bearing is remembered for frames that only glimpse the tip round the jamb. Hits
  within the plank's angular width are dropped, and so is the 24 cm strip of the opening
  itself. Anything else near a door is mapped like anything else.

## Planning and re-planning

* `house.plan_path` is Jack's BFS + line-of-sight shortening + squared doorway crossings, on the
  perceived clearance. New: the roomier tiers only win if they don't send the robot the long
  way round (≤ 1.3 × shortest + 0.5 m) – with a half-seen chair near a doorway, the 0.40 m
  tier can close that doorway while it is wide open at 0.28 m; the answer is through the
  door, not round the house. The squared doorway waypoints are checked against the map
  (lead 0.65 → 0.45 → 0.30 m) so they can't cut the corner of something standing beside a
  doorway, and a start already in a doorway gets no "pre" point (no doubling back). On the
  static map the routes are exactly as before.
* `skills._walk_path`: after every map update (10 Hz) the rest of the route is checked
  against the live map. Only cells that have become blocked *since the plan* count (the
  planner's own compromises near furniture are not news), with 8 cm of hysteresis (a
  half-seen chair "grows" as more of it comes into view), from 30 cm ahead of the robot
  (it weaves ±0.2 m about the waypoints), and not within 0.5 s of the last re-plan. When
  the route is cut, a new one is planned from where the robot is and walking continues at
  the current velocity – no stop. `MAX_REPLANS = 12` per `go_to`; no route → the skill
  returns `route to the <station> is blocked` for the brain to deal with.
* The final approach to a station is unchanged (stations sit inside the inflation).

## Evidence: `nav_test.py` (tidy house, seed 0)

Two chairs the floor plan does not know about:

* **A.** A chair 0.9 m ahead-right of the robot's start, on the straight line to the kitchen
  door. The robot has to see it before its first plan.
* **B.** On the way back from the drawers to the person, once the robot has come through the
  doorway, a chair is put down 1.3 m ahead of it, on the route it has already planned.

| | live map (`G1_NAV=live`, default) | static map (`G1_NAV=static`) |
|---|---|---|
| leg 1, home → kitchen drawers (chair A) | arrived, 26 s, 2 route refinements as more of the chair came into view, **0.06 s** of contact (a fingertip on the backrest while passing on the tight side; the hands hang ~0.4 m out when walking, Jack's ROBOT_R is 0.28) | arrived, 27 s, **3.6 s** of contact (its grid knows the chair exactly, but it planned the tight side at ROBOT_R) |
| leg 2, drawers → person (chair B) | re-planned **0.2 s** after the chair was put down, went round it on the north side, arrived, 26 s, **no contact** | never re-plans: pushed against the chair for **13 s**, did not arrive |

Localisation error during all of this: ≤ 2 cm, < 0.3°. Cost: +10 % wall time for the LiDAR
(1920 beams at 10 Hz) and the 1 Hz depth frames; the whole `house_demo --with-cup` still runs
at about real time headless.

`nav_test.py --video` records the run with the robot's own map as an inset (grey = floor
plan, black = seen, blue = route, red = where it thinks it is); `house_demo.py` has the same
inset.

## Honest limitations

* **Partial observation.** An obstacle is mapped where it has been seen (plus the 15 cm
  volume behind the hits). The first route past a half-seen chair can run close to its
  unseen side; the robot refines the route as it sees more, which is where the two leg-1
  re-plans and the fingertip brush come from. The planner's lowest tier (ROBOT_R = 0.28 m)
  is tighter than the walking G1 with swinging arms really is.
* **Starting next to an obstacle.** If the robot starts inside an obstacle's inflation (a
  chair 0.6 m from the dock) its first turn can sweep a hand into it; the planner starts from
  the nearest free cell but doesn't choose a "move away first" manoeuvre.
* **Door swings.** A leaf is recognised by its shape; a chair exactly on the leaf's bearing,
  or anything inside the 24 cm opening strip, is not mapped (Jack's layouts keep swings
  clear, so this is cheap; the dynamic person crossing a doorway is briefly invisible).
* **Low things.** The LiDAR's steepest beam is −60°, so anything lower than ~0.45 m within
  0.5 m, or lower than 0.12 m beyond 4 m, is only seen by the depth camera when the robot
  looks at it. A clothes pile (12 cm) is seen late.
* **No decay.** A cell stays occupied until a beam passes through it. A person who walked
  past leaves a trace until the robot looks there again (beams through that space clear it).
* The map is per-run (seeded at start); it is not saved between runs.
