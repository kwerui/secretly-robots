# Grasping: why physics + a verified scripted pinch, not a VLA or an RL policy

Branch `grasp-physical`, 8 Nov 2026. Companion to `assistive_g1/README.md` ("Physical grasp").

## The problem we had

The `G1-Integration` branch reached for items and **welded** them to the palm. That was not
laziness: NVIDIA's GR00T `g1.xml` (the model the branch uses for its locomotion policy) ships
the Dex3-1 hand with all seven finger joints and their motors **commented out**, so the hand was
a rigid block and a constraint was the only way to hold anything.

## Options considered

### 1. A vision-language-action model (GR00T N1.6 / N1.7 for the G1)

Checked, not guessed:

- Zero-shot, the public GR00T N1.6 checkpoint scores **0 %** on manipulation in scenes it was
  not post-trained for: 0/20 on SimplerEnv pick-coke-can (ETH Robotics Club write-up), 0/68 on
  the RoboGate industrial scenarios (arXiv 2603.22126, all 68 failures grasp misses). The
  fine-tuned variants work only in the scene/embodiment they were fine-tuned on.
- The one G1 pick-and-place fine-tune that exists (`nvidia/GR00T-N1.6-G1-PnPAppleToPlate`,
  community re-run ~50 % closed-loop) is for the GR00T-WholeBodyControl apple-to-plate scene with
  its own cameras and object; adapting it to our flat, our bottle and our Dex3 grasp means
  collecting teleop demonstrations, converting to LeRobot format and post-training on a CUDA
  GPU (the published recipes use 10 k steps on an 80 GB card), then serving a 3 B-parameter
  model next to the sim. Not a 24-hour job, and not an offline-first one.
- The control group in the RoboGate paper is the useful fact for the pitch: on the same 68
  scenarios where the VLA scored 0, **a scripted IK grasp scored 100 %**.

### 2. An existing RL grasping policy

There is no published RL policy for a Dex3-equipped G1 grasping tabletop objects that we could
drop into a custom MuJoCo scene; the available G1 policies are locomotion (GR00T WBC, MuJoCo
Playground, Holosoma). Training one ourselves (MJX/Playground, domain-randomised) is a sound
two-week project and a poor overnight one.

### 3. Physics + a scripted, verified pinch (what we did)

Enable the fingers, tune the contacts the way MuJoCo's documentation recommends against slip,
plan the approach with the existing collision-aware IK, close under position control up to the
hardware torque limits, and **verify** the grasp from contact forces before trusting it. This
is what the real robot stack does for a first deployment too (perception -> grasp pose ->
scripted closure -> force check), and it is auditable: every grasp logs its contact evidence.

## What it took to make it real

Things that were invisible with the weld and bit immediately with physics:

| Symptom | Cause | Fix |
|---|---|---|
| Fingers slammed to their joint limits | MjSpec actuators need `biastype=affine` for a position servo; array fields must be assigned whole (item assignment hits a copy) | `hand.py` |
| Fingers jammed in the palm | MuJoCo only skips parent-child contacts; finger meshes sit inside the palm hull | hand-internal exclusions |
| Item "between the fingers" was not where the fingers close | measured the real pinch equilibrium of a 6-7 cm item: x 0.10, y 0.07 in the wrist frame, not the palm centre | `right_grasp` site, second IK view |
| Bottle knocked over before the grasp started | the open hand is 22 cm long; the old "raise hand" pose put the fingertips at the item's stand-off distance | fingers-left raise when the item is within 50 cm |
| Item slid out during the lift from a top-down (floor) pinch | pinch near the fingertips; gravity along the finger axis squeezes it out | documented limitation; 45-deg candidates tried, not reliable from the squat |
| Item knocked over at release | arm springs back ~3 cm when the finger torques come off; thumb swing-back arc passes through the item | closed-loop release (site servoed in place + 2 cm ease), `Q_RELEASE` without swing-back, withdraw sideways -> up -> back |
| Second item placed into the first / wrist swept the first over | placed position recorded before the item finished sliding; clearance measured from the pinch, not the hand/forearm | record settled position; capsule footprint of hand + forearm; fill the table far-left first |
| Cupboard pick failed intermittently | open leaf (0.05 N*m hinge friction) drifted shut when the elbow brushed it; stance blocked by the leaf | hinge friction 0.5; stance retry further left; fingers close around the handle |

## Merge onto the house (8 Nov, 03:50-05:30)

While this was being built, `G1-Integration` moved the robot into Jack's `HouseEnv` (`house.py`:
drawers, Jack's bottle as a cylinder, a delivery tray, his planner) and refactored `skills.py`
around a `world` object. The grasp was ported onto that version by hand (nothing of theirs
dropped), the fingers are added in `scene.add_robot_extras()` so both worlds get them, the
house's item welds and hand/item exclusions are gone (drawer-handle welds stay), and the
from-above drawer pick is now physical. What that cost:

| Symptom | Cause | Fix |
|---|---|---|
| Raise jammed the arm behind the back at the 0.9 m counter | raise target above chest height flips the IK basin | cap the raise at chest height, retry from home |
| Drawer pick: hand rolled 26 deg on the way down, pinch off-centre | planner scored position only; the open drawer front is at elbow height | orientation residual in the planner score, steep-first candidates, firm orientation on the descent, roll gate |
| Bottle flipped at the tray | the "up" direction was recorded after the item had pivoted in the pinch | track the item's own axis in the hand; level during the carry, not inside the drawer |
| Bottle lost on the 4 m walk | gait sway works a fingertip pinch loose | walk at 0.3 m/s with something in the hand |

## Numbers (`python assistive_g1/eval_grasp.py --trials 6 --jitter 0.03`)

- Bottle from the table: 5/6 delivered upright. Cup from the closed cupboard: 6/6.
- 11/11 formed pinches survived lift + carry + walk; 41-45 N mean total normal force;
  0.5-0.7 mm mean slip after the lift, 1.3 mm max.
- Perception-error tolerance of the pinch: -2/+3 cm along the pinch axis, +-3 cm along the
  approach (bias injected into the detected pose).
- MPPI for the transport motions: same task success; commanded joint acceleration RMS -17 %,
  jerk RMS -18 %; +40 % wall time.

House demo (`house_demo.py --no-video`): drawer opened 37 cm, bottle pinched from the drawer
(42 N), carried across the house, delivered upright on the tray; the optional second item (cup)
is also fetched and delivered, but the tray is too small for two.

## How to say it to the judges

"The hand is the real Dex3 kinematics with the real torque limits; the item is held by friction
in MuJoCo's contact model, and the robot checks its own grasp from contact forces before it
trusts it. We evaluated VLA policies for this and chose not to ship one: the public evidence is
that they score zero outside their post-training scene, and the same papers show a verified
scripted grasp at 100 %. Where our approach fails - the floor pick from a deep squat - the
skill says so instead of faking it."

## Next steps if there is time

1. Regrasp: on `no stable pinch`, shift 1-2 cm towards the thumb side and retry once.
2. Physical door grip (same recipe; the fingers already close around the bar).
3. Floor pick: side approach from a kneel rather than a top-down pinch from a squat.
4. MPPI with batched numpy forward kinematics (STORM-style) to make it real-time, then use it
   for every arm move including the approach.
5. Domain randomisation of friction/mass in `eval_grasp.py` to show margin, not just success.
