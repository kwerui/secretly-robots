# G1: self-localization (LiDAR + RGB-D + IMU + leg odometry) and the drawer fix

Overnight changes on the G1-Integration working tree (uncommitted). Everything here runs in
Jack's house (`house.py`); the old one-room scene is unchanged.

## How to run

```bash
cd assistive_g1
python house_demo.py --viewer --with-cup         # drawer: open -> pick from above -> push shut -> deliver; then the cup
python house_brain_demo.py --viewer --reuse-demo # Gemini brain on top (the demo is re-recorded once if
                                                  # drawer_model.json predates the depth camera; see below)
G1_LOCALIZE=0 python house_demo.py --viewer      # same, but on the simulator's root pose (A/B)
ASSISTIVE_DEBUG=1 python house_demo.py --no-video  # prints the reach/close internals
```

Both summaries print the localization error at the end (estimate vs simulator truth,
evaluation only) and the LiDAR scan stats.

Delete `drawer_model.json` once (or run without `--reuse-demo`): the demo camera now uses depth,
and the fit is much tighter (residual 1.4 mm, opening 36 cm = the true 36 cm; it was 38-39 cm,
11-20 mm).

## Localization: what the robot uses instead of `data.qpos`

`state_estimation.py` (new) + `sensors.py` (copied from the Gemini-G1-Localization branch,
one bug fixed: `camera_pose_for_base` needed `mj_camlight`, it returned zeros).

| Sensor | Rate | Used for |
|---|---|---|
| Pelvis IMU (gyro + accelerometer, `data.sensordata`) | 500 Hz | attitude (roll, pitch, gravity), yaw rate; also the GR00T policy's angular-velocity and gravity inputs |
| Leg odometry (29 joint encoders + IMU tilt, stance foot) | 50 Hz | pelvis height, body velocity -> dead reckoning |
| 360° LiDAR, 8 horizontal slices x 180 rays (`mj_multiRay`) | 10 Hz | scan-to-map matching (Gauss-Newton, trimmed + Huber) against a 2.5 cm distance field of the house's static structure, built once at reset |
| Head RGB-D camera | per look | ArUco marker positions refined with depth (falls back to PnP if they disagree by > 10 cm); camera pose from the *estimated* base + encoders |

Fusion: complementary filter (odometry + gyro predict, an accepted LiDAR match pulls the pose
50% of the way each scan). A scan is accepted with > 40% inliers, RMS < 3 cm, jump < 0.3 m.
Open drawers, doors, people and items are not in the map; their points are trimmed as outliers.

Where the estimate is used: `G1WBC.base_pos / base_yaw / base_vel_body / fallen()` (so
navigation, the brain bridge and every skill), the arm IK's base (`ArmIK.root_pose`: planning and
`palm_pose` are in the frame the robot believes it is in, so camera targets and the arm agree),
the head camera's pose (`HeadCamera.pose_source`), and the GR00T observation (IMU gyro + AHRS
gravity). `true_base_*` on `G1WBC` is the simulator truth, for evaluation and sim setup only.

Measured (headless, tidy layout): 1-2 cm and < 0.3 deg error at the stations, worst 6-7 cm
while walking, every scan accepted across the house. Full `house_demo --with-cup` works on the
estimate end to end (bottle from the drawer and the cup, both delivered upright).

Still read from the simulator: the sensors themselves (a simulated LiDAR/camera is computed
from the true state), the start pose (a known dock pose at reset), arm gravity feed-forward
(`qfrc_bias`), and `holding_ok()` (item-in-hand check). The live occupancy map is done on the
grasp-physical branch (`costmap.py`, see `docs/NAV_NOTES.md`): the planner runs on what the
robot's LiDAR and depth camera have seen, over the floor plan.
`person_too_close` could come from the LiDAR (points not in the map, < 0.5 m).

## Arm / drawer: what was wrong and what changed

1. **Plans weren't executed.** `_choose` found a collision-free, elbow-down configuration and the
   arm then went somewhere else (Cartesian re-solve from the current pose, pulled to home).
   Now one planner (`_plan`) and one executor (`_reach`): joint-space swing to the planned
   pre-pose, straight-line approach with the IK's null space pulled to the *planned* angles,
   and a short joint-space finish to the planned target if the tracked approach stalls
   (collision rows near furniture). Used by pick (side / from above / floor), open and close.
2. **No overhead poses.** Comfort joint ranges in the IK (shoulder pitch >= -1.9, limited roll),
   and plans with the elbow > 5 cm above the shoulder are rejected (20 cm for top-down grasps,
   `G1_ELBOW_MAX`, `G1_ABOVE_ELBOW_MAX`).
3. **Drawer pick.** Hand comes in 12 cm above the item (was 30 cm: hand at shoulder height,
   elbow over the head), fingers pitched 20-50 deg forward first (from the grasp-physical
   branch: the forearm clears the drawer front lower), lifts 32 cm so the bottle clears the front.
4. **Held item stays upright.** The hold constraint is now "the item's axis stays vertical"
   (world up as seen from the palm at the grasp), not "the finger direction stays fixed": with a
   pitched hand the latter let the hand roll and tilt the bottle, which then fell off the tray.
5. **Drawer close = push.** The left fingertips push the front shut, 8 cm above the bar (the
   grip + walk lost the bar part-way). The path is paced by the feet (the target only moves on
   while the hand keeps up). Before the sideways move to the close stance the robot backs off
   first (the tucked bottle caught on the handle bar). `house.py`: no more left-hand/drawer
   contact excludes (the push needs the contact).
6. **Open from any stance.** If the handle can't be planned or the reach falls short, step in
   to 0.36 / 0.31 m, look again, re-plan (the brain run arrives from the counter, off-station).

The grasp-physical branch (Dex3 finger pinch + MPPI transport) was used for ideas only (pitched
top-down candidates, joint-space finish to the planned configuration); it is not merged.
