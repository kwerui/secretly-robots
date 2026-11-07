# Secretly Robots — two-day hackathon execution roadmap

**Dates:** 7–8 November 2026. **Target:** a simulated assistive robot that fetches a light bottle for an older adult seated on a sofa. The bottle starts in a drawer; the scene may be changed during the run. **Stack:** MuJoCo + Python, optional local Gemma 4, optional cloud Gemini demonstration.

## Non-negotiable scope

ONE room layout, ONE bottle, ONE drawer, ONE delivery target, ONE unannounced change. Prioritise a reliable complete sequence over adding extra tasks. If opening a real drawer/grasping is too hard, demo only the verified subset and explain the limitation; do not script or fake the rest.

## Priority 0: team agreement (30–45 min)

- [ ] Decide which MuJoCo quadruped/mobile manipulator model and arm controller to use.
- [ ] Confirm whether the carer demonstration is **symbolic task teaching** or a genuine teleoperated robot demo; do not claim one-shot motor imitation unless built.
- [ ] Agree on the `Observation` and `Decision` fields; identify who builds each bridge.
- [ ] Write down baseline success criteria: reach target, open drawer, grasp, deliver, verify, zero unsafe contacts.
- [ ] Assign a demo operator and an integration owner.

## Saturday — make one end-to-end loop

| Priority | Work | Exit criteria | Suggested owner |
|---|---|---|---|
| P0 | Load robot and mock home in MuJoCo | Robot, drawer, table, bottle and sofa visible | Simulation team |
| P0 | Implement basic locomotion and manipulation calls | Controller can navigate/open/grasp/place with manually chosen targets | Control team |
| P0 | Agree on sensor observations | Position and drawer/object state derived from the actual simulation | Perception team |
| P0 | Integrate AI brain with the fresh observation bridge | `Brain.next_action()` emits correct grounded high-level action | You |
| P0 | Validate command, log action and outcome | Failure cannot be mistaken for success | You + controller team |

**Checkpoint:** A real (possibly manually assisted) MuJoCo pick-and-deliver trial that uses fresh state. If not achievable, simplify robot/environment by end of Saturday rather than working on extra features.

## Sunday morning — earn the high-value judging marks

| Priority | Work | Exit criteria |
|---|---|---|
| P0 | Replace fixed object position with live perception | Moved bottle updates detection and selected action |
| P0 | Demonstrate replan | Object moved mid-run -> robot revises goal/route rather than replaying old positions |
| P0 | Safety stop | Person-approach or sensor-failure event visibly halts action |
| P1 | Local model inference | Gemma 4 via Ollama interprets a real request without Wi-Fi; record actual latency |
| P1 | Teach-once symbolic goal | Carer stores object/destination once, robot reattempts using current perception |
| P2 | Cloud Gemini prize path | One meaningful optional Gemini call, verified locally, never required for basic autonomy |

**Freeze:** Stop feature work at least **two hours before the confirmed submission deadline**, then rehearse and fix only blockers.

## Submission and demo

- [ ] **One-page technical summary:** user, problem, system diagram, components/models, where inference runs, observed latency/compute, safety, limitation and failure response.
- [ ] **Up to 60-second continuous simulation video** showing perception alongside real actions and an unscripted change (from the track slide).
- [ ] **Five-minute live pitch** for the track judges: need, working demo, evidence of adaptation, limits, impact.
- [ ] Backup screen recording; explain honestly if used instead of live.
- [ ] Public repository copy with source, setup instructions, test output, acknowledgement of dependencies, no API keys.

The general Google Builder Guide also suggests a sub-three-minute live *working-app demonstration*. Rehearse a short demo segment inside the longer five-minute track pitch; confirm timings with organisers.

## Fast no-go decisions

1. **Drawer/arm not controllable:** switch to pickup from an open shelf; mention drawer as future work.
2. **Object detection unreliable:** start with simulated ground-truth object poses explicitly disclosed as an engineering baseline; do not pretend camera AI works.
3. **Local model too slow/no RAM:** use deterministic offline rules for demo; show model upgrade separately only if actually working.
4. **No safe grasp:** stop rather than forcibly moving object; show a verified navigation and planning demo.
5. **No Wi-Fi:** continue the offline pipeline; do not depend on cloud to progress.

## Ownership proposal

- **You:** local/offline AI brain, intent interpretation, high-level task planning, adaptive replanning, structural validation and decision logs.
- **Robotics teammate(s):** quadruped/arm in MuJoCo, low-level control, grasp and placement.
- **Perception teammate:** camera/sensor-to-observation translation and confidence.
- **Whole team:** integration, live scenario changes, realistic testing and submission.
