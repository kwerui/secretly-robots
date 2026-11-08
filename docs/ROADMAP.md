# Secretly Robots — final demo roadmap

**Scope:** a simulation-only Unitree G1 assistive-fetch prototype. The final story is perception-driven search with adaptive location memory, learned drawer motion, verified delivery, and honest reporting of failure modes.

## What we can claim

- The G1 uses head-camera observations and a recency-weighted memory to choose where to search.
- The three-episode relocation benchmark verified physical delivery in **3/3** episodes.
- Only **1/3** episodes achieved the full-task criterion: episodes 1–2 left the drawer open, while episode 3 reloaded memory and searched the counter first.
- The system runs in MuJoCo. It is not real-world validated, medically certified, or safe for unsupervised use.

## What we must not claim

- Do not call 3/3 physical deliveries “3/3 task success.”
- Do not describe simulator truth checks as real-world perception or safety validation.
- Do not claim a carer demonstration teaches location memory live unless that exact flow is shown; the current memory benchmark reloads saved memory.
- Do not hide the weld-based grasp abstraction or the drawer-closing failures if asked.

## Recording run sheet

1. Start from a clean, known seed and show the on-screen perception/memory state.
2. Show the robot search according to its current memory ranking.
3. Relocate the target as part of the scenario; show that the next search decision follows the new observation or reloaded memory.
4. Show delivery verification separately from drawer-close verification.
5. If drawer closing fails, say so on camera and use the recorded successful full-task episode as the backup—label it as a recording.

## Pitch structure (five minutes)

1. **Need (20 seconds):** people with limited mobility can benefit from help with simple retrieval; this prototype supports carers rather than replacing them.
2. **System (45 seconds):** G1 in MuJoCo; camera observations → memory-ranked search → high-level planner → locomotion/manipulation → independent simulator verification.
3. **Evidence (60 seconds):** 3/3 verified physical deliveries; 1/3 full-task successes; drawer closing was the failure point in two episodes; episode 3 reloaded memory and searched the counter first.
4. **Demo (up to 60 seconds):** show a continuous run, including the changed-location or memory behavior.
5. **Limits and next step (30 seconds):** improve drawer-close reliability, test across more randomized layouts, and validate sensing/manipulation beyond simulation.

## Before publishing or submitting

- [ ] Keep only the current judge-facing README, roadmap, architecture/evaluation evidence and run instructions.
- [ ] Remove or redact internal handovers, team rosters, personal names, local setup notes and stale branch-specific claims from any public copy.
- [ ] Confirm `git status` is clean and that generated logs, models, videos, and `.env*` files are not tracked.
- [ ] Search the complete history for credentials before changing repository visibility.
- [ ] Label any prerecorded video as prerecorded and do not edit out a material failure without saying so.
