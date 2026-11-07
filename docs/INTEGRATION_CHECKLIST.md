# Integrating your AI brain with the robot team

Ask teammates for these five concrete details before touching their MuJoCo model:

1. Exact robot / MJCF model and names of actuators, drawers, objects and user target.
2. Which sensors produce object locations and how confidence is estimated.
3. Navigation API: how to command a named target location, and what signals completion.
4. Manipulation API: how to command open-drawer / grasp / place, and how to verify success.
5. Safe stop: how the controller actually arrests motion, not just stops issuing commands.

The interface in `brain/integration.py` has two protocols. The robotics team must implement both; the AI brain is decoupled from the joint-control code. Do not call the LLM on every MuJoCo physics step. Only re-evaluate the high-level plan after important scene changes or a completed/failed action.

## Definition of done (technical)

- One genuine MuJoCo simulation with object and robot bodies.
- At least one *actual* simulated sensor influences the next action.
- A user can ask for an object and the robot retrieves and delivers it (or fails safely).
- Moving the item during a run produces a changed observation AND different next action.
- An unknown/unsafe situation leads to a stop or request for assistance.
- Logs explicitly distinguish model proposals, accepted commands, attempts and verified outcomes.
- At least 5 live trials and a measured success rate; describe failure modes honestly.

## Demo proof requirement

Always show the robot's sensor perception, selected action and changed world state simultaneously. Don't pass typed fixture state, a hardcoded route or generated video off as sensor-driven autonomy. `python -m brain.demo` is a synthetic architecture tutorial ONLY.
