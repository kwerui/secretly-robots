"""High-level task planning grounded in the *latest* observation.

Never commands joint motors. The teammate-owned MuJoCo controller receives the
validated command and must handle motion, grasp planning and collisions.
"""
from .contracts import Decision, FetchTask, Observation

MIN_CONFIDENCE = 0.70


def choose_action(task: FetchTask, obs: Observation) -> Decision:
    """Choose the NEXT action; call again after every observed state change."""
    if not obs.sensors_ok:
        return Decision("stop", reason="Missing/unreliable sensors")
    if obs.person_too_close:
        return Decision("stop", reason="Person is within protected clearance")
    if obs.delivered_item == task.item_id:
        return Decision("complete", reason="Delivery verified by simulator state")
    if obs.holding_item is not None:
        if obs.holding_item != task.item_id:
            return Decision("ask_for_help", reason="Gripper holds an unexpected object")
        if obs.robot_area != task.destination:
            return Decision("navigate", task.destination, "Deliver the currently held item")
        return Decision("deliver", task.destination, "Place held item at user station")
    sightings = [o for o in obs.visible_objects if o.item_id == task.item_id and o.confidence >= MIN_CONFIDENCE]
    if not sightings:
        return Decision("search", task.item_id, "Object not confidently located")
    sighting = max(sightings, key=lambda obj: obj.confidence)
    if obs.robot_area != sighting.area:
        return Decision("navigate", sighting.area, "Move to freshly observed object location")
    if sighting.container == "drawer" and not obs.drawer_open:
        return Decision("open_drawer", "drawer", "Object is inside a closed drawer")
    return Decision("grasp", task.item_id, "Object observed at current location")
