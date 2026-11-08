"""Deterministic checks, independent from any model's proposed decisions."""
from .contracts import Decision, FetchTask, Observation

class RejectedAction(ValueError):
    pass


def validate(decision: Decision, task: FetchTask, obs: Observation) -> None:
    if not obs.sensors_ok or obs.person_too_close:
        if decision.action != "stop":
            raise RejectedAction("Unsafe to issue any non-stop action")
        return
    allowed_targets = {task.item_id, task.destination, "drawer"} | {o.area for o in obs.visible_objects}
    if decision.target and decision.target not in allowed_targets:
        raise RejectedAction("Unknown action target")
    if decision.action == "grasp":
        if decision.target != task.item_id:
            raise RejectedAction("Grasp target does not match request")
        candidates = [o for o in obs.visible_objects if o.item_id == task.item_id and o.confidence >= 0.70]
        if not any(o.area == obs.robot_area and (o.container != "drawer" or obs.drawer_open) for o in candidates):
            raise RejectedAction("Item not verified at accessible grasp location")
    # (G1 integration, 8 Nov: was `obs.robot_area != "cupboard"`; Jack's house has several
    # containers, so: the robot must be at an area where the item was seen inside a container)
    if decision.action == "open_drawer" and obs.robot_area not in {o.area for o in obs.visible_objects if o.container}:
        raise RejectedAction("Cannot open a drawer remotely")
    if decision.action == "deliver" and (obs.holding_item != task.item_id or obs.robot_area != task.destination):
        raise RejectedAction("Cannot claim delivery unless robot has the item at recipient")
    if decision.action == "complete" and obs.delivered_item != task.item_id:
        raise RejectedAction("Task completion has not been verified")
