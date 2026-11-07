"""Basic structural checks, *not* robot-specific collision safety checks."""
import math
from .contracts import Action, Observation

class InvalidAction(ValueError):
    pass

def validate_action(action: Action, observation: Observation) -> None:
    """Raise if a command has inconsistent or invalid arguments.

    A real controller must also check reachability, collisions, limits, etc.
    """
    if action.kind == "stop":
        if action.target_id is not None or action.target_xyz is not None:
            raise InvalidAction("stop must not have a target")
        return
    if action.kind == "report":
        return
    if action.kind == "navigate":
        if (action.target_id is None) == (action.target_xyz is None):
            raise InvalidAction("navigate needs exactly one target_id or target_xyz")
    else:
        if action.target_id is None:
            raise InvalidAction(f"{action.kind} needs target_id")
    if action.target_xyz is not None:
        if len(action.target_xyz) != 3 or not all(math.isfinite(v) for v in action.target_xyz):
            raise InvalidAction("target_xyz must be three finite numbers")
    if action.target_id is not None and action.kind in ("pick", "inspect"):
        known = {obj.id for obj in observation.objects}
        if action.target_id not in known:
            raise InvalidAction("target_id was not found in current observations")
