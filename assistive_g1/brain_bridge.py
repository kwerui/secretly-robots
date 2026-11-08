"""Kristina's brain (brain/) driving our G1 in Jack's house.

`RobotBridge` implements both interfaces in brain/integration.py:

  * PerceptionBridge.latest_observation(): an `Observation` built from what the robot itself
    knows: the items its HEAD CAMERA has seen (and where), the station it is at, whether it has
    a drawer open, what it holds and what it has delivered. Not simulator ground truth (unlike
    brain/mujoco_observation.py); the base pose comes from its state estimator (or the
    documented simulator-truth A/B fallback), not object ground truth.
  * ControllerBridge.execute_high_level(decision): runs our skills (skills.py) and reports an
    honest `ActionFeedback`:
        search(item)       -> visit stations in LOCATION-MEMORY order (memory.py); at a
                              container (drawer) open it with the learned one-shot model; stop
                              as soon as the head camera sees the item
        navigate(area)     -> go_to(station)       ("sofa" = the "person" station)
        open_drawer        -> open_articulated()   with the carer-demo model for this station
        grasp(item)        -> pick() (from above if it is in a drawer), then close_opened()
                              with the LEFT hand if something is open
        deliver(sofa)      -> hand_over() onto the user's tray
        stop               -> stand still

The brain's planner (rules) or Gemini proposes; brain/safety.py validates every decision
against the observation; this bridge executes. Areas = station names (see house.py), except
the user's station which the brain calls "sofa".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))  # repo root, so `brain` and `sim` import

import numpy as np  # noqa: E402

from brain.contracts import ActionFeedback, Decision, ObjectSighting, Observation  # noqa: E402

from articulation import ArticulationModel  # noqa: E402
from memory import LocationMemory  # noqa: E402
from skills import AssistiveRobot  # noqa: E402

SOFA = "sofa"
SEEN_CONFIDENCE = 0.95  # ArUco detection from the head camera (6-9 mm error): confident


@dataclass
class Container:
    """A thing at a station that has to be opened to see/take what is inside."""
    model: ArticulationModel  # learned from the carer's one demonstration
    anchor_marker: int
    weld: str
    part_markers: tuple[int, ...] = ()
    kind: str = "drawer"
    drawer: object | None = None  # optional simulator handle, evaluation-only


class RobotBridge:
    def __init__(self, robot: AssistiveRobot, memory: LocationMemory, containers: dict[str, Container],
                 say: Callable[[str], None] = print, search_first: tuple[str, ...] = ()):
        self.robot, self.memory, self.containers, self.say = robot, memory, containers, say
        self.sightings: dict[str, ObjectSighting] = {}  # item -> where the head camera last saw it
        self.delivered: str | None = None
        self.open_station: str | None = None  # station whose container this robot has open
        self.searched: set[str] = set()  # stations already searched for the current task
        self.search_first = search_first
        # Evaluation metadata; it is not exposed through latest_observation() and therefore
        # cannot become an input to the AI brain.
        self.last_execution: dict[str, object] = {}

    # ------------------------------------------------------------------ areas <-> stations
    @staticmethod
    def area(station: str | None) -> str:
        return SOFA if station == "person" else (station or "unknown")

    @staticmethod
    def station(area: str) -> str:
        return "person" if area == SOFA else area

    def new_task(self, item: str | None = None) -> None:
        self.searched.clear()
        self.memory.new_episode()
        self.task_item = item

    # ------------------------------------------------------------------ PerceptionBridge
    def latest_observation(self) -> Observation:
        r = self.robot
        person_close = False
        world = r.world
        if world is not None and getattr(world, "dynamic_controller", None) is not None:
            person_close = bool(getattr(world, "dynamic_state", {}).get("person_too_close", False))
        return Observation(
            timestamp_s=float(r.data.time),
            robot_area=self.area(r.station),
            visible_objects=tuple(self.sightings.values()),
            drawer_open=r.open_state is not None and self.open_station == r.station,
            holding_item=r.holding,
            delivered_item=self.delivered,
            person_too_close=person_close,
            sensors_ok=not r.wbc.fallen(),
            scene_note=self._context(),
        )

    def _context(self) -> str:
        """Plain-language context for an LLM planner (Gemini reads the whole Observation)."""
        areas = [self.area(s) for s in self.robot.stations if s != "home"]
        cont = ", ".join(f"{a} ({c.kind}: closed unless drawer_open)" for a, c in self.containers.items())
        task_item = getattr(self, "task_item", None)
        ranked = ""
        if task_item:
            ranked = "; likely places for the item, most likely first: " + ", ".join(
                f"{st} {p:.0%}" for st, p in self.memory.ranked(task_item) if st not in self.searched)
        return (
            "Unitree G1 humanoid in a house. Sightings come from its head camera (ArUco); its base pose "
            "from the simulator. Skills: search(item) walks to the likely places in order and looks, "
            "opening any drawer there with the motion learned from the carer; navigate(area); "
            "open_drawer (at the current area); grasp(item) (from above if it is in a drawer, then the "
            "other hand closes the drawer); deliver(sofa) puts it on the user's tray; stop; ask_for_help. "
            f"Areas: {', '.join(areas)}. Containers: {cont or 'none'}. Items inside a closed container "
            "cannot be seen until it is opened" + ranked + "."
        )

    def _look_here(self) -> set[str]:
        """Look with the head camera at the current station; update what we know is where."""
        st = self.robot.station
        seen = set(self.robot.look()["items"])
        container = self.containers[st].kind if st in self.containers else None
        for item in seen:
            if item != self.robot.holding:
                self.sightings[item] = ObjectSighting(item, self.area(st), container, SEEN_CONFIDENCE)
        for item, s in list(self.sightings.items()):  # looked here, it's gone: forget the old sighting
            if s.area == self.area(st) and item not in seen:
                del self.sightings[item]
        return seen

    # ------------------------------------------------------------------ ControllerBridge
    def execute_high_level(self, decision: Decision) -> ActionFeedback:
        a, t = decision.action, decision.target
        self.last_execution = {}
        try:
            ok, detail = getattr(self, f"_do_{a}")(t)
        except AttributeError:
            ok, detail = False, f"no skill for action {a!r}"
        self.say(f"  -> {a}({t or ''}): {'ok' if ok else 'FAILED'} {detail}")
        return ActionFeedback(a, t, ok, detail)

    def _square_up(self, station: str, tol: float = 0.05) -> float:
        """Re-drive onto the exact station pose if we stopped off it (the reach plans assume the
        station stance). Returns how far off we still are (m)."""
        s = self.robot.stations[station]
        def off():
            p = self.robot.wbc.base_pos
            return float(np.hypot(p[0] - s.x, p[1] - s.y))
        yaw_err = abs((self.robot.wbc.base_yaw - s.yaw + np.pi) % (2 * np.pi) - np.pi)
        if off() > tol or yaw_err > 0.1:
            self.robot._drive_to(s.x, s.y, s.yaw, tol=0.03, timeout=10.0)
        return off()

    def _do_search(self, item: str):
        """Visit likely stations in location-memory order until the head camera sees `item`."""
        ranked = self.memory.ranked(item)
        if self.search_first:
            priority = {station: index for index, station in enumerate(self.search_first)}
            ranked = sorted(ranked, key=lambda pair: (priority.get(pair[0], len(priority)), -pair[1]))
        for station, p in ranked:
            if station in self.searched:
                continue
            self.say(f"  searching for the {item.replace('_', ' ')} at the {station} (belief {p:.0%})")
            res = self.robot.go_to(station)
            if not res["ok"]:
                # "not arrived" often means a few cm off the exact pose: close enough to look/open
                s = self.robot.stations[station]
                off = float(np.hypot(*(self.robot.wbc.base_pos[:2] - np.array([s.x, s.y]))))
                if off > 0.35:
                    self.last_execution.setdefault("navigation_attempts", []).append(
                        {"station": station, "reported_success": False, "accepted_for_search": False,
                         "error": res.get("error", ""), "distance_from_station_m": round(off, 3)})
                    self.say(f"    couldn't get to the {station} ({res.get('error', '')}, {off:.2f} m off)")
                    self.searched.add(station)
                    continue
                self.robot.station = station
                off = self._square_up(station)
                self.last_execution.setdefault("navigation_attempts", []).append(
                    {"station": station, "reported_success": False, "accepted_for_search": True,
                     "error": res.get("error", ""), "distance_from_station_m": round(off, 3)})
                if off > 0.08:
                    self.say(f"    note: {off:.2f} m off the {station} pose")
            else:
                self.last_execution.setdefault("navigation_attempts", []).append(
                    {"station": station, "reported_success": True, "accepted_for_search": True, "error": ""})
            seen = self._look_here()
            if item not in seen and station in self.containers and self.open_station != station:
                ok, why = self._do_open_drawer(None)  # look inside, the way the carer showed us
                self.say(f"    opening the {self.containers[station].kind}: {'ok' if ok else 'FAILED ' + why}")
                if ok:
                    seen = self._look_here()
            self.searched.add(station)
            self.memory.observe_search(item, station, item in seen)
            if item in seen:
                return True, f"seen at {station}"
        return False, "searched every likely place"

    def _do_navigate(self, area: str):
        res = self.robot.go_to(self.station(area))
        self.last_execution.setdefault("navigation_attempts", []).append(
            {"station": self.station(area), "reported_success": bool(res["ok"]),
             "accepted_for_search": bool(res["ok"]), "error": res.get("error", "")})
        if res["ok"]:
            self._look_here()
        return res["ok"], res.get("error", "")

    def _do_open_drawer(self, _target):
        st = self.robot.station
        c = self.containers.get(st)
        if c is None:
            return False, f"nothing to open at {st}"
        if self.open_station == st and self.robot.open_state is not None:
            return True, "already open"
        self._square_up(st)
        self.last_execution["drawer_open_attempted"] = True
        attempt = {"station": st, "before": self._drawer_opening_state(c)}
        res = self.robot.open_articulated(c.model, c.anchor_marker, c.weld, part_markers=c.part_markers)
        self.last_execution["drawer_open_succeeded"] = bool(res["ok"])
        attempt.update(result_ok=bool(res["ok"]), result_error=res.get("error", ""), after=self._drawer_opening_state(c))
        self.last_execution.setdefault("drawer_opening_attempts", []).append(attempt)
        if res["ok"]:
            self.open_station = st
            self._look_here()
        return res["ok"], res.get("error", "")

    def _drawer_opening_state(self, container: Container) -> dict:
        """Evaluation metadata only; none of it is exposed in brain observations."""
        r = self.robot
        base = r.wbc.base_pos
        true_base = r.wbc.true_base_pos
        state = {
            "base_pose": {"x": round(float(base[0]), 4), "y": round(float(base[1]), 4),
                          "yaw": round(float(r.wbc.base_yaw), 4)},
            "base_pose_truth": {"x": round(float(true_base[0]), 4), "y": round(float(true_base[1]), 4),
                                "yaw": round(float(r.wbc.true_base_yaw), 4)},
            "right_arm_command": np.round(r.arm_cmd, 4).tolist(),
            "left_arm_command": np.round(r.left_cmd, 4).tolist(),
            "right_palm_xyz": np.round(r.ik.palm_pose()[0], 4).tolist(),
        }
        world = r.world
        drawer = getattr(container, "drawer", None)
        if world is not None and drawer is not None:
            state["drawer_open_m"] = round(float(world.drawer_open_m(drawer)), 4)
        else:
            state["drawer_open_m"] = None
        return state

    def _do_grasp(self, item: str):
        s = self.sightings.get(item)
        from_above = bool(s and s.container)
        res = self.robot.pick(item, from_above=from_above)
        retry_count = 0
        if not res["ok"] and self.robot.holding is None:
            # The grasp search depends on where we stand: square up on the station again with
            # the arm home, then one more try (the brain's own retry would ask from the same spot)
            self.say(f"    grasp failed ({res.get('error', '')}); re-approaching and retrying")
            st = self.robot.stations.get(self.robot.station)
            self.robot.arm_home()
            if st is not None:
                self.robot._step_back(0.2)
                self.robot._drive_to(st.x, st.y, st.yaw, tol=0.03, timeout=10.0)
            res = self.robot.pick(item, from_above=from_above)
            retry_count = 1
        self.last_execution["grasp_retry_count"] = retry_count
        if not res["ok"]:
            return False, res.get("error", "")
        self.sightings.pop(item, None)
        if self.robot.open_state is not None:  # tidy up: close it with the other hand
            self.last_execution["drawer_close_attempted"] = True
            c = self.robot.close_opened()
            self.last_execution["drawer_close_succeeded"] = bool(c.get("ok"))
            if c.get("ok"):
                self.open_station = None
        return True, ""

    def _do_deliver(self, _area):
        item = self.robot.holding
        res = self.robot.hand_over()
        if res["ok"]:
            self.delivered = item
        return res["ok"], res.get("error", "")

    def _do_stop(self, _target):
        self.robot.wbc.set_velocity(0, 0, 0)
        self.robot.wait(0.5)
        return True, "standing still"
