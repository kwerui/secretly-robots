"""Randomised fetch scenarios for showing adaptation: where things are, which door is locked, clutter.

Each item (water bottle, cup) has five spots it can spawn at, each with a fixed probability.
The robot's location memory starts from the same distribution (the carer has seen things put
away there over the last week), so it searches the likeliest place first and has to adapt when
the item isn't there. On top of that a door can be locked (the robot must take the long way
round) and clutter can be scattered on the floor (it must walk round it).

    from scenario import Scenario, apply
    sc = Scenario.sample(rng, lock_door="kitchen_study", clutter=4)
    apply(env, sc)               # after G1House construction, instead of env.reset(...)

Nothing here changes Jack's house or Eren's robot code; it uses their public functions
(HouseEnv layouts, G1House.place_item, stations, LocationMemory).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import house as G
from house import Station, _station_facing
from memory import LocationMemory
from sim import house_env as H

DAY = 86400.0


@dataclass(frozen=True)
class Spot:
    """Where an item can spawn and where the robot stands to look at / pick it."""
    station: str  # robot station name (also the location-memory key)
    surface: str | None  # SURFACE_TARGETS key the item sits on (None: inside the kitchen drawer)
    offset: tuple[float, float] = (0.0, 0.0)  # metres from the surface target, so two items never overlap


# Five spots per item and how likely each is. Kept to surfaces the G1 can reach and see from a
# station; the bottle can also be in the top kitchen drawer (opened with the learned motion).
SPAWNS: dict[str, list[tuple[Spot, float]]] = {
    "water_bottle": [
        (Spot("kitchen_counter", "kitchen_counter"), 0.30),
        (Spot("kitchen_drawers", None), 0.25),
        (Spot("kitchen_table", "kitchen_table"), 0.20),
        (Spot("study_desk", "study_desk"), 0.15),
        (Spot("coffee_table", "coffee_table"), 0.10),
    ],
    "cup": [
        (Spot("kitchen_counter", "kitchen_counter", (0.35, 0.0)), 0.35),
        (Spot("kitchen_table", "kitchen_table", (0.25, 0.10)), 0.25),
        (Spot("study_desk", "study_desk", (0.0, 0.35)), 0.15),
        (Spot("coffee_table", "coffee_table", (0.0, 0.25)), 0.15),
        (Spot("tv_unit", "tv_unit", (-0.70, 0.08)), 0.10),  # the open end of the unit, clear of the TV
    ],
}

# Where the robot stands for the spots that the G1 team's default_stations() doesn't cover:
# (point to face, heading, standoff). Headings face the furniture from open floor.
EXTRA_STATIONS = {
    "study_desk": ((3.45, 1.35), 0.0, 0.60),           # from the study floor, facing +x (wall behind desk)
    "coffee_table": ((-2.80, -1.05), math.pi, 0.60),   # G1House moves it to (-2.8, -1.15); face it from +x
    "tv_unit": ((-2.20, -2.70), -math.pi / 2, 0.50),   # close, so the head camera looks down on the cup
}

DOORS = list(H.DOORWAYS)  # living_kitchen, bedroom_study, living_bedroom, kitchen_study

# Clutter comes from the floor spots Jack placed by hand in his preset layouts (laundry_day,
# door_shut, moving_house, long_way_round), so every piece is somewhere sensible; a random
# subset is used and each piece is kept only if the house's own checks still pass with it.
CLUTTER_SPOTS = [o for name, lay in H.LAYOUTS.items() if name != "scenario" for o in lay.get("obstacles", [])]


def add_stations(env) -> None:
    for name, (xy, yaw, standoff) in EXTRA_STATIONS.items():
        if name not in env.stations:
            p, y = _station_facing(xy, yaw, standoff)
            env.stations[name] = Station(name, float(p[0]), float(p[1]), float(y))


def searchable(items=("water_bottle", "cup")) -> list[str]:
    out = []
    for it in items:
        for spot, _ in SPAWNS[it]:
            if spot.station not in out:
                out.append(spot.station)
    return out


def prior_memory(days: int = 7, per_day: float = 3.0, now: float | None = None) -> LocationMemory:
    """The carer's last week of putting things away, matching the spawn probabilities."""
    now = time.time() if now is None else now
    mem = LocationMemory(stations=searchable())
    for item, spots in SPAWNS.items():
        for spot, p in spots:
            n = int(round(p * days * per_day))
            for k in range(n):
                mem.observe_placement(item, spot.station, t=now - DAY * (k % days + 0.5))
    return mem


@dataclass
class Scenario:
    spots: dict[str, Spot]  # item -> where it spawned
    lock_door: str | None = None
    clutter: list[tuple[str, float, float, float]] = field(default_factory=list)  # (type, x, y, yaw)
    seed: int | None = None
    clutter_wanted: int = 0
    open_doors: bool = True  # unlocked doors wide open (half-open leaves are hard for the G1 to pass)

    @classmethod
    def sample(cls, rng: np.random.Generator, lock_door: str | None = None, clutter: int = 0,
               fixed: dict[str, str] | None = None, seed: int | None = None) -> "Scenario":
        spots = {}
        for item, options in SPAWNS.items():
            if fixed and item in fixed:
                spots[item] = next(s for s, _ in options if s.station == fixed[item])
            else:
                probs = np.array([p for _, p in options])
                spots[item] = options[rng.choice(len(options), p=probs / probs.sum())][0]
        return cls(spots, lock_door, _draw_clutter(rng, clutter), seed, clutter)

    def describe(self) -> str:
        parts = [f"{it} @ {s.station}" for it, s in self.spots.items()]
        if self.lock_door:
            parts.append(f"door {self.lock_door} LOCKED")
        if self.clutter:
            parts.append(f"{len(self.clutter)} clutter")
        return ", ".join(parts)


def _draw_clutter(rng, n: int) -> list[tuple[str, float, float, float]]:
    """Up to n candidate pieces, at most as many of each type as the house's obstacle pool has."""
    pool = H._obstacle_pool()
    out, used = [], {}
    for i in rng.permutation(len(CLUTTER_SPOTS)):
        typ, x, y, yaw = CLUTTER_SPOTS[int(i)]
        if used.get(typ, 0) < pool.get(typ, 0):
            used[typ] = used.get(typ, 0) + 1
            out.append((typ, float(x), float(y), float(yaw)))
    return out[: max(0, n) * 3]  # spare candidates: some get rejected


def _try(env, sc: Scenario, target: str, clutter) -> dict | None:
    H.LAYOUTS["scenario"] = dict(obstacles=list(clutter), doors_locked=[sc.lock_door] if sc.lock_door else [])
    try:
        info = env.reset("scenario", target, seed=sc.seed)
    except RuntimeError:
        return None
    return info if _items_reachable(env, sc) else None


def apply(env, sc: Scenario) -> dict:
    """Reset the house to this scenario. Clutter is added a piece at a time and a piece is
    dropped if it would block a doorway, a door swing, a drawer, or the route to an item or the
    person (the house's own checks plus a path check per station). Items never move."""
    add_stations(env)
    target = "kitchen_drawer" if sc.spots["water_bottle"].surface is None else sc.spots["water_bottle"].surface
    want = sc.clutter_wanted
    kept: list = []
    info = _try(env, sc, target, kept)
    if info is None:
        raise RuntimeError(f"scenario not valid even without clutter: {sc.describe()}")
    for piece in sc.clutter:
        if len(kept) >= want:
            break
        if _try(env, sc, target, kept + [piece]) is not None:
            kept.append(piece)
    sc.clutter = kept
    info = _try(env, sc, target, kept)  # final state with exactly the kept pieces
    if sc.open_doors:
        open_unlocked_doors(env)
    for item, spot in sc.spots.items():
        if item == "water_bottle" and spot.surface is None:
            continue  # G1House.reset already put it in the kitchen drawer
        place_on(env, item, spot)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    info = dict(info, scenario=sc.describe(), spots={k: v.station for k, v in sc.spots.items()},
                clutter=len(kept), locked=[sc.lock_door] if sc.lock_door else [])
    env.info = info
    return info


def open_unlocked_doors(env) -> None:
    """Swing every unlocked door fully open. The house's checks already keep clutter out of
    every door's swing, so an open leaf never hits anything."""
    for dn in H.DOORWAYS:
        if dn not in env.locked_doors:
            lo, hi = env.door_range[dn]
            env.set_door(dn, hi if abs(hi) >= abs(lo) else lo)
    env.info = dict(getattr(env, "info", {}) or {})


def place_on(env, item: str, spot: Spot) -> None:
    x, y, z = env.surface_xyz(spot.surface)
    it = env.items[item]
    env.place_item(item, "xyz", (x + spot.offset[0], y + spot.offset[1], z + it.size[2] + 0.002))


def _items_reachable(env, sc: Scenario) -> bool:
    """Every station the robot may need (item spots, the person) has a path from home."""
    home = env.stations["home"]
    for name in {s.station for s in sc.spots.values()} | {"person"}:
        st = env.stations[name]
        if env.plan_path((home.x, home.y), (st.x, st.y)) is None:
            return False
    return True


# ---------------------------------------------------------------------------------------------
# The 25 demo scenarios: 5 item placements x 5 floor setups, numbered 0-24 (= 5 * placement + setup).
# Fixed and validated with the real G1, so a demo or a video always gets the same house.

# Each bottle spot and each cup spot appears once, from the likeliest (found on the first look)
# to the least likely (the robot must rule out the others first). The SPAWNS probabilities are
# still the robot's starting beliefs (prior_memory) and are used by Scenario.sample().
PLACEMENTS = [
    {"water_bottle": "kitchen_counter", "cup": "kitchen_counter"},  # both where they usually are
    {"water_bottle": "kitchen_drawers", "cup": "kitchen_table"},    # bottle put away in the drawer
    {"water_bottle": "kitchen_table", "cup": "study_desk"},
    {"water_bottle": "study_desk", "cup": "coffee_table"},
    {"water_bottle": "coffee_table", "cup": "tv_unit"},             # both in their least likely spots
]

# Floor setups. Clutter pieces are Jack's hand-placed ones (sim/house_env.py LAYOUTS); a piece is
# dropped if it would block a route (scenario.apply). The locked door is living-kitchen: the
# robot then reaches the kitchen through the bedroom and study (the G1 handles every leg of that;
# locking kitchen-study or living-bedroom strands it on some legs).
_LAUNDRY = H.LAYOUTS["laundry_day"]["obstacles"]
SETUPS = [
    {"name": "tidy", "lock_door": None, "clutter": []},
    {"name": "light clutter", "lock_door": None, "clutter": [_LAUNDRY[0], _LAUNDRY[1], _LAUNDRY[4]]},
    {"name": "heavy clutter", "lock_door": None, "clutter": list(H.LAYOUTS["moving_house"]["obstacles"])},
    {"name": "locked door", "lock_door": "living_kitchen", "clutter": []},
    {"name": "locked door + clutter", "lock_door": "living_kitchen",
     "clutter": list(H.LAYOUTS["door_shut"]["obstacles"]) + [_LAUNDRY[0], _LAUNDRY[2]]},
]
N_SCENARIOS = len(PLACEMENTS) * len(SETUPS)


def scenario(n: int) -> Scenario:
    """Demo scenario n (0-24): placement n // 5, setup n % 5."""
    if not 0 <= n < N_SCENARIOS:
        raise ValueError(f"scenario must be 0-{N_SCENARIOS - 1}")
    place, setup = PLACEMENTS[n // len(SETUPS)], SETUPS[n % len(SETUPS)]
    spots = {item: next(s for s, _ in SPAWNS[item] if s.station == st) for item, st in place.items()}
    clutter = [tuple(c) for c in setup["clutter"]]
    return Scenario(spots, setup["lock_door"], clutter, seed=n, clutter_wanted=len(clutter))


def describe_scenario(n: int) -> str:
    place, setup = PLACEMENTS[n // len(SETUPS)], SETUPS[n % len(SETUPS)]
    prob = {it: dict((s.station, p) for s, p in SPAWNS[it]) for it in SPAWNS}
    return (f"{n:2d}: bottle {place['water_bottle']} ({prob['water_bottle'][place['water_bottle']]:.0%}), "
            f"cup {place['cup']} ({prob['cup'][place['cup']]:.0%}) | {setup['name']}")


# ---------------------------------------------------------------------------------------------
# Finding out that a door is locked, on the way.

class DoorLocked(Exception):
    def __init__(self, door: str):
        super().__init__(door)
        self.door = door


class DoorDiscovery:
    """The robot doesn't know in advance which doors are locked.

    Its planner starts out treating every door as pushable (env.locked_doors holds only what the
    robot has found out). It walks the shortest route; when it gets to a doorway on that route
    whose door stays shut, it stops, says so, remembers the door as locked and replans the same
    trip through the other doorways. The physics is unchanged: a locked door is welded shut
    whether or not the robot knows.

        disc = DoorDiscovery(env, robot, say=print)   # after apply(); wraps robot.go_to
    """

    def __init__(self, env, robot, say=print, on_found=None, radius: float = 0.85):
        self.env, self.robot, self.say, self.on_found = env, robot, say, on_found
        self.truth = set(env.locked_doors)  # what the house really has locked
        self.known: set[str] = set()  # what the robot has found out
        env.locked_doors = self.known  # the planner only avoids doors the robot knows are locked
        if getattr(robot, "costmap", None) is not None:
            # Perceived navigation (costmap.py, G1_NAV=live): the robot built its map while
            # env.locked_doors still held the truth, so the lock is in its floor plan. Rebuild it
            # from what the robot knows (every door pushable); mark_door_locked adds a found one.
            from costmap import make_costmap
            robot.costmap = env.costmap = make_costmap(env)
            robot._map_depth()
        self.radius = radius
        self._route_doors: set[str] = set()
        go_to = robot.go_to

        def go_to_finding_locks(station: str, _tries: int = 4):
            for _ in range(_tries):
                st, p = env.stations.get(station), robot.wbc.base_pos
                path = env.plan_path((float(p[0]), float(p[1])), (st.x, st.y)) if st else None
                self._route_doors = self._doors_on(path)
                frame = robot.on_frame

                def watch(r, _frame=frame):
                    if _frame is not None:
                        _frame(r)
                    self.check()
                robot.on_frame = watch
                try:
                    return go_to(station)
                except DoorLocked as e:
                    robot.wbc.set_velocity(0.0, 0.0, 0.0)
                    self.known.add(e.door)
                    if getattr(robot, "costmap", None) is not None:
                        robot.costmap.mark_door_locked(e.door)
                    a, b = e.door.split("_")
                    self.say(f"the {a}-{b} door is locked: rerouting the other way")
                    if self.on_found is not None:
                        self.on_found(e.door)
                    robot.wait(0.8)  # come to a stop before setting off again
                finally:
                    robot.on_frame = frame
            return {"ok": False, "error": f"couldn't find a way to the {station}"}
        robot.go_to = go_to_finding_locks

    @staticmethod
    def _doors_on(path) -> set[str]:
        if not path:
            return set()
        out = set()
        for dn, dw in H.DOORWAYS.items():
            c = np.array(dw["center"], float)
            pts = np.array(path, float)
            for a, b in zip(pts, pts[1:]):
                ab = b - a
                t = 0.0 if not ab.any() else float(np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0, 1))
                if np.linalg.norm(a + t * ab - c) < 0.35:
                    out.add(dn)
                    break
        return out

    def check(self) -> None:
        """Called every control tick while walking: at a doorway on the route whose door is
        locked (it stays shut), stop."""
        p = self.robot.wbc.base_pos
        for dn in (self.truth - self.known) & self._route_doors:
            c = H.DOORWAYS[dn]["center"]
            if math.hypot(p[0] - c[0], p[1] - c[1]) < self.radius:
                raise DoorLocked(dn)
