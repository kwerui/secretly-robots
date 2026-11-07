"""Where do things usually live? A per-item belief over stations, learned by watching the carer.

Learning: every time the carer puts an item down (seen as an ArUco marker coming to rest),
call `observe_placement(item, station)`. Recent placements count more (exponential decay).

Using it: `ranked(item)` gives stations most-likely-first. While searching, the robot
reports what it saw with `observe_search(item, station, found)`: not finding it lowers that
station for this episode; finding it is also a placement observation.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scene import STATIONS

SEARCHABLE = ["table", "shelf", "cupboard"]  # stations that hold items
DECAY_PER_DAY = 0.85
PRIOR = 0.2  # pseudo-count per station so nothing is ever impossible


class LocationMemory:
    def __init__(self, stations: list[str] | None = None):
        self.stations = stations or SEARCHABLE
        self.events: dict[str, list[tuple[str, float]]] = defaultdict(list)  # item -> [(station, t)]
        self._searched_empty: dict[str, set[str]] = defaultdict(set)  # per-episode negatives

    # ------------------------------------------------------------ learning
    def observe_placement(self, item: str, station: str, t: float | None = None) -> None:
        self.events[item].append((station, time.time() if t is None else t))

    def observe_search(self, item: str, station: str, found: bool) -> None:
        if found:
            self.observe_placement(item, station)
            self._searched_empty[item].clear()
        else:
            self._searched_empty[item].add(station)

    def new_episode(self) -> None:
        self._searched_empty.clear()

    # ------------------------------------------------------------ queries
    def belief(self, item: str, now: float | None = None) -> dict[str, float]:
        now = time.time() if now is None else now
        w = {s: PRIOR for s in self.stations}
        for station, t in self.events.get(item, []):
            if station in w:
                w[station] += DECAY_PER_DAY ** ((now - t) / 86400.0)
        for s in self._searched_empty.get(item, ()):  # looked there this episode, not found
            w[s] *= 0.05
        total = sum(w.values())
        return {s: v / total for s, v in w.items()}

    def ranked(self, item: str) -> list[tuple[str, float]]:
        b = self.belief(item)
        return sorted(b.items(), key=lambda kv: -kv[1])

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({k: v for k, v in self.events.items()}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LocationMemory":
        mem = cls()
        for item, evs in json.loads(Path(path).read_text()).items():
            mem.events[item] = [tuple(e) for e in evs]
        return mem


def nearest_station(position: np.ndarray) -> str:
    """Map a world position (where an item came to rest) to the station that serves it."""
    from scene import TABLE, SHELF, COUNTER

    surfaces = {"table": TABLE["pos"], "shelf": SHELF["pos"], "cupboard": COUNTER["pos"]}
    return min(surfaces, key=lambda s: np.hypot(position[0] - surfaces[s][0], position[1] - surfaces[s][1]))
