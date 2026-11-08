"""Where do things usually live? A per-item belief over stations.

Learning: a confirmed placement or a confident head-camera sighting contributes one
recency-weighted event.  Sightings are deliberately deduplicated while an object stays
in view: repeated video frames are not repeated storage events.

Using it: `ranked(item)` gives stations most-likely-first. While searching, the robot
reports what it saw with `observe_search(item, station, found)`: only a confirmed,
adequately observed absence lowers that station for the current episode.  Search
evidence is never written to persistent history.
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
        self.stations = list(stations or SEARCHABLE)
        self.events: dict[str, list[tuple[str, float]]] = defaultdict(list)  # item -> [(station, t)]
        self.event_metadata: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._searched_empty: dict[str, set[str]] = defaultdict(set)  # per-episode negatives
        self._active_sightings: dict[str, str] = {}  # item -> station, while continuously visible

    # ------------------------------------------------------------ learning
    def _record(self, item: str, station: str, t: float | None, source: str) -> None:
        self.events[item].append((station, time.time() if t is None else t))
        self.event_metadata[item].append({"evidence_source": source})

    def observe_placement(self, item: str, station: str, t: float | None = None) -> None:
        """Record an independently confirmed placement (not an ordinary camera frame)."""
        self._record(item, station, t, "confirmed_placement")

    def observe_sighting(self, item: str, station: str, t: float | None = None) -> bool:
        """Record one confident head-camera sighting, returning whether history changed.

        A new station is evidence that the item can be found there.  The same item at the
        same station remains one continuous sighting until it is absent from a sufficiently
        good look (or the task episode is reset), so video frame rate cannot inflate beliefs.
        """
        if station not in self.stations or self._active_sightings.get(item) == station:
            return False
        self._record(item, station, t, "head_camera_sighting")
        self._active_sightings[item] = station
        return True

    def observe_search(self, item: str, station: str, found: bool, *, absence_confirmed: bool = True) -> bool:
        """Record temporary search evidence; return whether a negative was accepted.

        ``found`` clears earlier negatives but does not assert a placement: it may be a
        robot-held item or merely a sighting.  Call :meth:`observe_placement` only when the
        camera observation actually establishes a placement event.
        """
        if found:
            self._searched_empty[item].clear()
            return False
        if absence_confirmed:
            self._searched_empty[item].add(station)
            self._active_sightings.pop(item, None)
            return True
        return False

    def new_episode(self) -> None:
        self._searched_empty.clear()

    # ------------------------------------------------------------ queries
    def belief(self, item: str, now: float | None = None) -> dict[str, float]:
        now = time.time() if now is None else now
        w = {s: PRIOR for s in self.stations}
        for station, t in self.events.get(item, []):
            if station in w:
                # A clock correction can make an event appear slightly in the future; treat it
                # as current rather than amplifying its weight above one or overflowing.
                age_days = max(0.0, (now - t) / 86400.0)
                w[station] += DECAY_PER_DAY ** age_days
        for s in self._searched_empty.get(item, ()):  # looked there this episode, not found
            w[s] *= 0.05
        total = sum(w.values())
        return {s: v / total for s, v in w.items()}

    def ranked(self, item: str, now: float | None = None) -> list[tuple[str, float]]:
        b = self.belief(item, now)
        return sorted(b.items(), key=lambda kv: -kv[1])

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 2,
            "stations": self.stations,
            "events": {k: v for k, v in self.events.items()},
            "event_metadata": {k: v for k, v in self.event_metadata.items()},
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, stations: list[str] | None = None) -> "LocationMemory":
        """Load v2 memory, or the original event-only JSON format.

        Supplying ``stations`` lets a caller deliberately migrate a legacy file to a new
        house layout; otherwise v2 restores the station list that produced the beliefs.
        """
        value = json.loads(Path(path).read_text())
        is_v2 = isinstance(value, dict) and "events" in value and "version" in value
        stored_stations = value.get("stations") if is_v2 else None
        mem = cls(stations if stations is not None else stored_stations)
        events = value["events"] if is_v2 else value
        metadata = value.get("event_metadata", {}) if is_v2 else {}
        for item, evs in events.items():
            mem.events[item] = [tuple(e) for e in evs]
            mem.event_metadata[item] = list(metadata.get(item, [{} for _ in evs]))
        return mem


def nearest_station(position: np.ndarray) -> str:
    """Map a world position (where an item came to rest) to the station that serves it."""
    from scene import TABLE, SHELF, COUNTER

    surfaces = {"table": TABLE["pos"], "shelf": SHELF["pos"], "cupboard": COUNTER["pos"]}
    return min(surfaces, key=lambda s: np.hypot(position[0] - surfaces[s][0], position[1] - surfaces[s][1]))
