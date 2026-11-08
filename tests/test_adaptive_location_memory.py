"""Fast, sensor-facing checks for the G1 adaptive location memory."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

ASSISTIVE = Path(__file__).resolve().parents[1] / "assistive_g1"
sys.path.insert(0, str(ASSISTIVE))

from brain_bridge import RobotBridge  # noqa: E402
from house import BOTTLE_TARGET_STATIONS, SUPPORTED_BOTTLE_TARGETS  # noqa: E402
from memory import LocationMemory  # noqa: E402
from sim.house_env import ALL_TARGETS  # noqa: E402


def test_head_camera_sighting_updates_priority_and_learns_incidental_items():
    memory = LocationMemory(["kitchen_counter", "kitchen_drawers"])
    robot = SimpleNamespace(
        station="kitchen_drawers", holding=None, open_state=None,
        look=lambda: {"items": ["water_bottle", "cup"]},
    )
    bridge = RobotBridge(robot, memory, {}, say=lambda _message: None)

    now = time.time()
    before = memory.belief("water_bottle", now=now)
    bridge._look_here()

    assert memory.ranked("water_bottle")[0][0] == "kitchen_drawers"
    assert memory.ranked("cup")[0][0] == "kitchen_drawers"
    assert memory.belief("water_bottle", now=now)["kitchen_drawers"] > before["kitchen_drawers"]
    assert bridge.last_execution["memory_updates"][0]["evidence_source"] == "head_camera_sighting"


def test_repeated_unchanged_camera_frames_do_not_add_location_events():
    memory = LocationMemory(["kitchen_counter", "kitchen_drawers"])

    assert memory.observe_sighting("water_bottle", "kitchen_counter", t=100.0)
    assert not memory.observe_sighting("water_bottle", "kitchen_counter", t=101.0)
    assert memory.events["water_bottle"] == [("kitchen_counter", 100.0)]


def test_confirmed_absence_is_temporary_but_inaccessible_drawer_is_not_penalised():
    memory = LocationMemory(["kitchen_counter", "kitchen_drawers"])
    memory.observe_placement("water_bottle", "kitchen_drawers", t=100.0)
    baseline = memory.belief("water_bottle", now=100.0)

    assert memory.observe_search("water_bottle", "kitchen_counter", found=False, absence_confirmed=True)
    assert memory.belief("water_bottle", now=100.0)["kitchen_counter"] < baseline["kitchen_counter"]
    memory.new_episode()
    assert memory.belief("water_bottle", now=100.0) == baseline

    assert not memory.observe_search("water_bottle", "kitchen_drawers", found=False, absence_confirmed=False)
    assert memory.belief("water_bottle", now=100.0) == baseline


def test_bridge_does_not_treat_a_failed_drawer_open_as_negative_evidence():
    memory = LocationMemory(["kitchen_drawers"])
    memory.observe_placement("water_bottle", "kitchen_drawers", t=100.0)
    baseline = memory.belief("water_bottle", now=100.0)
    robot = SimpleNamespace(
        station="home", holding=None, open_state=None,
        go_to=lambda station: {"ok": True},
        look=lambda: {"items": []},
    )
    drawer = SimpleNamespace(kind="drawer")
    bridge = RobotBridge(robot, memory, {"kitchen_drawers": drawer}, say=lambda _message: None)
    bridge._do_open_drawer = lambda _target: (False, "drawer locked")

    ok, _ = bridge._do_search("water_bottle")

    assert not ok
    assert memory.belief("water_bottle", now=100.0) == baseline
    assert bridge.last_execution["inaccessible_or_unobserved_stations"] == ["kitchen_drawers"]


def test_saved_memory_preserves_active_stations_and_legacy_files_still_load(tmp_path):
    path = tmp_path / "memory.json"
    memory = LocationMemory(["kitchen_counter", "kitchen_drawers"])
    now = time.time()
    memory.observe_sighting("water_bottle", "kitchen_drawers", t=now)
    memory.save(path)

    restored = LocationMemory.load(path)
    assert restored.stations == ["kitchen_counter", "kitchen_drawers"]
    assert restored.ranked("water_bottle", now=now)[0][0] == "kitchen_drawers"

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"cup": [["kitchen_counter", 100.0]]}))
    assert LocationMemory.load(legacy, stations=["kitchen_counter"]).events["cup"] == [("kitchen_counter", 100.0)]


def test_memory_survives_multiple_tasks_while_search_negatives_do_not():
    memory = LocationMemory(["kitchen_counter", "kitchen_drawers"])
    memory.observe_sighting("water_bottle", "kitchen_drawers", t=time.time())
    memory.observe_search("water_bottle", "kitchen_drawers", found=False, absence_confirmed=True)
    memory.new_episode()

    assert memory.ranked("water_bottle")[0][0] == "kitchen_drawers"


def test_supported_bottle_targets_are_simulator_targets_with_g1_search_stations():
    assert SUPPORTED_BOTTLE_TARGETS == ("kitchen_drawer", "kitchen_counter", "kitchen_table")
    assert set(SUPPORTED_BOTTLE_TARGETS) <= set(ALL_TARGETS)
    assert BOTTLE_TARGET_STATIONS == {
        "kitchen_drawer": "kitchen_drawers",
        "kitchen_counter": "kitchen_counter",
        "kitchen_table": "kitchen_table",
    }


def test_relocation_persists_camera_evidence_and_prioritises_newer_counter_sighting(tmp_path):
    path = tmp_path / "relocation-memory.json"
    stations = ["kitchen_counter", "kitchen_drawers", "kitchen_table"]
    first = LocationMemory(stations)
    assert first.observe_sighting("water_bottle", "kitchen_drawers", t=100.0)
    first.save(path)

    second = LocationMemory.load(path, stations=stations)
    assert second.ranked("water_bottle", now=101.0)[0][0] == "kitchen_drawers"
    assert second.observe_search("water_bottle", "kitchen_drawers", found=False, absence_confirmed=True)
    assert second.observe_sighting("water_bottle", "kitchen_counter", t=102.0)
    second.save(path)

    third = LocationMemory.load(path, stations=stations)
    assert third.ranked("water_bottle", now=103.0)[0][0] == "kitchen_counter"
    assert third.event_metadata["water_bottle"][-1]["evidence_source"] == "head_camera_sighting"


def test_counter_target_separates_cup_and_bottle_without_contact():
    from house import G1House

    env = G1House(seed=0)
    env.reset("tidy", "kitchen_counter", seed=0)
    cup = env.data.body(env.item_body["cup"]).xpos
    bottle = env.data.body(env.item_body["water_bottle"]).xpos
    cup_geom, bottle_geom = env.model.geom("cup_geom").id, env.model.geom("waterbottle").id

    assert np.linalg.norm((cup - bottle)[:2]) >= 0.12
    assert mujoco.mj_geomDistance(env.model, env.data, cup_geom, bottle_geom, 1.0, None) > 0.0


def test_inadequate_camera_coverage_is_unobserved_not_confirmed_empty():
    memory = LocationMemory(["kitchen_counter"])
    memory.observe_placement("water_bottle", "kitchen_counter", t=100.0)
    baseline = memory.belief("water_bottle", now=100.0)
    robot = SimpleNamespace(
        station="home", holding=None, open_state=None,
        go_to=lambda station: {"ok": True},
        look=lambda: {"ok": True, "items": {}, "absence_confirmed": False, "marker_ids": [3]},
    )
    bridge = RobotBridge(robot, memory, {}, say=lambda _message: None)

    ok, _ = bridge._do_search("water_bottle")

    assert not ok
    assert memory.belief("water_bottle", now=100.0) == baseline
    assert bridge.last_execution["inaccessible_or_unobserved_stations"] == ["kitchen_counter"]


def test_counter_camera_detection_records_marker_diagnostics_and_learns_relocation():
    memory = LocationMemory(["kitchen_counter", "kitchen_drawers"])
    robot = SimpleNamespace(
        station="kitchen_counter", holding=None, open_state=None,
        look=lambda: {"ok": True, "items": {"water_bottle": [1.8, -2.62, 1.11]},
                      "marker_ids": [2, 3], "absence_confirmed": False},
    )
    bridge = RobotBridge(robot, memory, {}, say=lambda _message: None)

    bridge._look_here()

    assert memory.ranked("water_bottle")[0][0] == "kitchen_counter"
    assert bridge.last_execution["positive_camera_observations"] == [
        {"item": "water_bottle", "station": "kitchen_counter"},
    ]
    assert bridge.last_execution["camera_diagnostics"] == [
        {"station": "kitchen_counter", "marker_ids": [2, 3], "absence_confirmed": False},
    ]
