"""Check every spawn spot in scenario.SPAWNS with the real G1: walk to its station, look, pick.

    MUJOCO_GL=cgl G1_CHEAT_GRASP=fallback ../sim/.venv/bin/python probe_spawns.py   # macOS
    G1_CHEAT_GRASP=fallback python probe_spawns.py --item cup                          # Linux
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from house import G1House  # noqa: E402
from scenario import SPAWNS, Scenario, apply  # noqa: E402
from skills import AssistiveRobot  # noqa: E402


def probe(env, item: str, station: str) -> dict:
    sc = Scenario.sample(np.random.default_rng(0), fixed={item: station,
                         **({} if item == "cup" else {"cup": "kitchen_counter"}),
                         **({} if item == "water_bottle" else {"water_bottle": "kitchen_counter"})}, seed=0)
    if sc.spots["water_bottle"].station == sc.spots["cup"].station and item == "cup":
        sc.spots["water_bottle"] = next(s for s, _ in SPAWNS["water_bottle"] if s.station == "kitchen_drawers")
    apply(env, sc)
    t = time.time()
    r = AssistiveRobot(world=env)
    out = {"item": item, "station": station}
    out["go_to"] = bool(r.go_to(station).get("ok"))
    if station == "kitchen_drawers":
        out["seen"] = "(in drawer: opened by search)"
    else:
        seen = r.look().get("items", {})
        out["seen"] = item in seen
        if out["seen"] and out["go_to"]:
            res = r.pick(item)
            out["pick"] = bool(res.get("ok"))
            out["detail"] = str(res.get("reason") or res.get("error") or "")[:60]
    out["fallen"] = r.wbc.fallen()
    out["wall_s"] = round(time.time() - t, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", choices=list(SPAWNS))
    args = ap.parse_args()
    for item in [args.item] if args.item else list(SPAWNS):
        for spot, p in SPAWNS[item]:
            # A fresh house per spot: a picked item keeps its collisions off (abstracted grasp).
            print(probe(G1House(seed=0), item, spot.station), flush=True)


if __name__ == "__main__":
    main()
