"""Run the full fetch (search -> open drawer if needed -> grasp -> deliver) in every demo scenario.

    MUJOCO_GL=cgl ../sim/.venv/bin/python validate_scenarios.py            # macOS, all 25 x both items
    python validate_scenarios.py --scenarios 3 18 19 --items water_bottle -j 2

Same robot code as voice_demo.py (RobotBridge, prior memory, locked-door discovery, grasp
fallback), without the voice part. Writes logs/scenarios/results.csv and summary.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "logs" / "scenarios"


def run_one(job: tuple[int, str]) -> dict:
    n, item = job
    os.environ.setdefault("G1_CHEAT_GRASP", "fallback")
    sys.path[:0] = [str(HERE), str(HERE.parent)]
    from articulation import ArticulationModel
    from brain.contracts import FetchTask
    from brain.engine import Brain
    from brain.safety import RejectedAction
    from brain_bridge import Container, RobotBridge
    from house import G1House, KITCHEN_TOP
    from scenario import DoorDiscovery, apply, describe_scenario, prior_memory, scenario
    from skills import AssistiveRobot

    t0 = time.time()
    steps: list[str] = []
    env = G1House(seed=n)
    sc = scenario(n)
    info = apply(env, sc)
    robot = AssistiveRobot(world=env)
    doors = DoorDiscovery(env, robot, say=steps.append)
    drawer = ArticulationModel.load(HERE / "drawer_model.json")
    containers = {"kitchen_drawers": Container(drawer, KITCHEN_TOP.anchor_marker, f"grasp_{KITCHEN_TOP.body}",
                                               (KITCHEN_TOP.front_marker,), drawer=KITCHEN_TOP)}
    bridge = RobotBridge(robot, prior_memory(), containers, say=lambda m: steps.append(m.strip()))
    brain = Brain()
    brain.set_task(FetchTask(item), f"validate {item}")
    bridge.new_task(item)
    outcome = "turn limit"
    try:
        for _ in range(15):
            try:
                d = brain.next_action(bridge.latest_observation())
            except RejectedAction as e:
                outcome = f"safety: {e}"
                break
            if d.action in ("complete", "ask_for_help", "stop"):
                outcome = d.action
                break
            fb = bridge.execute_high_level(d)
            brain.record_feedback(fb)
    except Exception as e:  # noqa: BLE001 - one bad run shouldn't stop the batch
        outcome = f"error: {type(e).__name__}: {e}"
    searched = [s.split(" at the ")[1].split(" (")[0] for s in steps if s.startswith("searching") and " at the " in s]
    return {
        "scenario": n, "item": item, "setup": describe_scenario(n).split("| ")[1],
        "spot": sc.spots[item].station, "outcome": outcome,
        "searched": " > ".join(searched), "doors_found_locked": " ".join(sorted(doors.known)),
        "clutter": info["clutter"], "fallen": bool(robot.wbc.fallen()),
        "sim_s": round(float(env.data.time), 1), "wall_s": round(time.time() - t0, 1),
        "notes": " | ".join(s for s in steps if "couldn't" in s or "FAIL" in s or "locked" in s)[:200],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=int, nargs="*", default=list(range(25)))
    ap.add_argument("--items", nargs="*", default=["water_bottle", "cup"])
    ap.add_argument("-j", type=int, default=3, help="parallel sims (~2.5 GB RAM each)")
    args = ap.parse_args()
    jobs = [(n, it) for n in args.scenarios for it in args.items]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.j) as ex:
        for row in ex.map(run_one, jobs):
            rows.append(row)
            ok = "OK  " if row["outcome"] == "complete" else "FAIL"
            print(f"{ok} {row['scenario']:2d} {row['item']:12} {row['setup']:22} {row['outcome'][:40]:40} "
                  f"searched: {row['searched']}  {'door found locked' if row['doors_found_locked'] else ''} "
                  f"({row['wall_s']}s)", flush=True)
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    done = sum(r["outcome"] == "complete" for r in rows)
    lines = [f"# Demo scenarios: {done}/{len(rows)} fetches delivered", "",
             "| # | item | setup | spot | outcome | searched (in order) | door found locked | fell |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['scenario']} | {r['item']} | {r['setup']} | {r['spot']} | {r['outcome']} | {r['searched']} | "
              f"{r['doors_found_locked']} | {'yes' if r['fallen'] else ''} |" for r in rows]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\n{done}/{len(rows)} delivered. Results: {OUT / 'results.csv'}, {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
