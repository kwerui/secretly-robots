"""Experiments over Jack's 25 demo scenarios: per-stage success of the full fetch.

    export MUJOCO_GL=egl
    python run_experiments.py --grasp cheat --tag cheat -j 4      # abstracted grasp (G1_CHEAT_GRASP=1)
    python run_experiments.py --grasp real  --tag real  -j 4      # real grasp only  (G1_CHEAT_GRASP=0)
    python run_experiments.py --scenarios 9 --grasp cheat --tag video -j 1 --video   # one recorded run

Same robot code as voice_demo.py / validate_scenarios.py (scenario.apply -> AssistiveRobot ->
DoorDiscovery -> RobotBridge -> brain), one request per scenario (--items alternate: the water
bottle in even-numbered scenarios, the cup in odd ones; or name items to run each).

Per run it records the stages: found the item (and after how many places, first guess right?),
drawer opened / closed again (when the item was in it), grasped, delivered (reported by the
robot) and delivery VERIFIED from the simulator state (item on the user's tray; evaluation only,
never given to the robot), locked door discovered, falls, re-plans, times. Writes
logs/experiments/<tag>/results.csv and summary.md.
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
OUT_ROOT = HERE.parent / "logs" / "experiments"
GRASP_ENV = {"cheat": "1", "real": "0", "fallback": "fallback"}
ITEMS = ("water_bottle", "cup")


def _on_tray(env, item: str) -> bool:
    """Evaluation only: is the item resting on the user's delivery tray (simulator state)?"""
    from house import DELIVERY
    p = env.data.body(env.item_body[item]).xpos
    base_z = p[2] if item == "water_bottle" else p[2] - env.items[item].size[2]  # bottle origin = its base
    (tx, ty), (hx, hy), top = DELIVERY["pos"], DELIVERY["half"], DELIVERY["top"]
    return abs(p[0] - tx) <= hx + 0.03 and abs(p[1] - ty) <= hy + 0.03 and top - 0.03 <= base_z <= top + 0.15


def run_one(job: tuple) -> dict:
    n, item, grasp, video_dir, size, fps = job
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
    rec = None
    if video_dir:
        from house_demo import LightRecorder
        rec = LightRecorder(Path(video_dir) / f"scenario{n:02d}_{item}_{grasp}.mp4", fps=fps, size=size)
    robot = AssistiveRobot(world=env, on_frame=rec)

    def say(msg: str) -> None:
        steps.append(msg.strip())
        if rec is not None:
            rec.caption = msg.strip()[:110]

    doors = DoorDiscovery(env, robot, say=say)
    drawer = ArticulationModel.load(HERE / "drawer_model.json")
    containers = {"kitchen_drawers": Container(drawer, KITCHEN_TOP.anchor_marker, f"grasp_{KITCHEN_TOP.body}",
                                               (KITCHEN_TOP.front_marker,), drawer=KITCHEN_TOP)}
    bridge = RobotBridge(robot, prior_memory(), containers, say=say)
    brain = Brain()
    brain.set_task(FetchTask(item), f"experiment {item}")
    bridge.new_task(item)
    say(f"Request: bring me the {item.replace('_', ' ')}  |  {describe_scenario(n).split('| ')[1]}")

    def searched_so_far() -> list[str]:
        return [s.split(" at the ")[1].split(" (")[0] for s in steps if s.startswith("searching") and " at the " in s]

    actions: list[tuple[str, bool, str]] = []
    found_after = None
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
            actions.append((d.action, bool(fb.succeeded), str(fb.detail)[:80]))
            if found_after is None and (item in bridge.sightings or robot.holding == item or bridge.delivered == item):
                found_after = len(searched_so_far())
        robot.wait(1.0)  # let a released item settle before the tray check
    except Exception as e:  # noqa: BLE001 - one bad run shouldn't stop the batch
        outcome = f"error: {type(e).__name__}: {e}"
    finally:
        if rec is not None:
            rec.close()

    searched = searched_so_far()
    spot = sc.spots[item].station
    ok = lambda a: any(x == a and s for x, s, _ in actions)  # noqa: E731
    in_drawer = spot == "kitchen_drawers"
    opened = ok("open_drawer") or any(s.startswith("opening the drawer: ok") for s in steps)
    drawer_m = float(env.drawer_open_m(KITCHEN_TOP))
    verified = _on_tray(env, item)
    fallen = bool(robot.wbc.fallen())
    drawer_ok = (drawer_m < 0.02) if opened else True
    return {
        "scenario": n, "item": item, "grasp": grasp, "setup": describe_scenario(n).split("| ")[1],
        "spot": spot, "outcome": outcome,
        "found": found_after is not None, "places_searched": found_after if found_after is not None else len(searched),
        "first_guess_right": bool(searched) and searched[0] == spot,
        "drawer_needed": in_drawer, "drawer_opened": opened, "drawer_closed_after": drawer_ok,
        "drawer_open_m_end": round(drawer_m, 3),
        "grasped": ok("grasp"), "delivered_reported": ok("deliver"), "delivered_verified": verified,
        "locked_door": sc.lock_door or "", "door_discovered": bool(sc.lock_door) and sc.lock_door in doors.known,
        "clutter": info["clutter"], "replans": len(getattr(robot, "nav_log", [])), "fallen": fallen,
        "success": outcome == "complete" and verified and drawer_ok and not fallen,
        "search_order": " > ".join(searched),
        "sim_s": round(float(env.data.time), 1), "wall_s": round(time.time() - t0, 1),
        "failures": " | ".join(f"{a}: {det}" for a, s, det in actions if not s)[:200],
    }


def _rate(rows, key, where=None) -> str:
    sel = [r for r in rows if where is None or where(r)]
    if not sel:
        return "n/a"
    k = sum(bool(r[key]) for r in sel)
    return f"{k}/{len(sel)} ({100 * k / len(sel):.0f}%)"


def summarise(rows: list[dict], tag: str, grasp: str) -> str:
    n = len(rows)
    mean = lambda k, sel=rows: sum(r[k] for r in sel) / max(len(sel), 1)  # noqa: E731
    found = [r for r in rows if r["found"]]
    lines = [
        f"# Experiments `{tag}` (grasp: {grasp}): {sum(r['success'] for r in rows)}/{n} full successes", "",
        "Full success = brain says complete AND item verified on the tray (sim state) AND drawer closed again "
        "if it was opened AND no fall.", "",
        "| stage | success |", "|---|---|",
        f"| item found | {_rate(rows, 'found')} |",
        f"| found on first guess | {_rate(rows, 'first_guess_right')} |",
        f"| drawer opened (item in drawer) | {_rate(rows, 'drawer_opened', lambda r: r['drawer_needed'])} |",
        f"| drawer closed again (when opened) | {_rate(rows, 'drawer_closed_after', lambda r: r['drawer_opened'])} |",
        f"| grasped (when found) | {_rate(rows, 'grasped', lambda r: r['found'])} |",
        f"| delivered, robot's report | {_rate(rows, 'delivered_reported')} |",
        f"| delivered, verified on tray | {_rate(rows, 'delivered_verified')} |",
        f"| locked door discovered | {_rate(rows, 'door_discovered', lambda r: r['locked_door'])} |",
        f"| no fall | {n - sum(r['fallen'] for r in rows)}/{n} |",
        f"| **full success** | **{_rate(rows, 'success')}** |", "",
        f"Places searched before finding (mean, found runs): {mean('places_searched', found):.2f}. "
        f"Re-plans per run (mean): {mean('replans'):.1f}. Sim time per run: {mean('sim_s'):.0f} s, "
        f"wall: {mean('wall_s'):.0f} s.", "",
        "## By floor setup", "", "| setup | full success | found | verified delivery |", "|---|---|---|---|",
    ]
    for setup in dict.fromkeys(r["setup"] for r in rows):
        f = lambda r, s=setup: r["setup"] == s  # noqa: E731
        lines.append(f"| {setup} | {_rate(rows, 'success', f)} | {_rate(rows, 'found', f)} | "
                     f"{_rate(rows, 'delivered_verified', f)} |")
    lines += ["", "## Runs", "",
              "| # | item | setup | spot | outcome | found (places) | drawer open/closed | grasp | delivered (verified) "
              "| door | fell | success |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    for r in rows:
        dr = f"{yn(r['drawer_opened'])}/{yn(r['drawer_closed_after'])}" if r["drawer_needed"] or r["drawer_opened"] else ""
        lines.append(f"| {r['scenario']} | {r['item']} | {r['setup']} | {r['spot']} | {r['outcome'][:30]} | "
                     f"{yn(r['found'])} ({r['places_searched']}) | {dr} | {yn(r['grasped'])} | "
                     f"{yn(r['delivered_reported'])} ({yn(r['delivered_verified'])}) | "
                     f"{'found' if r['door_discovered'] else ('missed' if r['locked_door'] else '')} | "
                     f"{'yes' if r['fallen'] else ''} | {'OK' if r['success'] else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=int, nargs="*", default=list(range(25)))
    ap.add_argument("--items", nargs="*", default=["alternate"],
                    help="'alternate' (bottle on even scenarios, cup on odd) or item names to run each")
    ap.add_argument("--grasp", choices=list(GRASP_ENV), required=True,
                    help="cheat: abstracted grasp always; real: real grasp only; fallback: real, then abstracted")
    ap.add_argument("--tag", required=True, help="results go to logs/experiments/<tag>/")
    ap.add_argument("-j", type=int, default=3, help="parallel sims (~2.5 GB RAM each)")
    ap.add_argument("--video", action="store_true", help="record each run to logs/experiments/<tag>/*.mp4")
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--fps", type=float, default=15)
    args = ap.parse_args()

    os.environ["G1_CHEAT_GRASP"] = GRASP_ENV[args.grasp]  # before any worker imports skills
    out = OUT_ROOT / args.tag
    if (out / "results.csv").exists():
        ap.error(f"{out} already has results; pick another --tag")
    out.mkdir(parents=True, exist_ok=True)
    video_dir = str(out) if args.video else None
    if args.items == ["alternate"]:
        jobs = [(n, ITEMS[n % 2]) for n in args.scenarios]
    else:
        jobs = [(n, it) for n in args.scenarios for it in args.items]
    jobs = [(n, it, args.grasp, video_dir, tuple(args.size), args.fps) for n, it in jobs]
    print(f"{len(jobs)} runs, grasp={args.grasp} (G1_CHEAT_GRASP={os.environ['G1_CHEAT_GRASP']}), "
          f"-j {args.j} -> {out}", flush=True)

    rows = []
    fields = None
    with ProcessPoolExecutor(max_workers=args.j) as ex, (out / "results.csv").open("w", newline="") as f:
        for row in ex.map(run_one, jobs):
            rows.append(row)
            if fields is None:
                fields = list(row)
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
            w.writerow(row)
            f.flush()  # partial results survive an interrupted batch
            print(f"{'OK  ' if row['success'] else 'FAIL'} {row['scenario']:2d} {row['item']:12} {row['setup']:22} "
                  f"found={row['found']} grasp={row['grasped']} tray={row['delivered_verified']} "
                  f"drawer_closed={row['drawer_closed_after']} door={row['door_discovered'] or '-'} "
                  f"{row['outcome'][:30]} ({row['wall_s']}s)", flush=True)
    (out / "summary.md").write_text(summarise(rows, args.tag, args.grasp))
    print(f"\n{sum(r['success'] for r in rows)}/{len(rows)} full successes. {out / 'summary.md'}")


if __name__ == "__main__":
    main()
