"""Run the scripted house demo over many seeds x layouts, headless and in parallel, and
print a per-skill success table with the failure reasons.

    python batch_eval.py                                   # tidy + laundry_day, seeds 0-3, 4 workers
    python batch_eval.py --seeds 0-9 --layouts tidy laundry_day moving_house -j 6
    python batch_eval.py --tag before   (then change code)   python batch_eval.py --tag after
    python batch_eval.py --compare before after

Each run is `house_demo.py --no-video --with-cup --seed S --layout L` in its own process;
its final `key: value` summary lines are parsed. Results go to logs/batch/<tag>/ (one .log and
one .json per run, plus summary.json). Extra environment variables (e.g. G1_LOCALIZE=0,
G1_STANCE_SEARCH=0) are passed through to the runs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAY_XY = None  # filled lazily from house.DELIVERY


def tray():
    global TRAY_XY
    if TRAY_XY is None:
        sys.path.insert(0, str(HERE))
        from house import DELIVERY
        TRAY_XY = (DELIVERY["pos"], DELIVERY["top"])
    return TRAY_XY


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_log(text: str) -> dict:
    res = {}
    for line in text.splitlines():
        if ": " not in line or line.startswith((" ", "[", "WARNING")):
            continue
        k, v = line.split(": ", 1)
        if not k.replace("_", "").isalnum():
            continue
        try:
            res[k] = ast.literal_eval(v)
        except Exception:
            res[k] = v
    return res


def on_tray(p) -> bool:
    (cx, cy), top = tray()
    return p is not None and abs(p[0] - cx) < 0.25 and abs(p[1] - cy) < 0.25 and p[2] > top - 0.02


def score(r: dict) -> dict:
    """Per-skill pass/fail (None = not reached) and a reason for the first failure."""
    def ok(key, extra=True):
        if key not in r:
            return None
        v = r[key]
        return bool(isinstance(v, dict) and v.get("ok") and extra)

    final = r.get("final_item_pos", {}) if isinstance(r.get("final_item_pos"), dict) else {}
    s = {
        "walk": ok("go_drawers"),
        "open": ok("open", isinstance(r.get("open"), dict) and r["open"].get("drawer_now_m", 0) > 0.25),
        "pick_drawer": ok("pick_bottle"),
        "close": ok("close", isinstance(r.get("close"), dict) and r["close"].get("drawer_now_m", 1) < 0.03),
        "deliver_1": None if "hand_over" not in r else bool(ok("hand_over") and on_tray(final.get("water_bottle"))),
        "pick_counter": ok("pick_cup"),
        "deliver_2": None if "hand_over_2" not in r else bool(ok("hand_over_2") and on_tray(final.get("cup"))),
    }
    s["fallen"] = bool(r.get("fallen"))
    s["full"] = all(s[k] for k in ("open", "pick_drawer", "close", "deliver_1", "pick_counter", "deliver_2")) and not s["fallen"]
    why = ""
    for key, rk in (("walk", "go_drawers"), ("open", "open"), ("pick_drawer", "pick_bottle"), ("close", "close"),
                    ("deliver_1", "hand_over"), ("pick_counter", "pick_cup"), ("deliver_2", "hand_over_2")):
        if s[key] is False:
            v = r.get(rk, {})
            why = f"{key}: {v.get('error') or v}" if isinstance(v, dict) else f"{key}: {v}"
            break
    s["why"] = why[:160]
    return s


def run_one(seed: int, layout: str, outdir: Path, timeout: float) -> dict:
    tag = f"{layout}_s{seed}"
    t0 = time.time()
    cmd = [sys.executable, str(HERE / "house_demo.py"), "--no-video", "--with-cup", "--seed", str(seed), "--layout", layout]
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), OMP_NUM_THREADS="1")
    try:
        p = subprocess.run(cmd, cwd=outdir, env=env, capture_output=True, text=True, timeout=timeout)
        text, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
    (outdir / f"{tag}.log").write_text(text)
    r = parse_log(text)
    s = score(r)
    if rc != 0 and not s["why"]:
        tail = [ln for ln in text.splitlines() if ln.strip()][-1:] or ["?"]
        s["why"] = f"crashed ({rc}): {tail[0][:140]}"
        s["full"] = False
    out = dict(seed=seed, layout=layout, rc=rc, wall_s=round(time.time() - t0), skills=s, results=r)
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    mark = "OK  " if s["full"] else "FAIL"
    print(f"  {mark} {tag:22s} {out['wall_s']:4d}s  {s['why']}", flush=True)
    return out


SKILLS = ("walk", "open", "pick_drawer", "close", "deliver_1", "pick_counter", "deliver_2", "full")


def table(runs: list[dict], title: str) -> str:
    lines = [f"\n== {title}: {len(runs)} runs"]
    for k in SKILLS:
        vals = [r["skills"][k] for r in runs if r["skills"].get(k) is not None]
        n_ok = sum(vals)
        lines.append(f"  {k:13s} {n_ok:3d}/{len(vals):<3d} {'#' * n_ok}{'.' * (len(vals) - n_ok)}")
    fails = [f"    {r['layout']}_s{r['seed']}: {r['skills']['why']}" for r in runs if not r["skills"]["full"]]
    if fails:
        lines.append("  failures:")
        lines += fails
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--layouts", nargs="+", default=["tidy", "laundry_day"])
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--tag", default=time.strftime("%H%M%S"))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()
    root = HERE / "logs" / "batch"

    if args.compare:
        for t in args.compare:
            runs = json.loads((root / t / "summary.json").read_text())
            print(table(runs, t))
        return

    outdir = root / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    # each run writes drawer_model.json into its cwd: give every run the same fresh copy dir
    jobs = [(s, lay) for lay in args.layouts for s in parse_range(args.seeds)]
    print(f"{len(jobs)} runs, {args.jobs} at a time -> {outdir}")
    work = []
    for s, lay in jobs:
        d = outdir / f"cwd_{lay}_s{s}"
        d.mkdir(exist_ok=True)
        work.append((s, lay, d))
    with ThreadPoolExecutor(args.jobs) as ex:
        runs = list(ex.map(lambda w: run_one(w[0], w[1], w[2], args.timeout), work))
    for r in runs:  # move logs up next to the summary
        d = outdir / f"cwd_{r['layout']}_s{r['seed']}"
        for f in d.glob("*_s*.*"):
            f.replace(outdir / f.name)
    (outdir / "summary.json").write_text(json.dumps(runs, indent=1, default=str))
    print(table(runs, args.tag))


if __name__ == "__main__":
    main()
