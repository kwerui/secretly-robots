"""Run the fixed tidy-house baseline repeatedly and collect its episode records.

The current baseline calls ``G1House.reset(..., jitter=False)`` and fixes the
bottle in the kitchen drawer. Seeds are retained for reproducibility, not
claimed as randomized bottle placements; each result records reset details.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluation import EpisodeMetrics, write_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/baseline"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=15)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least one")

    run_root = args.output / "runs"
    records: list[EpisodeMetrics] = []
    demo = Path(__file__).with_name("house_brain_demo.py")
    for index in range(args.trials):
        seed = args.seed + index
        experiment_id = f"tidy-baseline-seed-{seed}"
        run_dir = run_root / experiment_id
        command = [sys.executable, str(demo), "--offline", "--request", "I'm thirsty", "--layout", "tidy",
                   "--seed", str(seed), "--max-turns", str(args.max_turns), "--experiment-id", experiment_id,
                   "--results-dir", str(run_dir)]
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=Path(__file__).parent)
        result_path = run_dir / f"{experiment_id}.json"
        if result_path.exists():
            records.append(EpisodeMetrics.from_dict(json.loads(result_path.read_text())))
        else:
            failed = EpisodeMetrics.baseline(experiment_id, "water_bottle")
            failed.model_mode, failed.layout, failed.seed = "offline", "tidy", seed
            failed.failure_stage = "runner"
            failed.failure_reason = f"demo exited with code {completed.returncode} before writing a result"
            records.append(failed)
    paths = write_results(records, args.output)
    print(f"wrote {len(records)} trial records to {paths['csv']} and {paths['summary']}")


if __name__ == "__main__":
    main()
