"""Run the three-episode, perception-driven bottle-relocation benchmark.

The runner configures simulator placement only.  It never reads a bottle coordinate or
target into the brain or location memory; the child demo must acquire its evidence through
the G1 head camera and the shared persistent-memory JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from evaluation import EpisodeMetrics, write_results


@dataclass(frozen=True)
class RelocationEpisode:
    name: str
    bottle_target: str


EPISODES = (
    RelocationEpisode("episode-1-drawer", "kitchen_drawer"),
    RelocationEpisode("episode-2-counter-relocated", "kitchen_counter"),
    RelocationEpisode("episode-3-counter-reload", "kitchen_counter"),
)


def build_episode_command(*, python: str, demo: Path, episode: RelocationEpisode, memory_path: Path,
                          results_dir: Path, seed: int, max_turns: int) -> list[str]:
    """Build a child command without passing evaluation ground truth to its planner."""
    return [
        python, str(demo), "--offline", "--reuse-demo", "--live-memory",
        "--memory-path", str(memory_path), "--bottle-target", episode.bottle_target,
        "--request", "I'm thirsty", "--seed", str(seed), "--max-turns", str(max_turns),
        "--experiment-id", episode.name, "--results-dir", str(results_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/relocation"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--run-id", default=None, help="directory name; defaults to relocation-seed-<seed>")
    args = parser.parse_args()
    if args.max_turns < 1:
        parser.error("--max-turns must be at least one")

    run_id = args.run_id or f"relocation-seed-{args.seed}"
    run_root = args.output / run_id
    if run_root.exists():
        parser.error(f"refusing to overwrite existing benchmark run: {run_root}")
    run_root.mkdir(parents=True)
    memory_path = run_root / "location_memory.json"
    demo = Path(__file__).with_name("house_brain_demo.py")
    records: list[EpisodeMetrics] = []

    for episode in EPISODES:
        episode_dir = run_root / episode.name
        command = build_episode_command(
            python=sys.executable, demo=demo, episode=episode, memory_path=memory_path,
            results_dir=episode_dir, seed=args.seed, max_turns=args.max_turns,
        )
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=Path(__file__).parent)
        result_path = episode_dir / f"{episode.name}.json"
        if result_path.exists():
            records.append(EpisodeMetrics.from_dict(json.loads(result_path.read_text())))
            continue
        failed = EpisodeMetrics.baseline(episode.name, "water_bottle")
        failed.layout, failed.seed = "tidy", args.seed
        failed.bottle_placement = {"actual_target": episode.bottle_target}
        failed.failure_stage = "runner"
        failed.failure_reason = f"demo exited with code {completed.returncode} before writing a result"
        records.append(failed)

    paths = write_results(records, run_root)
    print(f"wrote {len(records)} relocation records to {paths['csv']} and {paths['summary']}")


if __name__ == "__main__":
    main()
