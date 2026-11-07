"""Deliberate, optional live smoke test for Gemini 4 Argon.

Normal unit tests mock this boundary. Run only with ``--live`` and an API key.
"""
from __future__ import annotations

import argparse
import os

from brain.contracts import FetchTask, ObjectSighting, Observation
from brain.gemini_optional import GeminiIntentParser, GeminiPlanner


def main() -> int:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--live", action="store_true", help="make real Gemini API calls")
    args = arguments.parse_args()
    if not args.live:
        print("No network call made. Re-run with --live to perform the optional paid API smoke test.")
        return 0
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is required for --live; no network call made.")
        return 2

    task = GeminiIntentParser().parse("Bring the water bottle to the person on the sofa")
    observation = Observation(
        timestamp_s=0.0,
        robot_area="sofa",
        visible_objects=(ObjectSighting("water_bottle", "cupboard", "drawer", 0.95),),
    )
    decision = GeminiPlanner().propose(task, observation)
    print(f"task={task}")
    print(f"proposed_decision={decision}")
    print("Proposal only: no MuJoCo action or robot controller was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
