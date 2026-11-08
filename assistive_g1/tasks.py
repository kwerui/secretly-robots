"""Task level: "I'm thirsty" -> which item -> search the likely places -> fetch -> hand over.

`fetch()` is a deterministic planner over the skills, useful as the demo backbone and as
the fallback when no LLM is available. agent.py exposes the same skills as LLM tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from articulation import ArticulationModel
from memory import LocationMemory
from skills import AssistiveRobot

# Fallback request -> item mapping (the LLM agent replaces this)
KEYWORDS = {
    "water_bottle": ["thirsty", "water", "drink", "bottle"],
    "cup": ["cup", "tea", "coffee", "mug"],
    "pill_box": ["pill", "medicine", "medication", "tablet"],
}


def item_for_request(text: str) -> str | None:
    t = text.lower()
    for item, words in KEYWORDS.items():
        if any(w in t for w in words):
            return item
    return None


def fetch(
    robot: AssistiveRobot,
    memory: LocationMemory,
    item: str,
    cupboard_model: ArticulationModel | None = None,
    say: Callable[[str], None] = print,
    debug_dir: Path | None = Path("debug"),
) -> dict:
    memory.new_episode()
    for station, p in memory.ranked(item):
        say(f"Looking for the {item.replace('_', ' ')} at the {station} (belief {p:.0%})")
        robot.go_to(station)
        seen = robot.look()
        if item not in seen["items"] and station == "cupboard" and "cupboard_anchor" in seen:
            if cupboard_model is None:
                say("The cupboard is closed and I haven't been shown how to open it.")
                memory.observe_search(item, station, False)
                continue
            say("Opening the cupboard the way the carer showed me")
            res = robot.open_cupboard(cupboard_model)
            if not res["ok"]:
                say(f"Couldn't open it fully: {res}")
            seen = robot.look()
        if item not in seen["items"] and station == "cupboard":
            seen = robot.look_closer()  # backed off while opening: step back in, lean, look
        if item not in seen["items"]:
            say(f"Not at the {station}")
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                robot.save_view(debug_dir / f"{item}_not_at_{station}.png")
            memory.observe_search(item, station, False)
            continue
        memory.observe_search(item, station, True)
        say(f"Found it - picking up the {item.replace('_', ' ')}")
        res = robot.pick(item)
        if not res["ok"]:
            say(f"Pick failed: {res['error']}")
            return {"ok": False, "stage": "pick", **res}
        say("Bringing it over")
        res = robot.go_to("person")
        if not res["ok"]:
            say(f"Sorry, I got stuck: {res['error']}")
            return {"ok": False, "stage": "deliver", **res}
        res = robot.hand_over()
        if not res["ok"]:
            say(f"Sorry, something went wrong: {res['error']}")
            return {"ok": False, "stage": "hand_over", **res}
        say("Here you go")
        return {"ok": True, "found_at": station}
    say(f"I couldn't find the {item.replace('_', ' ')}. Asking the carer where it lives.")
    return {"ok": False, "stage": "search", "ask_carer": f"Where do you keep the {item}?"}
