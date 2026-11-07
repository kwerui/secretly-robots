"""LLM agent layer: the person's request in, robot skills as tools.

The model decides what is meant ("I'm thirsty" -> water), where to look (location memory),
when to open the cupboard, and when to ask the carer instead of guessing. Every skill
returns a JSON result so the model can recover from failures.

    export ANTHROPIC_API_KEY=...
    python agent.py "I'm thirsty"

Untested against the live API in this repo (no key in the build environment): treat it as
the starting point for the agent teammates, and keep tasks.fetch() as the demo fallback.
"""

from __future__ import annotations

import os
import sys

# Offscreen rendering (robot camera, video) through EGL: uses the GPU without X/GLX. The
# MuJoCo --viewer window still needs GLX. Must run before anything imports mujoco.
# Override with MUJOCO_GL=glfw/osmesa.
os.environ.setdefault("MUJOCO_GL", "egl")

import json

from articulation import ArticulationModel
from memory import LocationMemory
from scene import ITEMS, STATIONS

SYSTEM = """You control a home-assistance humanoid robot helping an elderly person with limited mobility.
You can only act through the tools. Items you can fetch: {items}. Stations: {stations}.
Rules: check the location memory before searching; look() after arriving anywhere; if an item is
in the cupboard and the cupboard is closed, open it with open_cupboard (it was learned from the carer);
never guess about medication: if unsure which item or the item is missing after searching the likely
places, call ask_carer. Keep messages to the person short and warm."""

TOOLS = [
    {"name": "get_belief", "description": "Probability of where an item usually is, learned from watching the carer.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string", "enum": list(ITEMS)}}, "required": ["item"]}},
    {"name": "go_to", "description": "Walk to a station.",
     "input_schema": {"type": "object", "properties": {"station": {"type": "string", "enum": list(STATIONS)}}, "required": ["station"]}},
    {"name": "look", "description": "Detect tagged items and the cupboard with the head camera at the current station.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "open_cupboard", "description": "Open the cupboard door using the motion learned from the carer's demonstration. Must be at the cupboard station.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "pick", "description": "Pick up a visible item at the current station.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string", "enum": list(ITEMS)}}, "required": ["item"]}},
    {"name": "hand_over", "description": "At the person's station, place the held item on the side table next to them.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "say", "description": "Say something to the person.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "ask_carer", "description": "Send the carer a question (e.g. where an item is kept, or to show a new task).",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
]


class Agent:
    def __init__(self, robot, memory: LocationMemory, cupboard_model: ArticulationModel | None,
                 model: str = os.environ.get("AGENT_MODEL", "claude-sonnet-4-5")):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.robot, self.memory, self.cupboard = robot, memory, cupboard_model
        self.system = SYSTEM.format(items=", ".join(ITEMS), stations=", ".join(STATIONS))

    def _call(self, name: str, args: dict) -> dict:
        r = self.robot
        if name == "get_belief":
            return {"ranked": [(s, round(p, 2)) for s, p in self.memory.ranked(args["item"])]}
        if name == "go_to":
            return r.go_to(args["station"])
        if name == "look":
            res = r.look()
            for item in ITEMS:  # update memory with what was / wasn't seen at this station
                if r.station in self.memory.stations:
                    if item in res["items"]:
                        self.memory.observe_search(item, r.station, True)
            return res
        if name == "open_cupboard":
            if self.cupboard is None:
                return {"ok": False, "error": "no demonstration recorded for the cupboard yet"}
            return r.open_cupboard(self.cupboard)
        if name == "pick":
            return r.pick(args["item"])
        if name == "hand_over":
            return r.hand_over()
        if name == "say":
            print(f"ROBOT: {args['text']}")
            return {"ok": True}
        if name == "ask_carer":
            print(f"TO CARER: {args['question']}")
            return {"ok": True, "note": "message sent; carer will reply later"}
        return {"ok": False, "error": f"unknown tool {name}"}

    def run(self, request: str, max_turns: int = 25) -> None:
        self.memory.new_episode()
        messages = [{"role": "user", "content": f'The person says: "{request}"'}]
        for _ in range(max_turns):
            resp = self.client.messages.create(model=self.model, max_tokens=1024, system=self.system,
                                               tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                for block in resp.content:
                    if getattr(block, "type", "") == "text":
                        print(f"ROBOT: {block.text}")
                return
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = self._call(block.name, block.input)
                    print(f"  [{block.name}] {json.dumps(block.input)} -> {json.dumps(out)[:200]}")
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    from pathlib import Path

    from run_demo import learned_memory
    from skills import AssistiveRobot

    model_path = Path("cupboard_model.json")
    cup = ArticulationModel.load(model_path) if model_path.exists() else None
    Agent(AssistiveRobot(), learned_memory(), cup).run(" ".join(sys.argv[1:]) or "I'm thirsty")
