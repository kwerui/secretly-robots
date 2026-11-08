"""Request understanding: fully offline baseline and optional local Gemma.

Intent parsing only: neither parser commands the robot's motors.
"""
from __future__ import annotations
import json
import re
from urllib.request import Request, urlopen
from urllib.error import URLError
from .contracts import FetchTask

ALLOWED_OBJECTS = {"water_bottle", "cup", "book"}

class IntentError(ValueError):
    pass


def normalize_item(item: str) -> str:
    name = item.strip().lower().replace(" ", "_")
    if name not in ALLOWED_OBJECTS:
        raise IntentError(f"Unsupported object: {item!r}; ask a human to clarify")
    return name


class RulesIntentParser:
    """Predictable offline fallback, not an LLM or imitation learner."""
    def parse(self, request: str) -> FetchTask:
        text = request.lower()
        # (G1 integration, 8 Nov) needs, not just commands: "I'm thirsty" -> the water bottle
        if re.search(r"\b(thirsty|drink)\b", text):
            return FetchTask("water_bottle")
        if not re.search(r"\b(bring|fetch|get|pass|hand)\b", text):
            raise IntentError("I can only handle a simple fetch request for now")
        if "bottle" in text or "water" in text:
            return FetchTask("water_bottle")
        if re.search(r"\bcup\b", text):
            return FetchTask("cup")
        if re.search(r"\bbook\b", text):
            return FetchTask("book")
        raise IntentError("Item not recognised; ask the person to clarify")


class OllamaIntentParser:
    """Real local model call via Ollama's loopback API; no cloud access.

    Requires separately installed Ollama and downloaded model weights.
    If unavailable, raises IntentError: it does NOT silently fake model output.
    """
    def __init__(self, model: str = "gemma4:e2b", endpoint: str = "http://127.0.0.1:11434/api/chat"):
        if not endpoint.startswith("http://127.0.0.1:") and not endpoint.startswith("http://localhost:"):
            raise ValueError("For the offline configuration, only loopback Ollama endpoints are allowed")
        self.model = model
        self.endpoint = endpoint

    def parse(self, request: str) -> FetchTask:
        schema = {
            "type": "object", "properties": {
                "intent": {"type": "string", "enum": ["fetch", "unknown"]},
                "item_id": {"type": "string", "enum": sorted(ALLOWED_OBJECTS) + ["unknown"]}
            }, "required": ["intent", "item_id"], "additionalProperties": False,
        }
        payload = {
            "model": self.model, "stream": False, "format": schema,
            "messages": [
                {"role": "system", "content": "Extract a fetch request only. Return schema-compliant JSON. Do not execute actions. If ambiguous, use unknown."},
                {"role": "user", "content": request},
            ],
            "options": {"temperature": 0},
        }
        req = Request(self.endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=60) as response:
                answer = json.loads(response.read().decode())
            result = json.loads(answer["message"]["content"])
        except (URLError, TimeoutError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise IntentError(f"Local model unavailable or invalid output: {exc}") from exc
        if result.get("intent") != "fetch":
            raise IntentError("Local model could not confidently identify a fetch request")
        return FetchTask(normalize_item(result.get("item_id", "unknown")))
