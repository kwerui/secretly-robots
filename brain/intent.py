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


# A request that names what's wanted: act on it directly.
#   "I need water", "Could I have some water?", "Can you get me something to drink?"
NEED_PATTERNS = {
    "water_bottle": re.compile(
        r"\b(need|want|have|could do with|would love|'d love|love|fancy|like|get|bring|fetch)\b"
        r"[\w\s']{0,25}?\b(drink|water|sip|glass of water)\b"
        r"|\b(a|some)\s+(drink|water)\s+please\b"
    ),
}

# A hint at a need that doesn't name the item: the robot should ask first.
#   "My mouth is so dry", "I'm parched", "I am thirsty"
HINT_PATTERNS = {
    "water_bottle": re.compile(
        r"\b(thirsty|parched|dehydrated|dry mouth|mouth('s| is)( so| really| very)? dry"
        r"|throat('s| is)( so| really| very)? dry|lips are( so)? dry)\b"
    ),
}

YES = re.compile(r"\b(yes|yeah|yep|yup|please|sure|ok|okay|alright|go on|that would be (great|lovely|nice))\b")
NO = re.compile(r"\b(no|nope|not now|no thanks|no thank you|i'm fine|i'm okay|don't|do not|never mind)\b")


def _clean(text: str) -> str:
    return text.lower().replace("\u2019", "'")


def classify_need(text: str) -> tuple[str, str] | None:
    """("direct", item) for a request naming the item, ("hint", item) for an
    indirect hint the robot should confirm, None otherwise."""
    t = _clean(text)
    for item, pattern in NEED_PATTERNS.items():
        if pattern.search(t):
            return "direct", item
    for item, pattern in HINT_PATTERNS.items():
        if pattern.search(t):
            return "hint", item
    return None


def classify_answer(text: str) -> str | None:
    """"yes", "no" or None for a reply to a yes/no question. No wins a tie."""
    t = _clean(text)
    if NO.search(t):
        return "no"
    if YES.search(t):
        return "yes"
    return None


class RulesIntentParser:
    """Predictable offline fallback, not an LLM or imitation learner."""
    def parse(self, request: str) -> FetchTask:
        text = _clean(request)
        need = classify_need(text)
        if need is not None:  # hints too: confirming them is the listener's job
            return FetchTask(need[1])
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


GEMMA_SYSTEM_PROMPT = (
    "You interpret what a person seated on a sofa says to a household helper robot. "
    "Return schema-compliant JSON only. Do not execute actions. "
    "A direct request ('bring me my water') OR a stated need that a supported item "
    "would meet ('I am thirsty', 'I need a drink') counts as intent fetch. "
    "Thirst or wanting a drink means item water_bottle. "
    "Anything else (tiredness, pain, chat, questions) is intent unknown with item unknown. "
    "If ambiguous, use unknown."
)


UTTERANCE_KINDS = ("request", "hint", "yes", "no", "other")

GEMMA_CLASSIFY_PROMPT = (
    "You are the language understanding of a home-care robot. A person with limited "
    "mobility, sitting on a sofa, has just said something. Classify it. Return JSON only.\n"
    "kind:\n"
    "- request: they ask for an item, or say they need/want one and name it "
    "('bring me my book', 'I need some water', 'could I get a drink').\n"
    "- hint: they describe a feeling or problem that an item would help with, without "
    "asking for or naming it ('my mouth is so dry', 'I'm parched', 'I'm a bit thirsty'). "
    "The robot will ask before acting.\n"
    "- yes / no: an answer to a yes-or-no question ('sure', 'that would be lovely' = yes; "
    "'I'm alright', 'not now' = no). Use these only when the robot is waiting for an answer.\n"
    "- other: anything else (pain, tiredness, chat, questions, unclear). Never guess a request.\n"
    "item_id: water_bottle for thirst or any drink; cup or book if named; otherwise unknown. "
    "Medication is never a supported item: use other."
)


class OllamaIntentParser:
    """Real local model call via Ollama's loopback API; no cloud access.

    Requires separately installed Ollama and downloaded model weights.
    If unavailable, raises IntentError: it does NOT silently fake model output.
    """
    def __init__(self, model: str = "gemma4:e4b", endpoint: str = "http://127.0.0.1:11434/api/chat", timeout_s: float = 60):
        if not endpoint.startswith("http://127.0.0.1:") and not endpoint.startswith("http://localhost:"):
            raise ValueError("For the offline configuration, only loopback Ollama endpoints are allowed")
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def classify(self, text: str, expecting_answer: bool = False) -> tuple[str, str]:
        """(kind, item_id) for one utterance, kind in UTTERANCE_KINDS.

        Raises IntentError if the local model is unavailable or returns junk.
        """
        schema = {
            "type": "object", "properties": {
                "kind": {"type": "string", "enum": list(UTTERANCE_KINDS)},
                "item_id": {"type": "string", "enum": sorted(ALLOWED_OBJECTS) + ["unknown"]},
            }, "required": ["kind", "item_id"], "additionalProperties": False,
        }
        context = ("The robot just asked them a yes-or-no question and is waiting for the answer."
                   if expecting_answer else "The robot has not asked them anything.")
        result = self._chat(GEMMA_CLASSIFY_PROMPT, f"{context}\nThey said: {text!r}", schema)
        kind, item = result.get("kind"), result.get("item_id", "unknown")
        if kind not in UTTERANCE_KINDS:
            raise IntentError(f"Local model returned an unknown kind: {kind!r}")
        if kind in ("request", "hint"):
            item = normalize_item(item)  # raises for unknown/unsupported
        if kind in ("yes", "no") and not expecting_answer:
            kind = "other"
        return kind, item

    def _chat(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model, "stream": False, "format": schema, "think": False,
            "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0},
        }
        req = Request(self.endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout_s) as response:
                answer = json.loads(response.read().decode())
            return json.loads(answer["message"]["content"])
        except (URLError, TimeoutError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise IntentError(f"Local model unavailable or invalid output: {exc}") from exc

    def warm_up(self) -> None:
        """Load the model into memory before anyone speaks (first call is slow)."""
        self.classify("hello")

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
                {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            "options": {"temperature": 0},
        }
        req = Request(self.endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout_s) as response:
                answer = json.loads(response.read().decode())
            result = json.loads(answer["message"]["content"])
        except (URLError, TimeoutError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise IntentError(f"Local model unavailable or invalid output: {exc}") from exc
        if result.get("intent") != "fetch":
            raise IntentError("Local model could not confidently identify a fetch request")
        return FetchTask(normalize_item(result.get("item_id", "unknown")))
