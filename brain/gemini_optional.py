"""Opt-in CLOUD request parser using official Gemini Interactions API.

Never enabled automatically. Requires google-genai SDK and GEMINI_API_KEY.
This module makes a REAL network call if selected; it is not an offline path.
"""
import json
import os
from .contracts import FetchTask
from .intent import IntentError, normalize_item

class GeminiIntentParser:
    def __init__(self, model: str = "gemini-4-argon"):
        self.model = model

    def parse(self, request: str) -> FetchTask:
        if not os.getenv("GEMINI_API_KEY"):
            raise IntentError("GEMINI_API_KEY missing; cloud parser disabled")
        try:
            from google import genai
        except ImportError as exc:
            raise IntentError("Install optional google-genai dependency") from exc
        client = genai.Client()
        prompt = (
            "Extract ONLY a fetch request into exactly JSON with fields "
            "intent ('fetch' or 'unknown') and item_id (water_bottle, cup, book or unknown). "
            "Do not add any explanatory text. Request: " + request
        )
        try:
            result = client.interactions.create(model=self.model, input=prompt)
            value = json.loads(result.output_text)
        except (AttributeError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise IntentError(f"Cloud model did not return valid JSON: {exc}") from exc
        if value.get("intent") != "fetch":
            raise IntentError("Cloud model did not identify a fetch request")
        return FetchTask(normalize_item(value.get("item_id", "unknown")))
