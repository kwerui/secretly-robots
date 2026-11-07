"""Thin, opt-in client for Gemini's structured-output API.

This module is the only cloud boundary. It deliberately returns decoded JSON
objects, never free-form model text, so planning code cannot act on an explanation.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping, Protocol

GEMINI_ARGON_MODEL = "gemini-4-argon"


class GeminiApiError(RuntimeError):
    """The optional cloud planner cannot provide a trustworthy response."""


class _ModelsApi(Protocol):
    def generate_content(self, **kwargs: Any) -> Any: ...


class _SdkClient(Protocol):
    models: _ModelsApi


class GeminiArgonClient:
    """Call Gemini only when explicitly constructed by application code."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = GEMINI_ARGON_MODEL,
        timeout_ms: int = 30_000,
        client: _SdkClient | None = None,
    ):
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.timeout_ms = timeout_ms
        self._client = client

    def generate_json(self, prompt: str, schema: Mapping[str, object]) -> dict[str, object]:
        """Return a JSON object conforming to the supplied response schema."""
        client = self._client or self._make_sdk_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._structured_config(schema),
            )
            value = json.loads(response.text)
        except GeminiApiError:
            raise
        except Exception as exc:
            raise GeminiApiError("Gemini request failed or returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise GeminiApiError("Gemini response must be a JSON object")
        return value

    def _make_sdk_client(self) -> _SdkClient:
        if not self.api_key:
            raise GeminiApiError("GEMINI_API_KEY is missing; cloud planning is disabled")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiApiError("Install requirements-gemini.txt to enable cloud planning") from exc
        return genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=self.timeout_ms),
        )

    @staticmethod
    def _structured_config(schema: Mapping[str, object]) -> object:
        """Use the official typed SDK config; allow SDK-free injected test clients."""
        try:
            from google.genai import types
        except ImportError:
            return {
                "response_mime_type": "application/json",
                "response_json_schema": dict(schema),
                "temperature": 0,
            }
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=dict(schema),
            temperature=0,
        )
