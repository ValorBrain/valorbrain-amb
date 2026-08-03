"""
Gateway LLM for AMB — talks to an OpenAI-compatible endpoint without
response_format support (e.g. opencode-go gateway serving deepseek-v4-flash).

Uses prompt-based JSON extraction instead of structured output.
"""

import json
import os
import re
import time
import urllib.request

from .base import LLM, Schema

_MAX_RETRIES = 4
_RETRY_DELAY = 5


class GatewayLLM(LLM):
    """OpenAI-compatible LLM via plain HTTP, no response_format dependency."""

    def __init__(self, model: str | None = None):
        self._base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8201/v1").rstrip("/")
        self._key = os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("OMB_GATEWAY_MODEL", "deepseek-v4-flash")

    @property
    def model_id(self) -> str:
        return f"gateway:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        # Inject JSON instruction into the prompt
        fields_desc = ", ".join(schema.required)
        json_instruction = (
            f"\n\nRespond as a JSON object with these fields: {fields_desc}. "
            "Output ONLY valid JSON, no markdown fences, no prose."
        )
        full_prompt = prompt + json_instruction

        delay = _RETRY_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                data = self._call(full_prompt)
                return self._parse_json(data)
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(f"GatewayLLM failed after {_MAX_RETRIES} retries: {e}")

    def _call(self, prompt: str) -> str:
        url = f"{self._base}/chat/completions"
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self._key}"} if self._key else {}),
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _parse_json(self, text: str) -> dict:
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: return empty fields
        return {field: "" for field in ["reasoning", "answer", "choice"]}
