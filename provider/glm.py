"""
GLM-5.2 LLM for AMB — works with any OpenAI-compatible endpoint.

GLM-5.2 is a reasoning model: outputs reasoning_content separately from content.
This provider handles that fallback and uses prompt-based JSON extraction
(endpoints that don't support response_format).

Configuration:
  GLM_BASE_URL  — your GLM API endpoint (OpenAI-compatible)
  GLM_API_KEY   — your API key
  GLM_MODEL     — model name (default: glm-5.2)
"""

import json
import os
import urllib.request

from .gateway import GatewayLLM


class GlmLLM(GatewayLLM):
    """GLM-5.2 via any OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None):
        self._base = os.environ.get("GLM_BASE_URL", "").rstrip("/")
        self._key = os.environ.get("GLM_API_KEY", "")
        self._model = model or os.environ.get("GLM_MODEL", "glm-5.2")
        if not self._base:
            raise ValueError("GLM_BASE_URL environment variable is required. Set it to your GLM provider's OpenAI-compatible endpoint.")

    @property
    def model_id(self) -> str:
        return f"glm:{self._model}"

    def _call(self, prompt: str) -> str:
        url = f"{self._base}/chat/completions"
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {self._key}"} if self._key else {}),
        }, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "")
            # GLM-5.2 puts output in reasoning_content when content is empty
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
            return content
