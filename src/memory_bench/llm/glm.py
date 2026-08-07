"""
GLM-5.2 LLM for AMB — via Z.ai coding plan endpoint.

GLM-5.2 is a reasoning model: outputs reasoning_content separately from content.
Uses prompt-based JSON extraction (Z.ai doesn't support response_format).
"""

import json
import os
import urllib.request

from .gateway import GatewayLLM


class GlmLLM(GatewayLLM):
    """GLM-5.2 via Z.ai coding plan. Expensive per-request, use sparingly."""

    def __init__(self, model: str | None = None):
        self._base = os.environ.get(
            "GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4"
        ).rstrip("/")
        self._key = os.environ.get("GLM_API_KEY", "")
        self._model = model or os.environ.get("GLM_MODEL", "glm-5.2")

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
        import time as _time
        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    msg = data.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content", "")
                    # GLM-5.2 puts output in reasoning_content when content is empty
                    if not content and msg.get("reasoning_content"):
                        content = msg["reasoning_content"]
                    if content.strip():
                        return content
                    # Empty response — retry
                    last_err = "empty response"
            except Exception as e:
                last_err = str(e)
            if attempt < 2:
                _time.sleep(5 * (attempt + 1))  # 5s, 10s backoff
        # All retries failed — return what we have
        return ""
