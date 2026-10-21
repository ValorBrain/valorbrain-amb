"""
Gateway LLM for AMB — talks to an OpenAI-compatible endpoint without
response_format support (e.g. opencode-go gateway serving deepseek-v4-flash).

Uses prompt-based JSON extraction instead of structured output.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from .base import LLM, Schema

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_DELAY = 5

# Payload keys an extra body may not replace: they define what is being asked
# and of which model, so the strict served-model check would be meaningless.
_RESERVED_EXTRA_BODY_KEYS = ("model", "messages")

# Response headers that name the model a gateway actually served, in priority
# order; the JSON "model" field is the fallback.
_SERVED_MODEL_HEADERS = ("x-valorbrain-gateway-model", "x-manifest-model")

# HTTP statuses after which another try may succeed: rate limit, server and
# gateway errors, Anthropic's 529 overloaded.
TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504, 529})


def normalize_model_name(name: str) -> str:
    """Compare-friendly model name: last '/' segment, without a '-subscription' suffix.

    'anthropic/claude-haiku-5-5' and 'claude-haiku-5-5-subscription' both become
    'claude-haiku-5-5'.
    """
    base = name.strip().rsplit("/", 1)[-1]
    return base.removesuffix("-subscription")


def parse_extra_body(raw: str, source: str) -> dict:
    """Parse an extra-body JSON object from `source` (an env var name, for errors)."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source} is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object, got {type(value).__name__}")
    return value


class ServedModelMismatch(RuntimeError):
    """Strict model check failed: the gateway served (or reported) another model.

    Not retried, and scoring code must re-raise it instead of scoring 0: a run
    judged or answered by a substituted model has to fail, not look like a
    regression.
    """


class TransientLLMError(RuntimeError):
    """The gateway gave up on an error that may pass on its own: an HTTP status
    in TRANSIENT_HTTP_STATUS or a timeout ("The read operation timed out").

    The runner retries the call that raised it; any other gateway error is final.
    """


def is_transient_error(e: BaseException) -> bool:
    """An HTTP status in TRANSIENT_HTTP_STATUS, or a timeout."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code in TRANSIENT_HTTP_STATUS
    if isinstance(e, TimeoutError):  # socket.timeout during the read
        return True
    # A timeout before the response arrives comes wrapped in a URLError.
    return isinstance(e, urllib.error.URLError) and isinstance(e.reason, TimeoutError)


class GatewayLLM(LLM):
    """OpenAI-compatible LLM via plain HTTP, no response_format dependency."""

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 extra_body: dict | None = None, strict_model: bool = False):
        # Base/key por papel: o leitor glm roda no coding plan da Z.ai
        # (assinatura), o juiz deepseek no gateway OpenCode local — quem paga
        # por token é só quem não tem assinatura.
        self._base = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8201/v1")).rstrip("/")
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("OMB_GATEWAY_MODEL", "deepseek-v4-flash")
        # Portado do provider openai (f283496): o research de 75,5% comprou
        # +3,2pp com reasoning effort max — suporte que nunca chegou ao gateway.
        # Por papel: leitor lê OMB_ANSWER_REASONING_EFFORT, juiz OMB_JUDGE_REASONING_EFFORT.
        self._reasoning_effort = reasoning_effort or os.environ.get("OMB_GATEWAY_REASONING_EFFORT")
        # Provider-specific request fields (e.g. Anthropic adaptive thinking via
        # Manifest), merged into the payload last.
        extra_body = dict(extra_body or {})
        reserved = [k for k in _RESERVED_EXTRA_BODY_KEYS if k in extra_body]
        if reserved:
            raise ValueError(
                f"GatewayLLM extra_body may not set {', '.join(reserved)}; "
                "use the model argument instead"
            )
        self._extra_body = extra_body
        # Strict: fail the call when the gateway served a different model than
        # requested (a gateway that silently substitutes models otherwise passes
        # the requested name into the results).
        self._strict_model = strict_model

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
            except ServedModelMismatch:
                # Deterministic: the gateway keeps substituting, retrying only burns time.
                raise
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    error = TransientLLMError if is_transient_error(e) else RuntimeError
                    raise error(f"GatewayLLM failed after {_MAX_RETRIES} retries: {e}") from e

    def _call(self, prompt: str) -> str:
        url = f"{self._base}/chat/completions"
        # Reasoning models (deepseek-v4-flash) spend thousands of tokens on
        # thinking before the content lands — 8192 starved long reader prompts
        # and the empty content silently became an empty answer (same failure
        # class the GLM judge had before its fix).
        max_tokens = int(os.environ.get("OMB_GATEWAY_MAX_TOKENS", "16384"))
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        payload.update(self._extra_body)
        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                # opencode.ai's edge 403s the default "Python-urllib/3.x" UA
                # (WAF) while the same request from curl passes — send a
                # neutral client UA instead.
                "User-Agent": os.environ.get("OMB_GATEWAY_USER_AGENT", "amb-gateway/1.0"),
                **({"Authorization": f"Bearer {self._key}"} if self._key else {}),
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            served = self._served_model(resp.headers, data)
        logger.debug("GatewayLLM requested=%s served=%s", self._model, served)
        if self._strict_model:
            if not served:
                raise ServedModelMismatch(
                    f"strict model check: gateway did not report the served model "
                    f"(requested {self._model})"
                )
            if normalize_model_name(served) != normalize_model_name(self._model):
                raise ServedModelMismatch(
                    f"strict model check: requested {self._model} but gateway served {served}"
                )
        choice = data.get("choices", [{}])[0]
        if choice.get("finish_reason") == "content_filter":
            # Provider refusal (e.g. Anthropic stop_reason "refusal"): the answer
            # stays empty and scores 0, but it must not pass unnoticed.
            logger.warning(
                "GatewayLLM content_filter: model %s refused (requested %s)",
                served or self._model, self._model,
            )
        msg = choice.get("message", {})
        content = (msg.get("content") or "").strip()
        if content:
            return content
        # Reasoning fallback: when content is truncated away, the whole
        # answer can live inside reasoning_content.
        return (msg.get("reasoning_content") or "").strip()

    @staticmethod
    def _served_model(headers, data: dict) -> str | None:
        for name in _SERVED_MODEL_HEADERS:
            value = headers.get(name)
            if value:
                return value
        return data.get("model") or None

    def _parse_json(self, text: str) -> dict:
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Try direct parse first
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        # Try to extract JSON object from text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # The model answered in prose (or a bare array) instead of the JSON
        # object — that IS its answer, not an empty one. Ship it as answer
        # instead of silently zeroing the query.
        if isinstance(parsed, list):
            return {"answer": json.dumps(parsed, ensure_ascii=False)}
        if text:
            return {"answer": text}

        # Last resort: return empty fields
        return {field: "" for field in ["reasoning", "answer", "choice", "reason", "correct"]}
