"""Reader: one LLM call per (system, question), identical prompt for every system.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint. The reader must
return a single code or NONE; scoring is exact match on that value (see
scoring.py), so no LLM judge is involved.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

PROMPT = """You are an AI agent starting a work session. Today is {today}. You are acting as {persona}.
Your memory system returned the context below for the question. Use only this context.

Rules:
- Answer with the single code (shaped like WORD-123) that answers the question as of today.
- If the context offers several candidate codes, choose the one that is current, still in force today and still pending for you.
- If the context does not establish such an answer, answer NONE.

Question: {question}

<context>
{context}
</context>

Respond with only a JSON object: {{"value": "<CODE or NONE>", "reason": "<one short sentence>"}}"""

_CODE = re.compile(r"\b[A-Z]{3,12}-\d{3}\b")
_MAX_CONTEXT_CHARS = 24000


class Reader:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ.get("CB_READER_BASE_URL", "http://127.0.0.1:8201/v1")).rstrip("/")
        self.model = model or os.environ.get("CB_READER_MODEL", "deepseek-v4.1-flash")
        self._key = api_key if api_key is not None else os.environ.get("CB_READER_API_KEY", "")

    def answer(self, today: str, persona: str, question: str, context: str) -> tuple[str, str, str]:
        """Return (value, raw_text, served_model)."""
        prompt = PROMPT.format(today=today, persona=persona, question=question,
                               context=context[:_MAX_CONTEXT_CHARS] or "(empty)")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": int(os.environ.get("CB_READER_MAX_TOKENS", "4096")),
        }).encode()
        headers = {"content-type": "application/json", "user-agent": "continuity-bench/1"}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        last: Exception | None = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                if not text:
                    raise ValueError("empty reader response")
                return parse_value(text), text, str(data.get("model", self.model))
            except Exception as e:  # network, rate limit, malformed body: retry with backoff
                last = e
                time.sleep(3 * 2 ** attempt)
        raise RuntimeError(f"reader failed after retries: {last}")


def parse_value(text: str) -> str:
    """Extract the answered value: the JSON ``value`` field, else the last code or NONE."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = str(json.loads(match.group(0)).get("value", "")).strip().upper()
            if value == "NONE" or _CODE.fullmatch(value):
                return value
        except (json.JSONDecodeError, AttributeError):
            pass
    codes = _CODE.findall(text)
    if codes:
        return codes[-1]
    return "NONE" if re.search(r"\bNONE\b", text) else "UNPARSED"
