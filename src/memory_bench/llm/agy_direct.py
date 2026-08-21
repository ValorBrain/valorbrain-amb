"""
AGY direct LLM — calls Antigravity CLI via subprocess, no HTTP gateway.
Eliminates the gateway as a failure point.
"""

import json
import os
import re
import subprocess
import time

from .base import LLM, Schema

_MAX_RETRIES = 4


class AgyDirectLLM(LLM):
    """Calls agy CLI directly. No HTTP gateway, no intermediary process."""

    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get("AGY_MODEL", "gemini-3.6-flash")
        self._effort = os.environ.get("AGY_EFFORT", "low")
        # Long BEAM prompts (10k+ tokens of context) need more wall clock.
        self._timeout = int(os.environ.get("AGY_TIMEOUT", "180"))

    @property
    def model_id(self) -> str:
        return f"agy:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        fields_desc = ", ".join(schema.required)
        full_prompt = (
            # The CLI exposes file tools in print mode; at effort=high the model
            # sometimes tries to read paths mentioned in the retrieved context,
            # which trips the CLI's hardcoded FS boundary and kills the whole
            # run (observed 2026-08-20, unit 6). A benchmark reader answers
            # from the provided context — pin that contract in the prompt.
            "Answer strictly from the provided context. Do NOT use tools, "
            "read files, or run commands — respond directly.\n\n"
            + prompt
            + f"\n\nRespond as a JSON object with these fields: {fields_desc}. "
            "Output ONLY valid JSON. The 'answer' field must be a non-empty string "
            "with your final answer (or a clear abstention if context lacks info)."
        )

        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = subprocess.run(
                    [
                        "agy",
                        "--model",
                        self._model,
                        "--effort",
                        self._effort,
                        "-p",
                        full_prompt,
                        "--output-format",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env={**os.environ},
                )
                if result.returncode != 0:
                    # Sometimes JSON error is on stdout
                    err_blob = (result.stdout or result.stderr or "")[:400]
                    raise RuntimeError(f"agy exit {result.returncode}: {err_blob}")

                data = json.loads(result.stdout)
                if data.get("status") == "SUCCESS":
                    response = data.get("response", "") or ""
                    parsed = self._parse_json(response, schema)
                    # Retry if required text fields came back empty (parse fail
                    # or model refused to fill answer). Empty answer = free zero
                    # on BEAM; better to retry once than accept it.
                    answer = str(parsed.get("answer") or parsed.get("choice") or "").strip()
                    if not answer and attempt < _MAX_RETRIES - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    # Last resort: put raw response into answer so judge has text
                    if not answer and response.strip():
                        parsed["answer"] = response.strip()[:4000]
                        if "reasoning" in schema.required and not parsed.get("reasoning"):
                            parsed["reasoning"] = "raw_response_fallback"
                    return parsed

                raise RuntimeError(f"agy error: {data.get('error', 'unknown')}")

            except subprocess.TimeoutExpired as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise

        raise RuntimeError(f"agy failed after {_MAX_RETRIES} attempts: {last_err}")

    def _parse_json(self, text: str, schema: Schema) -> dict:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        def _ok(obj: dict) -> dict:
            # Coerce non-string answer to string (model sometimes returns list)
            if "answer" in obj and not isinstance(obj["answer"], str):
                obj["answer"] = json.dumps(obj["answer"], ensure_ascii=False)
            return obj

        try:
            return _ok(json.loads(text))
        except json.JSONDecodeError:
            pass

        # Greedy object match (handles nested braces better than naive regex)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return _ok(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        # Non-JSON prose: treat as the answer rather than empty fields
        out = {f: "" for f in schema.required}
        if "answer" in out:
            out["answer"] = text[:4000]
        if "reasoning" in out and text:
            out["reasoning"] = "non_json_response"
        if "choice" in out:
            m = re.search(r"\b([abcd])\b", text.lower())
            if m:
                out["choice"] = m.group(1)
        return out
