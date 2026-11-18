"""
AGY direct LLM — calls Antigravity CLI (`agy` 1.5+) via subprocess in a private
(`0700`) hermetic HOME/workspace (no host MCP servers, no global rules, no tools)
with inline `--settings`, `--print-timeout`, process-group cleanup, and native
`--json-schema` structured output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .base import LLM, Schema

_MAX_RETRIES = 4
_AUTH_FILES = ("antigravity-oauth-token", "installation_id")
_TIERED_FAMILIES: dict[str, tuple[str, ...]] = {
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-4-argon": ("low", "medium", "high"),
    "gemini-4-carbon": ("low", "medium", "high"),
    "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
}
_UNTIERED_MODELS: dict[str, dict[str, str]] = {
    "claude-sonnet-4-6": {"default": "medium"},
    "claude-opus-4-6-thinking": {"default": "high"},
    "gpt-oss-120b-medium": {"fixed": "medium", "default": "medium"},
    "gemini-pro-orchestrator": {"default": "high"},
    "gemini-pro-orchestrator-argon": {"default": "high"},
}
_HERMETIC_SETTINGS_JSON = json.dumps(
    {"allowNonWorkspaceAccess": False, "permissions": {"allow": []}},
    separators=(",", ":"),
)
_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "quota",
    "500",
    "502",
    "503",
    "504",
    "529",
    "overloaded",
    "unavailable",
    "timed out",
    "timeout",
    "connection reset",
    "eof",
)


def resolve_agy_cli_model(raw_model: str | None, default_effort: str | None = None) -> tuple[str, str]:
    """Resolve model + effort for agy 1.5+, preserving embedded tier suffixes."""
    explicit_eff = (default_effort or "").strip().lower()
    if explicit_eff not in {"low", "medium", "high", "xhigh", "max"}:
        explicit_eff = ""
    fallback_eff = explicit_eff or "low"

    candidate = (raw_model or "gemini-3.8-flash").strip()
    for prefix in ("agy:", "agy/"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]

    if candidate in _UNTIERED_MODELS:
        meta = _UNTIERED_MODELS[candidate]
        eff = meta.get("fixed") or explicit_eff or meta["default"]
        return candidate, eff

    for family, tiers in _TIERED_FAMILIES.items():
        for tier in tiers:
            full_id = f"{family}-{tier}"
            if candidate == full_id:
                return full_id, tier
        if candidate == family:
            clamped = (
                fallback_eff
                if fallback_eff in tiers
                else ("xhigh" if "xhigh" in tiers and fallback_eff == "max" else ("high" if fallback_eff in {"high", "xhigh", "max"} else tiers[0]))
            )
            return f"{family}-{clamped}", clamped

    m = re.search(r"-(low|medium|high|xhigh)$", candidate, re.IGNORECASE)
    if m:
        return candidate, m.group(1).lower()

    return candidate, fallback_eff


def _ensure_owned_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = os.lstat(path)
    if os.path.islink(path) or not Path(path).is_dir():
        raise RuntimeError(f"Refusing unsafe AGY eval path (not a real directory): {path}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise RuntimeError(f"Refusing AGY eval directory owned by uid={st.st_uid} (expected {os.getuid()}): {path}")
    os.chmod(path, 0o700)


def _ensure_atomic_auth_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    try:
        dst_st = os.lstat(dst)
        if os.path.islink(dst):
            if os.readlink(dst) == str(src):
                return
        else:
            # Never overwrite a refreshed regular token file that is newer than src.
            if dst_st.st_mtime >= src.stat().st_mtime:
                return
    except FileNotFoundError:
        pass

    tmp_link = dst.with_name(f"{dst.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        tmp_link.symlink_to(src)
        os.replace(tmp_link, dst)
    finally:
        if tmp_link.is_symlink() or tmp_link.exists():
            try:
                tmp_link.unlink()
            except FileNotFoundError:
                pass


def ensure_isolated_agy_env(
    eval_home: str | None = None,
    auth_source_dir: str | None = None,
) -> tuple[str, str, str]:
    """Create a private (0700) hermetic HOME + empty .git workspace outside /tmp."""
    default_home = Path.home() / ".cache" / "valorbrain-agy-eval-home"
    default_src = Path.home() / ".gemini" / "antigravity-cli"
    home = Path(eval_home or os.environ.get("AGY_EVAL_HOME") or default_home)
    src_dir = Path(auth_source_dir or os.environ.get("AGY_AUTH_SOURCE_DIR") or default_src)

    _ensure_owned_private_dir(home)
    gemini_dir = home / ".gemini"
    _ensure_owned_private_dir(gemini_dir)
    cli_dir = gemini_dir / "antigravity-cli"
    _ensure_owned_private_dir(cli_dir)
    ws_dir = home / "empty-ws"
    _ensure_owned_private_dir(ws_dir)
    (ws_dir / ".git").mkdir(parents=True, exist_ok=True, mode=0o700)

    for name in _AUTH_FILES:
        _ensure_atomic_auth_symlink(src_dir / name, cli_dir / name)

    return str(home), str(ws_dir), _HERMETIC_SETTINGS_JSON


def normalize_json_schema(schema: Schema) -> dict:
    """Build a valid JSON Schema object, ensuring array properties declare `items`."""
    props: dict[str, dict] = {}
    for key, spec in (schema.properties or {}).items():
        prop = dict(spec) if isinstance(spec, dict) else {"type": "string"}
        if prop.get("type") == "array" and "items" not in prop:
            prop["items"] = {"type": "object"}
        props[key] = prop
    return {
        "type": "object",
        "properties": props,
        "required": list(schema.required or []),
    }


def _is_retryable_error(msg: str) -> bool:
    lower = msg.lower()
    return any(marker in lower for marker in _TRANSIENT_MARKERS)


class AgyDirectLLM(LLM):
    """Calls `agy` CLI 1.5+ directly in a hermetic no-MCP environment with `--json-schema`."""

    def __init__(self, model: str | None = None):
        raw_effort = os.environ.get("AGY_EFFORT")
        raw_model = model or os.environ.get("AGY_MODEL", "gemini-3.8-flash")
        self._model, self._effort = resolve_agy_cli_model(raw_model, raw_effort)
        base_timeout = int(os.environ.get("AGY_TIMEOUT", "180"))
        self._timeouts = [base_timeout + 30 * i for i in range(_MAX_RETRIES)]
        self._use_json_schema = os.environ.get("AGY_USE_JSON_SCHEMA", "1") != "0"

    @property
    def model_id(self) -> str:
        return f"agy:{self._model}"

    def _build_json_schema(self, schema: Schema) -> dict:
        return normalize_json_schema(schema)

    def generate(self, prompt: str, schema: Schema) -> dict:
        fields_desc = ", ".join(schema.required)
        full_prompt = (
            "Answer strictly from the provided context. Do NOT use tools, "
            "read files, or run commands — respond directly.\n\n"
            + prompt
            + f"\n\nRespond as a JSON object with these fields: {fields_desc}. "
            "Output ONLY valid JSON. The 'answer' field (if requested) must be a non-empty string "
            "with your final answer (or a clear abstention if context lacks info)."
        )

        home_dir, ws_dir, settings_json = ensure_isolated_agy_env()
        child_env = {
            "PATH": os.environ.get("PATH", "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"),
            "HOME": home_dir,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": "dumb",
        }
        use_schema = self._use_json_schema and bool(schema.properties)

        last_err: Exception | None = None
        backoff = 5.0
        for attempt in range(_MAX_RETRIES):
            timeout_s = self._timeouts[attempt]
            cmd = [
                "agy",
                "--model",
                self._model,
                "--effort",
                self._effort,
                "--disable-slash-commands",
                "--print-timeout",
                f"{timeout_s}s",
                "--settings",
                settings_json,
                "--output-format",
                "json",
            ]
            if use_schema:
                cmd.extend(["--json-schema", json.dumps(self._build_json_schema(schema))])
            cmd.extend(["-p", full_prompt])

            proc = subprocess.Popen(
                cmd,
                cwd=ws_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s + 5)
            except subprocess.TimeoutExpired as e:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            try:
                if proc.returncode != 0:
                    err_blob = (stdout or stderr or "")[:400]
                    err_low = err_blob.lower()
                    if use_schema and any(k in err_low for k in ("json_schema", "json-schema", "invalid_argument")):
                        use_schema = False
                        continue
                    raise RuntimeError(f"agy exit {proc.returncode}: {err_blob}")

                data = json.loads(stdout)
                if data.get("status") == "SUCCESS":
                    structured = data.get("structured_output")
                    response = data.get("response", "") or ""
                    if isinstance(structured, dict):
                        parsed = self._coerce_fields(dict(structured))
                    else:
                        parsed = self._parse_json(response, schema)

                    if "choice" in schema.required:
                        choice = str(parsed.get("choice") or "").strip()
                        if not choice and response.strip():
                            m = re.search(r"\b([abcd])\b", response.lower())
                            if m:
                                parsed["choice"] = m.group(1)
                                choice = m.group(1)
                        if not choice and attempt < _MAX_RETRIES - 1:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                    elif "answer" in schema.required:
                        answer = str(parsed.get("answer") or "").strip()
                        if not answer and attempt < _MAX_RETRIES - 1:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                        if not answer and response.strip():
                            parsed["answer"] = response.strip()[:4000]
                            if "reasoning" in schema.required and not parsed.get("reasoning"):
                                parsed["reasoning"] = "raw_response_fallback"
                    return parsed

                err_msg = str(data.get("error", "unknown"))
                if use_schema and any(k in err_msg.lower() for k in ("json_schema", "json-schema", "invalid_argument")):
                    use_schema = False
                    continue
                raise RuntimeError(f"agy error: {err_msg}")

            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1 and _is_retryable_error(str(e)):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

        raise RuntimeError(f"agy failed after {_MAX_RETRIES} attempts: {last_err}")

    def _coerce_fields(self, obj: dict) -> dict:
        if "answer" in obj and obj["answer"] is not None and not isinstance(obj["answer"], str):
            obj["answer"] = json.dumps(obj["answer"], ensure_ascii=False)
        return obj

    def _parse_json(self, text: str, schema: Schema) -> dict:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return self._coerce_fields(loaded)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                loaded = json.loads(match.group())
                if isinstance(loaded, dict):
                    return self._coerce_fields(loaded)
            except json.JSONDecodeError:
                pass

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
