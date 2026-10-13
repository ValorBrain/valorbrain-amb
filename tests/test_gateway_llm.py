"""GatewayLLM: per-role extra body and strict served-model check, with HTTP mocked.

Run with `python3 -m unittest discover -s tests -v` or `uv run pytest tests/`.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_bench.llm import get_answer_llm, get_judge_llm  # noqa: E402
from memory_bench.llm import gateway  # noqa: E402
from memory_bench.llm.base import Schema  # noqa: E402
from memory_bench.llm.gateway import GatewayLLM, normalize_model_name  # noqa: E402

SCHEMA = Schema(properties={"answer": {"type": "string"}}, required=["answer"])
READER_EXTRA = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}


class _Response:
    def __init__(self, body: dict, headers: dict[str, str]):
        self._body = json.dumps(body).encode("utf-8")
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _completion(content: str = '{"answer": "42"}', model: str | None = None,
                finish_reason: str = "stop") -> dict:
    body = {"choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}]}
    if model is not None:
        body["model"] = model
    return body


class _FakeHTTP:
    """Records request payloads and answers each call with the same response."""

    def __init__(self, body: dict, headers: dict[str, str] | None = None):
        self.body = body
        self.headers = headers or {}
        self.payloads: list[dict] = []

    def __call__(self, req, timeout=None):
        self.payloads.append(json.loads(req.data.decode("utf-8")))
        return _Response(self.body, self.headers)


class GatewayTestCase(unittest.TestCase):
    def setUp(self):
        # Keep the host env (gate env files, OMB_GATEWAY_*) out of these tests.
        self._env = mock.patch.dict(os.environ, {}, clear=True)
        self._env.start()
        self.addCleanup(self._env.stop)
        sleep = mock.patch.object(gateway.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def run_generate(self, llm: GatewayLLM, fake: _FakeHTTP) -> dict:
        with mock.patch.object(gateway.urllib.request, "urlopen", fake):
            return llm.generate("question", SCHEMA)


class ExtraBodyTest(GatewayTestCase):
    def test_extra_body_merged_last(self):
        fake = _FakeHTTP(_completion())
        llm = GatewayLLM(
            "anthropic/claude-haiku-5-5",
            reasoning_effort="high",
            extra_body={**READER_EXTRA, "max_tokens": 1024, "reasoning_effort": "low"},
        )
        self.assertEqual(self.run_generate(llm, fake), {"answer": "42"})
        payload = fake.payloads[0]
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "low"})
        # Applied after the built-in fields, so it wins over them.
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["model"], "anthropic/claude-haiku-5-5")
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_model_and_messages_keys_rejected(self):
        for key in ("model", "messages"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                GatewayLLM("m", extra_body={key: "x"})

    def test_env_extra_body_reaches_each_role(self):
        os.environ.update({
            "OMB_ANSWER_LLM": "gateway", "OMB_ANSWER_MODEL": "anthropic/claude-haiku-5-5",
            "OMB_ANSWER_EXTRA_BODY": json.dumps(READER_EXTRA),
            "OMB_JUDGE_LLM": "gateway", "OMB_JUDGE_MODEL": "glm-5.3-flash",
            "OMB_JUDGE_EXTRA_BODY": '{"top_p": 0.5}',
        })
        answer, judge = get_answer_llm(), get_judge_llm()
        fake = _FakeHTTP(_completion())
        self.run_generate(answer, fake)
        self.run_generate(judge, fake)
        self.assertEqual(fake.payloads[0]["thinking"], {"type": "adaptive"})
        self.assertNotIn("top_p", fake.payloads[0])
        self.assertEqual(fake.payloads[1]["top_p"], 0.5)
        self.assertNotIn("thinking", fake.payloads[1])

    def test_env_reserved_key_rejected_at_construction(self):
        os.environ.update({"OMB_ANSWER_LLM": "gateway",
                           "OMB_ANSWER_EXTRA_BODY": '{"model": "other"}'})
        with self.assertRaisesRegex(ValueError, "model"):
            get_answer_llm()

    def test_env_invalid_json_raises(self):
        os.environ.update({"OMB_JUDGE_LLM": "gateway", "OMB_JUDGE_EXTRA_BODY": "{thinking:"})
        with self.assertRaisesRegex(ValueError, "OMB_JUDGE_EXTRA_BODY is not valid JSON"):
            get_judge_llm()

    def test_env_non_object_json_raises(self):
        os.environ.update({"OMB_ANSWER_LLM": "gateway", "OMB_ANSWER_EXTRA_BODY": "[1, 2]"})
        with self.assertRaisesRegex(ValueError, "OMB_ANSWER_EXTRA_BODY must be a JSON object"):
            get_answer_llm()


class StrictModelTest(GatewayTestCase):
    def test_mismatch_fails_after_retries_naming_both(self):
        fake = _FakeHTTP(_completion(model="deepseek-v4.1-flash"),
                         {"x-valorbrain-gateway-model": "longcat-2.5-preview-free"})
        llm = GatewayLLM("deepseek-v4.1-flash", strict_model=True)
        with self.assertRaises(RuntimeError) as ctx:
            self.run_generate(llm, fake)
        message = str(ctx.exception)
        self.assertIn(f"failed after {gateway._MAX_RETRIES} retries", message)
        self.assertIn("deepseek-v4.1-flash", message)
        self.assertIn("longcat-2.5-preview-free", message)
        self.assertEqual(len(fake.payloads), gateway._MAX_RETRIES)

    def test_served_model_source_priority(self):
        cases = [
            # (headers, json model, served) -> the first present source wins
            ({"x-valorbrain-gateway-model": "a", "x-manifest-model": "b"}, "c", "a"),
            ({"x-manifest-model": "b"}, "c", "b"),
            ({}, "c", "c"),
        ]
        for headers, body_model, served in cases:
            with self.subTest(served=served):
                fake = _FakeHTTP(_completion(model=body_model), headers)
                llm = GatewayLLM("requested", strict_model=True)
                with self.assertRaisesRegex(RuntimeError, f"requested requested but gateway served {served}"):
                    self.run_generate(llm, fake)

    def test_missing_served_model_fails(self):
        fake = _FakeHTTP(_completion())
        llm = GatewayLLM("m", strict_model=True)
        with self.assertRaisesRegex(RuntimeError, "did not report the served model"):
            self.run_generate(llm, fake)

    def test_normalized_names_match(self):
        cases = [
            ("anthropic/claude-haiku-5-5", {"x-manifest-model": "claude-haiku-5-5"}, None),
            ("claude-haiku-5-5", {}, "anthropic/claude-haiku-5-5"),
            ("anthropic/claude-haiku-5-5", {"x-manifest-model": "claude-haiku-5-5-subscription"}, None),
            ("glm-5.3-flash", {}, "zai/glm-5.3-flash"),
        ]
        for requested, headers, body_model in cases:
            with self.subTest(requested=requested, headers=headers, body_model=body_model):
                fake = _FakeHTTP(_completion(model=body_model), headers)
                llm = GatewayLLM(requested, strict_model=True)
                self.assertEqual(self.run_generate(llm, fake), {"answer": "42"})
                self.assertEqual(len(fake.payloads), 1)

    def test_normalize_model_name(self):
        self.assertEqual(normalize_model_name("anthropic/claude-haiku-5-5"), "claude-haiku-5-5")
        self.assertEqual(normalize_model_name("claude-haiku-5-5-subscription"), "claude-haiku-5-5")
        self.assertEqual(normalize_model_name("a/b/glm-5.3-flash"), "glm-5.3-flash")
        self.assertNotEqual(normalize_model_name("deepseek-v4.1-flash"),
                            normalize_model_name("longcat-2.5-preview-free"))

    def test_env_strict_flag(self):
        os.environ.update({"OMB_JUDGE_LLM": "gateway", "OMB_JUDGE_MODEL": "deepseek-v4.1-flash",
                           "OMB_JUDGE_STRICT_MODEL": "1"})
        fake = _FakeHTTP(_completion(model="longcat-2.5-preview-free"))
        with self.assertRaisesRegex(RuntimeError, "longcat-2.5-preview-free"):
            self.run_generate(get_judge_llm(), fake)

    def test_env_strict_flag_rejects_unknown_value(self):
        os.environ.update({"OMB_ANSWER_LLM": "gateway", "OMB_ANSWER_STRICT_MODEL": "true"})
        with self.assertRaisesRegex(ValueError, "OMB_ANSWER_STRICT_MODEL must be 0 or 1"):
            get_answer_llm()


class DefaultOffTest(GatewayTestCase):
    def test_default_payload_and_no_strict(self):
        os.environ.update({"OMB_ANSWER_LLM": "gateway", "OMB_ANSWER_MODEL": "deepseek-v4.1-flash"})
        llm = get_answer_llm()
        # A substituted model passes unchecked, exactly as before.
        fake = _FakeHTTP(_completion(model="longcat-2.5-preview-free"),
                         {"x-valorbrain-gateway-model": "longcat-2.5-preview-free"})
        self.assertEqual(self.run_generate(llm, fake), {"answer": "42"})
        self.assertEqual(
            fake.payloads[0],
            {"model": "deepseek-v4.1-flash",
             "messages": fake.payloads[0]["messages"],
             "max_tokens": 16384,
             "temperature": 0},
        )

    def test_served_model_logged_at_debug(self):
        fake = _FakeHTTP(_completion(), {"x-manifest-model": "claude-haiku-5-5"})
        with self.assertLogs(gateway.logger, "DEBUG") as logs:
            self.run_generate(GatewayLLM("anthropic/claude-haiku-5-5"), fake)
        self.assertIn("served=claude-haiku-5-5", logs.output[0])
        self.assertTrue(logs.output[0].startswith("DEBUG:"))


class ContentFilterTest(GatewayTestCase):
    def test_refusal_warns_and_answer_stays_empty(self):
        fake = _FakeHTTP(_completion(content=None, finish_reason="content_filter"),
                         {"x-manifest-model": "claude-sonnet-5-5"})
        with self.assertLogs(gateway.logger, "WARNING") as logs:
            result = self.run_generate(GatewayLLM("anthropic/claude-sonnet-5-5"), fake)
        self.assertEqual(result["answer"], "")
        self.assertEqual(len(fake.payloads), 1)
        self.assertIn("content_filter", logs.output[0])
        self.assertIn("claude-sonnet-5-5", logs.output[0])


if __name__ == "__main__":
    unittest.main()
