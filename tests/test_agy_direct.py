"""AgyDirectLLM unit tests — runs under both `python3 -m unittest discover -s tests` and `pytest`."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_bench.llm.agy_direct import (  # noqa: E402
    AgyDirectLLM,
    ensure_isolated_agy_env,
    resolve_agy_cli_model,
)
from memory_bench.llm.base import Schema  # noqa: E402


class AgyDirectTest(unittest.TestCase):
    def test_resolve_agy_cli_model_preserves_embedded_tiers_and_untiered_defaults(self):
        self.assertEqual(resolve_agy_cli_model("gemini-3.8-flash-high"), ("gemini-3.8-flash-high", "high"))
        self.assertEqual(resolve_agy_cli_model("gemini-4-hydrogen-xhigh"), ("gemini-4-hydrogen-xhigh", "xhigh"))
        self.assertEqual(resolve_agy_cli_model("gemini-3.8-flash", "medium"), ("gemini-3.8-flash-medium", "medium"))
        self.assertEqual(resolve_agy_cli_model("gemini-3.8-flash", "high"), ("gemini-3.8-flash-high", "high"))
        self.assertEqual(resolve_agy_cli_model("gpt-oss-120b-medium", "low"), ("gpt-oss-120b-medium", "medium"))
        self.assertEqual(resolve_agy_cli_model("claude-opus-4-6-thinking"), ("claude-opus-4-6-thinking", "high"))
        self.assertEqual(resolve_agy_cli_model("claude-sonnet-4-6"), ("claude-sonnet-4-6", "medium"))

    def test_ensure_isolated_agy_env_is_private_0700_and_preserves_refreshed_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            eval_home = root / "eval"
            src.mkdir()
            (src / "antigravity-oauth-token").write_text("oauth-old", encoding="utf-8")
            (src / "installation_id").write_text("inst-id", encoding="utf-8")
            (src / "jetski_state.pbtxt").write_text("should-not-symlink", encoding="utf-8")

            home_str, ws_str, settings_json = ensure_isolated_agy_env(str(eval_home), str(src))
            self.assertEqual(stat.S_IMODE(Path(home_str).stat().st_mode), 0o700)
            self.assertTrue(Path(ws_str, ".git").is_dir())
            tok_dst = Path(home_str, ".gemini/antigravity-cli/antigravity-oauth-token")
            self.assertTrue(tok_dst.is_symlink())
            self.assertEqual(tok_dst.read_text(encoding="utf-8"), "oauth-old")
            self.assertFalse(Path(home_str, ".gemini/antigravity-cli/jetski_state.pbtxt").exists())
            self.assertEqual(
                json.loads(settings_json),
                {"allowNonWorkspaceAccess": False, "permissions": {"allow": []}},
            )

            # Simulate agy refreshing token atomically into a newer regular file:
            tok_dst.unlink()
            tok_dst.write_text("oauth-refreshed", encoding="utf-8")
            ensure_isolated_agy_env(str(eval_home), str(src))
            self.assertFalse(tok_dst.is_symlink())
            self.assertEqual(tok_dst.read_text(encoding="utf-8"), "oauth-refreshed")

    def test_agy_direct_normalizes_array_schema_and_handles_non_dict_json(self):
        llm = AgyDirectLLM("gemini-3.8-flash-high")
        self.assertEqual(llm.model_id, "agy:gemini-3.8-flash-high")
        judge_schema = Schema(properties={"scores": {"type": "array"}}, required=["scores"])
        built = llm._build_json_schema(judge_schema)
        self.assertEqual(built["properties"]["scores"]["items"], {"type": "object"})

        ans_schema = Schema(properties={"answer": {"type": "string"}}, required=["answer"])
        self.assertEqual(llm._parse_json("null", ans_schema), {"answer": "null"})
        self.assertEqual(llm._parse_json('["x"]', ans_schema), {"answer": '["x"]'})

        mcq_schema = Schema(
            properties={"reasoning": {"type": "string"}, "choice": {"type": "string"}},
            required=["reasoning", "choice"],
        )
        self.assertEqual(llm._parse_json("The correct option is B.", mcq_schema)["choice"], "b")


if __name__ == "__main__":
    unittest.main()
