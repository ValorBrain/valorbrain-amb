import json
from pathlib import Path
import stat
import tempfile

from memory_bench.llm.agy_direct import (
    AgyDirectLLM,
    ensure_isolated_agy_env,
    resolve_agy_cli_model,
)
from memory_bench.llm.base import Schema


def test_resolve_agy_cli_model_preserves_embedded_tiers_and_untiered_defaults():
    assert resolve_agy_cli_model("gemini-3.8-flash-high") == ("gemini-3.8-flash-high", "high")
    assert resolve_agy_cli_model("gemini-4-hydrogen-xhigh") == ("gemini-4-hydrogen-xhigh", "xhigh")
    assert resolve_agy_cli_model("gemini-3.8-flash", "medium") == ("gemini-3.8-flash-medium", "medium")
    assert resolve_agy_cli_model("gemini-3.8-flash", "high") == ("gemini-3.8-flash-high", "high")
    assert resolve_agy_cli_model("gpt-oss-120b-medium", "low") == ("gpt-oss-120b-medium", "medium")
    assert resolve_agy_cli_model("claude-opus-4-6-thinking") == ("claude-opus-4-6-thinking", "high")
    assert resolve_agy_cli_model("claude-sonnet-4-6") == ("claude-sonnet-4-6", "medium")


def test_ensure_isolated_agy_env_is_private_0700_and_preserves_refreshed_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        eval_home = root / "eval"
        src.mkdir()
        (src / "antigravity-oauth-token").write_text("oauth-old", encoding="utf-8")
        (src / "installation_id").write_text("inst-id", encoding="utf-8")
        (src / "jetski_state.pbtxt").write_text("should-not-symlink", encoding="utf-8")

        home_str, ws_str, settings_json = ensure_isolated_agy_env(str(eval_home), str(src))
        assert stat.S_IMODE(Path(home_str).stat().st_mode) == 0o700
        assert Path(ws_str, ".git").is_dir()
        tok_dst = Path(home_str, ".gemini/antigravity-cli/antigravity-oauth-token")
        assert tok_dst.is_symlink()
        assert tok_dst.read_text(encoding="utf-8") == "oauth-old"
        assert not Path(home_str, ".gemini/antigravity-cli/jetski_state.pbtxt").exists()
        assert json.loads(settings_json) == {"allowNonWorkspaceAccess": False, "permissions": {"allow": []}}

        # Simulate agy refreshing token atomically into a newer regular file:
        tok_dst.unlink()
        tok_dst.write_text("oauth-refreshed", encoding="utf-8")
        ensure_isolated_agy_env(str(eval_home), str(src))
        assert not tok_dst.is_symlink()
        assert tok_dst.read_text(encoding="utf-8") == "oauth-refreshed"


def test_agy_direct_normalizes_array_schema_and_handles_non_dict_json():
    llm = AgyDirectLLM("gemini-3.8-flash-high")
    assert llm.model_id == "agy:gemini-3.8-flash-high"
    judge_schema = Schema(properties={"scores": {"type": "array"}}, required=["scores"])
    built = llm._build_json_schema(judge_schema)
    assert built["properties"]["scores"]["items"] == {"type": "object"}

    ans_schema = Schema(properties={"answer": {"type": "string"}}, required=["answer"])
    assert llm._parse_json("null", ans_schema) == {"answer": "null"}
    assert llm._parse_json('["x"]', ans_schema) == {"answer": '["x"]'}

    mcq_schema = Schema(properties={"reasoning": {"type": "string"}, "choice": {"type": "string"}}, required=["reasoning", "choice"])
    assert llm._parse_json("The correct option is B.", mcq_schema)["choice"] == "b"
