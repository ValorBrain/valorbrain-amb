"""A strict served-model mismatch must fail the run, never score 0 and carry on.

BEAM scoring falls back to per-item scoring, and the runner retries transient judge
errors; ServedModelMismatch has to cut through both, with no retry.

These tests import the full runner (memory providers, datasets), so they need the
project environment: `uv run --with pytest pytest tests/`. They are skipped under the
standard-library-only CI runner.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from memory_bench.dataset.beam import BEAMDataset
    from memory_bench.llm import gateway
    from memory_bench.llm.gateway import GatewayLLM, ServedModelMismatch
    from memory_bench.memory.base import MemoryProvider
    from memory_bench.models import AnswerResult, Document, Query, QueryResult
    from memory_bench.modes.base import ResponseMode
    from memory_bench.runner import EvalRunner
except ImportError as e:  # pragma: no cover - depends on the environment
    _IMPORT_ERROR: ImportError | None = e
else:
    _IMPORT_ERROR = None

MISMATCH = ("strict model check: requested deepseek-v4.1-flash "
            "but gateway served longcat-2.5-preview-free")


def _substituting_urlopen(calls: list):
    """urlopen stand-in for a gateway that serves LongCat whatever is asked."""
    class _Response:
        def __init__(self):
            self.headers = Message()
            self.headers["x-valorbrain-gateway-model"] = "longcat-2.5-preview-free"

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"scores": []}'},
                                            "finish_reason": "stop"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=None):
        calls.append(req)
        return _Response()

    return urlopen


@unittest.skipIf(_IMPORT_ERROR is not None, f"project dependencies not installed: {_IMPORT_ERROR}")
class BeamScoreResultTest(unittest.TestCase):
    def _result(self, answer="The meeting moved to Friday.") -> "QueryResult":
        return QueryResult(
            query_id="q1", query="When is the meeting?", answer=answer, reasoning="",
            context="ctx", context_tokens=0, retrieve_time_ms=0.0, gold_answers=["Friday"],
            correct=False, judge_reason="",
            meta={"question_category": "information_extraction",
                  "rubric": ["mentions Friday", "mentions the move"]},
        )

    def test_score_result_propagates_mismatch_without_fallback(self):
        llm = mock.Mock()
        llm.generate.side_effect = ServedModelMismatch(MISMATCH)
        with self.assertRaisesRegex(ServedModelMismatch, "longcat-2.5-preview-free"):
            BEAMDataset().score_result(self._result(), llm)
        # The batch call raised; the per-item fallback never ran.
        self.assertEqual(llm.generate.call_count, 1)

    def test_rubric_item_and_equivalence_propagate(self):
        llm = mock.Mock()
        llm.generate.side_effect = ServedModelMismatch(MISMATCH)
        with self.assertRaises(ServedModelMismatch):
            BEAMDataset._rubric_item_score("q", "a", "item", llm)
        with self.assertRaises(ServedModelMismatch):
            BEAMDataset._llm_equivalence("a", "b", llm)

    def test_other_judge_errors_are_never_a_zero(self):
        # FB-0220 (ValorBrain): a judge that answers 429 used to score the item 0.0 and the run
        # ended normally. The error now reaches the runner (it retries transient ones).
        llm = mock.Mock()
        llm.generate.side_effect = RuntimeError("GatewayLLM failed after 4 retries: HTTP Error 429: Too Many Requests")
        with self.assertRaisesRegex(RuntimeError, "429"):
            BEAMDataset().score_result(self._result(), llm)
        # Batch call, then the per-item fallback's first call, which raises.
        self.assertEqual(llm.generate.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, "429"):
            BEAMDataset._rubric_item_score("q", "a", "item", llm)
        with self.assertRaisesRegex(RuntimeError, "429"):
            BEAMDataset._llm_equivalence("a", "b", llm)

    def test_a_judge_that_answers_still_scores(self):
        llm = mock.Mock()
        llm.generate.side_effect = [RuntimeError("batch prompt failed"), {"score": 1.0, "reason": "ok"},
                                    {"score": 0.5, "reason": "partial"}]
        self.assertEqual(BEAMDataset().score_result(self._result(), llm), 0.75)


if _IMPORT_ERROR is None:
    class _FakeMemory(MemoryProvider):
        name = "fake"

        def ingest(self, documents):
            pass

        def retrieve(self, query, k=10, user_id=None, query_timestamp=None):
            return [], None

    class _FakeMode(ResponseMode):
        name = "fake"
        description = "fixed answer"

        def answer(self, query, memory, task_type="open", user_id=None):
            return AnswerResult(answer="Friday", reasoning="", context="ctx", retrieve_time_ms=0.0)

        def answer_from_context(self, query, context, task_type="open"):
            return self.answer(query, None)


@unittest.skipIf(_IMPORT_ERROR is not None, f"project dependencies not installed: {_IMPORT_ERROR}")
class RunnerStrictJudgeTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {
            "OMB_JUDGE_LLM": "gateway",
            "OMB_JUDGE_MODEL": "deepseek-v4.1-flash",
            "OMB_JUDGE_BASE_URL": "http://127.0.0.1:1/v1",
            "OMB_JUDGE_STRICT_MODEL": "1",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_substituted_judge_fails_the_run(self):
        dataset = BEAMDataset()
        queries = [
            Query(id=f"q{i}", query="When is the meeting?", gold_ids=[], gold_answers=["Friday"],
                  user_id="conv-1",
                  meta={"question_category": "information_extraction", "rubric": ["mentions Friday"]})
            for i in range(3)
        ]
        docs = [Document(id="d1", content="The meeting is on Friday.", user_id="conv-1")]
        calls: list = []
        runner = EvalRunner(output_dir=Path(self.tmp.name))
        self.assertIsInstance(runner._get_judge(dataset)._llm, GatewayLLM)
        with mock.patch.object(dataset, "load_queries", return_value=queries), \
                mock.patch.object(dataset, "load_documents", return_value=docs), \
                mock.patch.object(gateway.urllib.request, "urlopen", _substituting_urlopen(calls)), \
                mock.patch.object(gateway.time, "sleep") as gateway_sleep, \
                self.assertRaisesRegex(ServedModelMismatch, "longcat-2.5-preview-free"):
            runner.run(dataset, "100K", _FakeMemory(), _FakeMode(), run_name="strict-test")
        # No judge retries, no per-item fallback, no re-score: at most one judge call per
        # query that was in flight when the first mismatch surfaced.
        self.assertLessEqual(len(calls), len(queries))
        gateway_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
