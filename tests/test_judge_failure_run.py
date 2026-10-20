"""A judge that keeps failing fails the run; one that recovers scores for real (FB-0220).

ValorBrain's BEAM gate measured 80 rubric items scored 0 in one batch of parallel runs: the ZAI
judge answered HTTP 429, `_rubric_item_score` turned the error into 0.0 and the run ended
normally. The error now reaches the runner, which retries transient errors (429 and friends) and
lets anything left fail the run. A recorded 0 only ever means the judge said 0.

Like test_strict_judge_run.py, this imports the full runner: `uv run --with pytest pytest tests/`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from memory_bench import runner as runner_module
    from memory_bench.dataset.beam import BEAMDataset
    from memory_bench.llm import gateway
    from memory_bench.memory.base import MemoryProvider
    from memory_bench.models import AnswerResult, Document, Query
    from memory_bench.modes.base import ResponseMode
    from memory_bench.runner import EvalRunner
except ImportError as e:  # pragma: no cover - depends on the environment
    _IMPORT_ERROR: ImportError | None = e
else:
    _IMPORT_ERROR = None


def _judge_urlopen(calls: list, failures: int):
    """A judge gateway that answers 429 to the first `failures` calls, then scores 1.0."""
    class _Response:
        def __init__(self):
            self.headers = Message()

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"scores": [{"score": 1.0}]}'},
                                            "finish_reason": "stop"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) <= failures:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", Message(), None)
        return _Response()

    return urlopen


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


async def _no_wait(*_args, **_kwargs):
    return None


@unittest.skipIf(_IMPORT_ERROR is not None, f"project dependencies not installed: {_IMPORT_ERROR}")
class RunnerJudgeFailureTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {
            "OMB_JUDGE_LLM": "gateway",
            "OMB_JUDGE_MODEL": "glm-5.3-flash",
            "OMB_JUDGE_BASE_URL": "http://127.0.0.1:1/v1",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dataset = BEAMDataset()
        self.queries = [Query(id="q1", query="When is the meeting?", gold_ids=[], gold_answers=["Friday"],
                              user_id="conv-1",
                              meta={"question_category": "information_extraction", "rubric": ["mentions Friday"]})]
        self.docs = [Document(id="d1", content="The meeting is on Friday.", user_id="conv-1")]

    def _run(self, failures: int, calls: list):
        runner = EvalRunner(output_dir=Path(self.tmp.name))
        with mock.patch.object(self.dataset, "load_queries", return_value=self.queries), \
                mock.patch.object(self.dataset, "load_documents", return_value=self.docs), \
                mock.patch.object(gateway.urllib.request, "urlopen", _judge_urlopen(calls, failures)), \
                mock.patch.object(gateway.time, "sleep"), \
                mock.patch.object(runner_module.asyncio, "sleep", _no_wait), \
                mock.patch("time.sleep"):
            return runner.run(self.dataset, "100K", _FakeMemory(), _FakeMode(), run_name="judge-failure-test")

    def test_a_judge_that_keeps_answering_429_fails_the_run(self):
        calls: list = []
        with self.assertRaisesRegex(Exception, "429"):
            self._run(failures=10_000, calls=calls)
        self.assertGreater(len(calls), 1)

    def test_a_judge_that_recovers_scores_for_real(self):
        # The first query attempt exhausts the gateway's retries on the batch call (4) and on
        # the per-item fallback (4); the runner retries the query and the judge answers.
        calls: list = []
        result = self._run(failures=8, calls=calls)
        rows = result.results if hasattr(result, "results") else result["results"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        score_text = getattr(row, "judge_reason", None) if not isinstance(row, dict) else row.get("judge_reason")
        self.assertEqual(score_text, "score=1.000")


if __name__ == "__main__":
    unittest.main()
