"""A judge that keeps failing fails the run; one that recovers scores for real (FB-0220).

ValorBrain's BEAM gate measured 80 rubric items scored 0 in one batch of parallel runs: the ZAI
judge answered HTTP 429, `_rubric_item_score` turned the error into 0.0 and the run ended
normally. The error now reaches the runner, which retries transient errors (429 and friends) and
lets anything left fail the run. A recorded 0 only ever means the judge said 0.

Second review: a batch answer that does not grade every rubric item (no `scores`, fewer or more
scores than items, an item without a numeric score) goes to the per-item path instead of being
padded with 0 or cut; a per-item answer without `score` and an equivalence answer without
`answer` raise; a transient judge failure repeats only the judge, never the reader; a genuine 0 is
not re-scored; a timeout is transient and a parse error that mentions "char 429" is not.

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
    from memory_bench.models import AnswerResult, Document, Query, QueryResult
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


def _scripted_urlopen(calls: list, reply):
    """A judge gateway whose n-th call (1-based) runs `reply(n)`: it raises, or returns the reply.

    A str reply is the model's message content; bytes are the raw HTTP body.
    """
    class _Response:
        def __init__(self, body: bytes):
            self.headers = Message()
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout=None):
        calls.append(req)
        content = reply(len(calls))
        if isinstance(content, bytes):
            return _Response(content)
        return _Response(json.dumps({"choices": [{"message": {"content": content},
                                                  "finish_reason": "stop"}]}).encode())

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

        def __init__(self):
            self.answers = 0

        def answer(self, query, memory, task_type="open", user_id=None):
            self.answers += 1
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

    def _run(self, failures: int, calls: list, mode=None, urlopen=None):
        runner = EvalRunner(output_dir=Path(self.tmp.name))
        with mock.patch.object(self.dataset, "load_queries", return_value=self.queries), \
                mock.patch.object(self.dataset, "load_documents", return_value=self.docs), \
                mock.patch.object(gateway.urllib.request, "urlopen", urlopen or _judge_urlopen(calls, failures)), \
                mock.patch.object(gateway.time, "sleep"), \
                mock.patch.object(runner_module.asyncio, "sleep", _no_wait), \
                mock.patch("time.sleep"):
            return runner.run(self.dataset, "100K", _FakeMemory(), mode or _FakeMode(), run_name="judge-failure-test")

    @staticmethod
    def _score_text(result):
        rows = result.results if hasattr(result, "results") else result["results"]
        assert len(rows) == 1, rows
        row = rows[0]
        return getattr(row, "judge_reason", None) if not isinstance(row, dict) else row.get("judge_reason")

    def test_a_judge_that_keeps_answering_429_fails_the_run(self):
        calls: list = []
        with self.assertRaisesRegex(Exception, "429"):
            self._run(failures=10_000, calls=calls)
        self.assertGreater(len(calls), 1)

    def test_a_judge_that_recovers_scores_for_real(self):
        # The first judge call exhausts the gateway's retries on the batch call (4) and on the
        # per-item fallback (4); the runner repeats the judge and it answers.
        calls: list = []
        result = self._run(failures=8, calls=calls)
        rows = result.results if hasattr(result, "results") else result["results"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        score_text = getattr(row, "judge_reason", None) if not isinstance(row, dict) else row.get("judge_reason")
        self.assertEqual(score_text, "score=1.000")


    def test_a_transient_judge_failure_keeps_the_reader_answer(self):
        # Only the judge failed: the runner repeats the judge call, not retrieval and the reader.
        calls: list = []
        mode = _FakeMode()
        result = self._run(failures=8, calls=calls, mode=mode)
        self.assertEqual(self._score_text(result), "score=1.000")
        self.assertEqual(mode.answers, 1)

    def test_a_genuine_zero_stays_zero(self):
        # The judge said 0: that is the score, with no second draw.
        calls: list = []
        result = self._run(failures=0, calls=calls,
                           urlopen=_scripted_urlopen(calls, lambda n: '{"scores": [{"score": 0.0, "reason": "wrong"}]}'))
        self.assertEqual(self._score_text(result), "score=0.000")
        self.assertEqual(len(calls), 1)

    def test_a_judge_timeout_is_transient(self):
        # urllib's "The read operation timed out" exhausts the gateway's retries on the batch and on
        # the per-item fallback; the runner repeats the judge, which answers.
        calls: list = []

        def reply(n):
            if n <= 8:
                raise TimeoutError("The read operation timed out")
            return '{"scores": [{"score": 1.0, "reason": "ok"}]}'

        result = self._run(failures=0, calls=calls, urlopen=_scripted_urlopen(calls, reply))
        self.assertEqual(self._score_text(result), "score=1.000")
        self.assertEqual(len(calls), 9)

    def test_a_parse_error_at_char_429_is_not_transient(self):
        # A body that is not JSON fails "(char 429)": not a rate limit. The run fails after the
        # gateway's own retries on the batch and on the per-item fallback, with no judge repeat.
        calls: list = []
        body = ("[" + "0," * 214 + "x]").encode()
        with self.assertRaisesRegex(Exception, r"char 429"):
            self._run(failures=0, calls=calls, urlopen=_scripted_urlopen(calls, lambda n: body))
        self.assertEqual(len(calls), 8)


@unittest.skipIf(_IMPORT_ERROR is not None, f"project dependencies not installed: {_IMPORT_ERROR}")
class TransientClassificationTest(unittest.TestCase):
    def test_types_first_then_words_never_a_bare_number(self):
        transient = runner_module._is_transient
        http = lambda code: urllib.error.HTTPError("http://judge", code, "x", Message(), None)
        cases = {
            "gateway 429": (gateway.TransientLLMError("GatewayLLM failed after 4 retries: HTTP Error 429"), True),
            "HTTPError 503 as the cause": (_caused(RuntimeError("judge"), http(503)), True),
            "read timeout": (TimeoutError("The read operation timed out"), True),
            "connect timeout": (urllib.error.URLError(TimeoutError("timed out")), True),
            "SDK rate limit text": (RuntimeError("OpenAI request failed after 3 retries: Error code: 429 - Rate limit reached"), True),
            "Gemini quota text": (RuntimeError("429 RESOURCE_EXHAUSTED"), True),
            "parse error at char 429": (RuntimeError("GatewayLLM failed after 4 retries: Expecting value: line 1 column 430 (char 429)"), False),
            "HTTPError 400": (http(400), False),
            "judge already retried": (_caused(runner_module.JudgeError("judge failed"), http(429)), False),
            "missing score": (KeyError("score"), False),
        }
        for name, (exc, expected) in cases.items():
            with self.subTest(name):
                self.assertIs(transient(exc), expected)


def _caused(exc: BaseException, cause: BaseException) -> BaseException:
    exc.__cause__ = cause
    return exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"project dependencies not installed: {_IMPORT_ERROR}")
class BeamJudgeGradesEveryItemTest(unittest.TestCase):
    """The score is what the judge said, item by item: never padded, cut or defaulted to 0."""

    def _result(self) -> "QueryResult":
        return QueryResult(
            query_id="q1", query="When is the meeting?", answer="The meeting moved to Friday.",
            reasoning="", context="ctx", context_tokens=0, retrieve_time_ms=0.0, gold_answers=["Friday"],
            correct=False, judge_reason="",
            meta={"question_category": "information_extraction",
                  "rubric": ["mentions Friday", "mentions the move"]},
        )

    def _score(self, batch_answer) -> tuple[float, mock.Mock]:
        """score_result with a batch answer, then per-item answers of 1.0 and 0.5."""
        llm = mock.Mock()
        llm.generate.side_effect = [batch_answer, {"score": 1.0, "reason": "ok"}, {"score": 0.5, "reason": "partial"}]
        return BEAMDataset().score_result(self._result(), llm), llm

    def test_one_score_for_two_items_goes_per_item(self):
        # Before: padded with 0.0 and recorded 0.5.
        score, llm = self._score({"scores": [{"score": 1.0, "reason": "ok"}]})
        self.assertEqual(score, 0.75)
        self.assertEqual(llm.generate.call_count, 3)

    def test_three_scores_for_two_items_go_per_item(self):
        # Before: the third was cut silently and the first two recorded.
        score, llm = self._score({"scores": [{"score": 1.0}, {"score": 1.0}, {"score": 0.0}]})
        self.assertEqual(score, 0.75)
        self.assertEqual(llm.generate.call_count, 3)

    def test_a_batch_without_scores_is_not_a_zero(self):
        # The gateway turns a prose answer into {"answer": ...}; before, that scored 0.
        for batch in ({"answer": "Both criteria are met."}, {"scores": "1.0, 0.5"}, {"scores": None}):
            with self.subTest(batch=batch):
                score, llm = self._score(batch)
                self.assertEqual(score, 0.75)
                self.assertEqual(llm.generate.call_count, 3)

    def test_an_item_without_a_numeric_score_goes_per_item(self):
        # Before: rs.get("score", 0) recorded the item as 0.
        for item in ({"reason": "no score"}, {"score": None}, {"score": "high"}, 1.0):
            with self.subTest(item=item):
                score, llm = self._score({"scores": [{"score": 1.0}, item]})
                self.assertEqual(score, 0.75)
                self.assertEqual(llm.generate.call_count, 3)

    def test_a_per_item_answer_without_score_raises(self):
        llm = mock.Mock()
        llm.generate.return_value = {"answer": "The response mentions Friday."}
        with self.assertRaises(Exception):
            BEAMDataset._rubric_item_score("q", "a", "item", llm)
        # Through score_result: a malformed batch, then a per-item answer without score.
        llm.generate.side_effect = [{"scores": []}, {"reason": "mentions Friday"}]
        with self.assertRaises(Exception):
            BEAMDataset().score_result(self._result(), llm)

    def test_an_equivalence_answer_without_answer_raises(self):
        # Before: a missing "answer" read as "NO".
        for answer in ({}, {"answer": ""}, {"answer": None}):
            with self.subTest(answer=answer):
                llm = mock.Mock()
                llm.generate.return_value = answer
                with self.assertRaises(Exception):
                    BEAMDataset._llm_equivalence("a", "b", llm)


if __name__ == "__main__":
    unittest.main()
