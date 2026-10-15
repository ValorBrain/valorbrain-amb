"""ValorBrain provider ingest: a unit is measured whole and refined, or the run fails.

The provider used to decide "collection already ingested" per session document:
after the first session's windows landed, every later session of the unit was
skipped (BEAM-100K unit 2: 14 of 71 windows) and refine never ran. A window the
engine deduplicated into another collection, a failed POST, a failed /stats
check or a failed refine were warnings, so the run measured a partial unit.

These tests import the provider package, so they need the project environment:
`uv run --with pytest pytest tests/`. They are skipped where it cannot import.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from memory_bench.memory.valorbrain import ValorBrainMemoryProvider
    from memory_bench.models import Document
except ImportError as e:  # pragma: no cover - depends on the environment
    _IMPORT_ERROR: ImportError | None = e
else:
    _IMPORT_ERROR = None

ENV = {
    "VALORBRAIN_URL": "http://localhost:1",
    "VALORBRAIN_TOKEN": "x",
    "VALORBRAIN_BENCHMARK_TENANT_ID": "00000000-0000-0000-0000-000000000001",
}


class FakeEngine:
    """Collections → document counts; records every POST."""

    def __init__(self, counts: dict[str, int] | None = None, post_action: str = "created"):
        self.counts = dict(counts or {})
        self.posted: list[str] = []
        self.refined: list[str] = []
        self.post_action = post_action
        self.stats_failures = 0
        self.post_failures: set[str] = set()
        self.refine_fails = False

    def stats(self, collection: str, timeout: float = 30) -> dict:
        if self.stats_failures > 0:
            self.stats_failures -= 1
            raise OSError("connection refused")
        return {"collectionDocuments": self.counts.get(collection, 0), "collectionHybridPending": 0}

    def post(self, path: str, body: dict, timeout: float = 0) -> dict:
        if path == "/documents":
            if body["path"] in self.post_failures:
                raise OSError("HTTP 500")
            self.posted.append(body["path"])
            if self.post_action == "created":
                self.counts[body["collection"]] = self.counts.get(body["collection"], 0) + 1
            return {"action": self.post_action, "existing_collection": "beam-100k-2"}
        if path == "/api/v1/memory/refine":
            if self.refine_fails:
                raise TimeoutError("timed out")
            self.refined.append(body["collection"])
            self.counts[f"{body['collection']}-obs"] = 1
            return {"observed": 1}
        raise AssertionError(path)


def unit(user: str, sessions: int, chars: int = 20_000) -> list:
    return [Document(id=f"{user}_s{i}", content="x" * chars, user_id=user, timestamp=None) for i in range(sessions)]


@unittest.skipIf(_IMPORT_ERROR is not None, f"project environment unavailable: {_IMPORT_ERROR}")
class ProviderIngestTest(unittest.TestCase):
    def provider(self, engine: FakeEngine):
        p = ValorBrainMemoryProvider()
        p._get_stats_collection = engine.stats
        p._post = engine.post
        return p

    def setUp(self):
        patcher = mock.patch.dict(os.environ, ENV, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("VALORBRAIN_ALLOW_UNREFINED", None)
        sleep = mock.patch("memory_bench.memory.valorbrain.time.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_new_unit_posts_every_session_and_refines_once(self):
        engine = FakeEngine()
        self.provider(engine).ingest(unit("2", 3))
        self.assertEqual(len(engine.posted), 9)  # 3 sessions x 3 windows of 8000/800
        self.assertEqual(engine.refined, ["beam-100k-2"])

    def test_ingested_and_refined_collection_is_reused(self):
        engine = FakeEngine({"beam-100k-1": 38, "beam-100k-1-obs": 167})
        self.provider(engine).ingest(unit("1", 3))
        self.assertEqual(engine.posted, [])
        self.assertEqual(engine.refined, [])

    def test_windows_without_refine_fail_unless_allowed(self):
        engine = FakeEngine({"beam-100k-2": 14})
        with self.assertRaisesRegex(RuntimeError, "never refined"):
            self.provider(engine).ingest(unit("2", 3))
        self.assertEqual(engine.posted, [])
        with mock.patch.dict(os.environ, {"VALORBRAIN_ALLOW_UNREFINED": "1"}):
            self.provider(engine).ingest(unit("2", 3))
        self.assertEqual(engine.posted, [])

    def test_deduplicated_window_fails_after_posting_all(self):
        engine = FakeEngine(post_action="deduplicated")
        with self.assertRaisesRegex(RuntimeError, "deduplicated"):
            self.provider(engine).ingest(unit("3", 2))
        self.assertEqual(len(engine.posted), 6)
        self.assertEqual(engine.refined, [])

    def test_failed_window_post_fails_the_run(self):
        engine = FakeEngine()
        engine.post_failures = {"4_s1_w2"}
        with self.assertRaisesRegex(RuntimeError, "1 window\\(s\\) failed to post"):
            self.provider(engine).ingest(unit("4", 2))
        self.assertEqual(engine.refined, [])

    def test_stats_failure_is_retried_then_fails(self):
        engine = FakeEngine({"beam-100k-1": 38, "beam-100k-1-obs": 167})
        engine.stats_failures = 2
        self.provider(engine).ingest(unit("1", 1))
        self.assertEqual(engine.posted, [])
        engine.stats_failures = 5
        with self.assertRaisesRegex(RuntimeError, "/stats failed"):
            self.provider(engine).ingest(unit("1", 1))
        self.assertEqual(engine.posted, [])

    def test_failed_refine_fails_the_run(self):
        engine = FakeEngine()
        engine.refine_fails = True
        with self.assertRaisesRegex(RuntimeError, "refine failed"):
            self.provider(engine).ingest(unit("5", 1))


if __name__ == "__main__":
    unittest.main()
