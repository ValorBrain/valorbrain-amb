"""
ValorBrain memory provider for the Agent Memory Benchmark.

ValorBrain is a hybrid memory engine: BM25 + dense vectors (pgvector) + RRF +
graph reranking + BGE cross-encoder rerank, all on PostgreSQL with RLS.

This provider talks to a running ValorBrain engine instance via its REST API.
Set VALORBRAIN_URL (default http://localhost:7438) and VALORBRAIN_TOKEN.

Each AMB isolation unit (e.g. BEAM conversation) becomes a ValorBrain collection.
"""

import json
import logging
import os
import time
import urllib.request

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)


class ValorBrainMemoryProvider(MemoryProvider):
    name = "valorbrain"
    description = (
        "ValorBrain hybrid memory engine — BM25 + dense (pgvector) + RRF + "
        "graph reranking + BGE cross-encoder rerank on PostgreSQL."
    )
    kind = "cloud"
    link = "https://valor.digital"
    concurrency = 4

    def __init__(self):
        self._base = os.environ.get("VALORBRAIN_URL", "http://localhost:7438").rstrip("/")
        self._token = os.environ.get("VALORBRAIN_TOKEN", "")
        self._tenant = os.environ.get("VALORBRAIN_BENCHMARK_TENANT_ID", "")
        self._ingested_collections: set[str] = set()

    # ── HTTP helper ──────────────────────────────────────────────────────

    def _post(self, path: str, body: dict, timeout: float = 120) -> dict:
        url = f"{self._base}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **({"x-tenant-id": self._tenant} if self._tenant else {}),
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── Provider interface ───────────────────────────────────────────────

    def ingest(self, documents: list[Document]) -> None:
        # AMB already chunks conversations into multiple Documents sharing
        # the same user_id. We must ingest ALL of them, not dedup by collection.
        collections_seen: set[str] = set()
        for doc in documents:
            collection = (doc.user_id or "amb-default").lower()
            collections_seen.add(collection)

            body = {
                "content": doc.content,
                "collection": collection,
                "path": doc.id or f"{collection}/doc-{id(doc)}",
                "content_type": "conversation",
            }
            if doc.timestamp:
                body["event_at"] = doc.timestamp

            try:
                self._post("/documents", body, timeout=300)
            except Exception as e:
                logger.warning("ValorBrain ingest failed for %s/%s: %s", collection, doc.id, e)

        # Wait for hybrid index on each collection that received documents.
        for collection in collections_seen:
            if collection not in self._ingested_collections:
                self._wait_index(collection)
                self._ingested_collections.add(collection)

    def _wait_index(self, collection: str, timeout: float = 120) -> None:
        """Poll /stats until the collection's hybrid index is ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                stats = self._post(
                    "/stats",
                    {"collection": collection},
                    timeout=30,
                )
                pending = stats.get("collectionHybridPending", 0)
                total = stats.get("collectionDocuments", 0)
                if total > 0 and pending == 0:
                    return
            except Exception:
                pass
            time.sleep(2)

    def retrieve(
        self,
        query: str,
        k: int = 20,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        collection = user_id or "amb-default"

        # Use /search (hybrid BM25+dense+RRF+rerank) with full content.
        # k defaults to 20 for broader coverage — the AMB RAG mode joins
        # multiple memories, and more chunks = more facts available.
        body: dict = {
            "query": query,
            "mode": "hybrid",
            "limit": k,
            "collection": collection,
            "compact": False,
        }

        try:
            data = self._post("/search", body, timeout=60)
        except Exception as e:
            logger.warning("ValorBrain retrieve failed: %s", e)
            return [], {"error": str(e)}

        results = data.get("results", [])
        docs: list[Document] = []
        for r in results:
            content = (
                r.get("body") or r.get("content")
                or r.get("text") or r.get("snippet") or ""
            )
            if content:
                docs.append(Document(
                    id=r.get("docid") or r.get("id") or r.get("path", ""),
                    content=content, user_id=user_id,
                ))
        return docs, data
