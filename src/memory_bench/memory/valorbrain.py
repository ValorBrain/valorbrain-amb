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
        # Re-window large chunks into ~8k-char windows (matching production's
        # 6-turn windows). The AMB chunks at 100k chars; our retrieval expects
        # ~38 smaller windows per conversation, not 7 huge ones. Without this,
        # the search pool is too small for good coverage.
        WINDOW_SIZE = 8000
        WINDOW_OVERLAP = 800  # ~1 turn overlap, like our janelaTurnos-1
        collections_seen: set[str] = set()
        for doc in documents:
            raw = (doc.user_id or "amb-default").lower()
            # Check if production-chunked collection already exists (beam-100k-N)
            existing = f"beam-100k-{raw}" if raw.isdigit() else raw
            try:
                stats = self._post("/stats", {"collection": existing}, timeout=15)
                if stats.get("collectionDocuments", 0) > 0:
                    self._ingested_collections.add(existing)
                    continue  # skip ingest — data already chunked and indexed
            except Exception:
                pass
            collection = raw
            collections_seen.add(collection)

            content = doc.content
            if len(content) <= WINDOW_SIZE:
                windows = [(doc.id or f"{collection}/0", content)]
            else:
                windows = []
                wi = 0
                pos = 0
                while pos < len(content):
                    end = min(pos + WINDOW_SIZE, len(content))
                    windows.append((f"{doc.id}_w{wi}", content[pos:end]))
                    wi += 1
                    if end >= len(content):
                        break
                    pos = end - WINDOW_OVERLAP

            for w_path, w_content in windows:
                body = {
                    "content": w_content,
                    "collection": collection,
                    "path": w_path,
                    "content_type": "conversation",
                }
                if doc.timestamp:
                    body["event_at"] = doc.timestamp
                try:
                    self._post("/documents", body, timeout=300)
                except Exception as e:
                    logger.warning("ValorBrain ingest failed for %s/%s: %s", collection, w_path, e)

        # Wait for hybrid index on each collection that received documents.
        for collection in collections_seen:
            if collection not in self._ingested_collections:
                self._wait_index(collection)
                self._refine_collection(collection)
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

    def _refine_collection(self, collection: str, timeout: float = 300) -> None:
        """Extract observations + consolidate after ingest, before queries.

        Without this, /memory/prepare queries arrive before Phase 1.5 has
        extracted observations from the conversation docs. The answering LLM
        gets raw text instead of pre-extracted facts.
        """
        try:
            result = self._post(
                "/api/v1/memory/refine",
                {"collection": collection},
                timeout=timeout,
            )
            logger.info(
                "Refine %s: observed=%s extracted=%s consolidated=%s",
                collection,
                result.get("observed", 0),
                result.get("extracted", 0),
                result.get("consolidated", 0),
            )
        except Exception as e:
            logger.warning("Refine failed for %s: %s (continuing)", collection, e)

    def retrieve(
        self,
        query: str,
        k: int = 20,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        # Map AMB user_id (e.g. "1") to our production chunked collections
        # (e.g. "beam-100k-1") which have proper 6-turn windows (38 docs/conv).
        raw = user_id or "amb-default"
        collection = f"beam-100k-{raw}" if raw.isdigit() else raw

        # /memory/prepare delivers the full pipeline (funnel + multitrecho + rerank).
        # delivered_documents are snippeted server-side (6k). This is the production
        # path — same endpoint Hermes uses, validated by the benchmark.
        body: dict = {"message": query, "collection": collection}
        try:
            data = self._post("/api/v1/memory/prepare", body, timeout=60)
            funnel = data.get("funnel") or {}
            delivered = funnel.get("delivered_documents", [])
            if delivered:
                docs = [Document(id=d.get("path",""), content=d.get("content",""), user_id=user_id)
                        for d in delivered if d.get("content")]
                if docs:
                    return docs, data
        except Exception as e:
            logger.warning("ValorBrain prepare failed: %s", e)

        # Fallback: /search
        try:
            data = self._post("/search", {"query": query, "mode": "hybrid", "limit": k,
                "collection": collection, "compact": False}, timeout=60)
        except Exception as e:
            return [], {"error": str(e)}
        docs = [Document(id=r.get("docid",""), content=(r.get("body") or r.get("snippet",""))[:6000], user_id=user_id)
                for r in data.get("results",[]) if r.get("body") or r.get("snippet")]
        return docs, data
