"""
ValorBrain memory provider for the Agent Memory Benchmark.

Retrieval pipeline: BM25 (Portuguese and English) and dense vectors (LFM2.5-Embedding-350M,
1024 dimensions) fused by reciprocal-rank fusion, then a cross-encoder rerank
(BGE-Reranker-v2-m3) and MMR for diversity, on PostgreSQL with row-level security. An LLM
only expands the query; it never ranks. Personalized PageRank exists behind a flag and is
off in production.

This provider talks to a running ValorBrain engine instance via its REST API.
VALORBRAIN_URL, VALORBRAIN_TOKEN and VALORBRAIN_BENCHMARK_TENANT_ID are all
REQUIRED — the provider refuses to start without them (see __init__).

Why: on 2026-09-18/19 six benchmark runs silently fell through the old defaults
(VALORBRAIN_URL -> https://valorbrain-api.valor.digital = PRODUCTION, tenant ->
whatever the token resolves to = the company dogfood tenant), ingesting a fresh
BEAM corpus into the real tenant and poisoning every score with dogfood
memories. Benchmarks run on a declared engine + dedicated bench tenant, never
anywhere by accident. See docs/research/AMB-CANONICO.md in the engine repo.

Each AMB isolation unit (e.g. BEAM conversation) becomes a ValorBrain collection.
"""

import json
import logging
import os
import time
import urllib.parse
import urllib.request

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)

# Real tenants that must NEVER receive benchmark data (Gus rule, 2026-09-19:
# "bench deve ser limpo e não no nosso tenant"). Add more if real tenants appear.
FORBIDDEN_TENANTS: dict[str, str] = {
    "3ceeb048-754c-4910-a798-112ae867e9a4": "valor-digital (company dogfood/production tenant)",
}


def _refuse(reason: str) -> "None":
    raise SystemExit(
        f"\nREFUSING TO RUN: {reason}\n"
        "A benchmark must declare its engine and dedicated bench tenant explicitly.\n"
        "Canonical setup: see docs/research/AMB-CANONICO.md in the engine repo\n"
        "(staging :7439 + bench tenant 00000000-0000-0000-0000-000000000001).\n"
    )


class ValorBrainMemoryProvider(MemoryProvider):
    name = "valorbrain"
    description = (
        "ValorBrain memory engine — BM25 + dense vectors fused by RRF, cross-encoder "
        "rerank and MMR on PostgreSQL."
    )
    kind = "cloud"
    link = "https://valor.digital"
    concurrency = 4

    def __init__(self):
        url = os.environ.get("VALORBRAIN_URL", "").strip()
        if not url:
            _refuse(
                "VALORBRAIN_URL is not set. The old silent default was the "
                "PRODUCTION engine (valorbrain-api.valor.digital)."
            )
        self._base = url.rstrip("/")
        self._token = os.environ.get("VALORBRAIN_TOKEN", "").strip()
        if not self._token:
            _refuse("VALORBRAIN_TOKEN is not set.")
        self._tenant = os.environ.get("VALORBRAIN_BENCHMARK_TENANT_ID", "").strip()
        if not self._tenant:
            _refuse(
                "VALORBRAIN_BENCHMARK_TENANT_ID is not set. Without it the engine "
                "resolves the tenant from the token — that is how the 2026-09-18/19 "
                "runs landed on the company dogfood tenant."
            )
        forbidden = FORBIDDEN_TENANTS.get(self._tenant.lower())
        if forbidden:
            _refuse(
                f"VALORBRAIN_BENCHMARK_TENANT_ID is {forbidden} — benchmark data "
                "must never be written to a real tenant."
            )
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

    def _get_stats_collection(self, collection: str, timeout: float = 30) -> dict:
        """GET /stats?collection=X.

        /stats is a GET endpoint; POSTing a body to it returns {"error": ...}.
        The provider used _post("/stats", {...}) for both the exists-check and
        the index-wait — both silently dead (every check saw an error response,
        existence never matched, _wait_index burned its full 120s timeout per
        collection). This GET uses the real contract: response carries
        collectionDocuments / collectionHybridPending / collectionEmbedFailed.
        """
        url = f"{self._base}/stats?collection={urllib.parse.quote(collection)}"
        req = urllib.request.Request(
            url,
            headers={
                **({"x-tenant-id": self._tenant} if self._tenant else {}),
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
            method="GET",
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
        # Decide "already ingested" ONCE per collection, before writing to it.
        # Checking per document skipped every session after the first one:
        # once the first session's windows landed, the collection had documents,
        # so the rest of the unit was dropped (14 of 71 windows on BEAM-100K
        # unit 2) and the collection was never refined.
        prefix = os.environ.get("VALORBRAIN_COLLECTION_PREFIX", "beam-100k")
        to_ingest: set[str] = set()
        collections_seen: set[str] = set()
        deduplicated: list[str] = []
        for doc in documents:
            raw = (doc.user_id or "amb-default").lower()
            # Prefix is configurable so other splits (e.g. beam-10m) don't collide.
            collection = f"{prefix}-{raw}" if raw.isdigit() else raw
            if collection in self._ingested_collections:
                continue
            if collection not in to_ingest:
                try:
                    stats = self._get_stats_collection(collection, timeout=15)
                    if stats.get("collectionDocuments", 0) > 0:
                        self._ingested_collections.add(collection)
                        continue  # skip ingest — data already chunked and indexed
                except Exception:
                    pass
                to_ingest.add(collection)
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

            def _post_window(args: tuple[str, str]) -> None:
                w_path, w_content = args
                body = {
                    "content": w_content,
                    "collection": collection,
                    "path": w_path,
                    "content_type": "conversation",
                    # BEAM 10m docs are giant config/log dumps that trip the
                    # engine's operational-noise heuristic (JSON lines ≤10 chars).
                    "skip_noise_filter": True,
                }
                if doc.timestamp:
                    body["event_at"] = doc.timestamp
                try:
                    result = self._post("/documents", body, timeout=300)
                except Exception as e:
                    logger.warning("ValorBrain ingest failed for %s/%s: %s", collection, w_path, e)
                    return
                # The engine stores identical content once per tenant: a window
                # already indexed in another collection answers 200 with
                # action "deduplicated" and is NOT written here. Measuring
                # such a collection would score an empty or partial unit.
                if isinstance(result, dict) and result.get("action") == "deduplicated":
                    deduplicated.append(f"{collection}/{w_path} -> {result.get('existing_collection')}")

            # Parallel ingest: ~240ms/POST sequential is hours at 10m scale;
            # 8 workers keep the embed service and DB both busy.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(_post_window, windows))

        if deduplicated:
            raise RuntimeError(
                f"ValorBrain deduplicated {len(deduplicated)} window(s) into other collections of the "
                f"bench tenant (e.g. {deduplicated[0]}): the same unit is already ingested under another "
                "name. Purge it or use that collection prefix — refusing to measure a partial unit."
            )

        # Wait for hybrid index on each collection that received documents.
        for collection in collections_seen:
            if collection not in self._ingested_collections:
                self._wait_index(collection)
                self._refine_collection(collection)
                self._ingested_collections.add(collection)

    def _wait_index(self, collection: str, timeout: float = 600) -> None:
        """Poll GET /stats?collection= until the collection's hybrid index is ready.

        Bails early when documents exist, nothing is pending, and some embeds
        have permanently failed — those will not recover within the timeout
        (the embed worker retries failed docs on an ~1h cadence), so waiting
        the full window only stalls the run.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                stats = self._get_stats_collection(collection, timeout=30)
                pending = stats.get("collectionHybridPending", 0)
                total = stats.get("collectionDocuments", 0)
                failed = stats.get("collectionEmbedFailed", 0)
                if total > 0 and pending == 0:
                    return
                if total > 0 and pending == failed and failed > 0:
                    return  # everything indexed that will be indexed
            except Exception:
                pass
            time.sleep(2)

    def _refine_collection(self, collection: str, timeout: float = 1800) -> None:
        """Extract observations + consolidate after ingest, before queries.

        Without this, /memory/prepare queries arrive before Phase 1.5 has
        extracted observations from the conversation docs. The answering LLM
        gets raw text instead of pre-extracted facts. Refine reads the whole
        conversation (one LLM pass per few windows plus a digest), so a long
        unit can take several minutes — a client timeout here would start the
        queries while the server is still refining.
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
        # Map AMB user_id (e.g. "1") to our chunked collections (prefix-configurable,
        # default beam-100k-N — production 6-turn windows, 38 docs/conv).
        raw = user_id or "amb-default"
        prefix = os.environ.get("VALORBRAIN_COLLECTION_PREFIX", "beam-100k")
        collection = f"{prefix}-{raw}" if raw.isdigit() else raw

        # /memory/prepare delivers the full pipeline (funnel + multitrecho + rerank).
        # delivered_documents are snippeted server-side (6k). This is the production
        # path — same endpoint Hermes uses, validated by the benchmark.
        body: dict = {"message": query, "collection": collection}
        try:
            data = self._post("/api/v1/memory/prepare", body, timeout=60)
            funnel = data.get("funnel") or {}
            delivered = funnel.get("delivered_documents", [])
            if delivered:
                # Separate synthetic docs (digest, facts) from conversation windows.
                # Embed synthetic as a header at the TOP of the first memory,
                # so the reader sees it as context, not as competing memories.
                synthetic = [d for d in delivered if d.get("path", "").startswith("__")]
                conversations = [d for d in delivered if not d.get("path", "").startswith("__")]
                
                synth_text = ""
                if synthetic:
                    parts = []
                    for d in synthetic:
                        c = d.get("content", "").strip()
                        if c:
                            parts.append(c)
                    synth_text = "\n\n".join(parts)
                
                docs = []
                for i, d in enumerate(conversations):
                    content = d.get("content", "")
                    if not content:
                        continue
                    # Prepend synthetic context to the FIRST conversation doc
                    if i == 0 and synth_text:
                        content = (
                            "=== CONVERSATION SUMMARY (use for factual questions) ===\n"
                            + synth_text
                            + "\n=== END SUMMARY ===\n\n"
                            + content
                        )
                    docs.append(Document(id=d.get("path", ""), content=content, user_id=user_id))
                
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

