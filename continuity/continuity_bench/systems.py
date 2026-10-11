"""The systems under test. Each returns the context it would hand an agent.

* ``valorbrain-prepare`` — MCP ``memory_prepare`` (the session-start entry point).
* ``valorbrain-retrieve`` — MCP ``memory_retrieve`` (question-time retrieval).
* ``rag`` — plain vector RAG: every text written to the workspace, embedded
  once, top-k by cosine similarity, one shared index per workspace.
* ``rag-per-user`` — the same, but one index per writer (the asker only sees
  what they wrote). Only meaningful for the permission scenarios.

The RAG baselines index exactly the text the ValorBrain writes carried. They do
not see the lifecycle metadata (``supersedes``, ``valid_until``, handoff
status, visibility) — that metadata is what this benchmark is about.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from .engine import Tenant

RAG_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RAG_TOP_K = 8


class System(Protocol):
    name: str

    def context(self, tenant: str, persona: str, question: str) -> str: ...


class ValorBrainSystem:
    def __init__(self, name: str, tool: str, tenants: dict[str, Tenant]):
        self.name = name
        self._tool = tool
        self._tenants = tenants

    def context(self, tenant: str, persona: str, question: str) -> str:
        arg = "message" if self._tool == "memory_prepare" else "query"
        return self._tenants[tenant].mcp(persona, self._tool, {arg: question})


class ValorBrainFlowSystem:
    """The documented product flow: ``memory_prepare`` at session start (once
    per flush of questions), then ``memory_retrieve`` for the specific
    question. The prepare context carries identity, pending handoffs and keyed
    facts; the retrieve context carries the ranked memories. The two texts are
    concatenated, in delivery order."""

    def __init__(self, name: str, tenants: dict[str, Tenant]):
        self.name = name
        self._tenants = tenants
        self._prepare: dict[tuple[str, str], str] = {}

    def prime(self, tenant: str, persona: str, opening: str) -> None:
        t = self._tenants[tenant]
        self._prepare[(tenant, persona)] = t.mcp(persona, "memory_prepare", {"message": opening})

    def context(self, tenant: str, persona: str, question: str) -> str:
        head = self._prepare.get((tenant, persona), "")
        retrieved = self._tenants[tenant].mcp(persona, "memory_retrieve", {"query": question})
        return "\n\n".join(part for part in (head, retrieved) if part)


class _Embedder:
    _lock = threading.Lock()
    _model = None

    @classmethod
    def embed(cls, texts: list[str]) -> np.ndarray:
        with cls._lock:
            if cls._model is None:
                from fastembed import TextEmbedding

                cls._model = TextEmbedding(RAG_EMBED_MODEL)
            vecs = np.array(list(cls._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


class _Index:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self._vecs = np.zeros((0, 384), dtype=np.float32)
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        vec = _Embedder.embed([text])
        with self._lock:
            self.texts.append(text)
            self._vecs = np.vstack([self._vecs, vec])

    def search(self, query: str, k: int) -> list[str]:
        q = _Embedder.embed([query])[0]
        with self._lock:
            if not self.texts:
                return []
            scores = self._vecs @ q
            order = np.argsort(-scores, kind="stable")[:k]
            return [self.texts[i] for i in order]


class RagSystem:
    """Plain vector RAG. ``per_user`` keys the index by writer instead of workspace."""

    def __init__(self, name: str, per_user: bool):
        self.name = name
        self._per_user = per_user
        self._indexes: dict[tuple[str, str], _Index] = {}

    def add(self, tenant: str, writer: str, text: str) -> None:
        key = (tenant, writer if self._per_user else "*")
        self._indexes.setdefault(key, _Index()).add(text)

    def context(self, tenant: str, persona: str, question: str) -> str:
        idx = self._indexes.get((tenant, persona if self._per_user else "*"))
        hits = idx.search(question, RAG_TOP_K) if idx else []
        return "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(hits))
