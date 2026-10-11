"""Executes one run of the timeline against fresh tenants and records every context."""

from __future__ import annotations

import datetime as dt
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .engine import EngineError, Tenant
from .reader import Reader
from .scoring import context_ok, leaked
from .systems import RagSystem, System, ValorBrainFlowSystem, ValorBrainSystem

PERSONA_LABEL = {
    "agent": "the agent that owns this workspace",
    "alice": "Alice, a member of this workspace",
    "bob": "Bob, a member of this workspace",
}

_DATE = re.compile(r"\{date:(-?\d+)\}")
_DOCID = re.compile(r"Docid:\s*#([0-9a-f]{6,})")
_COLLECTION = re.compile(r"Collection:\s*(\S+)")


def render(text: str, today: dt.date) -> str:
    return _DATE.sub(lambda m: (today + dt.timedelta(days=int(m.group(1)))).isoformat(), text)


class Run:
    def __init__(self, dataset: dict[str, Any], tenants: dict[str, Tenant], run_id: int,
                 today: dt.date, workers: int = 4):
        self.dataset = dataset
        self.tenants = tenants
        self.run_id = run_id
        self.today = today
        self.workers = workers
        self.vb: list[System] = [
            ValorBrainFlowSystem("valorbrain", tenants),
            ValorBrainSystem("valorbrain-prepare", "memory_prepare", tenants),
            ValorBrainSystem("valorbrain-retrieve", "memory_retrieve", tenants),
        ]
        self.rag = RagSystem("rag", per_user=False)
        self.rag_user = RagSystem("rag-per-user", per_user=True)
        self.rows: list[dict[str, Any]] = []
        self.properties: list[dict[str, Any]] = []
        self._handoffs: dict[str, str] = {}
        self._collections: dict[str, set[str]] = {}
        self._agent_member: str | None = None

    # ── writes ──────────────────────────────────────────────────────────────
    def _remember(self, tenant: str, writer: str, text: str) -> None:
        self.rag.add(tenant, writer, text)
        self.rag_user.add(tenant, writer, text)

    def _doc(self, s: dict[str, Any]) -> None:
        t = self.tenants[s["tenant"]]
        text = render(s["text"], self.today)
        body: dict[str, Any] = {"collection": s["collection"], "path": s["path"], "content": text,
                                "content_type": s["content_type"]}
        if s["supersedes"]:
            body["supersedes"] = s["supersedes"]
        resp = t.rest("POST", "/documents", body)
        self._collections.setdefault(s["tenant"], set()).add(s["collection"])
        self.properties.append({"property": "agent-write-default-sensitivity", "surface": "POST /documents",
                                "ref": f"{s['collection']}/{s['path']}", "value": resp.get("sensitivity")})
        self._remember(s["tenant"], "agent", text)

    def _fact(self, s: dict[str, Any]) -> None:
        text = render(s["text"], self.today)
        self.tenants[s["tenant"]].rest("POST", "/api/v1/keyed-facts", {
            "fact_key": s["key"], "fact_value": text,
            "as_of": (self.today + dt.timedelta(days=s["as_of_offset"])).isoformat(),
            "valid_until": (self.today + dt.timedelta(days=s["valid_until_offset"])).isoformat(),
        })
        self._remember(s["tenant"], "agent", text)

    def _member_name(self, tenant: Tenant) -> str:
        if self._agent_member is None:
            members = tenant.rest("GET", "/api/v1/team/members").get("members", [])
            agents = [m for m in members if m.get("kind") == "agent"]
            if not agents:
                raise EngineError("agent workspace has no agent team member to hand off to")
            self._agent_member = str(agents[0]["name"])
        return self._agent_member

    def _handoff(self, s: dict[str, Any]) -> None:
        t = self.tenants[s["tenant"]]
        resp = t.rest("POST", "/api/v1/team/handoffs", {"to": self._member_name(t), "summary": render(s["text"], self.today)})
        self._handoffs[s["ref"]] = str(resp["handoff"]["id"])
        self._remember(s["tenant"], "agent", s["text"])

    def _consume(self, s: dict[str, Any]) -> None:
        hid = self._handoffs[s["ref"]]
        self.tenants[s["tenant"]].rest("POST", f"/api/v1/team/handoffs/{hid}/consume", {"note": "done"})

    def _memory(self, s: dict[str, Any]) -> None:
        t = self.tenants[s["tenant"]]
        args: dict[str, Any] = {"type": s["type"], "title": s["title"], "content": s["text"]}
        if s["visibility"]:
            args["visibility"] = s["visibility"]
        out = t.mcp(s["persona"], "memory_store", args)
        coll = _COLLECTION.search(out)
        if coll:
            self._collections.setdefault(s["tenant"], set()).add(coll.group(1))
        docid = _DOCID.search(out)
        if s["visibility"] is None and docid:
            sens = t.document_sensitivity(docid.group(1))
            if sens is not None:
                self.properties.append({"property": "agent-write-default-sensitivity", "surface": "MCP memory_store",
                                        "ref": s["ref"], "value": sens})
        self._remember(s["tenant"], s["persona"], f"{s['title']}\n{s['text']}")

    def _barrier(self, s: dict[str, Any]) -> None:
        t = self.tenants[s["tenant"]]
        if any(p.rest_key for p in t.personas.values()):
            t.wait_indexed(self._collections.get(s["tenant"], set()))

    # ── questions ───────────────────────────────────────────────────────────
    def _ask_one(self, system: System, q: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        error = None
        try:
            ctx = system.context(q["tenant"], q["persona"], q["question"])
        except EngineError as e:
            ctx, error = "", str(e)
        return {
            "run": self.run_id, "system": system.name, "id": q["id"], "category": q["category"],
            "variant": q["variant"], "persona": q["persona"], "question": q["question"],
            "gold": q["gold"], "forbidden": q["forbidden"], "latency_ms": round((time.monotonic() - t0) * 1000),
            "context": ctx, "context_chars": len(ctx), "error": error,
            "context_ok": context_ok(ctx, q["gold"], q["forbidden"]),
            "leak": leaked(ctx, q["forbidden"]),
        }

    def _flush(self, asks: list[dict[str, Any]]) -> None:
        flow = next((s for s in self.vb if isinstance(s, ValorBrainFlowSystem)), None)
        if flow:
            for tenant, persona in sorted({(q["tenant"], q["persona"]) for q in asks}):
                flow.prime(tenant, persona,
                           "New session: give me the context I need — anything waiting for me, "
                           "rules in force and current decisions.")
        jobs = []
        for q in asks:
            systems: list[System] = [*self.vb, self.rag]
            if q["category"] == "permission":
                systems.append(self.rag_user)
            jobs += [(s, q) for s in systems]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            self.rows += list(pool.map(lambda job: self._ask_one(*job), jobs))

    def execute(self, tenants_in_scope: set[str]) -> None:
        handlers = {"doc": self._doc, "fact": self._fact, "handoff": self._handoff, "consume": self._consume,
                    "memory": self._memory, "barrier": self._barrier}
        pending: list[dict[str, Any]] = []
        for step in self.dataset["steps"]:
            if step["tenant"] not in tenants_in_scope:
                continue
            if step["op"] == "ask":
                pending.append(step)
                continue
            if pending:
                self._flush(pending)
                pending = []
            handlers[step["op"]](step)
        if pending:
            self._flush(pending)

    def read(self, reader: Reader, workers: int = 6) -> None:
        def one(row: dict[str, Any]) -> None:
            value, raw, model = reader.answer(self.today.isoformat(), PERSONA_LABEL[row["persona"]],
                                              row["question"], row["context"])
            row.update({"answer": value, "answer_raw": raw, "reader_model": model, "correct": value == row["gold"]})

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, self.rows))
