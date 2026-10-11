"""Thin ValorBrain client: REST for writes, MCP (streamable HTTP, JSON) for reads.

Writes use the documented REST surfaces an integration calls (POST /documents
with ``supersedes``, POST /api/v1/keyed-facts with ``valid_until``,
/api/v1/team/handoffs). Reads go through the MCP tools an agent calls
(``memory_prepare`` at session start, ``memory_retrieve`` for a specific
question), with the asking persona's own MCP token.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

USER_AGENT = "continuity-bench/1"


class EngineError(RuntimeError):
    pass


def http_json(method: str, url: str, body: Any = None, headers: dict[str, str] | None = None,
              timeout: float = 180) -> Any:
    h = {"content-type": "application/json", "user-agent": USER_AGENT, **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise EngineError(f"{method} {url} -> {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    return json.loads(raw) if raw else None


@dataclass
class Persona:
    """One caller identity: an MCP token for reads, optionally a REST key for writes."""

    name: str
    mcp_token: str
    rest_key: str | None = None


@dataclass
class Tenant:
    rest_url: str
    mcp_url: str
    tenant_id: str
    personas: dict[str, Persona] = field(default_factory=dict)
    # Credential allowed to read document metadata (sensitivity) — optional.
    inspect_token: str | None = None

    # ── REST ────────────────────────────────────────────────────────────────
    def rest(self, method: str, path: str, body: Any = None, token: str | None = None) -> Any:
        tok = token or self._rest_key()
        return http_json(method, self.rest_url.rstrip("/") + path, body,
                         {"authorization": f"Bearer {tok}", "x-tenant-id": self.tenant_id})

    def _rest_key(self) -> str:
        for p in self.personas.values():
            if p.rest_key:
                return p.rest_key
        raise EngineError("no REST credential configured for this tenant")

    def wait_indexed(self, collections: set[str], timeout: float = 240) -> None:
        """Poll GET /stats?collection= until every collection has nothing pending."""
        deadline = time.monotonic() + timeout
        pending = set(collections)
        while pending and time.monotonic() < deadline:
            for c in list(pending):
                stats = self.rest("GET", f"/stats?collection={urllib.parse.quote(c)}")
                total = stats.get("collectionDocuments", 0)
                waiting = stats.get("collectionHybridPending", 0)
                failed = stats.get("collectionEmbedFailed", 0)
                if total > 0 and waiting <= failed:
                    pending.discard(c)
            if pending:
                time.sleep(2)  # polling a real server-side condition (embedding queue)
        if pending:
            raise EngineError(f"index not ready after {timeout}s: {sorted(pending)}")

    def document_sensitivity(self, docid: str) -> str | None:
        if not self.inspect_token:
            return None
        doc = self.rest("GET", f"/documents/{docid}", token=self.inspect_token)
        scope = doc.get("scope") if isinstance(doc, dict) else None
        return scope.get("sensitivity") if isinstance(scope, dict) else None

    # ── MCP ─────────────────────────────────────────────────────────────────
    def mcp(self, persona: str, tool: str, args: dict[str, Any]) -> str:
        token = self.personas[persona].mcp_token
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}}
        resp = http_json("POST", self.mcp_url, body, {
            "authorization": f"Bearer {token}",
            "accept": "application/json, text/event-stream",
        })
        if "error" in resp:
            raise EngineError(f"MCP {tool}: {resp['error']}")
        result = resp.get("result") or {}
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if result.get("isError"):
            raise EngineError(f"MCP {tool} error: {text[:300]}")
        return text


def health(rest_url: str) -> dict[str, Any]:
    return http_json("GET", rest_url.rstrip("/") + "/health", timeout=30)


def signup_agent(rest_url: str, mcp_url: str, agent_caller: str) -> Tenant:
    """Self-service agent workspace — the same call `valorbrain init --agent` makes."""
    r = http_json("POST", rest_url.rstrip("/") + "/api/v1/agents/signup", {"agent_caller": agent_caller})
    t = Tenant(rest_url=rest_url, mcp_url=mcp_url, tenant_id=r["tenant_id"])
    t.personas["agent"] = Persona("agent", mcp_token=r["mcp_token"], rest_key=r["api_key"])
    return t


def provision_team(rest_url: str, mcp_url: str, admin_secret: str, platform_token: str, tag: str) -> Tenant:
    """Operator path: a workspace with two people and an agent, like the SaaS creates.

    ENGINE_ADMIN_SECRET creates the tenant and its second member; the platform
    credential mints one MCP token per identity (member tokens are bound to the
    member; the agent token is a workspace token with no person behind it).
    """
    base = rest_url.rstrip("/")
    admin = {"authorization": f"Bearer {admin_secret}"}
    t = http_json("POST", base + "/api/v1/admin/tenants", {
        "name": f"Continuity bench {tag}", "slug": "continuity-bench",
        "admin_email": f"alice+{tag}@continuity-bench.invalid",
    }, admin)
    tid, alice = t["tenant_id"], t["admin_user_id"]
    bob = http_json("POST", f"{base}/api/v1/admin/tenants/{tid}/users", {
        "email": f"bob+{tag}@continuity-bench.invalid", "role": "member",
    }, admin)["user_id"]
    team = Tenant(rest_url=rest_url, mcp_url=mcp_url, tenant_id=tid, inspect_token=platform_token)
    plat = {"authorization": f"Bearer {platform_token}", "x-tenant-id": tid}
    for name, uid in (("alice", alice), ("bob", bob), ("agent", None)):
        body: dict[str, Any] = {"name": f"continuity-{name}", "scopes": ["read", "write"], "user_id": uid}
        if uid is None:
            body["intended_agent_platform"] = "continuity-agent"
        tok = http_json("POST", base + "/api/v1/mcp/tokens", body, plat)["token"]
        team.personas[name] = Persona(name, mcp_token=tok, rest_key=t["api_token"] if name == "alice" else None)
    return team
