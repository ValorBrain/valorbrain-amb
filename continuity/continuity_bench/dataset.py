"""Deterministic scenario generator for the continuity benchmark.

The dataset is a timeline of steps. Writes go to one of two isolated tenants:

* ``agent`` — a single agent's workspace (handoff, supersession, expiry);
* ``team`` — a workspace with two people (alice, bob) and an agent (permissions).

Every answer is a unique code shaped ``WORD-123``. Questions carry the gold code
(or ``NONE``) and the codes that must NOT be given (the stale version, the
expired rule, the consumed handoff, the other person's private note). Nothing
here depends on the clock: dates are stored as day offsets and rendered against
the run date, so the same seed yields the same timeline on any day.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

DATASET_VERSION = "continuity-v1"

PROJECTS = [
    "Atlas", "Borealis", "Cobalt", "Delta", "Ember", "Fjord", "Garnet", "Halcyon",
    "Indigo", "Juniper", "Kepler", "Lumen", "Meridian", "Nimbus", "Onyx", "Pioneer",
    "Quasar", "Redwood", "Sierra", "Tundra", "Umbra", "Vertex", "Willow", "Xenon",
    "Yarrow", "Zephyr", "Aurora", "Basalt", "Cascade", "Dune",
]

CODE_WORDS = [
    "KESTREL", "HERON", "OSPREY", "PLOVER", "WREN", "FINCH", "LARK", "EGRET",
    "ROBIN", "SWIFT", "TERN", "GANNET", "IBIS", "MERLIN", "PIPIT", "RAVEN",
    "SISKIN", "THRUSH", "VIREO", "AVOCET", "BITTERN", "CRANE", "DUNLIN", "EIDER",
    "FALCON", "GODWIT", "HOOPOE", "JACANA", "KITE", "LINNET", "MAGPIE", "NIGHTJAR",
    "ORIOLE", "PETREL", "QUAIL", "REDSTART", "SANDERLING", "TOWHEE", "WAGTAIL", "AUKLET",
]

# (subject, unit) pairs for decisions that get revised.
DECISION_TOPICS = [
    ("data retention policy", "policy code"),
    ("on-call escalation route", "route code"),
    ("release branch naming scheme", "scheme code"),
    ("primary database failover plan", "plan code"),
    ("customer refund approval flow", "flow code"),
    ("log shipping destination", "destination code"),
    ("staging deploy approver group", "group code"),
    ("incident severity matrix", "matrix code"),
    ("API rate limit tier", "tier code"),
    ("backup encryption key profile", "profile code"),
]

RULE_TOPICS = [
    ("deploy freeze", "only change ticket {code} may ship"),
    ("vendor purchase hold", "only purchase order {code} may be approved"),
    ("schema change embargo", "only migration {code} may run"),
    ("travel restriction", "only trip request {code} may be booked"),
    ("hiring pause", "only requisition {code} may be filled"),
]

HANDOFF_TASKS = [
    ("billing migration", "run job {code} and then verify the row counts"),
    ("search index rebuild", "start rebuild batch {code} and watch the queue"),
    ("certificate rotation", "apply certificate bundle {code} on the edge nodes"),
    ("cost report", "reconcile invoice batch {code} against the ledger"),
    ("data export for audit", "regenerate export {code} with the new filters"),
    ("flaky test triage", "re-run suite {code} with tracing enabled"),
    ("onboarding guide rewrite", "merge draft {code} after the screenshots land"),
    ("queue consumer upgrade", "roll out consumer build {code} to the canary"),
]

PRIVATE_SUBJECTS = [
    ("locker code for the {p} offsite", "Note to self: my locker code for the {p} offsite is {code}."),
    ("door code of my home office I shared for the {p} courier", "Note to self: the courier for {p} needs my home office door code {code}."),
    ("personal salary band I am asking for after {p}", "Note to self: after {p} ships I will ask for salary band {code}."),
    ("medical leave reference for the {p} schedule", "Note to self: my medical leave reference that affects the {p} schedule is {code}."),
]

SHARED_SUBJECTS = [
    ("budget code the {p} offsite is billed to", "Decision: the {p} offsite is billed to budget code {code} for all attendees."),
    ("meeting room booked for the {p} kickoff", "Decision: the {p} kickoff happens in meeting room {code}."),
    ("vendor chosen for the {p} catering", "Decision: catering for {p} goes to vendor {code}."),
    ("shared drive folder for {p} contracts", "Decision: all {p} contracts live in shared folder {code}."),
]

AGENT_SUBJECTS = [
    ("guest wifi network name at the {p} venue", "Observed: the {p} venue guest wifi network is named {code}."),
    ("parking lot assigned to the {p} team", "Observed: the {p} team parks in lot {code} this quarter."),
    ("print queue used by the {p} team", "Observed: the {p} team prints through queue {code}."),
    ("shuttle route serving the {p} site", "Observed: the shuttle serving the {p} site is route {code}."),
]

FILLER = [
    "{p} weekly sync moved to Wednesdays; notes go in the team channel.",
    "{p} stakeholders asked for a one-page summary before the quarterly review.",
    "The {p} dashboard now shows p95 latency next to error rate.",
]


class _Codes:
    """Unique ``WORD-123`` codes drawn from a seeded RNG."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._used: set[str] = set()

    def next(self) -> str:
        while True:
            code = f"{self._rng.choice(CODE_WORDS)}-{self._rng.randint(100, 999)}"
            if code not in self._used:
                self._used.add(code)
                return code


def _ask(qid: str, tenant: str, persona: str, category: str, variant: str, question: str,
         gold: str, forbidden: list[str]) -> dict[str, Any]:
    return {
        "op": "ask", "id": qid, "tenant": tenant, "persona": persona, "category": category,
        "variant": variant, "question": question, "gold": gold, "forbidden": forbidden,
    }


def _supersession(codes: _Codes, n: int, projects: list[str]) -> tuple[list, list]:
    writes: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    for i in range(n):
        for variant in ("implicit", "explicit"):
            p = projects[(i * 2 + (variant == "explicit")) % len(projects)]
            subject, unit = DECISION_TOPICS[(i + (variant == "explicit") * 5) % len(DECISION_TOPICS)]
            versions = 3 if i % 3 == 2 else 2
            vcodes = [codes.next() for _ in range(versions)]
            slug = f"{p.lower()}/{subject.replace(' ', '-')}"
            prev_path = None
            for v, code in enumerate(vcodes):
                path = f"{slug}-v{v + 1}.md"
                if variant == "explicit":
                    day = -30 + v * 10
                    lead = "Initial decision" if v == 0 else "Updated decision, replaces the previous one"
                    text = f"{lead} ({{date:{day}}}): the {p} {subject} uses {unit} {code}."
                else:
                    text = f"Decision: the {p} {subject} uses {unit} {code}."
                writes.append({
                    "op": "doc", "tenant": "agent", "collection": "cb-decisions", "path": path,
                    "text": text, "content_type": "decision", "supersedes": prev_path,
                })
                prev_path = path
            asks.append(_ask(
                f"S-{variant}-{i:02d}", "agent", "agent", "supersession", variant,
                f"What {unit} does the {p} {subject} use now? Answer with the code.",
                vcodes[-1], vcodes[:-1],
            ))
        # Control: a decision that was never revised.
        p = projects[(i * 2 + 1) % len(projects)]
        subject, unit = DECISION_TOPICS[(i + 3) % len(DECISION_TOPICS)]
        code = codes.next()
        writes.append({
            "op": "doc", "tenant": "agent", "collection": "cb-decisions",
            "path": f"{p.lower()}/single-{subject.replace(' ', '-')}.md",
            "text": f"Decision: the {p} {subject} uses {unit} {code}.",
            "content_type": "decision", "supersedes": None,
        })
        asks.append(_ask(
            f"S-control-{i:02d}", "agent", "agent", "supersession", "control",
            f"What {unit} does the {p} {subject} use now? Answer with the code.", code, [],
        ))
    return writes, asks


def _expiry(codes: _Codes, n: int, projects: list[str]) -> tuple[list, list]:
    writes: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    for i in range(n):
        for variant in ("implicit", "explicit"):
            for state in ("expired", "valid"):
                k = i * 4 + (variant == "explicit") * 2 + (state == "valid")
                # (project, topic) stays unique for k < 150: k and k+30 share the
                # project but land on the next topic.
                p = projects[(k + 7) % len(projects)]
                topic, clause = RULE_TOPICS[(k + k // len(projects)) % len(RULE_TOPICS)]
                code = codes.next()
                until = -10 if state == "expired" else 90
                body = f"{p} {topic}: {clause.format(code=code)}"
                if variant == "explicit":
                    body += f" (in force from {{date:-40}} until {{date:{until}}})"
                writes.append({
                    "op": "fact", "tenant": "agent", "key": f"rule:{p.lower()}-{topic.replace(' ', '-')}",
                    "text": body, "as_of_offset": -40, "valid_until_offset": until,
                })
                asks.append(_ask(
                    f"E-{variant}-{state}-{i:02d}", "agent", "agent", "expiry", f"{variant}-{state}",
                    f"Is a {p} {topic} in force today? If yes, give the one code it still allows; "
                    f"if no such rule is in force today, answer NONE.",
                    code if state == "valid" else "NONE",
                    [] if state == "valid" else [code],
                ))
    return writes, asks


def _handoffs(codes: _Codes, n: int, projects: list[str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for i in range(n):
        p = projects[(i + 13) % len(projects)]
        task, nxt = HANDOFF_TASKS[i % len(HANDOFF_TASKS)]
        old, new = codes.next(), codes.next()
        steps.append({"op": "handoff", "tenant": "agent", "ref": f"H{i:02d}-old",
                      "text": f"Handoff to my next session — {p} {task}: next step is to {nxt.format(code=old)}."})
        steps.append({"op": "consume", "tenant": "agent", "ref": f"H{i:02d}-old"})
        steps.append({"op": "handoff", "tenant": "agent", "ref": f"H{i:02d}-new",
                      "text": f"Handoff to my next session — {p} {task}: next step is to {nxt.format(code=new)}."})
        steps.append(_ask(
            f"H-topic-{i:02d}", "agent", "agent", "handoff", "topic",
            f"I am resuming the {p} {task} in a new session. What is the next step that was handed off "
            f"to me and is still pending? Answer with the code.", new, [old],
        ))
        steps.append(_ask(
            f"H-generic-{i:02d}", "agent", "agent", "handoff", "generic",
            "New session. Is there a handoff still waiting for me? Answer with the code of its next step.",
            new, [old],
        ))
        steps.append({"op": "consume", "tenant": "agent", "ref": f"H{i:02d}-new"})
        steps.append(_ask(
            f"H-none-{i:02d}", "agent", "agent", "handoff", "none",
            "New session. Is there a handoff still waiting for me? Answer with the code of its next step, "
            "or NONE if nothing is pending.", "NONE", [old, new],
        ))
    return steps


def _permissions(codes: _Codes, n: int, projects: list[str]) -> tuple[list, list]:
    writes: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    for i in range(n):
        p = projects[(i + 19) % len(projects)]
        priv_q, priv_t = PRIVATE_SUBJECTS[i % len(PRIVATE_SUBJECTS)]
        shared_q, shared_t = SHARED_SUBJECTS[i % len(SHARED_SUBJECTS)]
        agent_q, agent_t = AGENT_SUBJECTS[i % len(AGENT_SUBJECTS)]
        c_priv, c_shared, c_agent = codes.next(), codes.next(), codes.next()
        writes.append({"op": "memory", "tenant": "team", "persona": "alice", "type": "note",
                       "title": priv_q.format(p=p).capitalize(), "text": priv_t.format(p=p, code=c_priv),
                       "visibility": "private", "ref": f"P{i:02d}-private"})
        writes.append({"op": "memory", "tenant": "team", "persona": "alice", "type": "decision",
                       "title": shared_q.format(p=p).capitalize(), "text": shared_t.format(p=p, code=c_shared),
                       "visibility": "tenant", "ref": f"P{i:02d}-shared"})
        writes.append({"op": "memory", "tenant": "team", "persona": "agent", "type": "observation",
                       "title": agent_q.format(p=p).capitalize(), "text": agent_t.format(p=p, code=c_agent),
                       "visibility": None, "ref": f"P{i:02d}-agent"})
        asks.append(_ask(f"P-private-other-{i:02d}", "team", "bob", "permission", "private-other",
                         f"What is the {priv_q.format(p=p)}? Answer with the code.", "NONE", [c_priv]))
        asks.append(_ask(f"P-private-owner-{i:02d}", "team", "alice", "permission", "private-owner",
                         f"What is the {priv_q.format(p=p)}? Answer with the code.", c_priv, []))
        asks.append(_ask(f"P-shared-{i:02d}", "team", "bob", "permission", "shared",
                         f"What is the {shared_q.format(p=p)}? Answer with the code.", c_shared, []))
        asks.append(_ask(f"P-agent-default-{i:02d}", "team", "bob", "permission", "agent-default",
                         f"What is the {agent_q.format(p=p)}? Answer with the code.", c_agent, []))
    return writes, asks


def _fillers(projects: list[str], tenant: str) -> list[dict[str, Any]]:
    out = []
    for j, p in enumerate(projects):
        tmpl = FILLER[j % len(FILLER)]
        if tenant == "agent":
            out.append({"op": "doc", "tenant": "agent", "collection": "cb-notes", "path": f"{p.lower()}/note.md",
                        "text": tmpl.format(p=p), "content_type": "note", "supersedes": None})
        else:
            out.append({"op": "memory", "tenant": "team", "persona": "alice", "type": "note",
                        "title": f"{p} team note", "text": tmpl.format(p=p), "visibility": "tenant",
                        "ref": f"filler-{p.lower()}"})
    return out


def generate(seed: int = 20261011, per_cell: int = 10) -> dict[str, Any]:
    """Build the full timeline. ``per_cell`` = cases per scenario cell."""
    rng = random.Random(seed)
    codes = _Codes(rng)
    projects = PROJECTS[:]
    rng.shuffle(projects)

    s_writes, s_asks = _supersession(codes, per_cell, projects)
    e_writes, e_asks = _expiry(codes, per_cell, projects)
    h_steps = _handoffs(codes, per_cell, projects)
    p_writes, p_asks = _permissions(codes, per_cell, projects)

    steps: list[dict[str, Any]] = []
    steps += _fillers(projects, "agent")
    steps += s_writes + e_writes
    steps.append({"op": "barrier", "tenant": "agent"})
    steps += s_asks + e_asks
    steps += h_steps
    steps += _fillers(projects[:10], "team")
    steps += p_writes
    steps.append({"op": "barrier", "tenant": "team"})
    steps += p_asks

    body = {"version": DATASET_VERSION, "seed": seed, "per_cell": per_cell, "steps": steps}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "sha256": digest}


def questions(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in dataset["steps"] if s["op"] == "ask"]
