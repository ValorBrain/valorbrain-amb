"""Command line: generate | run | report."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from . import dataset as ds
from .engine import Persona, Tenant, health, provision_team, signup_agent
from .reader import PROMPT, Reader
from .runner import Run
from .scoring import summarize
from .systems import RAG_EMBED_MODEL, RAG_TOP_K


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"missing environment variable {name}")
    return value


def cmd_generate(a: argparse.Namespace) -> None:
    data = ds.generate(seed=a.seed, per_cell=a.per_cell)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    qs = ds.questions(data)
    print(f"{a.out}: {len(data['steps'])} steps, {len(qs)} questions, sha256 {data['sha256']}")


def _agent_tenant(a: argparse.Namespace, run: int) -> Tenant | None:
    if a.agent_mode == "skip":
        return None
    if a.agent_mode == "signup":
        return signup_agent(a.rest_url, a.mcp_url, f"continuity-bench-r{run}")
    t = Tenant(a.rest_url, a.mcp_url, _env("CB_AGENT_TENANT_ID"))
    t.personas["agent"] = Persona("agent", _env("CB_AGENT_MCP_TOKEN"), _env("CB_AGENT_REST_KEY"))
    return t


def _team_tenant(a: argparse.Namespace, run: int) -> Tenant | None:
    if a.team_mode == "skip":
        return None
    if a.team_mode == "operator":
        tag = f"{dt.date.today():%Y%m%d}-r{run}-{secrets.token_hex(3)}"
        return provision_team(a.rest_url, a.mcp_url, _env("ENGINE_ADMIN_SECRET"),
                              _env("VALORBRAIN_PLATFORM_TOKEN"), tag)
    t = Tenant(a.rest_url, a.mcp_url, _env("CB_TEAM_TENANT_ID"))
    for name in ("alice", "bob", "agent"):
        t.personas[name] = Persona(name, _env(f"CB_{name.upper()}_MCP_TOKEN"))
    return t


def cmd_run(a: argparse.Namespace) -> None:
    data = json.loads(Path(a.dataset).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = None if a.no_reader else Reader()
    today = dt.date.today()
    for run in range(1, a.runs + 1):
        tenants: dict[str, Tenant] = {}
        agent = _agent_tenant(a, run)
        if agent:
            tenants["agent"] = agent
        team = _team_tenant(a, run)
        if team:
            tenants["team"] = team
        if not tenants:
            sys.exit("both workspaces skipped: nothing to run")
        engine = health(a.rest_url)
        started = dt.datetime.now(dt.timezone.utc)
        r = Run(data, tenants, run, today, workers=a.workers)
        r.execute(set(tenants))
        if reader:
            r.read(reader, workers=a.reader_workers)
        served = sorted({row.get("reader_model", "") for row in r.rows if row.get("reader_model")})
        meta = {
            "run": run, "date": today.isoformat(), "started_at": started.isoformat(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset": {"version": data["version"], "seed": data["seed"], "per_cell": data["per_cell"],
                        "sha256": data["sha256"]},
            "engine": {"rest_url": a.rest_url, "mcp_url": a.mcp_url, "version": engine.get("version"),
                       "git_hash": engine.get("git_hash")},
            "tenants": sorted(tenants), "team_mode": a.team_mode, "agent_mode": a.agent_mode,
            "rag": {"embedder": RAG_EMBED_MODEL, "top_k": RAG_TOP_K, "index": "exact cosine, one per workspace"},
            "reader": None if not reader else {"base_url": reader.base_url, "requested_model": reader.model,
                                               "served_models": served, "temperature": 0},
            "judge": "exact match on the reader's value (deterministic, no LLM)",
            "properties": r.properties,
        }
        with gzip.open(out / f"run-{run}.jsonl.gz", "wt") as fh:
            for row in r.rows:
                fh.write(json.dumps(row) + "\n")
        (out / f"run-{run}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        errors = sum(1 for row in r.rows if row["error"])
        print(f"run {run}: {len(r.rows)} rows, {errors} engine errors, engine {engine.get('git_hash')}")
    (out / "reader-prompt.txt").write_text(PROMPT + "\n")


def _pct(cell: dict[str, Any] | None) -> str:
    if not cell:
        return "—"
    lo, hi = cell["ci95"]
    return f"{100 * cell['rate']:.1f}% ({cell['k']}/{cell['n']}; 95% CI {100 * lo:.1f}–{100 * hi:.1f})"


def cmd_report(a: argparse.Namespace) -> None:
    out = Path(a.results)
    rows: list[dict[str, Any]] = []
    metas = []
    for path in sorted(out.glob("run-*.jsonl.gz")):
        with gzip.open(path, "rt") as fh:
            rows += [json.loads(line) for line in fh]
    for path in sorted(out.glob("run-*.meta.json")):
        metas.append(json.loads(path.read_text()))
    if not rows:
        sys.exit(f"no run files in {out}")
    summary = summarize(rows)
    props: dict[str, dict[str, int]] = {}
    for m in metas:
        for p in m["properties"]:
            key = f"{p['property']} via {p['surface']}"
            props.setdefault(key, {})
            props[key][str(p["value"])] = props[key].get(str(p["value"]), 0) + 1
    report = {"runs": metas, "summary": summary, "properties": props}
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")

    systems = [s for s in ("valorbrain", "valorbrain-prepare", "valorbrain-retrieve", "rag", "rag-per-user") if s in summary]
    m0 = metas[0]
    lines = [
        f"# Continuity benchmark — {m0['date']}",
        "",
        f"- Dataset: `{m0['dataset']['version']}` seed {m0['dataset']['seed']}, per_cell {m0['dataset']['per_cell']}, sha256 `{m0['dataset']['sha256'][:16]}`",
        f"- Engine: {m0['engine']['rest_url']} version {m0['engine']['version']} git {m0['engine']['git_hash']}",
        f"- Runs: {len(metas)} (fresh workspaces each run)",
        f"- Reader: {m0['reader']['served_models'] if m0['reader'] else 'none (context metric only)'} at temperature 0",
        f"- Judge: {m0['judge']}",
        f"- RAG baseline: {m0['rag']['embedder']}, top-{m0['rag']['top_k']}, {m0['rag']['index']}",
        "",
    ]
    for metric, title in (("correct", "Answer accuracy"), ("context_ok", "Context accuracy")):
        if not any(metric in summary[s].get("*", {}).get("*", {}) for s in systems):
            continue
        lines += [f"## {title}", "", "| Scenario | Variant | " + " | ".join(systems) + " |",
                  "|---|---|" + "---|" * len(systems)]
        cats = sorted({c for s in systems for c in summary[s] if c != "*"})
        for cat in cats:
            variants = sorted({v for s in systems for v in summary[s].get(cat, {})})
            for v in variants:
                cells = [_pct(summary[s].get(cat, {}).get(v, {}).get(metric)) for s in systems]
                lines.append(f"| {cat} | {'all' if v == '*' else v} | " + " | ".join(cells) + " |")
        cells = [_pct(summary[s]["*"]["*"].get(metric)) for s in systems]
        lines += ["| **all** | | " + " | ".join(cells) + " |", ""]
    leak_cells = []
    for s in systems:
        leak = summary[s].get("permission", {}).get("*", {}).get("leak")
        leak_cells.append(f"{leak['k']}/{leak['n']}" if leak else "—")
    lines += ["## Private-note leaks (forbidden code present in the context handed to another person)", "",
              "| " + " | ".join(systems) + " |", "|" + "---|" * len(systems), "| " + " | ".join(leak_cells) + " |", ""]
    if props:
        lines += ["## Properties", "", "| Property | Observed values (count) |", "|---|---|"]
        for key, values in props.items():
            lines.append(f"| {key} | " + ", ".join(f"{v}: {c}" for v, c in sorted(values.items())) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(prog="continuity-bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the deterministic dataset")
    g.add_argument("--seed", type=int, default=20261011)
    g.add_argument("--per-cell", type=int, default=10)
    g.add_argument("--out", default="data/continuity-v1.json")
    g.set_defaults(fn=cmd_generate)
    r = sub.add_parser("run", help="run against a ValorBrain engine")
    r.add_argument("--dataset", default="data/continuity-v1.json")
    r.add_argument("--rest-url", default=os.environ.get("VALORBRAIN_URL", "http://127.0.0.1:7439"))
    r.add_argument("--mcp-url", default=os.environ.get("VALORBRAIN_MCP_URL", "http://127.0.0.1:7439/mcp"))
    r.add_argument("--agent-mode", choices=("signup", "tokens", "skip"), default="signup")
    r.add_argument("--team-mode", choices=("operator", "tokens", "skip"), default="operator")
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--reader-workers", type=int, default=6)
    r.add_argument("--no-reader", action="store_true", help="context metric only, zero LLM calls")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    rep = sub.add_parser("report", help="aggregate run files into summary.json and summary.md")
    rep.add_argument("--results", required=True)
    rep.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
