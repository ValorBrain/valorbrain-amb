#!/usr/bin/env python3
"""Recompute every published BEAM number from the run files in results/.

Two metrics, never mixed:

- avg-score: the AMB scoreboard metric. It is the `accuracy` field AMB writes: the mean
  judge score (0..1 per question, partial credit allowed) over the questions the judge
  scored. Questions the judge could not score (for example, empty retrieval) are left out of
  the denominator, exactly as AMB's runner does.
- binary: the share of ALL questions the judge marked correct (`correct / total_queries`).

The 95% interval is a percentile bootstrap that resamples whole conversations (questions
inside one conversation are not independent), 10,000 resamples, fixed seed. It measures
sampling error over questions for ONE run; it is not run-to-run variance.

Usage:
    python3 scripts/beam_summary.py                 # markdown summary of every run file
    python3 scripts/beam_summary.py --json          # same, as JSON

Standard library only, so the README test runs without installing AMB.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261011


@dataclass
class RunSummary:
    file: str
    run_name: str
    split: str
    reader: str
    judge: str
    total_queries: int
    scored_queries: int
    conversations: int
    avg_score: float
    avg_score_ci: tuple[float, float]
    binary_correct: int
    binary: float
    binary_ci: tuple[float, float]
    avg_context_tokens: float | None
    avg_retrieve_time_ms: float | None
    categories: dict[str, dict[str, float]] = field(default_factory=dict)


def load_run(path: Path) -> dict:
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def run_files() -> list[Path]:
    return sorted(p for p in RESULTS.iterdir() if p.name.endswith((".json", ".json.gz")))


def _avg_score(results: list[dict]) -> float:
    scored = [r["score"] for r in results if r.get("score") is not None]
    return sum(scored) / len(scored) if scored else 0.0


def _binary(results: list[dict]) -> float:
    return sum(1 for r in results if r.get("correct")) / len(results) if results else 0.0


def _conversation(result: dict) -> str:
    meta = result.get("meta") or {}
    return str(meta.get("conversation_id") or result["query_id"].split("_", 1)[0])


def _cluster_bootstrap(results: list[dict], metric) -> tuple[float, float]:
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_conv[_conversation(r)].append(r)
    clusters = list(by_conv.values())
    rng = random.Random(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample: list[dict] = []
        for _ in clusters:
            sample.extend(rng.choice(clusters))
        values.append(metric(sample))
    values.sort()
    lo = values[int(0.025 * BOOTSTRAP_RESAMPLES)]
    hi = values[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return (lo, hi)


def summarize(path: Path) -> RunSummary:
    data = load_run(path)
    results = data["results"]
    categories: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        categories[(r.get("meta") or {}).get("question_category", "unknown")].append(r)
    return RunSummary(
        file=str(path.relative_to(ROOT)),
        run_name=data.get("run_name") or path.stem,
        split=data["split"],
        reader=data.get("answer_llm") or "unknown",
        judge=data.get("judge_llm") or "unknown",
        total_queries=len(results),
        scored_queries=sum(1 for r in results if r.get("score") is not None),
        conversations=len({_conversation(r) for r in results}),
        avg_score=_avg_score(results),
        avg_score_ci=_cluster_bootstrap(results, _avg_score),
        binary_correct=sum(1 for r in results if r.get("correct")),
        binary=_binary(results),
        binary_ci=_cluster_bootstrap(results, _binary),
        avg_context_tokens=data.get("avg_context_tokens"),
        avg_retrieve_time_ms=data.get("avg_retrieve_time_ms"),
        categories={
            name: {"avg_score": _avg_score(rs), "binary": _binary(rs), "n": len(rs)}
            for name, rs in sorted(categories.items())
        },
    )


def pct(x: float) -> str:
    """One decimal, half-up on the 10-digit value (313/400 → 78.3%, not float-truncated 78.2%)."""
    value = Decimal(repr(round(x * 100, 10))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{value}%"


def to_markdown(s: RunSummary) -> str:
    lines = [
        f"### {s.file}",
        f"- run: `{s.run_name}` · split {s.split} · {s.conversations} conversations · {s.total_queries} questions",
        f"- reader: `{s.reader}` · judge: `{s.judge}`",
        f"- avg-score: {pct(s.avg_score)} (95% CI {pct(s.avg_score_ci[0])}–{pct(s.avg_score_ci[1])}; {s.scored_queries} scored questions)",
        f"- binary: {pct(s.binary)} ({s.binary_correct}/{s.total_queries}) (95% CI {pct(s.binary_ci[0])}–{pct(s.binary_ci[1])})",
        "",
        "| Category | avg-score |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {pct(v['avg_score'])} |" for name, v in s.categories.items()]
    lines += ["", "| Category | binary |", "|---|---:|"]
    lines += [f"| {name} | {pct(v['binary'])} |" for name, v in s.categories.items()]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    summaries = [summarize(p) for p in run_files()]
    if "--json" in argv:
        print(json.dumps([asdict(s) for s in summaries], indent=2))
    else:
        print("\n\n".join(to_markdown(s) for s in summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
