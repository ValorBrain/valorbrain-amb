"""Scoring. Declared judge: deterministic exact match, no LLM.

Two metrics per (system, question):

* **answer accuracy** — the reader's value equals the gold code (or NONE).
* **context accuracy** — reader-independent: the delivered context contains the
  gold code (when there is one) and none of the forbidden codes. For permission
  questions a forbidden code in the context is also counted as a **leak**.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def context_ok(context: str, gold: str, forbidden: list[str]) -> bool:
    if any(code in context for code in forbidden):
        return False
    return gold == "NONE" or gold in context


def leaked(context: str, forbidden: list[str]) -> bool:
    return any(code in context for code in forbidden)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate result rows by system → category → variant, pooled over runs."""
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["system"], r["category"], r["variant"])].append(r)
        groups[(r["system"], r["category"], "*")].append(r)
        groups[(r["system"], "*", "*")].append(r)
    for (system, category, variant), items in sorted(groups.items()):
        n = len(items)
        runs = sorted({r["run"] for r in items})
        cell: dict[str, Any] = {"n": n, "runs": len(runs)}
        for metric in ("context_ok", "correct"):
            scored = [r for r in items if r.get(metric) is not None]
            if not scored:
                continue
            k = sum(1 for r in scored if r[metric])
            lo, hi = wilson(k, len(scored))
            per_run = []
            for run in runs:
                rr = [r for r in scored if r["run"] == run]
                if rr:
                    per_run.append(sum(1 for r in rr if r[metric]) / len(rr))
            cell[metric] = {"k": k, "n": len(scored), "rate": k / len(scored), "ci95": [lo, hi],
                            "per_run": per_run}
        if category in ("permission", "*"):
            perm = [r for r in items if r["category"] == "permission" and r["forbidden"]]
            if perm:
                k = sum(1 for r in perm if r["leak"])
                cell["leak"] = {"k": k, "n": len(perm), "rate": k / len(perm)}
        out.setdefault(system, {}).setdefault(category, {})[variant] = cell
    return out
