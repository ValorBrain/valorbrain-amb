#!/usr/bin/env python3
"""
AMB checkpoint runner v2 — runs one conversation at a time, saves after each.
On restart, skips completed conversations. Uses `amb` CLI directly.

Usage:
  python3 scripts/run-with-checkpoint.py

Environment variables must be set before running (same as amb run).
"""
import json
import os
import subprocess
import time
from pathlib import Path

ALL_UNITS = [str(i) for i in range(1, 21)]
OUTPUT = Path("outputs/beam/valorbrain-gemini-judge-full/rag/100k.json")


def load_completed():
    """Load completed conversation IDs from main results file."""
    if not OUTPUT.exists():
        return set()
    try:
        data = json.loads(OUTPUT.read_text())
        return {r.get("query_id", "_").split("_")[0] for r in data.get("results", [])}
    except Exception:
        return set()


def merge_unit(unit_results):
    """Merge a single conversation's results into the main file."""
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        existing_results = existing.get("results", [])
    else:
        existing = {}
        existing_results = []

    # Dedup by query_id — new results replace old ones for same query
    by_qid = {r.get("query_id", f"_{i}"): r for i, r in enumerate(existing_results)}
    for r in unit_results:
        by_qid[r.get("query_id", f"new_{len(by_qid)}")] = r

    all_results = list(by_qid.values())
    correct = sum(1 for r in all_results if r.get("correct"))
    scores = [r.get("score") for r in all_results if r.get("score") is not None]
    accuracy = sum(scores) / len(scores) if scores else correct / max(1, len(all_results))

    merged = dict(existing)
    merged["results"] = all_results
    merged["total_queries"] = len(all_results)
    merged["correct"] = correct
    merged["accuracy"] = accuracy

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(merged, indent=2))
    return len(all_results), correct, accuracy


def main():
    completed = load_completed()
    print(f"[ckpt] completed: {sorted(completed, key=lambda x: int(x) if x.isdigit() else 99)}")
    remaining = [u for u in ALL_UNITS if u not in completed]
    print(f"[ckpt] remaining: {remaining}")

    for unit in remaining:
        print(f"\n[ckpt] === conversation {unit} ===")

        unit_name = f"ckpt-{unit}"
        cmd = ["/root/anaconda3/bin/amb", "run", "--dataset", "beam", "--split", "100k",
               "--memory", "valorbrain", "--unit", unit,
               "--llm", "glm", "--name", unit_name]
        t0 = time.time()
        result = subprocess.run(cmd, timeout=1800)  # 30 min max per conversation
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[ckpt] unit {unit} FAILED (exit {result.returncode}) in {elapsed:.0f}s — stopping")
            break

        # Load unit results and merge
        unit_path = Path(f"outputs/beam/{unit_name}/rag/100k.json")
        if not unit_path.exists():
            print(f"[ckpt] unit {unit} — no output file, stopping")
            break

        unit_data = json.loads(unit_path.read_text())
        unit_results = unit_data.get("results", [])
        total, correct, acc = merge_unit(unit_results)
        print(f"[ckpt] unit {unit} done in {elapsed:.0f}s — merged: {total}/400, {correct} correct, {acc*100:.1f}%")

        # Clean up unit-specific file
        unit_path.unlink(missing_ok=True)

    # Final summary
    print(f"\n[ckpt] === FINAL ===")
    final_completed = load_completed()
    print(f"[ckpt] conversations done: {sorted(final_completed, key=lambda x: int(x) if x.isdigit() else 99)}")
    if OUTPUT.exists():
        final = json.loads(OUTPUT.read_text())
        results = final.get("results", [])
        correct = sum(1 for r in results if r.get("correct"))
        scores = [r.get("score") for r in results if r.get("score") is not None]
        print(f"[ckpt] total: {len(results)}/400, correct: {correct}")
        if scores:
            print(f"[ckpt] avg score: {sum(scores)/len(scores)*100:.1f}%")


if __name__ == "__main__":
    main()
