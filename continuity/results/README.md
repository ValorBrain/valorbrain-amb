# Continuity benchmark results

| Directory | Date | Engine | Dataset | Runs × questions | Reader | Judge |
|---|---|---|---|---|---|---|
| [`2026-10-11-staging-n3`](2026-10-11-staging-n3/summary.md) | 2026-10-11 | staging, git `af530899` | `continuity-v1`, per_cell 3 (sha256 `db7fb526a9d6…`) | 3 × 42 | `longcat-2.5-preview-free`, temperature 0 | exact match, no LLM |

## 2026-10-11, staging, N = 3 runs × 42 questions

Answer accuracy, pooled over runs:

| Scenario | valorbrain (prepare + retrieve) | rag (vector, same texts) | rag-per-user |
|---|---|---|---|
| Supersession | 27/27 | 25/27 | — |
| Expired rule | 36/36 | 27/36 | — |
| Handoff | 27/27 | 8/27 | — |
| Permission | 36/36 | 27/36 | 18/36 |
| All | 126/126 | 87/126 | 18/36 (permission only) |
| Private-note leaks to another person | 0/9 | 9/9 | 0/9 |

Context accuracy is independent of the reader: the gold code is present and no stale or forbidden code is. On that measure `valorbrain` scored 126/126 and `rag` scored 54/126. The full per-variant tables, with 95% Wilson intervals, are in [`summary.md`](2026-10-11-staging-n3/summary.md).

How to read this run:

- **The reader that actually served the run.** We requested `deepseek-v4.1-flash` from our OpenAI-compatible gateway, which fell back to `longcat-2.5-preview-free`. `run-N.meta.json` records the served model, and every reader output is in `run-N.jsonl.gz`.
- **Single tools versus the product flow.** `valorbrain-prepare` alone kept a superseded version in its context in 18 of 27 supersession questions, and the reader still picked the current one. The cause was the engine's recall block (fixed in agentxagi/valorbrain#380, not yet deployed to staging at `af530899`). We'll re-run after that deploy and record it here. `valorbrain-retrieve` alone doesn't carry handoffs: they are delivered by `memory_prepare` at session start, which is why the documented flow calls both.
- **Agent-write default.** All 150 agent writes in this run were stored as `internal`: 141 through `POST /documents`, 9 through MCP `memory_store`.
- **Small N.** The cells hold 9 questions per variant. Treat differences of a few questions as noise, and use the intervals in `summary.md`. The reference dataset (`--per-cell 10`, 140 questions) is committed for larger runs.
