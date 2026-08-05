# valorbrain-amb

This is the **ValorBrain fork** of the [Agent Memory Benchmark (AMB)](https://agentmemorybenchmark.ai) — the open benchmark published by Hindsight/Vectorize (`vectorize-io/open-memory-benchmark`). We use AMB to evaluate the ValorBrain memory engine head-to-head against Hindsight, mem0, and other memory systems on long-conversation recall.

Public mirror: **[github.com/ValorBrain/valorbrain-amb](https://github.com/ValorBrain/valorbrain-amb)**

> AMB's own pitch (why it exists, what it measures, reproducibility rules) is preserved at the bottom of this file. This README documents **what the fork adds and how to run it**.

---

## What this fork adds

Five additions on top of upstream AMB. All plug into AMB's existing provider registry — no core harness changes.

| Addition | File | Registers as |
|---|---|---|
| **ValorBrain memory provider** | `src/memory_bench/memory/valorbrain.py` | `--memory valorbrain` |
| **GLM-5.2 LLM provider** | `src/memory_bench/llm/glm.py` | `--llm glm` |
| **Gateway LLM provider** (OpenAI-compatible) | `src/memory_bench/llm/gateway.py` | `--llm gateway` |
| **AGY direct LLM provider** (Gemini via Antigravity CLI) | `src/memory_bench/llm/agy_direct.py` | `--llm agy-direct` |
| **Checkpoint runner** | `scripts/run-with-checkpoint.py` | standalone script |

### ValorBrain memory provider

Talks to a running ValorBrain engine instance over its REST API. The engine is a hybrid memory pipeline — BM25 + dense vectors (pgvector) + reciprocal-rank fusion + graph reranking + BGE cross-encoder rerank — on PostgreSQL with row-level security.

Two behaviors worth knowing:

1. **Re-windowing.** AMB chunks each BEAM conversation at ~100k chars. ValorBrain's retrieval expects ~8k-char windows (matching production's 6-turn windows). On ingest, large chunks are re-windowed into 8000-char windows with 800-char overlap, unless a pre-chunked production collection already exists (`beam-100k-{id}`), in which case ingest is skipped entirely.
2. **Retrieval via `/api/v1/memory/prepare`.** This is the production memory path — the same endpoint the agent uses at runtime. It runs the full funnel (consolidation + timeline + snippet delivery). Falls back to `/search` (hybrid mode) if prepare returns nothing.

### GLM-5.2 LLM

GLM-5.2 is a reasoning model (Z.ai coding plan): the answer appears in `reasoning_content` when `content` is empty. Since Z.ai has no `response_format` support, JSON is extracted prompt-side. This is the reader that scored **70.9% on BEAM-100K (312/400)** — 5 of 10 category wins vs Hindsight.

### Gateway LLM

Plain-HTTP OpenAI-compatible client with no `response_format` dependency (built to talk to an `opencode-go` gateway serving e.g. `deepseek-v4-flash`). Prompt-based JSON extraction, exponential-backoff retries (4 attempts). The base for both the GLM provider (subclass) and any generic OpenAI-compatible endpoint.

### AGY direct LLM

Calls the `agy` (Antigravity) CLI via subprocess — Gemini without an HTTP gateway in the middle, so the gateway can't be a failure point. Used as an alternative path to Gemini.

### Checkpoint runner

BEAM-100K is 100 conversations × 20 questions = 2000 queries (the fork runs a 400-question subset, 20 conversations). A single `amb run` ingests and queries everything in one process, so a crash or rate-limit midway loses everything. The checkpoint runner (`scripts/run-with-checkpoint.py`) runs **one conversation at a time**, merges results into the main output file after each, and on restart skips conversations already completed. See [Checkpoint runner](#checkpoint-runner) below.

---

## Setup

### Prerequisites

- Python ≥ 3.11
- A running ValorBrain engine (default `http://localhost:7438`) with a benchmark tenant provisioned
- At least one LLM key (depends on which providers you use — see table below)

### Environment variables

Copy into a `.env` at the repo root (AMB auto-loads it), or export in the shell.

**ValorBrain memory provider**

| Variable | Default | Purpose |
|---|---|---|
| `VALORBRAIN_URL` | `http://localhost:7438` | Engine REST base URL |
| `VALORBRAIN_TOKEN` | _(empty)_ | Auth bearer token |
| `VALORBRAIN_BENCHMARK_TENANT_ID` | _(empty)_ | Tenant to scope all benchmark collections to |

**LLM providers**

| Variable | Default | Provider | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | `gemini` | Google AI key. `GEMINI_API_KEY` is mirrored to `GOOGLE_API_KEY` on startup (see [Gemini key gate](#gemini-key-gate)) |
| `GLM_API_KEY` | _(empty)_ | `glm` | Z.ai coding-plan key |
| `GLM_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` | `glm` | Z.ai endpoint base |
| `GLM_MODEL` | `glm-5.2` | `glm` | Model override |
| `OPENAI_BASE_URL` | `http://localhost:8201/v1` | `gateway` | OpenAI-compatible endpoint (e.g. opencode-go gateway) |
| `OPENAI_API_KEY` | _(empty)_ | `gateway` | Bearer key for the gateway |
| `OMB_GATEWAY_MODEL` | `deepseek-v4-flash` | `gateway` | Model served by the gateway |
| `AGY_MODEL` | `gemini-3.6-flash` | `agy-direct` | Antigravity model |
| `AGY_EFFORT` | `low` | `agy-direct` | Reasoning effort |

**Role wiring (which LLM answers vs which LLM judges)**

| Variable | Default | Purpose |
|---|---|---|
| `OMB_ANSWER_LLM` | `groq` | Provider for RAG answer generation |
| `OMB_ANSWER_MODEL` | _(none)_ | Optional model override for the answer LLM |
| `OMB_JUDGE_LLM` | `gemini` | Provider for evaluation/judging — see [Judge override](#judge-override) |
| `OMB_JUDGE_MODEL` | _(none)_ | Optional model override for the judge LLM |

**Datasets**

| Variable | Purpose |
|---|---|
| `BEAM_DATA_PATH` | Local JSON for BEAM 100K/500K/1M splits (else auto-downloaded from HuggingFace) |
| `BEAM_10M_DATA_PATH` | Local JSON for the BEAM 10M split |
| `MEMBENCH_DATA_PATH` | Local data directory for MemBench |

---

## Running benchmarks

```bash
# List available datasets, memory providers, and response modes
uv run amb providers

# List domains/splits for a dataset
uv run amb splits --dataset beam

# Full BEAM-100K run: ValorBrain memory + GLM-5.2 reader, Gemini judge (dataset default)
OMB_ANSWER_LLM=glm OMB_ANSWER_MODEL=glm-5.2 \
  uv run amb run --dataset beam --split 100k --memory valorbrain

# Single conversation (isolation unit) — e.g. conversation "7"
uv run amb run --dataset beam --split 100k --memory valorbrain --unit 7

# Filter to one category with a query cap
uv run amb run --dataset beam --split 100k --memory valorbrain \
  --category event_ordering --query-limit 20

# Re-judge cached answers from a previous run (no retrieval, no answer gen)
uv run amb run --dataset beam --split 100k --memory valorbrain --skip-answer

# Browse results in the browser
uv run amb view
```

### Output layout

Results are saved to `outputs/{dataset}/{name}/{mode}/{split}.json`, where `{name}` defaults to the memory provider name (override with `--name`). Each result JSON carries `answer_llm` and `judge_llm` (e.g. `glm:glm-5.2`, `gateway:deepseek-v4-flash`) so runs are self-describing.

---

## Checkpoint runner

For long runs that can't fit in one `amb run`, the checkpoint runner processes one conversation at a time and resumes on restart.

```bash
# Set the same env vars as `amb run`, then:
python3 scripts/run-with-checkpoint.py
```

What it does:

1. Reads completed conversation IDs from the merged output file (`outputs/beam/valorbrain-gemini-judge-full/rag/100k.json`).
2. For each remaining conversation, runs `amb run --dataset beam --split 100k --memory valorbrain --unit <n> --llm glm --name ckpt-<n>`.
3. Merges that conversation's results into the main file (deduplicating by `query_id`), recomputes totals/accuracy, and deletes the per-conversation file.
4. Stops on the first failed conversation (exit ≠ 0) so you can inspect and re-run. On restart it picks up from the last good checkpoint.

> The unit list, output path, and `amb` binary path are constants at the top of the script (`ALL_UNITS`, `OUTPUT`, the `cmd` list). Edit them before running a different configuration.

---

## Known patches

These are changes to upstream AMB behavior that this fork relies on.

### Judge override

By default each dataset picks its own judge model (e.g. BEAM uses `gemini-3.5-flash` for head-to-head parity with Hindsight/mem0, whose nugget-judge rubric BEAM mirrors). The runner's `_get_judge()` (`src/memory_bench/runner.py`) lets `OMB_JUDGE_LLM` override this:

- **`OMB_JUDGE_LLM` unset or `gemini`** → the dataset's `default_judge_llm()` wins (the parity path).
- **`OMB_JUDGE_LLM` set to another provider** (e.g. `glm`, `gateway`) → that provider judges instead, via `get_judge_llm()`. Pair with `OMB_JUDGE_MODEL` to pin the exact model.

This is how the fork can run a GLM reader with a Gemini judge, or a fully non-Gemini (reader + judge) configuration.

### Gemini key gate

Upstream AMB hard-fails without a Gemini key because the judge and reader default to Gemini. The fork's `_resolve_gemini_key()` (`src/memory_bench/cli.py`) makes the key **non-fatal**: it mirrors `GEMINI_API_KEY` → `GOOGLE_API_KEY` when present, but if no key is set the CLI still starts. `GeminiLLM` only errors if it is actually instantiated — so a run configured with `OMB_ANSWER_LLM=glm OMB_JUDGE_LLM=gateway` needs no Gemini key at all.

---

## Results

Headline: **70.9% on BEAM-100K** (312/400) with GLM-5.2 as reader — 5 of 10 category wins vs Hindsight. The full run lives at `outputs/beam/valorbrain-gemini-judge-full/rag/100k.json`.

---

## Requirements

- Python ≥ 3.11
- A running ValorBrain engine + benchmark tenant
- LLM keys for whichever providers you configure (see [Environment variables](#environment-variables))
- `uv` (recommended) — run every command as `uv run amb …`. `amb` and `omb` (legacy alias) are both installed as entry points.

---

## Appendix: what AMB is (upstream)

AMB was built to be honest about how Hindsight performs, because no existing benchmark gave the full picture. It is fully open: datasets, prompts, scoring logic, and results. Live leaderboard: **[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)**.

**The problem with existing benchmarks.** LoComo and LongMemEval are solid datasets, but they were designed for 32k context windows. State-of-the-art models now have million-token context windows — on most instances a naive "dump everything into context" approach scores competitively, not because it's a good memory architecture, but because retrieval has become the easy part. Both datasets were also built around chatbot use cases; AMB adds datasets focused on agentic tasks.

**What AMB measures.** Accuracy first (hardest to fake), with speed and token cost tracked alongside. A memory system that scores 90% accuracy but costs $10/user/day is not better than one that scores 82% at $0.10.

**How it works.**

1. **Ingest** — documents from a dataset are loaded into a memory provider
2. **Retrieve** — for each query the memory provider retrieves relevant context
3. **Generate** — a model produces an answer from the retrieved context
4. **Judge** — a second call scores the answer against gold answers

Retrieval time is tracked separately from generation; ingestion time is also recorded.
