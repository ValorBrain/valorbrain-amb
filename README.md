# valorbrain-amb

This is the **ValorBrain fork** of the [Agent Memory Benchmark (AMB)](https://agentmemorybenchmark.ai), the open benchmark published by Hindsight/Vectorize (`vectorize-io/agent-memory-benchmark`). We use AMB to measure the ValorBrain memory engine on long-conversation memory (BEAM).

Public mirror: **[github.com/ValorBrain/valorbrain-amb](https://github.com/ValorBrain/valorbrain-amb)**

> AMB's own pitch (why it exists, what it measures, reproducibility rules) is preserved at the bottom of this file. This README documents **what the fork adds, how to run it, and what we measured with it**.

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

Talks to a running ValorBrain engine instance over its REST API. Retrieval pipeline (the same description as valorbrain.com.br and docs.valorbrain.com.br): BM25 (Portuguese and English) and dense vectors (LFM2.5-Embedding-350M, 1024 dimensions) fused by reciprocal-rank fusion, then a cross-encoder rerank (BGE-Reranker-v2-m3) and MMR for diversity, on PostgreSQL with row-level security. An LLM only expands the query; it never ranks. Personalized PageRank exists behind a flag and is off in production.

Two behaviors worth knowing:

1. **Re-windowing.** AMB chunks each BEAM conversation at ~100k chars. ValorBrain's retrieval expects ~8k-char windows (matching production's 6-turn windows). On ingest, large chunks are re-windowed into 8000-char windows with 800-char overlap, unless a pre-chunked production collection already exists (`beam-100k-{id}`), in which case ingest is skipped entirely.
2. **Retrieval via `/api/v1/memory/prepare`.** This is the production memory path — the same endpoint the agent uses at runtime. It runs the full funnel (consolidation + timeline + snippet delivery). Falls back to `/search` (hybrid mode) if prepare returns nothing.

### GLM-5.2 LLM

GLM-5.2 is a reasoning model (Z.ai coding plan): the answer appears in `reasoning_content` when `content` is empty. Since Z.ai has no `response_format` support, JSON is extracted prompt-side. In the published runs below GLM-5.2 is the **judge**, not the reader.

### Gateway LLM

Plain-HTTP OpenAI-compatible client with no `response_format` dependency (built to talk to an `opencode-go` gateway serving e.g. `deepseek-v4-flash`). Prompt-based JSON extraction, exponential-backoff retries (4 attempts). The base for both the GLM provider (subclass) and any generic OpenAI-compatible endpoint, including OpenRouter (how the GLM-5.3 Flash reader below was called).

### AGY direct LLM

Calls the `agy` (Antigravity) CLI via subprocess — Gemini without an HTTP gateway in the middle, so the gateway can't be a failure point. Used as an alternative path to Gemini.

### Checkpoint runner

The BEAM 100K split is 20 conversations × 20 questions = 400 questions (BEAM as a whole has 100 conversations and 2,000 questions across the 100K, 500K, 1M and 10M splits). A single `amb run` ingests and queries everything in one process, so a crash or rate-limit midway loses everything. The checkpoint runner (`scripts/run-with-checkpoint.py`) runs **one conversation at a time**, merges results into the main output file after each, and on restart skips conversations already completed. See [Checkpoint runner](#checkpoint-runner-1) below.

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
3. Merges that conversation's results into the main file (deduplicating by `query_id`), recomputes totals, and deletes the per-conversation file.
4. Stops on the first failed conversation (exit ≠ 0) so you can inspect and re-run. On restart it picks up from the last good checkpoint.

> The unit list, output path, and `amb` binary path are constants at the top of the script (`ALL_UNITS`, `OUTPUT`, the `cmd` list). Edit them before running a different configuration.

---

## Known patches

These are changes to upstream AMB behavior that this fork relies on.

### Judge override

By default each dataset picks its own judge model (BEAM's `default_judge_llm()` returns `gemini-3.5-flash`). The runner's `_get_judge()` (`src/memory_bench/runner.py`) lets `OMB_JUDGE_LLM` override this:

- **`OMB_JUDGE_LLM` unset or `gemini`** → the dataset's `default_judge_llm()` wins.
- **`OMB_JUDGE_LLM` set to another provider** (e.g. `glm`, `gateway`) → that provider judges instead, via `get_judge_llm()`. Pair with `OMB_JUDGE_MODEL` to pin the exact model.

Every published ValorBrain run below used this override (judge `glm:glm-5.2`). The Hindsight runs on the AMB scoreboard were judged by Gemini models, so a ValorBrain number and a Hindsight number are never a head-to-head.

### Gemini key gate

Upstream AMB hard-fails without a Gemini key because the judge and reader default to Gemini. The fork's `_resolve_gemini_key()` (`src/memory_bench/cli.py`) makes the key **non-fatal**: it mirrors `GEMINI_API_KEY` → `GOOGLE_API_KEY` when present, but if no key is set the CLI still starts. `GeminiLLM` only errors if it is actually instantiated — so a run configured with `OMB_ANSWER_LLM=glm OMB_JUDGE_LLM=gateway` needs no Gemini key at all.

---

## Results

Every number in this section is recomputed from a file in `results/` by `scripts/beam_summary.py`, and `tests/test_readme_metrics.py` fails the build if a number here drifts from its file or if a line mixes two metrics.

**Metrics.** Each line carries exactly one metric:

- **avg-score** is the AMB scoreboard metric: the mean judge score (0 to 1 per question, partial credit allowed) over the questions the judge scored. It is the `accuracy` field AMB writes.
- **binary** is the share of all questions the judge marked correct (`correct / total_queries`).

**Interval.** The 95% CI is a percentile bootstrap over conversations (10,000 resamples, fixed seed). It covers sampling error over questions inside one run; it is not run-to-run variance.

**Runs.** The PRD target is N ≥ 3 runs per configuration with a confidence interval. We are not there: the BEAM-100K headline configuration has N = 1 run and the BEAM-10M configuration has N = 2 runs. The reader used for both (`stealth/ox-alpha` on OpenRouter) was a time-limited stealth listing, so new runs need a different reader and will be published as a new configuration, not as extra runs of this one.

**Reader and product default.** The reader is not part of the product: ValorBrain returns context and the customer's own LLM writes the answer. GLM-5.3 Flash at `reasoning.effort=max` is a benchmark configuration, not a product default. The memory side runs the production path (`/api/v1/memory/prepare`) on a staging engine.

**Reader identity.** `stealth/ox-alpha` was revealed on 2026-08-25 to be **GLM-5.3 Flash (Z.ai)** served by OpenRouter under a stealth alias. Run files keep the alias as recorded.

### BEAM-100K — headline run (2026-08-23)

- File: `results/beam-100k-ox-alpha-effortmax-400q.json`
- Dataset: BEAM 100K split, the whole split: 20 conversations, 400 questions
- Reader: GLM-5.3 Flash (`openai:stealth/ox-alpha`), `reasoning.effort=max`; not the product default
- Judge: GLM-5.2 (`glm:glm-5.2`)
- avg-score: 75.5% (95% CI 72.5%–78.5%; 398 scored questions)
- binary: 80.8% (323/400) (95% CI 78.0%–83.5%)
- Runs: N = 1

### BEAM-10M — run (2026-08-24)

- File: `results/beam-10m-ox-alpha-effortmax-200q.json`
- Dataset: BEAM 10M split, the whole split: 10 conversations of about 10M tokens each, 200 questions
- Reader: GLM-5.3 Flash (`openai:stealth/ox-alpha`), `reasoning.effort=max`; not the product default
- Judge: GLM-5.2 (`glm:glm-5.2`)
- avg-score: 51.2% (95% CI 44.9%–57.4%; 200 scored questions)
- binary: 56.0% (112/200) (95% CI 48.5%–63.0%)
- Runs: N = 2 (second run below)
- Ingest: about 15 minutes; the provider sends `skip_noise_filter=true`, which only disables the engine's heartbeat/ping rejection for these windows

### BEAM-10M — second run, same configuration (2026-08-24)

- File: `results/beam-10m-ox-alpha-effortmax-200q-run2.json.gz`
- Same run name, reader, judge and collections as the run above; 141 of 200 retrieved contexts are byte-identical between the two runs
- avg-score: 50.7% (95% CI 44.6%–57.0%; 200 scored questions)
- binary: 56.0% (112/200) (95% CI 49.0%–63.0%)

### Where the BEAM-100K headline came from (same harness, same judge, same 400 questions)

The three runs below used the same staging engine, the same judge (`glm:glm-5.2`) and the same 400 questions, two days apart; only the reader and its reasoning effort change. Retrieval is not bit-identical between runs (184 to 262 of 400 contexts match), so each step also carries run noise.

| Step | File | Reader | avg-score | Change in avg-score |
|---|---|---|---:|---:|
| Baseline | `results/beam-100k-gemini36flash-effortdefault-400q.json.gz` | Gemini 3.6 Flash (`agy:gemini-3.6-flash`) | 69.3% | — |
| Reader swap | `results/beam-100k-ox-alpha-effortdefault-400q.json.gz` | GLM-5.3 Flash, default effort | 72.3% | +3.0 pp |
| Reasoning effort | `results/beam-100k-ox-alpha-effortmax-400q.json` | GLM-5.3 Flash, `effort=max` | 75.5% | +3.2 pp |
| Total | | | | +6.2 pp |

| Step | File | Reader | binary | Questions gained |
|---|---|---|---:|---:|
| Baseline | `results/beam-100k-gemini36flash-effortdefault-400q.json.gz` | Gemini 3.6 Flash | 76.0% (304/400) | — |
| Reader swap | `results/beam-100k-ox-alpha-effortdefault-400q.json.gz` | GLM-5.3 Flash, default effort | 78.3% (313/400) | +9 |
| Reasoning effort | `results/beam-100k-ox-alpha-effortmax-400q.json` | GLM-5.3 Flash, `effort=max` | 80.8% (323/400) | +10 |
| Total | | | | +19 |

The whole gain from the baseline to the headline is on the reader side. We no longer publish a "memory engineering vs reader" split: the earlier split mixed runs with different judges, question counts and metrics, and does not survive recomputation from the run files.

### Per category — BEAM-100K, reasoning effort default vs max (same harness, same judge)

- Files: `results/beam-100k-ox-alpha-effortdefault-400q.json.gz` (default effort), `results/beam-100k-ox-alpha-effortmax-400q.json` (effort=max)

| Category | binary, default effort | binary, effort=max | Δ (pp) |
|---|---:|---:|---:|
| Preference Following | 90.0% | 100.0% | +10.0 |
| Information Extraction | 87.5% | 92.5% | +5.0 |
| Instruction Following | 87.5% | 92.5% | +5.0 |
| Temporal Reasoning | 70.0% | 75.0% | +5.0 |
| Contradiction Resolution | 92.5% | 95.0% | +2.5 |
| Knowledge Update | 62.5% | 65.0% | +2.5 |
| Multi-session Reasoning | 75.0% | 77.5% | +2.5 |
| Summarization | 90.0% | 87.5% | -2.5 |
| Abstention | 65.0% | 65.0% | 0.0 |
| Event Ordering | 62.5% | 57.5% | -5.0 |

### Context from the AMB scoreboard (not a head-to-head)

These are Hindsight's runs as published in `results-manifest.json` (the AMB scoreboard). Same dataset splits and the same avg-score metric, but a different reader, a different judge and a different AMB response mode, so the gap is context, not a controlled comparison.

| Split | System (mode) | Reader | Judge | avg-score |
|---|---|---|---|---:|
| 10M | Hindsight (single-query) | Gemini 3.1 Pro preview | Gemini 2.5 Flash Lite | 64.1% |
| 100K | Hindsight (single-query) | Gemini 3.1 Pro preview | Gemini 2.5 Flash Lite | 73.4% |
| 100K | Hindsight (rag) | not recorded in the manifest | not recorded in the manifest | 86.2% |

### Scales

- Measured: BEAM-100K (2026-08-23) and BEAM-10M (2026-08-24).
- Not run yet: BEAM-500K and BEAM-1M.

---

## Reproducing

ValorBrain is a hosted memory API. To reproduce our results:

1. **Create a ValorBrain account** at [valorbrain.valor.digital](https://valorbrain.valor.digital)
2. **Get your API key** from the dashboard
3. **Install the AMB** with our provider (see [REPRODUCE.md](REPRODUCE.md))
4. **Run the benchmark** with your API key
5. **Recompute the numbers** with `python3 scripts/beam_summary.py`

## What's in this repo

- `src/memory_bench/memory/valorbrain.py` — ValorBrain memory provider for AMB
- `src/memory_bench/llm/glm.py` — GLM-5.2 LLM provider (any OpenAI-compatible endpoint)
- `src/memory_bench/llm/gateway.py` — Generic OpenAI-compatible LLM provider
- `src/memory_bench/llm/agy_direct.py` — Direct AGY CLI caller (Gemini via Antigravity)
- `results/` — the run files every number above is computed from
- `scripts/beam_summary.py` — recomputes avg-score, binary, CIs and per-category tables from `results/`
- `tests/test_readme_metrics.py` — fails when this README mixes metrics or drifts from `results/`
- `scripts/run-with-checkpoint.py` — Checkpoint runner (resume after interruptions)
- `REPRODUCE.md` — Step-by-step reproduction guide

Run the README check with `python3 -m unittest discover -s tests -v` (standard library only).

## License

MIT for the provider code. Benchmark results are public domain.

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
