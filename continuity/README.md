# Continuity benchmark

Long-conversation benchmarks (LoCoMo, BEAM, LongMemEval) ask whether a memory system can find a fact. They don't ask whether it hands an agent the right fact *at the right moment*: the current decision rather than the replaced one, the rule still in force rather than the expired one, the handoff still open rather than the finished one, and only what the person asking is allowed to read.

That lifecycle is the part of agent memory that plain vector RAG doesn't do. This suite measures it. It is open: the data generator, harness, scoring and raw outputs live here, and anyone can run it against their own ValorBrain engine.

## Scenarios

Each question has one gold answer: a unique code shaped `WORD-123`, or `NONE`. It may also list **forbidden** codes, meaning the stale value that must not be served.

| Scenario | What is written | What is asked | Gold | Forbidden |
|---|---|---|---|---|
| **Supersession**, `implicit` | 2–3 versions of a decision, each declaring `supersedes` on the previous one. The text carries no date or "replaces" wording | "What code does X use now?" | latest version | earlier versions |
| **Supersession**, `explicit` | same, but each text says "updated decision, replaces the previous one" and carries a date | same | latest | earlier |
| **Supersession**, `control` | one decision, never revised | same | that version | — |
| **Expired rule**, `implicit-{expired,valid}` | a rule as a keyed fact with `as_of` 40 days ago and `valid_until` 10 days ago (expired) or 90 days ahead (valid). The text has no dates | "Is the rule in force today? Which code does it allow?" | `NONE` / the code | the expired code |
| **Expired rule**, `explicit-{expired,valid}` | same, with "in force from … until …" in the text | same | same | same |
| **Handoff**, `topic` / `generic` | the agent hands off to its next session, consumes that handoff, then hands off a newer one | "What is the next step still pending for me?" (naming the task / not naming it) | the open handoff's code | the consumed handoff's code |
| **Handoff**, `none` | the newer handoff is consumed too | "Is anything still waiting for me?" | `NONE` | both codes |
| **Permission**, `private-other` | Alice stores a `visibility: private` note | Bob asks for it | `NONE` | Alice's code |
| **Permission**, `private-owner` | same | Alice asks | the code | — |
| **Permission**, `shared` | Alice stores a tenant-visible decision | Bob asks | the code | — |
| **Permission**, `agent-default` | the workspace agent stores an observation with no visibility argument | Bob asks | the code | — |

The `agent-default` cell checks that agent writes stay readable by the team. The harness also records, as a property, the `sensitivity` each agent write was stored with. The product decision is that agent writes default to `internal`, and the property shows whether the engine you ran does that.

Dates are stored as day offsets and rendered against the run date. The same seed therefore produces the same timeline on any day. `data/continuity-v1.json` (`--per-cell 10`, 140 questions) is the reference dataset, and `data/continuity-v1-n3.json` (`--per-cell 3`, 42 questions) is the small one we ran first. A test checks that both committed files match the generator.

## Systems compared

| System | What the reader gets |
|---|---|
| `valorbrain` | The documented product flow. MCP `memory_prepare` once at session start (identity, pending handoffs, keyed facts in force, relevant memory), then MCP `memory_retrieve` for each question. Both contexts are concatenated |
| `valorbrain-prepare` | `memory_prepare` alone, called with the question |
| `valorbrain-retrieve` | `memory_retrieve` alone |
| `rag` | Plain vector RAG over **the same texts** written to ValorBrain. `BAAI/bge-small-en-v1.5` embeddings (fastembed), exact cosine, top 8, one index per workspace |
| `rag-per-user` | Same as `rag`, but with one index per writer. The asker only sees what they wrote themselves. This is the naive way to add privacy to RAG. Scored on permission questions only |

The RAG baselines see every text. They don't see the lifecycle metadata (`supersedes`, `valid_until`, handoff status, visibility), and that metadata is the thing being tested.

Every read uses the asking persona's own MCP token. Agent-workspace writes use the REST surfaces an integration calls: `POST /documents` with `supersedes`, `POST /api/v1/keyed-facts` with `valid_until`, and `/api/v1/team/handoffs` plus `/consume`. Permission writes use MCP `memory_store` with each person's token.

## Scoring

The judge is **deterministic exact match, with no LLM judge.**

- **Answer accuracy.** One reader call per (system, question), with the same prompt for every system (`results/*/reader-prompt.txt`) at temperature 0. The reader returns one code or `NONE`, and the answer counts as correct when it equals the gold value.
- **Context accuracy.** This one doesn't depend on the reader. The delivered context must contain the gold code (when there is one) and none of the forbidden codes. It measures what the memory hands over before any model reasons about it.
- **Leaks.** Permission questions where the other person's private code appears anywhere in the context.

Every cell reports k/n pooled over runs, with a 95% Wilson interval. Per-run rates are in `summary.json`. Every run uses fresh workspaces.

## Reproduce

You need Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), a ValorBrain engine you control, and any OpenAI-compatible chat endpoint for the reader. Don't point this at a production engine. It creates workspaces and writes data.

```bash
git clone https://github.com/ValorBrain/valorbrain-amb && cd valorbrain-amb/continuity
uv run --extra test pytest -q tests                      # generator + scoring checks
uv run continuity-bench generate --per-cell 3 --out data/continuity-v1-n3.json   # must reproduce the committed sha256

export VALORBRAIN_URL=http://127.0.0.1:7439              # engine REST base
export VALORBRAIN_MCP_URL=http://127.0.0.1:7439/mcp      # same engine, MCP endpoint
export ENGINE_ADMIN_SECRET=...                           # creates the two-person workspace
export VALORBRAIN_PLATFORM_TOKEN=...                     # engine VALORBRAIN_API_TOKEN: mints one MCP token per person
export CB_READER_BASE_URL=https://openrouter.ai/api/v1   # any OpenAI-compatible endpoint
export CB_READER_MODEL=deepseek/deepseek-chat CB_READER_API_KEY=...

uv run continuity-bench run --dataset data/continuity-v1-n3.json --runs 3 --out results/my-run
uv run continuity-bench report --results results/my-run   # writes summary.md + summary.json
```

- `--no-reader` skips the reader and reports context accuracy only, with zero LLM calls.
- `--agent-mode signup` (the default) creates the agent workspace through the public self-service signup (`POST /api/v1/agents/signup`, limited to 5 per network per day). `--agent-mode tokens` uses an existing workspace from `CB_AGENT_TENANT_ID`, `CB_AGENT_MCP_TOKEN` and `CB_AGENT_REST_KEY`.
- `--team-mode tokens` uses an existing two-person workspace from `CB_TEAM_TENANT_ID` and `CB_{ALICE,BOB,AGENT}_MCP_TOKEN`. Use it when you don't hold the operator secrets, for example on a hosted account where an admin minted the three tokens. `--team-mode skip` drops the permission scenarios.

Each run writes `run-N.jsonl.gz`, which holds every question, system, delivered context, reader output and score, plus `run-N.meta.json`, which records the engine git hash, dataset sha256, served reader model and observed properties.

## Results

See [`results/`](results/). Every result directory holds its own `summary.md` with N, date, engine version, reader and judge.
