# Reproducing ValorBrain BEAM results

The numbers in the README come from the run files in `results/`. You can recompute them without running anything, or rerun the benchmark against your own ValorBrain account.

## Recompute the published numbers (no API keys)

```bash
python3 scripts/beam_summary.py            # avg-score, binary, 95% CIs, per-category tables
python3 -m unittest discover -s tests -v   # checks the README against results/
```

## Rerun the benchmark

### Prerequisites

1. A [ValorBrain](https://valorbrain.valor.digital) account with an API key
2. A reader LLM on any OpenAI-compatible endpoint. The published runs used GLM-5.3 Flash through OpenRouter (alias `stealth/ox-alpha`, no longer listed) at `reasoning.effort=max`, and Gemini 3.6 Flash for the baseline.
3. A judge LLM. Every published ValorBrain run was judged by GLM-5.2 (`glm:glm-5.2`). AMB's own BEAM default judge is `gemini-3.5-flash`; a different judge gives a different number, so record it with the run.

### Step 1: install this fork

The ValorBrain provider and the GLM, Gateway and AGY LLM providers are already registered in this repository (`src/memory_bench/memory/valorbrain.py`, `src/memory_bench/llm/`).

```bash
git clone https://github.com/ValorBrain/valorbrain-amb.git
cd valorbrain-amb
uv sync            # or: pip install -e .
```

### Step 2: run

```bash
# ValorBrain API (from your dashboard)
export VALORBRAIN_URL=https://valorbrain-api.valor.digital
export VALORBRAIN_TOKEN=your-valorbrain-api-key

# Reader: any OpenAI-compatible endpoint through the gateway provider
export OMB_ANSWER_LLM=gateway
export OMB_ANSWER_MODEL=your-reader-model
export OPENAI_BASE_URL=your-endpoint
export OPENAI_API_KEY=your-key

# Judge: GLM-5.2 on the Z.ai coding plan, as in the published runs
export OMB_JUDGE_LLM=glm
export OMB_JUDGE_MODEL=glm-5.2
export GLM_API_KEY=your-key

uv run amb run --dataset beam --split 100k --memory valorbrain
```

Results are saved to `outputs/beam/valorbrain/rag/100k.json`. Copy the file to `results/` and run `python3 scripts/beam_summary.py` to get the same metrics the README uses.

## Notes

- The ValorBrain provider re-windows large AMB chunks into ~8K-char windows with overlap, matching ValorBrain's production ingest behavior.
- Retrieval goes through `/api/v1/memory/prepare`, the production path: the same retrieval pipeline the product uses, plus consolidated facts and a timeline delivered via `funnel.delivered_documents`.
- For long runs (400 queries), use `scripts/run-with-checkpoint.py`, which saves after each conversation and resumes on restart.
