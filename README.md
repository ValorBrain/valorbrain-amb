# ValorBrain × Agent Memory Benchmark

Reproducible BEAM-100K results for [ValorBrain](https://valorbrain.valor.digital) using the [Agent Memory Benchmark (AMB)](https://github.com/vectorize-io/open-memory-benchmark).

## Results

| Dataset | Score | Queries | Answer LLM | Judge LLM |
|---------|-------|---------|------------|-----------|
| BEAM-100K | **70.9%** | 400 (20 conv) | GLM-5.2 | Gemini 3.6 Flash |

Full 20-conversation, 400-query evaluation. 5 of 10 category wins vs Hindsight (single-query, 73.4%).

## Key finding

58% of our total improvement (22.5 points) came from reader-agnostic memory engineering. Consolidation, timeline, and delivery improvements that benefit any LLM. The reader swap (deepseek to GLM-5.2) contributed 42%.

Full analysis: [Memory Quality Beats Reader Quality](https://valorbrain.valor.digital/research/memory-quality-beats-reader-quality-beam-100k)

## Per-category (400 queries)

| Category | ValorBrain | Hindsight (sq) | Gap |
|----------|-----------|----------------|-----|
| Information Extraction | **88.9%** | 64.9% | +24.0 |
| Knowledge Update | **70.6%** | 58.8% | +11.8 |
| Multi-session Reasoning | **57.2%** | 47.4% | +9.8 |
| Temporal Reasoning | **71.9%** | 57.5% | +14.4 |
| Contradiction Resolution | **67.2%** | 61.6% | +5.6 |
| Preference Following | 91.0% | 95.0% | -4.0 |
| Instruction Following | 83.8% | 91.2% | -7.5 |
| Summarization | 64.5% | 79.3% | -14.8 |
| Event Ordering | 46.4% | 80.5% | -34.1 |
| Abstention | 67.5% | 97.5% | -30.0 |

## What's in this repo

- `provider/valorbrain.py` — ValorBrain memory provider for AMB
- `provider/glm.py` — GLM-5.2 LLM provider (any OpenAI-compatible endpoint, handles reasoning_content)
- `provider/gateway.py` — Generic OpenAI-compatible LLM provider (no response_format dependency)
- `provider/agy_direct.py` — Direct AGY CLI caller (Gemini models via Google Antigravity)
- `results/beam-100k-glm-gemini-judge-400q.json` — Official 400-query results
- `REPRODUCE.md` — Step-by-step reproduction guide

## Quick start

```bash
# 1. Install AMB
git clone https://github.com/vectorize-io/open-memory-benchmark.git
cd open-memory-benchmark
pip install -e .

# 2. Copy provider files
cp /path/to/valorbrain-amb/provider/*.py src/memory_bench/memory/
cp /path/to/valorbrain-amb/provider/glm.py src/memory_bench/llm/
cp /path/to/valorbrain-amb/provider/gateway.py src/memory_bench/llm/
cp /path/to/valorbrain-amb/provider/agy_direct.py src/memory_bench/llm/

# 3. Register providers (see REPRODUCE.md)

# 4. Configure your LLM endpoints
export VALORBRAIN_URL=http://localhost:7438
export VALORBRAIN_TOKEN=your-token
export GLM_BASE_URL=your-glm-endpoint
export GLM_API_KEY=your-key
export OMB_ANSWER_LLM=glm
export OMB_JUDGE_LLM=gateway

# 5. Run
amb run --dataset beam --split 100k --memory valorbrain --llm glm
```

## License

MIT for the provider code. Benchmark results are public domain.
