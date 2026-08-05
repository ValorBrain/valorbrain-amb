# ValorBrain × Agent Memory Benchmark

Reproducible BEAM-100K results for [ValorBrain](https://valorbrain.valor.digital) using the [Agent Memory Benchmark (AMB)](https://github.com/vectorize-io/open-memory-benchmark).

## Results

| Dataset | Score | Queries | Answer LLM | Judge LLM |
|---------|-------|---------|------------|-----------|
| BEAM-100K | **70.9%** | 400 (20 conv) | GLM-5.2 | Gemini 3.6 Flash |

Full 20-conversation, 400-query evaluation. 5 of 10 category wins vs Hindsight (single-query, 73.4%).

## Per-category (400 queries)

| Category | ValorBrain | Hindsight (sq) | Gap |
|----------|-----------|----------------|-----|
| Information Extraction | **88.9%** | 64.9% | +24.0 |
| Temporal Reasoning | **71.9%** | 57.5% | +14.4 |
| Knowledge Update | **70.6%** | 58.8% | +11.8 |
| Multi-session Reasoning | **57.2%** | 47.4% | +9.8 |
| Contradiction Resolution | **67.2%** | 61.6% | +5.6 |
| Preference Following | 91.0% | 95.0% | -4.0 |
| Instruction Following | 83.8% | 91.2% | -7.5 |
| Summarization | 64.5% | 79.3% | -14.8 |
| Event Ordering | 46.4% | 80.5% | -34.1 |
| Abstention | 67.5% | 97.5% | -30.0 |

## Key finding

58% of our total improvement came from reader-agnostic memory engineering. Consolidation, timeline, and delivery improvements that benefit any LLM. The reader swap contributed 42%.

Full analysis: [Memory Quality Beats Reader Quality](https://valorbrain.valor.digital/research/memory-quality-beats-reader-quality-beam-100k)

## Reproducing

ValorBrain is a hosted memory API. To reproduce our results:

1. **Create a ValorBrain account** at [valorbrain.valor.digital](https://valorbrain.valor.digital)
2. **Get your API key** from the dashboard
3. **Install the AMB** with our provider (see [REPRODUCE.md](REPRODUCE.md))
4. **Run the benchmark** with your API key

This is the same model as Hindsight and mem0-cloud: proprietary engine, public API, reproducible results.

## What's in this repo

- `provider/valorbrain.py` — ValorBrain memory provider for AMB
- `provider/glm.py` — GLM-5.2 LLM provider (any OpenAI-compatible endpoint)
- `provider/gateway.py` — Generic OpenAI-compatible LLM provider
- `provider/agy_direct.py` — Direct AGY CLI caller (Gemini via Antigravity)
- `results/beam-100k-glm-gemini-judge-400q.json` — Official 400-query results
- `scripts/run-with-checkpoint.py` — Checkpoint runner (resume after interruptions)
- `REPRODUCE.md` — Step-by-step reproduction guide

## License

MIT for the provider code. Benchmark results are public domain.
