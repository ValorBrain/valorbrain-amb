# ValorBrain × Agent Memory Benchmark

Reproducible BEAM-100K results for [ValorBrain](https://valorbrain.valor.digital) using the [Agent Memory Benchmark (AMB)](https://github.com/vectorize-io/open-memory-benchmark).

## Results

| Dataset | Score | Answer LLM | Judge LLM |
|---------|-------|------------|-----------|
| BEAM-100K | 71.4% | GLM-5.2 | deepseek-v4-flash |

5 conversations, 100 queries. Full 20-conversation run pending.

## What's in this repo

- `provider/valorbrain.py` — ValorBrain memory provider for AMB
- `provider/glm.py` — GLM-5.2 LLM provider (Z.ai coding plan, handles reasoning_content)
- `provider/gateway.py` — Generic OpenAI-compatible LLM provider (no response_format dependency)
- `results/` — Benchmark result JSONs
- `REPRODUCE.md` — Step-by-step reproduction guide

## Key finding

58% of our total improvement (22.5 points) came from reader-agnostic memory engineering — consolidation, timeline, and delivery improvements that benefit any LLM. The reader swap (deepseek → GLM-5.2) contributed 42%.

Full analysis: [Memory Quality Beats Reader Quality](https://valorbrain.valor.digital/research/memory-quality-beats-reader-quality-beam-100k)

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

# 3. Register providers (see REPRODUCE.md)

# 4. Run
export VALORBRAIN_URL=http://localhost:7438
export VALORBRAIN_TOKEN=your-token
export GLM_API_KEY=your-zai-key
export OMB_ANSWER_LLM=glm
export OMB_JUDGE_LLM=gateway

amb run --dataset beam --split 100k --memory valorbrain --llm glm
```

## License

MIT for the provider code. Benchmark results are public domain.
