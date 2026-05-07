# MemArena

MemArena is an ego-centric conversational benchmark for on-device personal-memory assistants. It evaluates whether an assistant can recover what a user observed, reason across a coherent multi-session social world, and withhold information under abstention and permission constraints.

This repository contains the simulator, evaluation pipeline, memory-backend adapters, experiment runners, dataset metadata, and result-reproduction utilities for the anonymous submission:

**MemArena: An Ego-Centric Conversational Benchmark for On-Device Agentic Personal Memory Assistants**

The hosted dataset link for the anonymous release should be configured through `MEMARENA_HF_REPO_ID` or passed to `scripts/download_dataset.py --repo-id`. This is the code and metadata repository; paper source is intentionally not included.

## What Is Released

MemArena-L is the headline benchmark split used in the paper:

| Property | Value |
|---|---:|
| Simulated agents | 50 |
| Simulated horizon | 15 days |
| Dialog-text tokens | 10.3M |
| Mean ego-observed tokens | 24.1K tokens / agent / day |
| Evaluation instances | 1,579 |
| Reader models | 5 open-weight readers |
| Main memory backends | Vanilla, BM25-RAG, Oracle, Memobase, MemSearch |
| Main trials | s2, s3, s4 |

MemArena is built with `MASim`, an evaluation-aware multi-agent simulator. MASim generates one coherent social world, projects it into each user's ego-centric history, and emits evidence-linked evaluation instances. The hosted dataset contains benchmark data only, not baseline result JSON files.

## Evaluation Dimensions

The paper reports six user-facing dimensions grouped into three categories.

| Category | Dimension | What It Tests |
|---|---|---|
| Recall | D1 Cloze Fidelity | Recover blanked or next-turn dialog content. |
| Recall | D2 Metadata Completeness | Recover speaker, timestamp, and participant metadata. |
| Reasoning | D3 Factual QA | Answer standard, temporal, and counterfactual memory questions. |
| Reasoning | D4 Cross-Session Reasoning | Resolve conflicts and anaphora across sessions. |
| Trustworthiness | D5 Calibrated Abstention | Refuse unsupported claims while answering grounded ones. |
| Trustworthiness | D6 Permission-Aware Access | Respect explicit, autonomous, and identity-conditioned access rules. |

Internally, the released dataset stores nine task files under `eval_instances/` (`d1`, `d2`, `d3`, `d4`, `d5`, `d6`, `d7`, `d8`, `d10`). The evaluation code maps those internal probes into the six paper dimensions.

## Main Findings

The current paper centers on three results:

1. **Permission-aware access remains the bottleneck.** No backend-reader cell clears `F1_PU = 50` on D6. Systems split into privacy-by-amnesia when evidence is not retrieved, and anti-policy disclosure when evidence is retrieved but still revealed despite access markers.
2. **Matched evidence dominates reader scale.** Oracle evidence with Qwen3-0.6B beats every non-Oracle cell on average, while stronger retrievers and an omniscient full-context backend remain far below the Oracle ceiling on cross-session reasoning.
3. **Search overhead matters mainly at the edge.** On Qwen3-0.6B, BM25 search can be a first-order TTFT cost; on Qwen3-32B, the same search cost is a small fraction of prefill latency.

The hosted dataset contains benchmark data only; baseline result JSON files are intentionally not included in this repository.

## Repository Layout

| Path | Purpose |
|---|---|
| [`MASim/`](MASim/) | Multi-agent world, persona, schedule, dialog, and ground-truth generation. |
| [`eval/`](eval/) | Evaluation CLI, scoring, answering, and memory-backend adapters. |
| [`scripts/`](scripts/) | Install helpers, data downloaders, service launchers, experiment runners, rejudge tools, latency scripts, and reproduction helpers. See [`scripts/README.md`](scripts/README.md). |
| [`memarena/figures/`](memarena/figures/) | Table and figure artifact generators for result reproduction. |
| [`docs/`](docs/) | Dataset card, Croissant metadata, citation metadata, and setup notes. |
| [`config/`](config/) | Model matrices and backend configuration. |
| [`tests/`](tests/) | Unit and smoke tests for scoring, adapters, launchers, and figure plumbing. |

## Install

```bash
git clone <ANONYMOUS-GITHUB-URL>
cd MemArena

scripts/setup_venv.sh
source scripts/activate_venv.sh
python scripts/verify_install.py
```

If Ubuntu is missing the venv package:

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv
```

Optional judge key setup:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-...
EOF
```

`scripts/llmjudge.py`, D6 rejudge scripts, and most sweep wrappers load `.env` from the repository root.

## Download The Dataset

```bash
python scripts/download_dataset.py --repo-id <ANONYMOUS-HF-DATASET-ID> --out data/
```

Or set:

```bash
export MEMARENA_HF_REPO_ID=<ANONYMOUS-HF-DATASET-ID>
python scripts/download_dataset.py --out data/
```

Expected local layout:

```text
data/benchmark/
|-- corpus_sessions.jsonl.gz
|-- agents_personas.jsonl.gz
|-- agent_schedules.jsonl.gz
|-- events.jsonl
|-- ego_projections.json
|-- graph_edges.jsonl
|-- groups.json
|-- locations.json
|-- pipeline_report.json
`-- eval_instances/
    |-- d1_conflict.jsonl
    |-- d2_anaphora.jsonl
    |-- d3_confabulation.jsonl
    |-- d4_permission.jsonl
    |-- d5_cloze.jsonl
    |-- d6_metadata.jsonl
    |-- d7_qa.jsonl
    |-- d8_temporal.jsonl
    `-- d10_counterfactual.jsonl
```

The dataset is synthetic and contains no real personal information.

## Smoke Test

This path uses deterministic stubs. It needs no GPU, Docker service, dataset download, or API key.

```bash
python run_masim.py --smoke --output out/smoke/masim --overwrite

python scripts/run_accuracy.py \
    --dry-run \
    --backend vanilla \
    --n 10 \
    --out-dir out/smoke/accuracy_vanilla

python scripts/run_latency.py \
    --dry-run \
    --backend vanilla \
    --n 10 \
    --out-dir out/smoke/latency_vanilla

python -m pytest tests/test_accuracy.py tests/test_latency.py -q
```

## Model And Service Conventions

Main paper readers:

| Tag | Model | Default endpoint |
|---|---|---|
| `0_6b` | Qwen/Qwen3-0.6B | `http://localhost:16000` |
| `llama3b` | meta-llama/Llama-3.2-3B-Instruct | `http://localhost:16001` |
| `7b` | mistralai/Mistral-7B-Instruct-v0.3 | `http://localhost:16002` |
| `8b` | Qwen/Qwen3-8B | `http://localhost:16003` |
| `32b` | Qwen/Qwen3-32B-AWQ | `http://localhost:16004` |

Put checkpoints under `~/models` by default:

```text
~/models/0_6b
~/models/llama3b
~/models/7b
~/models/8b
~/models/32b
```

Start SGLang:

```bash
MODEL_ROOT=~/models ./start_sglang_servers.sh \
    --image lmsysorg/sglang:latest \
    start 0_6b llama3b 7b 8b 32b

MODEL_ROOT=~/models ./start_sglang_servers.sh status
```

Start isolated memory-backend services:

```bash
scripts/setup_memory_backends.sh restart 0_6b llama3b 7b 8b 32b
scripts/check_service_health.sh
```

Useful service commands:

```bash
./start_sglang_servers.sh logs 8b
./start_sglang_servers.sh stop

scripts/setup_memory_backends.sh status
scripts/setup_memory_backends.sh stop 0_6b llama3b 7b 8b 32b
```

## Main Accuracy Evaluation

The paper-style grid is:

```text
5 readers x 5 backends x 3 seeds = 75 cells
readers:  0_6b, llama3b, 7b, 8b, 32b
backends: vanilla, inmem, oracle, memobase, memsearch
trials:   s2, s3, s4
```

For the downloaded MemArena-L dataset:

```bash
scripts/run_eval_matrix.sh \
    --paper-full \
    --run-dir data/benchmark \
    --models-file config/eval_matrix_paper_models.tsv \
    --backends vanilla,inmem,oracle,memobase,memsearch \
    --trials s2,s3,s4 \
    --answer-concurrency 32 \
    --eval-concurrency 32 \
    --memory-cache-concurrency 32 \
    --out-dir out/accuracy_memarena_l
```

For a quick single-reader debug run:

```bash
scripts/run_eval_matrix.sh \
    --run-dir data/benchmark \
    --models-file config/eval_matrix_qwen3_0_6b_models.tsv \
    --backends vanilla,oracle \
    --trials s2 \
    --test \
    --qa-limit 1 \
    --message-limit 20 \
    --out-dir out/debug_qwen0_6b
```

## Judging And D6 Rejudging

Run LLM-as-a-judge over completed answer files:

```bash
python scripts/summarize_eval_progress.py \
    out/accuracy_memarena_l \
    --paths-output out/eval_progress_trial_paths.json

python scripts/llmjudge.py \
    --from-progress-json out/eval_progress_trial_paths.json \
    --run-dir data/benchmark \
    --judge-preset openrouter \
    --judge-model openai/gpt-4o-mini \
    --concurrency 4 \
    --force \
    --keep-going \
    --manifest-out out/llmjudge_manifest.json
```

The current D6 pipeline uses a 5-label rubric and `F1_PU` aggregation. Use:

```bash
python scripts/rerun_d6_judge.py --workers 64
python scripts/judge_d6_self_probe.py --input-root out/d6_self_probe --workers 32
python scripts/rerun_d6_judge_ablations.py --root out/ablations_l_example --workers 64
```

## Ablations And Latency

Common paper ablation entry points:

```bash
bash scripts/run_d6_self_probe.sh
bash scripts/run_step5_oracle_gated.sh
bash scripts/run_step6_memobase_writer32.sh
bash scripts/run_step7_temporal_sweep.sh
bash scripts/run_step9_full_panel.sh
bash scripts/run_g3a_norm_binding.sh
bash scripts/run_b5_topk_sweep_v2.sh
```

Latency replay and appendix tables:

```bash
python scripts/select_latency_queries.py --data-dir data/benchmark --out out/latency_selection/queries_s2.json
bash scripts/run_latency_answer.sh
bash scripts/run_latency_answer_sweep.sh
bash scripts/run_latency_memobase_ingest.sh
bash scripts/run_latency_memsearch_ingest.sh
python scripts/fit_latency_model.py
python scripts/gen_ttft_table.py
python scripts/gen_latency_appendix_tables.py
```

Most result-producing scripts write to `out/`. This repository does not include baseline result JSON files.

## Figures And Tables

Table/figure artifact generators live in `memarena/figures/` and are dispatched by:

```bash
python scripts/reproduce_figures.py --list
python scripts/reproduce_figures.py --all --out-dir out/reproduced_figures
```

If you have staged accuracy inputs:

```bash
MEMARENA_RUN_DIR="$PWD/out/paper_accuracy_input" \
python scripts/reproduce_figures.py --all
```

## MASim Generation

The released MemArena-L corpus was generated with a large generator model. Smaller local reproduction configs can be run with a local OpenAI-compatible endpoint:

```bash
python run_masim.py \
    --config MASim/configs/memarena_5a10d_5k.yaml \
    --sglang-url http://localhost:16000 \
    --output out/masim_5a10d \
    --overwrite
```

For production-scale generation, inspect `MASim/configs/` and `MASim/README.md`.

## Script Catalogue

Every script under `scripts/` is summarized in [`scripts/README.md`](scripts/README.md). Start there when choosing an entry point; many files are specialized paper-run wrappers and assume existing `out/` artifacts.

## Dataset Quality Tools

```bash
python -m memarena.tools.scan_corpus_non_ascii path/to/corpus.jsonl
python -m memarena.tools.validate_croissant docs/memarena-croissant.json
python -m memarena.tools.bundle_memarena --version 1.0.0 --out out/bundle/
```

## License

- Code: [MIT](LICENSE)
- Dataset: [CC BY 4.0](docs/LICENSE)
