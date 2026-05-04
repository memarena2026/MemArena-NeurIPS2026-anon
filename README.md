# MemArena

An egocentric, permission-aware memory benchmark for personal-assistant LLMs.

[Paper (arXiv)](https://arxiv.org/abs/TODO) · [Dataset on Hugging Face](https://huggingface.co/datasets/zthsecondantigravity/memarena-l) · [Croissant metadata](docs/memarena-croissant.json)

MemArena contains:

- `MASim/`: multi-agent data generation.
- `eval/`: answering, retrieval, and scoring pipeline.
- `scripts/`: service launchers, evaluation runners, LLM judge, and figure reproduction.
- `memarena/figures/`: paper table and figure generators.

The hosted Hugging Face dataset contains the benchmark corpus and eval instances. It does not contain baseline result files; reproduce those by running the evaluation pipeline below.

The released MemArena-L corpus was generated with Qwen3-235B. The 5a10d and 10a5d workflows below are smaller local reproduction lines that use your SGLang endpoint by default.

## Quick Install

```bash
git clone <ANONYMOUS-GITHUB-URL>
cd MemArena

scripts/setup_venv.sh
source scripts/activate_venv.sh
python scripts/verify_install.py
```

Keep the virtual environment active for all MemArena commands. This avoids Ubuntu system-Python permission errors under `/usr/local/lib/python3.10/dist-packages`.

If `python3 -m venv` is missing on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv
```

Optional API key setup for OpenRouter judging:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-...
EOF
```

`scripts/llmjudge.py` loads `$PWD/.env` automatically. You can also export `OPENROUTER_API_KEY` in the shell.

## Smoke Test

This path uses deterministic stubs. It needs no GPU, no Docker services, and no API keys.

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

## Real Run Prerequisites

Reserve roughly **150 GB** of free disk space for Docker images, Memobase/MemOS service repos and volumes, and evaluation caches. This does not include the model checkpoint directories.

Put local model checkpoints under `~/models` by default:

```text
~/models/0_6b
~/models/llama3b
~/models/7b
~/models/8b
~/models/32b
```

Override the root with `MODEL_ROOT=/path/to/models`.

The five reader endpoints are fixed by convention:

```text
0_6b     Qwen/Qwen3-0.6B                         http://localhost:16000
llama3b  meta-llama/Llama-3.2-3B-Instruct        http://localhost:16001
7b       mistralai/Mistral-7B-Instruct-v0.3      http://localhost:16002
8b       Qwen/Qwen3-8B                           http://localhost:16003
32b      Qwen/Qwen3-32B-AWQ                      http://localhost:16004
```

`llama3b` is the 3B reader tag used by scripts.

## Start Services

Start SGLang. Pass your own image with `--image`; if omitted, the default is `lmsysorg/sglang:latest`.

```bash
MODEL_ROOT=~/models ./start_sglang_servers.sh \
    --image lmsysorg/sglang:latest \
    start 0_6b llama3b 7b 8b 32b

MODEL_ROOT=~/models ./start_sglang_servers.sh status
```

If the image is already local and you do not want Docker to pull:

```bash
PULL_IMAGE=never MODEL_ROOT=~/models ./start_sglang_servers.sh start 0_6b
```

SGLang defaults:

```text
--context-length 16384
--mem-fraction-static 0.85
--max-running-requests 64
```

The script starts requested models concurrently. GPU allocation is:

```text
0_6b     GPU 0
llama3b  GPU 1
7b       GPUs 2,3
8b       GPUs 4,5
32b      GPUs 6,7
```

Single-GPU models use `tp=1, dp=1`; multi-GPU models use pure data parallelism with `tp=1` and `dp` equal to the GPU count.

Start isolated Memobase and MemOS stacks for each reader:

```bash
scripts/setup_memory_backends.sh restart 0_6b llama3b 7b 8b 32b
```

Pinned service versions:

```text
Memobase  https://github.com/memodb-io/memobase.git  v0.0.42
MemOS     https://github.com/MemTensor/MemOS.git     v2.0.13
```

Per-reader memory endpoints:

```text
reader    Memobase                MemOS
0_6b      http://localhost:18100  http://localhost:18101
llama3b   http://localhost:18102  http://localhost:18103
7b        http://localhost:18104  http://localhost:18105
8b        http://localhost:18106  http://localhost:18107
32b       http://localhost:18108  http://localhost:18109
```

Check everything:

```bash
scripts/check_service_health.sh
```

Useful service commands:

```bash
./start_sglang_servers.sh logs 8b
./start_sglang_servers.sh stop

scripts/setup_memory_backends.sh status
scripts/setup_memory_backends.sh stop 0_6b llama3b 7b 8b 32b
```

## Evaluation Shape

The paper-style matrix is:

```text
5 readers x 5 backends x 3 trials = 75 cells
readers:  0_6b, llama3b, 7b, 8b, 32b
backends: vanilla, inmem, oracle, memobase, memos
trials:   s2, s3, s4
```

Different reader models run in parallel. Within one reader model, cells run sequentially. For `memobase` and `memos`, each cell first builds a frozen `memory_cache`, then answers from that cache. Cache files are isolated by backend, reader, and trial path; inside cache construction the memory namespace is the raw `ego_agent_id`, matching the upstream builder.

OpenRouter judge concurrency defaults to 4 in `scripts/llmjudge.py`. Answering, eval, and memory-cache concurrency default to 32 in the matrix runner.

## Workflow A: 5a10d Full Run

This is the smaller real run: 5 agents, 10 days, 5K tokens per agent per day.

### 1. Generate MASim

```bash
export MASIM_RUN_5A10D="$PWD/MASim/runs/5a10d_real"

python run_masim.py \
    --config MASim/configs/memarena_5a10d_5k.yaml \
    --sglang-url http://localhost:16000 \
    --output "$MASIM_RUN_5A10D" \
    --overwrite
```

### 2. Run The 5x5x3 Evaluation

```bash
scripts/run_eval_matrix.sh \
    --paper-full \
    --run-dir "$MASIM_RUN_5A10D" \
    --answer-concurrency 32 \
    --eval-concurrency 32 \
    --memory-cache-concurrency 32 \
    --out-dir out/accuracy_5a10d_5x5x3
```

### 3. Find Completed Cells

This scans manifests and `matrix_logs`, so partially completed matrices still count finished cells.

```bash
python scripts/summarize_eval_progress.py \
    out/accuracy_5a10d_5x5x3 \
    --paths-output eval_progress_trial_paths_5a10d.json
```

Expected full completion is `75/75`.

### 4. Run LLM-as-a-Judge

This judges completed answer files that do not already have a real LLM judge result.

```bash
python scripts/llmjudge.py \
    --from-progress-json eval_progress_trial_paths_5a10d.json \
    --run-dir "$MASIM_RUN_5A10D" \
    --judge-preset openrouter \
    --judge-model openai/gpt-4o-mini \
    --concurrency 4 \
    --force \
    --keep-going \
    --manifest-out out/llmjudge_5a10d_openrouter_c4_manifest.json
```

To use Qwen3-235B as judge through an OpenAI-compatible endpoint:

```bash
QWEN235B_BASE_URL=https://openrouter.ai/api/v1 \
QWEN235B_API_KEY="$OPENROUTER_API_KEY" \
python scripts/llmjudge.py \
    --from-progress-json eval_progress_trial_paths_5a10d.json \
    --run-dir "$MASIM_RUN_5A10D" \
    --judge-preset qwen235b \
    --concurrency 4 \
    --force \
    --keep-going \
    --manifest-out out/llmjudge_5a10d_qwen235b_c4_manifest.json
```

### 5. Stage Paper Inputs

Staging requires real LLM-judged files by default, so token-F1 fallback files are not used accidentally.

```bash
python scripts/stage_paper_accuracy_inputs.py \
    --progress-json eval_progress_trial_paths_5a10d.json \
    --masim-run "$MASIM_RUN_5A10D" \
    --out-dir out/paper_accuracy_5a10d_input \
    --force
```

### 6. Generate Figures And Tables

```bash
MEMARENA_RUN_DIR="$PWD/out/paper_accuracy_5a10d_input" \
python scripts/reproduce_figures.py --all
```

Outputs are written under:

```text
out/paper_accuracy_5a10d_input/tables/
out/paper_accuracy_5a10d_input/figures/
```

Use `python scripts/reproduce_figures.py --all --out-dir out/custom_artifacts`
to override the artifact destination explicitly.
If neither `MEMARENA_RUN_DIR` nor `--out-dir` is set, artifacts go under
`out/reproduced_figures/` rather than the source tree.

`fig_finding2` reads the staged input from `MEMARENA_RUN_DIR` by default. To reproduce the archived paper-era hardcoded version instead, add `MEMARENA_FINDING2_SOURCE=published`.

## Workflow B: 10a5d Full Run

This is an independent full workflow: 10 agents, 5 days, 5K tokens per agent per day. It uses its own MASim directory, eval output directory, progress JSON, judge manifest, and staged figure input directory.

### 1. Generate MASim

```bash
export MASIM_RUN_10A5D="$PWD/MASim/runs/10a5d_real"

python run_masim.py \
    --config MASim/configs/memarena_10a5d_5k.yaml \
    --sglang-url http://localhost:16000 \
    --output "$MASIM_RUN_10A5D" \
    --overwrite
```

### 2. Run The 5x5x3 Evaluation

```bash
scripts/run_eval_matrix.sh \
    --paper-full \
    --run-dir "$MASIM_RUN_10A5D" \
    --answer-concurrency 32 \
    --eval-concurrency 32 \
    --memory-cache-concurrency 32 \
    --out-dir out/accuracy_10a5d_5x5x3
```

### 3. Find Completed Cells

```bash
python scripts/summarize_eval_progress.py \
    out/accuracy_10a5d_5x5x3 \
    --paths-output eval_progress_trial_paths_10a5d.json
```

### 4. Run LLM-as-a-Judge

```bash
python scripts/llmjudge.py \
    --from-progress-json eval_progress_trial_paths_10a5d.json \
    --run-dir "$MASIM_RUN_10A5D" \
    --judge-preset openrouter \
    --judge-model openai/gpt-4o-mini \
    --concurrency 4 \
    --force \
    --keep-going \
    --manifest-out out/llmjudge_10a5d_openrouter_c4_manifest.json
```

### 5. Stage Paper Inputs

```bash
python scripts/stage_paper_accuracy_inputs.py \
    --progress-json eval_progress_trial_paths_10a5d.json \
    --masim-run "$MASIM_RUN_10A5D" \
    --out-dir out/paper_accuracy_10a5d_input \
    --force
```

### 6. Generate Figures And Tables

```bash
MEMARENA_RUN_DIR="$PWD/out/paper_accuracy_10a5d_input" \
python scripts/reproduce_figures.py --all
```

Outputs are written under `out/paper_accuracy_10a5d_input/{figures,tables}/`.

## Quick Single-Reader Evaluation

Use this to debug Qwen3-0.6B with only `vanilla` and `oracle`.

```bash
export MASIM_RUN_5A10D="$PWD/MASim/runs/5a10d_real"

scripts/run_eval_matrix.sh \
    --run-dir "$MASIM_RUN_5A10D" \
    --models-file config/eval_matrix_qwen3_0_6b_models.tsv \
    --backends vanilla,oracle \
    --trials s2 \
    --judge-preset remote \
    --answer-concurrency 32 \
    --eval-concurrency 32 \
    --out-dir out/accuracy_5a10d_qwen0_6b_vanilla_oracle
```

For shape-only testing, add `--test --qa-limit 1 --message-limit 20`.

## Progress And Logs

Matrix progress:

```bash
python scripts/summarize_eval_progress.py out --paths-output eval_progress_trial_paths.json
```

Cell logs:

```bash
OUT=out/accuracy_5a10d_5x5x3
tail -n 200 "$OUT/matrix_logs/32b/memos_s3.log"
```

Matrix manifest:

```bash
cat out/accuracy_5a10d_5x5x3/run_eval_matrix_manifest.tsv
```

SGLang logs:

```bash
./start_sglang_servers.sh logs 32b
```

## Latency Checks

Single reader:

```bash
python scripts/run_latency.py \
    --backend vanilla \
    --sglang-url http://localhost:16000 \
    --model-name Qwen/Qwen3-0.6B \
    --n 100 \
    --out-dir out/test_latency_qwen0_6b
```

All five readers:

```bash
while IFS='|' read -r model_tag model_name endpoint _memobase_url _memos_url; do
    case "$model_tag" in ""|\#*) continue ;; esac
    python scripts/run_latency.py \
        --backend vanilla \
        --sglang-url "$endpoint" \
        --model-name "$model_name" \
        --n 100 \
        --out-dir "out/test_latency_${model_tag}"
done < config/eval_matrix_paper_models.tsv
```

## Download The Hosted Dataset

```bash
python scripts/download_dataset.py --out data/
```

Default dataset repo: `zthsecondantigravity/memarena-l`. Override with `MEMARENA_HF_REPO_ID` or `--repo-id`.

After download:

```text
data/benchmark/
├── corpus_sessions.jsonl.gz
├── agents_personas.jsonl.gz
├── agent_schedules.jsonl.gz
├── events.jsonl
├── ego_projections.json
├── graph_edges.jsonl
├── groups.json
├── locations.json
├── pipeline_report.json
└── eval_instances/
    ├── d1_conflict.jsonl
    ├── d2_anaphora.jsonl
    ├── d3_confabulation.jsonl
    ├── d4_permission.jsonl
    ├── d5_cloze.jsonl
    ├── d6_metadata.jsonl
    ├── d7_qa.jsonl
    ├── d8_temporal.jsonl
    └── d10_counterfactual.jsonl
```

## Human-Judge Calibration

```bash
python memarena/human_calibration/server.py --port 8080
# open http://localhost:8080/
python -m memarena.human_calibration.analyze_judge_human --labels labels.json
```

## Dataset Quality Tools

```bash
python -m memarena.tools.scan_corpus_non_ascii path/to/corpus.jsonl
python -m memarena.tools.validate_croissant docs/memarena-croissant.json
python -m memarena.tools.bundle_memarena --version 1.0.0 --out out/bundle/
```

## License

- Code: [MIT](LICENSE)
- Dataset: [CC BY 4.0](docs/LICENSE)
