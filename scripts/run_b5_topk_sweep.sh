#!/usr/bin/env bash
# B5 top-k sweep: 20 cells (5 readers × 4 top-k {4,8,16,32} × BM25 × s2)
# Uses run_eval_matrix.sh (run_accuracy.py path), same as run_l_ablations.sh.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[b5-topk $(date -u +%H:%M:%S)] $*"; }

declare -A PORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004)
declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
READERS=(0_6b llama3b 7b 8b 32b)
TOPKS=(4 8 16 32)
OUT_ROOT="out/b5_topk_sweep"
RUN_DIR="data/benchmark"
JUDGE_MODEL="openai/gpt-4o-mini-2024-07-18"
JUDGE_ENDPOINT="https://openrouter.ai/api/v1"

log "=== B5 top-k sweep: 20 cells (5 readers × 4 top-k × BM25 × s2) ==="

run_reader_allk() {
  local reader="$1"
  local port="${PORT[$reader]}"
  local mname="${MODEL_NAME[$reader]}"

  for k in "${TOPKS[@]}"; do
    local out_dir="${OUT_ROOT}/${reader}/k${k}"
    mkdir -p "${out_dir}/_models_files" "${out_dir}/wrapper_logs"

    # models-file TSV: model_tag|model_name|endpoint
    printf '%s|%s|http://localhost:%s\n' \
      "$reader" "$mname" "$port" > "${out_dir}/_models_files/${reader}.tsv"

    local lf="${out_dir}/wrapper_logs/b5_${reader}_k${k}_s2.log"
    log "[$reader] k=$k → $out_dir"

    bash "$SCRIPT_DIR/run_eval_matrix.sh" \
      --run-dir "$RUN_DIR" \
      --out-dir "$out_dir" \
      --models-file "${out_dir}/_models_files/${reader}.tsv" \
      --backends "inmem_text_sessions" \
      --trials "s2" \
      --top-k "$k" \
      --judge-preset remote \
      --remote-judge-model "$JUDGE_MODEL" \
      --remote-judge-endpoint "$JUDGE_ENDPOINT" \
      --judge-api-key "$OPENROUTER_API_KEY" \
      --answer-concurrency 64 \
      --force \
      > "$lf" 2>&1

    local rc=$?
    if [[ $rc -eq 0 ]]; then
      log "[$reader] k=$k done"
    else
      log "[$reader] k=$k FAILED (rc=$rc)"
    fi
  done
  log "[$reader] all 4 k-values done"
}

# Launch all 5 readers in parallel (each runs k=4,8,16,32 sequentially)
declare -A PIDS
for reader in "${READERS[@]}"; do
  run_reader_allk "$reader" &
  PIDS[$reader]=$!
  log "  spawned $reader (PID=${PIDS[$reader]})"
done

log "Waiting for all 5 reader workers..."
ok=0; fail=0
for reader in "${READERS[@]}"; do
  if wait "${PIDS[$reader]}"; then
    log "  ✓ $reader"; ok=$((ok+1))
  else
    log "  ✗ $reader (rc=$?)"; fail=$((fail+1))
  fi
done
log "Inference done: ok=$ok fail=$fail / 5 readers"

# LLM judge: run_accuracy.py only does answering; judging is a separate step
log "=== LLM judge (llmjudge.py) — scoring all 20 answer files ==="
judge_pids=()
for reader in "${READERS[@]}"; do
  for k in "${TOPKS[@]}"; do
    ans="${OUT_ROOT}/${reader}/k${k}/eval_results_s2/inmem_text_sessions/answer_results_inmem_text_sessions_${reader}_s2.json"
    if [[ ! -f "$ans" ]]; then
      log "  WARN: missing $ans — skipping judge"
      continue
    fi
    .venv/bin/python scripts/llmjudge.py \
      --run-dir "$RUN_DIR" \
      --answer-path "$ans" \
      --judge-preset remote \
      --judge-model "$JUDGE_MODEL" \
      --judge-endpoint "$JUDGE_ENDPOINT" \
      --judge-api-key "$OPENROUTER_API_KEY" \
      --concurrency 512 --force \
      > "/tmp/b5_judge_${reader}_k${k}.log" 2>&1 &
    judge_pids+=($!)
    log "  judging $reader k$k (PID=$!)"
  done
done
for p in "${judge_pids[@]}"; do wait "$p" || true; done
log "LLM judge done"

# D6 rejudge (5-label policy_category / rationale_v2)
log "=== D6 rejudge ==="
.venv/bin/python scripts/rerun_d6_judge_ablations.py \
  --root "$OUT_ROOT" --workers 256 2>&1 | tee /tmp/b5_d6_rejudge.log
log "D6 rejudge done"

# Append 20 rows to experiments_index.csv
log "=== Updating experiments_index.csv ==="
declare -A FULLMODEL=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
for reader in "${READERS[@]}"; do
  for k in "${TOPKS[@]}"; do
    ns="inmem_text_sessions_${reader}_s2_judge_remote"
    path="${OUT_ROOT}/${reader}/k${k}/eval_results_s2/inmem_text_sessions/evaluation_results_${ns}.json"
    echo "ablation,b5_topk_sweep,inmem_text_sessions_k${k},${reader},${FULLMODEL[$reader]},s2,1002,${path}"
  done
done >> experiments_index.csv
log "experiments_index.csv updated (+20 rows)"

log "Outputs are left under $OUT_ROOT and experiments_index.csv was updated; no repository staging, commit, or push is performed."
log "=== B5 top-k sweep DONE ==="
