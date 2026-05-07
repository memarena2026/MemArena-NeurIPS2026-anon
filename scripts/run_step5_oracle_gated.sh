#!/usr/bin/env bash
# Step 5: A7 oracle_gated sweep — 15 cells (5 readers × 3 trials)
# Runs after D6 self-probe pipeline (Steps 1-4) is complete.
#
# GPU layout (reuses all 5 sglang servers from Phase 3+4):
#   GPU 1: sglang-0_6b  GPU 2: sglang-llama3b  GPU 3: sglang-7b
#   GPU 4: sglang-8b    GPU 5,6: sglang-32b (TP=2)
#
# Output: out/d6_gating/oracle_gated_<model>/eval_results_<trial>/oracle_gated/

set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[step5-gating $(date -u +%H:%M:%S)] $*"; }

declare -A MPORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004)
declare -A MNAME=([0_6b]="Qwen/Qwen3-0.6B" [llama3b]="meta-llama/Llama-3.2-3B-Instruct" [7b]="mistralai/Mistral-7B-Instruct-v0.3" [8b]="Qwen/Qwen3-8B" [32b]="Qwen/Qwen3-32B-AWQ")
declare -A MGPU=([0_6b]="1" [llama3b]="2" [7b]="3" [8b]="4" [32b]="5,6")
declare -A MTP=([0_6b]=1 [llama3b]=1 [7b]=1 [8b]=1 [32b]=2)
TRIALS=(s2 s3 s4)
declare -A SEED=([s2]=1002 [s3]=1003 [s4]=1004)

# ── Step 1: start 5 sglang servers ───────────────────────────────────────────
log "=== starting 5 sglang servers ==="
docker ps -aq --filter 'name=memarena-sglang-' | xargs -r docker rm -f >/dev/null 2>&1 || true

for model in 0_6b llama3b 7b 8b 32b; do
  port="${MPORT[$model]}"
  gpus="${MGPU[$model]}"
  tp="${MTP[$model]}"
  mname="${MNAME[$model]}"
  pip_extra=""; [[ "$model" == "32b" ]] && pip_extra="vllm==0.7.2"
  docker run -d --name "memarena-sglang-${model}" \
    --gpus "\"device=${gpus}\"" --ipc=host --shm-size 32g --restart no \
    -p "${port}:${port}" \
    -v "${HOME}/models:/models:ro" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --env "SGLANG_PIP_INSTALL=protobuf sentencepiece ${pip_extra}" \
    lmsysorg/sglang:latest \
    bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"' bash python3 -m sglang.launch_server \
    --model-path "/models/${model}" --served-model-name "${mname}" \
    --host 0.0.0.0 --port "$port" --tp "$tp" --dp 1 \
    --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null
done

for model in 0_6b llama3b 7b 8b 32b; do
  port="${MPORT[$model]}"
  log "  waiting sglang-$model :$port..."
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do sleep 10; done
  log "  ✓ sglang-$model ready"
done

# ── Step 2: launch 15 oracle_gated cells in parallel ─────────────────────────
log "=== launching 15 oracle_gated cells ==="
declare -A PIDS

for model in 0_6b llama3b 7b 8b 32b; do
  port="${MPORT[$model]}"
  mname="${MNAME[$model]}"
  out_base="out/d6_gating/oracle_gated_${model}"
  mkdir -p "${out_base}/wrapper_logs"

  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    namespace="oracle_gated_${model}_${trial}_judge_remote"
    logf="${out_base}/wrapper_logs/${trial}.log"
    log "  → oracle_gated_${model}/${trial}"

    .venv/bin/python eval/cli.py \
      --run-dir data/benchmark \
      --output-dir "${out_base}" \
      --system oracle_gated \
      --model "$mname" \
      --endpoint "http://localhost:${port}/v1" \
      --trial-name "$trial" \
      --namespace "$namespace" \
      --stages answer evaluate \
      --dimensions d4_permission \
      --d6-arm B \
      --answer-concurrency 8 \
      --judge-model openai/gpt-4o-mini-2024-07-18 \
      --judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      > "$logf" 2>&1 &
    PIDS["${model}_${trial}"]=$!
  done
done

log "waiting for ${#PIDS[@]} cells..."
ok=0; fail=0
for label in "${!PIDS[@]}"; do
  if wait "${PIDS[$label]}"; then
    log "  ✓ $label"; ok=$((ok+1))
  else
    log "  ✗ $label (rc=$?)"; fail=$((fail+1))
  fi
done
log "oracle_gated inference: ok=$ok fail=$fail / 15 cells"

# ── Step 3: append to experiments_index.csv ──────────────────────────────────
log "=== updating experiments_index.csv ==="
declare -A MODEL_NAME_MAP=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)

for model in 0_6b llama3b 7b 8b 32b; do
  mname="${MODEL_NAME_MAP[$model]}"
  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    json_path="out/d6_gating/oracle_gated_${model}/eval_results_${trial}/oracle_gated/evaluation_results_oracle_gated_${model}_${trial}_judge_remote.json"
    # Only add if not already present
    if ! grep -qF "$json_path" experiments_index.csv 2>/dev/null; then
      echo "ablation,oracle_gated,${model},${mname},${trial},${seed},${json_path}" >> experiments_index.csv
      log "  added oracle_gated_${model}/${trial} to experiments_index.csv"
    fi
  done
done

# ── Step 4: D6 rejudge (canonical 5-label path) ──────────────────────────────
log "=== running rerun_d6_judge.py for oracle_gated cells ==="
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee /tmp/step5_d6_rejudge.log

log "Outputs are left under out/d6_gating/ and experiments_index.csv was updated; no repository staging, commit, or push is performed."
log "=== Step 5 DONE ==="
