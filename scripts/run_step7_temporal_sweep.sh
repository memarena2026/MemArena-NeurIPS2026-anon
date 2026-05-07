#!/usr/bin/env bash
# Step 7: A8 temporal-window sweep — 15 cells (5 windows × 3 trials, Qwen3-8B)
# Runs after Step 5 (oracle_gated) is complete.
#
# GPU: uses sglang-8b on GPU 4 (TP=1). Other GPUs idle.
# Output: out/temporal_sweep/window_<N>d/eval_results_<trial>/temporal/

set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[step7-temporal $(date -u +%H:%M:%S)] $*"; }

MODEL_TAG="8b"
MODEL_NAME="Qwen/Qwen3-8B"
PORT=16003
GPU="4"
TRIALS=(s2 s3 s4)
WINDOWS=(1 3 7 15 all)
declare -A SEED=([s2]=1002 [s3]=1003 [s4]=1004)

# ── Start sglang-8b ───────────────────────────────────────────────────────────
log "=== starting sglang-8b on GPU 4 :${PORT} ==="
docker ps -aq --filter 'name=memarena-sglang-' | xargs -r docker rm -f >/dev/null 2>&1 || true

docker run -d --name "memarena-sglang-${MODEL_TAG}" \
  --gpus "\"device=${GPU}\"" --ipc=host --shm-size 32g --restart no \
  -p "${PORT}:${PORT}" \
  -v "${HOME}/models:/models:ro" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --env "SGLANG_PIP_INSTALL=protobuf sentencepiece" \
  lmsysorg/sglang:latest \
  bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"' bash python3 -m sglang.launch_server \
  --model-path "/models/${MODEL_TAG}" --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" --tp 1 --dp 1 \
  --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null

log "  waiting sglang-8b :${PORT}..."
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do sleep 10; done
log "  ✓ sglang-8b ready"

# ── Launch 15 cells in parallel ───────────────────────────────────────────────
log "=== launching 15 temporal-window cells (5 windows × 3 trials) ==="
declare -A PIDS

for w in "${WINDOWS[@]}"; do
  win_label="window_${w}d"
  out_base="out/temporal_sweep/${win_label}"
  mkdir -p "${out_base}/wrapper_logs"

  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    namespace="temporal_${MODEL_TAG}_${trial}_${win_label}_judge_remote"
    logf="${out_base}/wrapper_logs/${trial}.log"

    window_arg=""
    [[ "$w" != "all" ]] && window_arg="--temporal-window-days $w"

    log "  → ${win_label}/${trial} (window=$w days)"
    # shellcheck disable=SC2086
    .venv/bin/python eval/cli.py \
      --run-dir data/benchmark \
      --output-dir "${out_base}" \
      --system temporal \
      $window_arg \
      --model "$MODEL_NAME" \
      --endpoint "http://localhost:${PORT}/v1" \
      --trial-name "$trial" \
      --namespace "$namespace" \
      --stages add answer evaluate \
      --dimensions d4_permission \
      --d6-arm B \
      --answer-concurrency 8 \
      --judge-model openai/gpt-4o-mini-2024-07-18 \
      --judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      > "$logf" 2>&1 &
    PIDS["${win_label}_${trial}"]=$!
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
log "temporal sweep inference: ok=$ok fail=$fail / 15 cells"

# ── Append to experiments_index.csv ──────────────────────────────────────────
log "=== updating experiments_index.csv ==="

for w in "${WINDOWS[@]}"; do
  win_label="window_${w}d"
  backend_tag="temporal_${win_label}"
  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    namespace="temporal_${MODEL_TAG}_${trial}_${win_label}_judge_remote"
    json_path="out/temporal_sweep/${win_label}/eval_results_${trial}/temporal/evaluation_results_${namespace}.json"
    if ! grep -qF "$json_path" experiments_index.csv 2>/dev/null; then
      echo "ablation,${backend_tag},${MODEL_TAG},${MODEL_NAME},${trial},${seed},${json_path}" >> experiments_index.csv
      log "  added ${backend_tag}/${trial} to experiments_index.csv"
    fi
  done
done

# ── D6 rejudge ────────────────────────────────────────────────────────────────
log "=== running rerun_d6_judge.py for temporal-sweep cells ==="
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee /tmp/step7_d6_rejudge.log

log "Outputs are left under out/temporal_sweep/ and experiments_index.csv was updated; no repository staging, commit, or push is performed."
log "=== Step 7 DONE ==="
