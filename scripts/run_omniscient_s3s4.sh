#!/usr/bin/env bash
# B7 fix: run omniscient ablation for s3+s4 (s2 already exists)
# 5 readers × 2 new seeds = 10 cells
# Each cell: full 1579-record panel (all dimensions)
# Backend: omniscient (full session context, no retrieval)
# Ports: 0_6b→16000, llama3b→16001, 7b→16002, 8b→16003, 32b→16004

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[omniscient-s3s4 $(date -u +%H:%M:%S)] $*"; }

declare -A READER_PORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004)
declare -A READER_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
READERS=(0_6b llama3b 7b 8b 32b)
declare -A SEED=([s3]=1003 [s4]=1004)
TRIALS=(s3 s4)

log "Verifying sglang servers..."
for reader in "${READERS[@]}"; do
  port="${READER_PORT[$reader]}"
  curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 \
    || { log "ERROR: ${reader} not on :${port}"; exit 1; }
  log "  ✓ ${reader} on :${port}"
done

log "=== Omniscient s3+s4: 10 cells (5 readers × 2 seeds) ==="
declare -A PIDS

for reader in "${READERS[@]}"; do
  port="${READER_PORT[$reader]}"
  mname="${READER_NAME[$reader]}"
  out_base="out/ablations_l_omniscient_${reader}"
  mkdir -p "${out_base}/wrapper_logs"

  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    namespace="omniscient_${reader}_${trial}_judge_remote"
    logf="${out_base}/wrapper_logs/${trial}.log"

    log "  → omniscient/${reader}/${trial} (port=${port}, seed=${seed})"
    .venv/bin/python eval/cli.py \
      --run-dir data/benchmark \
      --output-dir "${out_base}" \
      --system omniscient \
      --model "$mname" \
      --endpoint "http://localhost:${port}/v1" \
      --trial-name "$trial" \
      --trial-seed "$seed" \
      --namespace "$namespace" \
      --stages answer evaluate \
      --temperature 0.3 \
      --answer-concurrency 64 \
      --force \
      --judge-model openai/gpt-4o-mini-2024-07-18 \
      --judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      > "$logf" 2>&1 &
    PIDS["${reader}_${trial}"]=$!
  done
done

log "Waiting for ${#PIDS[@]} cells..."
ok=0; fail=0
for label in "${!PIDS[@]}"; do
  if wait "${PIDS[$label]}"; then
    log "  ✓ $label"; ok=$((ok+1))
  else
    log "  ✗ $label (rc=$?)"; fail=$((fail+1))
  fi
done
log "Omniscient s3+s4 done: ok=$ok fail=$fail / 10 cells"

log "=== D6 rejudge for new omniscient cells ==="
.venv/bin/python scripts/rerun_d6_judge_ablations.py --workers 128 2>&1 | tee /tmp/omniscient_d6_rejudge.log
log "D6 rejudge done"

log "Outputs are left under out/ablations_l_omniscient_*; no repository staging, commit, or push is performed."
log "=== Omniscient s3+s4 DONE ==="
