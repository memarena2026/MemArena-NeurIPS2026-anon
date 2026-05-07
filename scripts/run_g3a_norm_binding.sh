#!/usr/bin/env bash
# G3a: Reader-side norm-binding test
# Injects [access:DENY]/[access:ALLOW] into the prompt at scoring time for
# d4_permission items. Tests whether the reader can bind explicit access
# markers to disclosure decisions (F2's central argument).
#
# Backend: oracle (ensures evidence is in context — cleanest test of binding)
# Readers: 0_6b, llama3b, 7b, 8b, 32b  (5 readers × 3 seeds = 15 cells)
# Dimensions: d4_permission only (~200 records/cell → ~3000 total)
# Stages: answer evaluate (oracle needs no add stage)
#
# Ports already running (from previous pipeline):
#   port 16000: Qwen/Qwen3-0.6B
#   port 16001: meta-llama/Llama-3.2-3B-Instruct
#   port 16002: mistralai/Mistral-7B-Instruct-v0.3
#   port 16003: Qwen/Qwen3-8B
#   port 16004: Qwen/Qwen3-32B-AWQ
#
# Output: out/g3a_access_marker_{reader}/
# After completion: rerun_d6_judge.py + commit

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[g3a-norm-binding $(date -u +%H:%M:%S)] $*"; }

TRIALS=(s2 s3 s4)
declare -A SEED=([s2]=1002 [s3]=1003 [s4]=1004)

declare -A READER_PORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004)
declare -A READER_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
READERS=(0_6b llama3b 7b 8b 32b)

# Verify all sglang servers are up
log "Verifying sglang servers..."
for reader in "${READERS[@]}"; do
  port="${READER_PORT[$reader]}"
  curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 \
    || { log "ERROR: sglang for ${reader} not on :${port}"; exit 1; }
  log "  ✓ ${reader} on :${port}"
done

log "=== G3a: 15 cells (5 readers × 3 seeds) with [access:*] injection ==="
declare -A PIDS

for reader in "${READERS[@]}"; do
  port="${READER_PORT[$reader]}"
  mname="${READER_NAME[$reader]}"
  out_base="out/g3a_access_marker_${reader}"
  mkdir -p "${out_base}/wrapper_logs"

  for trial in "${TRIALS[@]}"; do
    namespace="g3a_access_marker_${reader}_${trial}_judge_remote"
    logf="${out_base}/wrapper_logs/${trial}.log"

    log "  → g3a/${reader}/${trial} (port=${port})"
    .venv/bin/python eval/cli.py \
      --run-dir data/benchmark \
      --output-dir "${out_base}" \
      --system oracle \
      --model "$mname" \
      --endpoint "http://localhost:${port}/v1" \
      --trial-name "$trial" \
      --namespace "$namespace" \
      --stages answer evaluate \
      --force \
      --dimensions d4_permission \
      --d6-inject-access \
      --answer-concurrency 8 \
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
log "G3a inference done: ok=$ok fail=$fail / 15 cells"

log "=== D6 rejudge ==="
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee /tmp/g3a_d6_rejudge.log
log "D6 rejudge done"

log "Outputs are left under out/g3a_access_marker_*; no repository staging, commit, or push is performed."
log "=== G3a DONE ==="
