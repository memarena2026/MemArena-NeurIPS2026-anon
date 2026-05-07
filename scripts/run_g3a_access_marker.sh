#!/usr/bin/env bash
# G3a: norm-binding test — oracle + --d6-inject-access, 15 cells (5 readers × 3 trials)
# Tests whether readers can bind [access:DENY]/[access:ALLOW] markers to disclosure decisions.
# SGLang servers on ports 16000-16004 must already be up.
#
# Output: out/g3a_access_marker_<reader>/eval_results_<trial>/oracle/

set -uo pipefail
cd "$(dirname "$0")/.."

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate
PYTHON=".venv/bin/python"

log() { echo "[g3a $(date -u +%H:%M:%S)] $*"; }

declare -A MPORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004)
declare -A MNAME=([0_6b]="Qwen/Qwen3-0.6B" [llama3b]="meta-llama/Llama-3.2-3B-Instruct" \
                  [7b]="mistralai/Mistral-7B-Instruct-v0.3" [8b]="Qwen/Qwen3-8B" \
                  [32b]="Qwen/Qwen3-32B-AWQ")
TRIALS=(s2 s3 s4)

# ── Health-check existing servers ─────────────────────────────────────────────
for reader in 0_6b llama3b 7b 8b 32b; do
  port="${MPORT[$reader]}"
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    log "  waiting for sglang-$reader :$port..."
    sleep 10
  done
done
log "All 5 SGLang servers ready."

# ── Launch 15 cells in parallel ───────────────────────────────────────────────
declare -A PIDS
declare -A CELL_LOGS

for reader in 0_6b llama3b 7b 8b 32b; do
  port="${MPORT[$reader]}"
  mname="${MNAME[$reader]}"
  for trial in "${TRIALS[@]}"; do
    ns="g3a_access_marker_${reader}_${trial}_judge_remote"
    outdir="out/g3a_access_marker_${reader}"
    logfile="/tmp/g3a_${reader}_${trial}.log"
    CELL_LOGS["${reader}_${trial}"]="$logfile"

    log "Launching $reader/$trial → $outdir"
    "$PYTHON" eval/cli.py \
      --run-dir data/benchmark \
      --system oracle \
      --d6-inject-access \
      --model "$mname" \
      --endpoint "http://localhost:${port}/v1" \
      --trial-name "$trial" \
      --namespace "$ns" \
      --output-dir "$outdir" \
      --stages answer evaluate \
      --answer-concurrency 16 \
      --judge-model openai/gpt-4o-mini-2024-07-18 \
      --judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      > "$logfile" 2>&1 &
    PIDS["${reader}_${trial}"]=$!
  done
done

log "All 15 cells launched. Waiting..."

# ── Wait and report ───────────────────────────────────────────────────────────
ok=0; fail=0
for label in "${!PIDS[@]}"; do
  if wait "${PIDS[$label]}"; then
    log "  ✓ $label"; ok=$((ok+1))
  else
    log "  ✗ $label (rc=$?)"; fail=$((fail+1))
    tail -5 "${CELL_LOGS[$label]}"
  fi
done
log "Inference: ok=$ok fail=$fail / 15 cells"

# ── Verify outputs ────────────────────────────────────────────────────────────
out_ok=0
for reader in 0_6b llama3b 7b 8b 32b; do
  for trial in "${TRIALS[@]}"; do
    f="out/g3a_access_marker_${reader}/eval_results_${trial}/oracle/evaluation_results_g3a_access_marker_${reader}_${trial}_judge_remote.json"
    [[ -f "$f" ]] && out_ok=$((out_ok+1)) || log "MISSING: $f"
  done
done
log "Output files present: $out_ok/15"

# ── D6 canonical rejudge (idempotent) ─────────────────────────────────────────
log "Running D6 canonical rejudge on g3a cells..."
for reader in 0_6b llama3b 7b 8b 32b; do
  "$PYTHON" scripts/rerun_d6_judge_ablations.py \
    --root "out/g3a_access_marker_${reader}" \
    --workers 64 2>&1 | tail -3 &
done
wait
log "D6 rejudge complete."

log "Outputs are left under out/g3a_access_marker_*; no repository staging, commit, or push is performed."
