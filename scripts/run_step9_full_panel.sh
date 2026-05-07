#!/usr/bin/env bash
# Step 9: Re-run Step 7 (15 temporal cells) + Step 6/8B (3 memobase_writer32 cells)
# WITHOUT --d6-arm B and --dimensions d4_permission to produce the full D1-D10 panel.
#
# Prerequisite: all 5 sglang servers already running (from previous pipeline):
#   port 16003: Qwen/Qwen3-8B
# Both sglang and memobase services should be up from prior runs.
#
# Output: overwrites existing D6-only eval JSONs with full 1579-record panels.
# After completion, runs rerun_d6_judge.py for canonical D6 5-label rubric.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[step9-full-panel $(date -u +%H:%M:%S)] $*"; }

TRIALS=(s2 s3 s4)
declare -A SEED=([s2]=1002 [s3]=1003 [s4]=1004)

# Verify sglang-8b is running on port 16003
curl -fsS "http://127.0.0.1:16003/v1/models" >/dev/null 2>&1 \
  || { log "ERROR: sglang-8b not on :16003"; exit 1; }
log "✓ sglang-8b ready on :16003"

# ─────────────────────────────────────────────────────────────────────────────
# Part A: Temporal-window sweep re-run (15 cells)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-8B"
PORT=16003
WINDOWS=(1 3 7 15 all)

log "=== Part A: 15 temporal-window cells (full D1-D10 panel) ==="
declare -A PIDS_A

for w in "${WINDOWS[@]}"; do
  win_label="window_${w}d"
  out_base="out/temporal_sweep/${win_label}"
  mkdir -p "${out_base}/wrapper_logs"

  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    namespace="temporal_8b_${trial}_${win_label}_judge_remote"
    logf="${out_base}/wrapper_logs/${trial}_full.log"

    window_arg=""
    [[ "$w" != "all" ]] && window_arg="--temporal-window-days $w"

    log "  A → ${win_label}/${trial} (ns=$namespace)"
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
      --force \
      --answer-concurrency 8 \
      --judge-model openai/gpt-4o-mini-2024-07-18 \
      --judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      > "$logf" 2>&1 &
    PIDS_A["${win_label}_${trial}"]=$!
  done
done

log "waiting for ${#PIDS_A[@]} Part A cells..."
ok_a=0; fail_a=0
for label in "${!PIDS_A[@]}"; do
  if wait "${PIDS_A[$label]}"; then
    log "  ✓ A/$label"; ok_a=$((ok_a+1))
  else
    log "  ✗ A/$label (rc=$?)"; fail_a=$((fail_a+1))
  fi
done
log "Part A done: ok=$ok_a fail=$fail_a / 15 cells"

# ─────────────────────────────────────────────────────────────────────────────
# Part B: Memobase writer32 × 8B reader re-run (3 cells)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Part B: 3 memobase_writer32_8b cells (full D1-D10 panel) ==="
declare -A PIDS_B
CACHE_BASE="out/memobase_writer32_shared"

for trial in "${TRIALS[@]}"; do
  seed="${SEED[$trial]}"
  cache_path="${CACHE_BASE}/memcache_memobase_writer32_${trial}.jsonl"
  namespace="memobase_writer32_8b_${trial}_judge_remote"
  out_base="out/memobase_writer32_8b"
  logf="${out_base}/wrapper_logs/${trial}_full.log"
  mkdir -p "${out_base}/wrapper_logs"

  log "  B → memobase_writer32_8b/${trial} (ns=$namespace)"
  .venv/bin/python eval/cli.py \
    --run-dir data/benchmark \
    --output-dir "${out_base}" \
    --system memory_cache \
    --cache-path "$cache_path" \
    --no-cache-strict \
    --expected-memory-system memobase \
    --expected-config A_paired \
    --model "Qwen/Qwen3-8B" \
    --endpoint "http://localhost:16003/v1" \
    --trial-name "$trial" \
    --namespace "$namespace" \
    --stages answer evaluate \
    --force \
    --answer-concurrency 8 \
    --judge-model openai/gpt-4o-mini-2024-07-18 \
    --judge-endpoint https://openrouter.ai/api/v1 \
    --judge-api-key "$OPENROUTER_API_KEY" \
    > "$logf" 2>&1 &
  PIDS_B["$trial"]=$!
done

log "waiting for ${#PIDS_B[@]} Part B cells..."
ok_b=0; fail_b=0
for label in "${!PIDS_B[@]}"; do
  if wait "${PIDS_B[$label]}"; then
    log "  ✓ B/$label"; ok_b=$((ok_b+1))
  else
    log "  ✗ B/$label (rc=$?)"; fail_b=$((fail_b+1))
  fi
done
log "Part B done: ok=$ok_b fail=$fail_b / 3 cells"

# ─────────────────────────────────────────────────────────────────────────────
# D6 rejudge (canonical 5-label path)
# ─────────────────────────────────────────────────────────────────────────────
log "=== D6 rejudge via rerun_d6_judge.py ==="
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee /tmp/step9_d6_rejudge.log
log "D6 rejudge done"

# ─────────────────────────────────────────────────────────────────────────────
# Completion
# ─────────────────────────────────────────────────────────────────────────────
log "Outputs are left under out/temporal_sweep/ and out/memobase_writer32_8b/; no repository staging, commit, or push is performed."
log "=== Step 9 DONE — all 18 cells complete ==="
