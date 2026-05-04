#!/usr/bin/env bash
# D6 paired-probe self-ego inference — all 75 cells (5 backends × 5 readers × 3 trials).
# Runs --stages answer only; judging done separately via scripts/judge_d6_self_probe.py.
#
# Prerequisites:
#   - All 5 sglang servers running on ports 16000-16004
#   - Pre-built memory-cache JSONL files under out/accuracy_memarena_l_*/memory_cache/
#   - .env with OPENROUTER_API_KEY (or OPENAI_API_KEY)
#   - .venv activated or venv at .venv/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
if [[ -f .env ]]; then set -a; . .env; set +a; fi

PYTHON=".venv/bin/python"
RUN_DIR="data/benchmark"
OUT_BASE="out/d6_self_probe"
mkdir -p "$OUT_BASE"

declare -A MODEL_NAME MODEL_PORT MODEL_CACHE_DIR
MODEL_NAME[0_6b]="Qwen/Qwen3-0.6B"
MODEL_NAME[llama3b]="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME[7b]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_NAME[8b]="Qwen/Qwen3-8B"
MODEL_NAME[32b]="Qwen/Qwen3-32B-AWQ"
MODEL_PORT[0_6b]=16000
MODEL_PORT[llama3b]=16001
MODEL_PORT[7b]=16002
MODEL_PORT[8b]=16003
MODEL_PORT[32b]=16004
# Directory under out/ where the original eval stored memory-cache JSONL files
MODEL_CACHE_DIR[0_6b]="out/accuracy_memarena_l_0_6b/memory_cache"
MODEL_CACHE_DIR[llama3b]="out/accuracy_memarena_l_3b/memory_cache"
MODEL_CACHE_DIR[7b]="out/accuracy_memarena_l_7b/memory_cache"
MODEL_CACHE_DIR[8b]="out/accuracy_memarena_l_8b/memory_cache"
MODEL_CACHE_DIR[32b]="out/accuracy_memarena_l_32b/memory_cache"

MODELS=(0_6b llama3b 7b 8b 32b)
BACKENDS=(vanilla oracle baseline_simplerag memobase memsearch)
TRIALS=(s2 s3 s4)

log() { echo "[d6-self-probe $(date -u +%H:%M:%S)] $*"; }

run_cell() {
  local tag="$1" backend="$2" trial="$3"
  local model="${MODEL_NAME[$tag]}"
  local port="${MODEL_PORT[$tag]}"
  local out_dir="$OUT_BASE/${backend}_${tag}/${trial}"
  mkdir -p "$out_dir"

  local extra_args=()
  local stages=()
  case "$backend" in
    memobase|memsearch)
      local cache_path="${MODEL_CACHE_DIR[$tag]}/${backend}/${tag}/${trial}/memcache_${backend}_A_paired_${tag}_${trial}.jsonl"
      if [[ ! -f "$cache_path" ]]; then
        log "ERROR: missing cache $cache_path — skipping ${backend}/${tag}/${trial}"
        return 1
      fi
      extra_args+=(--system memory_cache --cache-path "$cache_path" --expected-memory-system "$backend")
      stages=(search answer)   # memory_cache needs search to build hit index first
      ;;
    baseline_simplerag)
      extra_args+=(--system baseline_simplerag)
      stages=(search answer)   # BM25 retriever also needs search stage
      ;;
    *)
      extra_args+=(--system "$backend")
      stages=(answer)          # vanilla/oracle use TEXT_SESSIONS, no search needed
      ;;
  esac

  local log_file="/tmp/d6sp_${backend}_${tag}_${trial}.log"
  log "  starting ${backend}/${tag}/${trial} → $out_dir"
  "$PYTHON" eval/cli.py \
    --run-dir "$RUN_DIR" \
    --output-dir "$out_dir" \
    "${extra_args[@]}" \
    --model "$model" \
    --endpoint "http://localhost:${port}/v1" \
    --trial-name "$trial" \
    --stages "${stages[@]}" \
    --dimensions d4_permission \
    --d6-arm B \
    --d6-probe-mode self_ego \
    --answer-concurrency 8 \
    >"$log_file" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FAILED ${backend}/${tag}/${trial} (rc=$rc) — see $log_file"
  else
    log "OK    ${backend}/${tag}/${trial}"
  fi
  return $rc
}

log "=== D6 self-probe: 75 cells ==="
log "backends: ${BACKENDS[*]}"
log "models:   ${MODELS[*]}"
log "trials:   ${TRIALS[*]}"

# Launch all 75 cells in parallel (each cell is 200 items answer-only).
pids=(); labels=()
for tag in "${MODELS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    for trial in "${TRIALS[@]}"; do
      run_cell "$tag" "$backend" "$trial" &
      pids+=("$!")
      labels+=("${backend}_${tag}_${trial}")
    done
  done
done

log "waiting for ${#pids[@]} cells..."
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    log "FAILED: ${labels[$i]}"
    failed=$((failed + 1))
  fi
done

ok=$((${#pids[@]} - failed))
log "=== done: ${ok}/${#pids[@]} cells succeeded, $failed failed ==="
