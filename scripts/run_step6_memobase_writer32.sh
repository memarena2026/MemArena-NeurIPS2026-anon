#!/usr/bin/env bash
# Step 6: A4 Memobase writer >> reader sweep — 9 cells (3 readers × 3 trials)
# Extractor: Qwen3-32B-AWQ; Readers: 0_6b, llama3b, 7b
# Runs after Step 5 (oracle_gated) is complete.
#
# Strategy:
#   1. Build 3 shared memobase caches (one per trial) using Qwen3-32B as extractor
#      (same cache used by all 3 readers — extractor is reader-independent)
#   2. Run 9 inference cells with small readers against the 32b-extracted caches
#   3. Add to experiments_index.csv + run rerun_d6_judge.py
#
# GPU layout:
#   GPU 5,6: sglang-32b (TP=2) for cache build
#   GPU 1,2,3: sglang-0_6b, sglang-llama3b, sglang-7b for inference

set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[step6-writer32 $(date -u +%H:%M:%S)] $*"; }

TRIALS=(s2 s3 s4)
declare -A SEED=([s2]=1002 [s3]=1003 [s4]=1004)

# Reader models (skip 8b/32b: reader=writer would be trivial case)
declare -A READER_PORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002)
declare -A READER_GPU=([0_6b]="1" [llama3b]="2" [7b]="3")
declare -A READER_TP=([0_6b]=1 [llama3b]=1 [7b]=1)
declare -A READER_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
)

WRITER_PORT=16004
WRITER_GPU="5,6"
WRITER_TP=2
WRITER_TAG="32b"
WRITER_NAME="Qwen/Qwen3-32B-AWQ"
CACHE_BASE="out/memobase_writer32_shared"

# ── Start sglang-32b for cache build ─────────────────────────────────────────
log "=== starting sglang-32b (writer) on GPU $WRITER_GPU :$WRITER_PORT ==="
docker ps -aq --filter 'name=memarena-sglang-' | xargs -r docker rm -f >/dev/null 2>&1 || true

docker run -d --name "memarena-sglang-${WRITER_TAG}" \
  --gpus "\"device=${WRITER_GPU}\"" --ipc=host --shm-size 32g --restart no \
  -p "${WRITER_PORT}:${WRITER_PORT}" \
  -v "${HOME}/models:/models:ro" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --env "SGLANG_PIP_INSTALL=protobuf sentencepiece vllm==0.7.2" \
  lmsysorg/sglang:latest \
  bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"' bash python3 -m sglang.launch_server \
  --model-path "/models/${WRITER_TAG}" --served-model-name "$WRITER_NAME" \
  --host 0.0.0.0 --port "$WRITER_PORT" --tp "$WRITER_TP" --dp 1 \
  --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null

log "  waiting sglang-32b :$WRITER_PORT..."
until curl -fsS "http://127.0.0.1:${WRITER_PORT}/v1/models" >/dev/null 2>&1; do sleep 10; done
log "  ✓ sglang-32b ready"

# ── Also start 3 memobase stacks for cache build ─────────────────────────────
log "=== starting 3 memobase docker stacks for trials s2/s3/s4 ==="
SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-/ephemeral/ubuntu/memarena-memory-services}"

for trial in "${TRIALS[@]}"; do
  # Re-use existing memobase stack launcher from the memory-services repo
  if [[ -f "$SERVICES_ROOT/docker-compose-${trial}.yml" ]]; then
    docker compose -f "$SERVICES_ROOT/docker-compose-${trial}.yml" up -d 2>/dev/null || true
  else
    log "WARNING: memobase compose file missing for $trial — manual stack start needed"
  fi
done

# ── Build shared caches (3 trials, one per seed) ──────────────────────────────
log "=== building 3 memobase caches with Qwen3-32B extractor ==="
mkdir -p "${CACHE_BASE}"
declare -A CACHE_PIDS

for trial in "${TRIALS[@]}"; do
  seed="${SEED[$trial]}"
  out_jsonl="${CACHE_BASE}/memcache_memobase_writer32_${trial}.jsonl"
  logf="${CACHE_BASE}/build_${trial}.log"

  if [[ -f "$out_jsonl" ]]; then
    existing=$(wc -l < "$out_jsonl")
    if [[ "$existing" -ge 1500 ]]; then
      log "  cache $trial already complete ($existing rows), skipping build"
      continue
    fi
    log "  cache $trial partial ($existing rows), rebuilding"
    rm -f "$out_jsonl"
  fi

  log "  → building cache for $trial (seed=$seed)"
  .venv/bin/python scripts/reproduce/build_memory_cache.py \
    --system memobase \
    --run-dir data/benchmark \
    --extractor-model "$WRITER_NAME" \
    --extractor-endpoint "http://localhost:${WRITER_PORT}/v1" \
    --extractor-api-key "EMPTY" \
    --extractor-config A_paired \
    --seed "$seed" \
    --output "$out_jsonl" \
    > "$logf" 2>&1 &
  CACHE_PIDS["$trial"]=$!
done

log "waiting for cache builds..."
for trial in "${!CACHE_PIDS[@]}"; do
  if wait "${CACHE_PIDS[$trial]}"; then
    rows=$(wc -l < "${CACHE_BASE}/memcache_memobase_writer32_${trial}.jsonl" 2>/dev/null || echo 0)
    log "  ✓ cache $trial ($rows rows)"
  else
    log "  ✗ cache $trial FAILED"
  fi
done

# ── Stop 32b writer, start 3 small reader sglang servers ─────────────────────
log "=== stopping sglang-32b, starting reader servers ==="
docker rm -f "memarena-sglang-${WRITER_TAG}" >/dev/null 2>&1 || true

for reader in 0_6b llama3b 7b; do
  port="${READER_PORT[$reader]}"
  gpus="${READER_GPU[$reader]}"
  tp="${READER_TP[$reader]}"
  mname="${READER_NAME[$reader]}"
  docker run -d --name "memarena-sglang-${reader}" \
    --gpus "\"device=${gpus}\"" --ipc=host --shm-size 32g --restart no \
    -p "${port}:${port}" \
    -v "${HOME}/models:/models:ro" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --env "SGLANG_PIP_INSTALL=protobuf sentencepiece" \
    lmsysorg/sglang:latest \
    bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"' bash python3 -m sglang.launch_server \
    --model-path "/models/${reader}" --served-model-name "$mname" \
    --host 0.0.0.0 --port "$port" --tp "$tp" --dp 1 \
    --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null
done

for reader in 0_6b llama3b 7b; do
  port="${READER_PORT[$reader]}"
  log "  waiting sglang-$reader :$port..."
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do sleep 10; done
  log "  ✓ sglang-$reader ready"
done

# ── Symlink shared caches to per-reader cell paths ────────────────────────────
log "=== symlinking caches to cell paths ==="
for reader in 0_6b llama3b 7b; do
  for trial in "${TRIALS[@]}"; do
    cell_dir="out/memobase_writer32_${reader}/memory_cache/memobase/${reader}/${trial}"
    cell_cache="${cell_dir}/memcache_memobase_writer32_${reader}_${trial}.jsonl"
    shared_cache="${REPO_ROOT}/${CACHE_BASE}/memcache_memobase_writer32_${trial}.jsonl"
    mkdir -p "$cell_dir"
    [[ ! -e "$cell_cache" ]] && ln -s "$shared_cache" "$cell_cache"
    log "  → $cell_cache"
  done
done

# ── Run 9 inference cells in parallel ─────────────────────────────────────────
log "=== launching 9 inference cells (3 readers × 3 trials) ==="
declare -A PIDS

for reader in 0_6b llama3b 7b; do
  port="${READER_PORT[$reader]}"
  mname="${READER_NAME[$reader]}"
  out_base="out/memobase_writer32_${reader}"
  mkdir -p "${out_base}/wrapper_logs"

  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    cache_path="${out_base}/memory_cache/memobase/${reader}/${trial}/memcache_memobase_writer32_${reader}_${trial}.jsonl"
    namespace="memobase_writer32_${reader}_${trial}_judge_remote"
    logf="${out_base}/wrapper_logs/${trial}.log"
    log "  → memobase_writer32_${reader}/${trial}"

    .venv/bin/python eval/cli.py \
      --run-dir data/benchmark \
      --output-dir "${out_base}" \
      --system memory_cache \
      --cache-path "$cache_path" \
      --no-cache-strict \
      --expected-memory-system memobase \
      --expected-config A_paired \
      --model "$mname" \
      --endpoint "http://localhost:${port}/v1" \
      --trial-name "$trial" \
      --namespace "$namespace" \
      --stages answer evaluate \
      --d6-arm B \
      --answer-concurrency 8 \
      --judge-model openai/gpt-4o-mini-2024-07-18 \
      --judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      > "$logf" 2>&1 &
    PIDS["${reader}_${trial}"]=$!
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
log "memobase_writer32 inference: ok=$ok fail=$fail / 9 cells"

# ── Append to experiments_index.csv ──────────────────────────────────────────
log "=== updating experiments_index.csv ==="
declare -A MODEL_NAME_MAP=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
)

for reader in 0_6b llama3b 7b; do
  mname="${MODEL_NAME_MAP[$reader]}"
  for trial in "${TRIALS[@]}"; do
    seed="${SEED[$trial]}"
    namespace="memobase_writer32_${reader}_${trial}_judge_remote"
    json_path="out/memobase_writer32_${reader}/eval_results_${trial}/memory_cache/evaluation_results_${namespace}.json"
    if ! grep -qF "$json_path" experiments_index.csv 2>/dev/null; then
      echo "ablation,memobase_writer32,${reader},${mname},${trial},${seed},${json_path}" >> experiments_index.csv
      log "  added memobase_writer32_${reader}/${trial} to experiments_index.csv"
    fi
  done
done

# ── D6 rejudge ────────────────────────────────────────────────────────────────
log "=== running rerun_d6_judge.py for memobase_writer32 cells ==="
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee /tmp/step6_d6_rejudge.log

log "Outputs are left under out/memobase_writer32*/ and experiments_index.csv was updated; no repository staging, commit, or push is performed."
log "=== Step 6 DONE ==="
