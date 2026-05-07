#!/usr/bin/env bash
# Full D6 self-probe completion pipeline.
# Phase 1 (vanilla/oracle/baseline_simplerag + 0_6b memobase) already running (PID 2629852).
# This script handles Phase 2 onward.
#
# Phase 2: memobase main-eval cache build for llama3b/7b/8b/32b (DP=8 sglang per model)
#          Also fills the 15 missing memobase main-eval cells.
# Phase 3: memobase self-probe cells (needs Phase 2 cache)
# Phase 4: memsearch shared cache symlink + memsearch self-probe
# Phase 5: judge_d6_self_probe.py
# Phase 6: rerun_d6_judge.py (for D6 labels in memobase main eval cells)
# Phase 7: commit + push

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing"; exit 2; }
source .venv/bin/activate

log() { echo "[d6-complete $(date -u +%H:%M:%S)] $*"; }

SERVICES_ROOT="/ephemeral/ubuntu/memarena-memory-services"
SELF_PROBE_PID="${SELF_PROBE_PID:-2629852}"
MEMSEARCH_BUILD_PID="${MEMSEARCH_BUILD_PID:-2634606}"

# ── Phase 2: wait for Phase 1 self-probe, then build memobase cache ───────────
log "=== Phase 2: waiting for Phase 1 self-probe (PID $SELF_PROBE_PID) ==="
wait "$SELF_PROBE_PID" || log "Phase 1 self-probe exited (some cells may have failed, continuing)"

log "Phase 1 done. tearing down 5-model sglang setup..."
docker ps -aq --filter 'name=memarena-sglang-' | xargs -r docker rm -f >/dev/null 2>&1 || true
docker network ls --format '{{.Name}}' | grep '^memarena_' | xargs -r docker network rm >/dev/null 2>&1 || true
log "sglang containers stopped"

log "=== Phase 2: building memobase cache for llama3b/7b/8b/32b ==="
log "(builds main eval + provides cache JSONL for Phase 3 memobase self-probe)"
MODELS="llama3b 7b 8b 32b" \
BACKENDS="memobase" \
SAFE_NO_RM=1 \
MEMARENA_MEMORY_SERVICES_ROOT="$SERVICES_ROOT" \
  bash scripts/run_l_all_models.sh 2>&1 | tee /tmp/d6_memobase_build.log

log "Phase 2 (memobase cache build + main eval) done"

# ── Phase 3: restart 5 sglang servers for memobase + memsearch self-probe ──
log "=== Phase 3+4: restart 5 sglang servers ==="
declare -A MGPU=([0_6b]="1" [llama3b]="2" [7b]="3" [8b]="4" [32b]="5,6")
declare -A MTP=([0_6b]=1 [llama3b]=1 [7b]=1 [8b]=1 [32b]=2)
declare -A MNAME=([0_6b]="Qwen/Qwen3-0.6B" [llama3b]="meta-llama/Llama-3.2-3B-Instruct" [7b]="mistralai/Mistral-7B-Instruct-v0.3" [8b]="Qwen/Qwen3-8B" [32b]="Qwen/Qwen3-32B-AWQ")
declare -A MPORT=([0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004)

for model in 0_6b llama3b 7b 8b 32b; do
  port="${MPORT[$model]}"
  gpus="${MGPU[$model]}"
  tp="${MTP[$model]}"
  pip_extra=""; [[ "$model" == "32b" ]] && pip_extra="vllm==0.7.2"
  docker rm -f "memarena-sglang-${model}" 2>/dev/null || true
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
    --model-path "/models/${model}" --served-model-name "${MNAME[$model]}" \
    --host 0.0.0.0 --port "$port" --tp "$tp" --dp 1 \
    --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null
done

for model_port in "0_6b:16000" "llama3b:16001" "7b:16002" "8b:16003" "32b:16004"; do
  model="${model_port%%:*}"; port="${model_port##*:}"
  log "  waiting sglang-$model :$port..."
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do sleep 10; done
  log "  ✓ sglang-$model ready"
done

# ── Wait for memsearch cache build, then symlink ─────────────────────────────
log "=== Phase 4: waiting for memsearch cache build (PID $MEMSEARCH_BUILD_PID) ==="
wait "$MEMSEARCH_BUILD_PID" || log "memsearch build exited"

SHARED_CACHE="$REPO_ROOT/out/_memsearch_shared/memcache_memsearch_A_paired_shared_s2.jsonl"
if [[ -f "$SHARED_CACHE" ]]; then
  log "memsearch shared cache ready: $(wc -l < "$SHARED_CACHE") rows. symlinking..."
  for model in 0_6b llama3b 7b 8b 32b; do
    slug="$model"; [[ "$model" == "llama3b" ]] && slug="3b"
    for t in s2 s3 s4; do
      cell_dir="out/accuracy_memarena_l_${slug}/memory_cache/memsearch/${model}/${t}"
      cell_cache="${cell_dir}/memcache_memsearch_A_paired_${model}_${t}.jsonl"
      mkdir -p "$cell_dir"
      [[ ! -e "$cell_cache" ]] && ln -s "$SHARED_CACHE" "$cell_cache"
      log "  → $cell_cache"
    done
  done
else
  log "WARNING: memsearch shared cache missing — memsearch self-probe cells will be skipped"
fi

# ── Phase 3+4: run self-probe again — memobase + memsearch cells now have caches ──
log "=== Phase 3+4: running self-probe for remaining cells (memobase/memsearch) ==="
bash scripts/run_d6_self_probe.sh 2>&1 | tee /tmp/d6_self_probe_phase2.log

log "Phase 3+4 self-probe done"

# ── Phase 5: judge all self-probe cells ──────────────────────────────────────
log "=== Phase 5: judging all d6_self_probe cells ==="
.venv/bin/python scripts/judge_d6_self_probe.py 2>&1 | tee /tmp/d6_self_probe_judge.log

# ── Phase 6: D6 5-label rejudge for new memobase main-eval cells ─────────────
log "=== Phase 6: rerun_d6_judge.py for new memobase main-eval cells ==="
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee /tmp/d6_rejudge.log

log "Outputs are left under out/d6_self_probe/, out/accuracy_memarena_l_*/, and out/_memsearch_shared/; no repository staging, commit, or push is performed."
log "=== ALL DONE ==="
