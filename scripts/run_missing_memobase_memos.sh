#!/usr/bin/env bash
# Run the 30 missing memobase+memos cells (5 models x 2 backends x 3 trials),
# then upgrade D6 labels with rerun_d6_judge.py. Results stay under out/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG="/tmp/run_missing_memobase_memos.log"
echo "[wrapper] started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

# Load env
if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then set -a; . .env; set +a; fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing" >&2; exit 2; }

# Run all 5 models, memobase+memos only, preserving existing results
MODELS="0_6b llama3b 7b 8b 32b" \
BACKENDS="memobase memos" \
SAFE_NO_RM=1 \
MEMARENA_MEMORY_SERVICES_ROOT=/ephemeral/ubuntu/memarena-memory-services \
  bash scripts/run_l_all_models.sh 2>&1 | tee -a "$LOG"

echo "[wrapper] eval done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

# Upgrade D6 labels to the 5-label rubric for all new cells
echo "[wrapper] running rerun_d6_judge.py ..." | tee -a "$LOG"
.venv/bin/python scripts/rerun_d6_judge.py 2>&1 | tee -a "$LOG"

echo "[wrapper] D6 rejudge done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

echo "[wrapper] outputs are left under out/accuracy_memarena_l_*/; no repository staging, commit, or push is performed" | tee -a "$LOG"
echo "[wrapper] all done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
