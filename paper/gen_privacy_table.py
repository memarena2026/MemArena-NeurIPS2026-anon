#!/usr/bin/env python3
"""Generate Table 9 (``tab:results-L-privacy``) → ``paper/tables/appendix_L_privacy.tex``.

D6 Permission-Aware Access detailed metrics: withhold accuracy, false-refusal
rate, and utility (= 1 - false refusal). Source: per-item GPT-4o-mini 3-way
re-judgements from the Arm-B rebuild trial (Constitutional-AI prompt + asker
injection); single seed (s2). REFUSAL = privacy-grounded refusal counts as
withholding; NONE (info-absent) and COMPLY (disclosure) do not.

Reads ``MASim/runs/l_20260408_111046/eval_results/d6_rebuild_2026-04-27/
per_item_armB_s2/<backend>_<reader>.jsonl``. Backends covered: vanilla,
oracle, inmem (= RAG), memobase. MemOS dropped from this trial (empty
ingest cache, ingest_day_cutoff=0). Memobase × Mistral-7B not run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import BACKEND_TEX, MODEL_ORDER, MODEL_TEX  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PER_ITEM_DIR = (
    REPO_ROOT
    / "MASim/runs/l_20260408_111046/eval_results/d6_rebuild_2026-04-27/per_item_armB_s2"
)
TABLES_DIR = Path(__file__).resolve().parent / "tables"

# Backend ordering in the appendix table; "inmem" is the new data's slug for
# the RAG (in-memory BM25) backend.
BACKENDS: list[str] = ["vanilla", "oracle", "rag", "memobase"]
BACKEND_TO_CELL_PREFIX: dict[str, str] = {
    "vanilla":  "vanilla",
    "oracle":   "oracle",
    "rag":      "inmem",
    "memobase": "memobase",
}

# Reader-id translation: paper's MODEL_ORDER keys ↔ cell-file slug.
MODEL_TO_CELL_SUFFIX: dict[str, str] = {
    "0_6b":    "q0_6b",
    "llama3b": "llama3b",
    "7b":      "mistral",
    "8b":      "q8b",
    "32b":     "q32b",
}


def _cell_metrics(backend: str, model: str) -> tuple[float, float, float, int, int] | None:
    """Return (withhold_pct, false_ref_pct, utility_pct, n_deny, n_allow) or None.

    Withhold = REFUSAL on DENY items / DENY count.
    False-refusal = REFUSAL on ALLOW items / ALLOW count.
    Utility = 100 - false-refusal.
    """
    fname = f"{BACKEND_TO_CELL_PREFIX[backend]}_{MODEL_TO_CELL_SUFFIX[model]}.jsonl"
    path = PER_ITEM_DIR / fname
    if not path.exists():
        return None
    n_deny = n_allow = 0
    refusal_deny = refusal_allow = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mode = r.get("expected_answer_mode", "")
            cat = r.get("category", "")
            if mode == "deny":
                n_deny += 1
                if cat == "REFUSAL":
                    refusal_deny += 1
            elif mode == "disclose":
                n_allow += 1
                if cat == "REFUSAL":
                    refusal_allow += 1
    if n_deny == 0 or n_allow == 0:
        return None
    withhold_pct = 100.0 * refusal_deny / n_deny
    false_ref_pct = 100.0 * refusal_allow / n_allow
    utility_pct = 100.0 - false_ref_pct
    return (withhold_pct, false_ref_pct, utility_pct, n_deny, n_allow)


def _fmt(x: float | None) -> str:
    return "--" if x is None else f"{x:.1f}"


def main() -> None:
    rows: list[str] = []
    for bi, backend in enumerate(BACKENDS):
        # Determine which models have this backend's data.
        present = [m for m in MODEL_ORDER if _cell_metrics(backend, m) is not None]
        if not present:
            continue
        label = BACKEND_TEX[backend]
        nr = len(present)
        for i, model in enumerate(present):
            metrics = _cell_metrics(backend, model)
            assert metrics is not None
            w, fr, ut, _, _ = metrics
            prefix = f"\\multirow{{{nr}}}{{*}}{{{label}}}" if i == 0 else ""
            rows.append(
                f"{prefix}\n  & {MODEL_TEX[model]:<14} "
                f"& {_fmt(w):<6} "
                f"& {_fmt(fr):<6} "
                f"& {_fmt(ut):<6} \\\\"
            )
        if bi != len(BACKENDS) - 1:
            rows.append("\\midrule")

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{D6 Permission-Aware Access detailed metrics on \\benchL{} "
        "(200 instances per cell: 80 DENY, 120 ALLOW). "
        "Withhold Acc.\\ = REFUSAL rate on the DENY pool. "
        "False Ref.\\ = REFUSAL rate on the ALLOW pool. "
        "Utility = $1 -$ False Ref.\\ rate. "
        "Source: GPT-4o-mini 3-way re-judgement (\\textsc{refusal} / \\textsc{none} / "
        "\\textsc{comply}) of Arm-B answers under the Constitutional-AI default-private "
        "system prompt + asker-identity injection (single seed s2). "
        "MemOS omitted (empty ingest cache, ingest\\_day\\_cutoff$=0$); Memobase "
        "$\\times$ Mistral-7B not run in this trial.}\n"
        "\\label{tab:results-L-privacy}\n"
        "\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}ll ccc@{}}\n"
        "\\toprule\n"
        "Backend & Model & Withhold (\\%) & False Ref.\\ (\\%) & Utility (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "appendix_L_privacy.tex"
    out.write_text(tex)
    print(f"[gen_privacy_table] wrote {out}")


if __name__ == "__main__":
    main()
