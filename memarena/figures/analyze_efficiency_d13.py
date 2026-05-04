#!/usr/bin/env python3
"""Rebuild tab:efficiency with per-backend rows for Memobase and MemOS
(D13 cells, 8-agent subset). Adds an Accuracy column alongside TTFT /
Decode / Total / Tokens.

Sources:
- Cells 1-5 (Q0.6B×{Memobase,MemOS}, Llama-3B×MemOS, Q8B×{Memobase,MemOS}):
    answer results: MASim/runs/L/spark_results_s1/paperhook/memory_cache/
                    answer_results_paperhook_<bk>_<model>_s1.json
    filter to 8-agent subset via question_ids listed in
                    memory_cache_8agent/evaluation_results_paperhook_<bk>_<model>_s1_8agent.json
    accuracy: the 8agent.json summary.accuracy
- Cell 6 (Q32B × Memobase):
    answer results: memory_cache_8agent/memory_cache/answer_results_paperhook_memobase_qwen3_32b_awq_s1.json
                    (already 383-Q filtered)
    accuracy: memory_cache_8agent/memory_cache/evaluation_results_paperhook_memobase_qwen3_32b_awq_s1.json
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

try:
    from .paths import table_path
except ImportError:  # pragma: no cover - direct script execution
    from paths import table_path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent


def _resolve_paperhook_root() -> Path:
    override = os.getenv("MEMARENA_EFFICIENCY_D13_ROOT") or os.getenv("MEMARENA_D13_ROOT")
    if override:
        return Path(override).expanduser()

    run_override = os.getenv("MEMARENA_RUN_DIR")
    if run_override:
        run_dir = Path(run_override).expanduser()
        candidates = [
            run_dir / "spark_results_s1" / "paperhook",
            run_dir / "paperhook",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    return REPO_ROOT / "MASim" / "runs" / "l_20260408_111046" / "spark_results_s1" / "paperhook"


ROOT = _resolve_paperhook_root()
SUB_OLD = ROOT / "memory_cache"                 # cells 1-5 answer-results
SUB_NEW = ROOT / "memory_cache_8agent"          # cells 1-5 eval 8-agent + cell 6 everything

OUT_TEX = table_path("efficiency.tex")

# Existing Vanilla / Oracle / Structured (inmem) numbers from current fig_efficiency.tex
# source (these are from analyze_efficiency.py which uses 250-Q SPARK subset timing).
# We KEEP these rows as reference baselines alongside the new D13 8-agent rows.
VANILLA_ORACLE_STRUCTURED_ROWS = {
    "Qwen3-0.6B": {
        "Vanilla":    {"ttft": 70,  "decode": 74.7, "total": 0.38, "tok": 24, "acc": None},
        "Oracle":     {"ttft": 95,  "decode": 68.0, "total": 0.52, "tok": 22, "acc": None},
    },
    "Llama-3.2-3B": {
        "Vanilla":    {"ttft": 127, "decode": 27.4, "total": 0.81, "tok": 17, "acc": None},
        "Oracle":     {"ttft": 262, "decode": 22.6, "total": 1.90, "tok": 30, "acc": None},
    },
    "Qwen3-8B": {
        "Vanilla":    {"ttft": 227, "decode": 12.6, "total": 1.12, "tok": 10, "acc": None},
        "Oracle":     {"ttft": 672, "decode": 10.6, "total": 4.14, "tok": 30, "acc": None},
    },
    "Qwen3-32B (AWQ)": {
        "Vanilla":    {"ttft": 1962, "decode": 1.5, "total": 9.29,  "tok": 10, "acc": None},
        "Oracle":     {"ttft": 3934, "decode": 1.3, "total": 29.82, "tok": 34, "acc": None},
    },
}

# D13 cell definitions: (paper-model-label, backend-tex-label, cell-key)
CELL_MAP = [
    ("Qwen3-0.6B",     "Memobase", "memobase", "qwen3_0.6b",  "old"),
    ("Qwen3-0.6B",     "MemOS",    "memos",    "qwen3_0.6b",  "old"),
    ("Llama-3.2-3B",   "MemOS",    "memos",    "llama3_3b",   "old"),
    ("Qwen3-8B",       "Memobase", "memobase", "qwen3_8b",    "old"),
    ("Qwen3-8B",       "MemOS",    "memos",    "qwen3_8b",    "old"),
    ("Qwen3-32B (AWQ)","Memobase", "memobase", "qwen3_32b_awq","new"),
]


def _load_answer_rows(backend: str, model_key: str, source: str) -> list[dict]:
    """Load per-query answer_results rows with timing fields."""
    if source == "old":
        p = SUB_OLD / f"answer_results_paperhook_{backend}_{model_key}_s1.json"
    else:
        p = SUB_NEW / "memory_cache" / f"answer_results_paperhook_{backend}_{model_key}_s1.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _load_eval_8agent(backend: str, model_key: str, source: str) -> dict:
    """Load 8-agent-filtered evaluation result (accuracy + details with question_ids)."""
    if source == "old":
        p = SUB_NEW / f"evaluation_results_paperhook_{backend}_{model_key}_s1_8agent.json"
    else:
        p = SUB_NEW / "memory_cache" / f"evaluation_results_paperhook_{backend}_{model_key}_s1.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _filter_to_8agent(answer_rows: list[dict], eval_data: dict) -> list[dict]:
    """Keep only answer rows whose question_id is in eval details (8-agent set)."""
    details = eval_data.get("details") or eval_data.get("details_light") or []
    if not details:
        # Cell 6 case: answer rows already filtered
        return answer_rows
    qids = {r["question_id"] for r in details if "question_id" in r}
    if not qids:
        return answer_rows
    return [r for r in answer_rows if r.get("question_id") in qids]


def _median_timing(rows: list[dict]) -> dict:
    """Median TTFT, answer_time (total), completion tokens, and decode tok/s."""
    ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    tot  = [r["answer_time_ms"] for r in rows if r.get("answer_time_ms") is not None]
    comp = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    # Decode tok/s = completion_tokens / ((answer_time_ms - ttft_ms)/1000)
    decode = []
    for r in rows:
        t = r.get("answer_time_ms")
        tt = r.get("ttft_ms")
        c  = r.get("completion_tokens")
        if t is not None and tt is not None and c is not None and t > tt:
            decode.append(c / ((t - tt) / 1000.0))
    def _m(xs): return statistics.median(xs) if xs else None
    return {
        "ttft":   _m(ttft),
        "decode": _m(decode),
        "total":  _m(tot),
        "tok":    _m(comp),
    }


def _fmt(v, kind):
    if v is None: return "--"
    if kind == "ttft":   return f"{int(v)}"
    if kind == "decode": return f"{v:.1f}"
    if kind == "total":  return f"{v/1000.0:.2f}" if v > 100 else f"{v:.2f}"
    if kind == "tok":    return f"{int(v)}"
    if kind == "acc":    return f"{v*100:.1f}"
    return str(v)


def main():
    if not ROOT.exists():
        print(
            f"[tab:efficiency-d13] skipped: D13 paper-hook root not found: {ROOT}. "
            "Set MEMARENA_EFFICIENCY_D13_ROOT to reproduce this optional Spark table."
        )
        return

    cell_stats = {}  # (model, backend) -> {"ttft","decode","total","tok","acc"}
    for model_label, backend_label, bk_key, model_key, source in CELL_MAP:
        rows = _load_answer_rows(bk_key, model_key, source)
        eval_data = _load_eval_8agent(bk_key, model_key, source)
        if not rows or not eval_data:
            print(f"[tab:efficiency-d13] missing {model_label} × {backend_label}; skipping")
            continue
        rows_8 = _filter_to_8agent(rows, eval_data)
        timing = _median_timing(rows_8)
        acc = eval_data.get("summary", {}).get("accuracy")
        cell_stats[(model_label, backend_label)] = {
            **timing,
            "acc": acc,
            "n": len(rows_8),
        }
        print(f"{model_label:15s} × {backend_label:8s}: n={len(rows_8):3d}  "
              f"ttft={_fmt(timing['ttft'],'ttft')}ms  "
              f"decode={_fmt(timing['decode'],'decode')}tok/s  "
              f"tot={_fmt(timing['total'],'total')}ms  "
              f"tok={_fmt(timing['tok'],'tok')}  "
              f"acc={_fmt(acc,'acc')}%")

    if not cell_stats:
        print(
            f"[tab:efficiency-d13] skipped: no D13 paper-hook cells found under {ROOT}. "
            "Keeping the base efficiency table, if one was generated earlier."
        )
        return

    # Build LaTeX table with Vanilla / Oracle / Memobase / MemOS rows per tier.
    tex = []
    tex.append("% Auto-generated by paper/analyze_efficiency_d13.py. Do not hand-edit.")
    tex.append(r"\begin{table}[h]")
    tex.append(r"\centering")
    tex.append(r"\caption{Per-query answering efficiency on \benchL{} (DGX Spark, SGLang, answer concurrency $=1$). "
               r"Vanilla and Oracle rows are medians across the full 250-question Spark subset "
               r"(no backend-specific memory cache). Memobase and MemOS rows are medians across the "
               r"383-question 8-agent cross-check subset (D13 paper-hook run) at the matching SGLang "
               r"SKU; the Accuracy column reports the 4o-mini-judged raw accuracy on the same 383 "
               r"questions. TTFT: time-to-first-token. Decode: completion-token decode throughput. "
               r"Total: end-to-end answer wall-time. Tokens: completion token count. "
               r"Llama-3.2-3B $\times$ Memobase is absent because that memcache was not built on Spark; "
               r"Qwen3-32B $\times$ MemOS is deferred to the camera-ready revision (ingest pending).}")
    tex.append(r"\label{tab:efficiency}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{@{}llrrrrr@{}}")
    tex.append(r"\toprule")
    tex.append(r"Model & Backend & Acc (\%) & TTFT (ms) & Decode (tok/s) & Total (s) & Tokens \\")
    tex.append(r"\midrule")

    for tier in ["Qwen3-0.6B", "Llama-3.2-3B", "Qwen3-8B", "Qwen3-32B (AWQ)"]:
        v = VANILLA_ORACLE_STRUCTURED_ROWS[tier]
        # Vanilla
        row = v["Vanilla"]
        tex.append(f"{tier} & Vanilla & -- & {row['ttft']} & {row['decode']:.1f} & {row['total']:.2f} & {row['tok']} \\\\")
        # Oracle
        row = v["Oracle"]
        tex.append(f" & Oracle & -- & {row['ttft']} & {row['decode']:.1f} & {row['total']:.2f} & {row['tok']} \\\\")
        # Memobase (if cell exists)
        if (tier, "Memobase") in cell_stats:
            s = cell_stats[(tier, "Memobase")]
            tex.append(f" & Memobase & {_fmt(s['acc'],'acc')} & "
                       f"{_fmt(s['ttft'],'ttft')} & {_fmt(s['decode'],'decode')} & "
                       f"{_fmt(s['total'],'total')} & {_fmt(s['tok'],'tok')} \\\\")
        else:
            tex.append(f" & Memobase & -- & -- & -- & -- & -- \\\\")
        # MemOS (if cell exists)
        if (tier, "MemOS") in cell_stats:
            s = cell_stats[(tier, "MemOS")]
            tex.append(f" & MemOS & {_fmt(s['acc'],'acc')} & "
                       f"{_fmt(s['ttft'],'ttft')} & {_fmt(s['decode'],'decode')} & "
                       f"{_fmt(s['total'],'total')} & {_fmt(s['tok'],'tok')} \\\\")
        else:
            tex.append(f" & MemOS & -- & -- & -- & -- & -- \\\\")
        tex.append(r"\midrule")

    # Remove trailing \midrule and close
    if tex[-1] == r"\midrule":
        tex.pop()
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex) + "\n")
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
