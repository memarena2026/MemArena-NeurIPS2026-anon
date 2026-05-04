#!/usr/bin/env python3
"""Generate Figure 6 (``fig:efficiency``): on-device efficiency on MemArena-L.

Aggregates per-query timing and token counts from the SPARK run at
``MASim/runs/l_20260408_111046/spark_results_s1/{vanilla,oracle,inmem}/<model>/
answer_results_run.json`` and produces a two-panel figure:

* (a) Time-to-first-token (TTFT, ms) per model tier, one bar per backend.
* (b) Decode throughput (completion tokens / decode seconds) per model tier,
      one bar per backend.

Only the answering phase is plotted here; ingest cost for structured-memory
backends (memobase / memos) is reported in a companion table from
``memory_cache/memcache_*_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
SPARK_ROOT = REPO_ROOT / "MASim" / "runs" / "l_20260408_111046" / "spark_results_s1"
OUT_PDF = PAPER_DIR / "figures" / "fig_efficiency.pdf"
OUT_TEX = PAPER_DIR / "tables" / "efficiency.tex"

# Model order (small → large) and display names.
MODEL_ORDER = ["qwen3_0.6b", "llama3_3b", "qwen3_8b", "qwen3_32b_awq"]
MODEL_LABEL = {
    "qwen3_0.6b":    "Qwen3-0.6B",
    "llama3_3b":     "Llama-3.2-3B",
    "qwen3_8b":      "Qwen3-8B",
    "qwen3_32b_awq": "Qwen3-32B (AWQ)",
}

# Backends to plot on the answering phase. In this SPARK sweep, ``inmem`` is
# the RAG baseline; structured-memory backends are profiled separately through
# the memcache answer probes and hardware summaries.
BACKEND_ORDER = ["vanilla", "oracle", "inmem"]
BACKEND_LABEL = {
    "vanilla": "Vanilla (no memory)",
    "oracle":  "Oracle",
    "inmem":   "RAG",
}
BACKEND_COLOR = {
    "vanilla": "#7f7f7f",
    "oracle":  "#2ca02c",
    "inmem":   "#1f77b4",
}


def _load_cell(backend: str, model: str) -> list[dict] | None:
    p = SPARK_ROOT / backend / model / "answer_results_run.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _ttft_ms(rows: list[dict]) -> list[float]:
    return [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]


def _decode_tps(rows: list[dict]) -> list[float]:
    out: list[float] = []
    for r in rows:
        n = r.get("completion_tokens")
        t_total = r.get("answer_time_ms")
        t_ttft  = r.get("ttft_ms")
        if not n or not t_total or t_ttft is None:
            continue
        decode_ms = t_total - t_ttft
        if decode_ms <= 0:
            continue
        out.append(n / (decode_ms / 1000.0))
    return out


def _median_iqr(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    arr = np.array(xs)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def _collect() -> dict:
    """Return dict: metric -> backend -> model -> (median, q25, q75, n)."""
    out: dict[str, dict[str, dict[str, tuple[float, float, float, int]]]] = {
        "ttft": {b: {} for b in BACKEND_ORDER},
        "tps":  {b: {} for b in BACKEND_ORDER},
    }
    for b in BACKEND_ORDER:
        for m in MODEL_ORDER:
            rows = _load_cell(b, m)
            if rows is None:
                continue
            ttft  = _ttft_ms(rows)
            tps   = _decode_tps(rows)
            med, q25, q75 = _median_iqr(ttft)
            out["ttft"][b][m] = (med, q25, q75, len(ttft))
            med, q25, q75 = _median_iqr(tps)
            out["tps"][b][m]  = (med, q25, q75, len(tps))
    return out


def _bar_panel(ax, data: dict, ylabel: str, title: str, log: bool = False) -> None:
    n_back = len(BACKEND_ORDER)
    width = 0.8 / n_back
    xs = np.arange(len(MODEL_ORDER))
    for i, b in enumerate(BACKEND_ORDER):
        ys, errs_lo, errs_hi = [], [], []
        for m in MODEL_ORDER:
            cell = data[b].get(m)
            if cell is None:
                ys.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
            else:
                med, q25, q75, _ = cell
                ys.append(med)
                errs_lo.append(max(med - q25, 0.0))
                errs_hi.append(max(q75 - med, 0.0))
        offset = (i - (n_back - 1) / 2) * width
        ax.bar(
            xs + offset, ys, width=width,
            yerr=[errs_lo, errs_hi],
            color=BACKEND_COLOR[b], edgecolor="black", linewidth=0.5,
            label=BACKEND_LABEL[b], capsize=2, error_kw={"elinewidth": 0.8},
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    if log:
        ax.set_yscale("log")


def _render(data: dict, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    _bar_panel(axes[0], data["ttft"], "Time-to-first-token (ms)",  "(a) TTFT by model tier",              log=True)
    _bar_panel(axes[1], data["tps"],  "Decode throughput (tok/s)", "(b) Generation throughput by model", log=False)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, title="Backend", title_fontsize=8)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_table(out_tex: Path) -> None:
    """Emit tables/efficiency.tex (Table ``tab:efficiency``) with medians per cell."""
    lines = [
        r"% Auto-generated by paper/analyze_efficiency.py. Do not hand-edit.",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-query answering efficiency on \benchL{} (DGX Spark, SGLang, answer concurrency $=1$; median across 250 questions per cell). "
        r"TTFT: time-to-first-token. Decode: completion-token decode throughput. Total: end-to-end answer wall-time. Tokens: completion token count.}",
        r"\label{tab:efficiency}",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Backend & TTFT (ms) & Decode (tok/s) & Total (s) & Tokens \\",
        r"\midrule",
    ]
    prev_model = None
    for m in MODEL_ORDER:
        for b in BACKEND_ORDER:
            rows = _load_cell(b, m)
            if rows is None:
                ttft_s, tps_s, tot_s, ct_s = "--", "--", "--", "--"
            else:
                ttft = _ttft_ms(rows)
                tps  = _decode_tps(rows)
                totals = [r["answer_time_ms"] / 1000.0 for r in rows if r.get("answer_time_ms") is not None]
                cts    = [r["completion_tokens"]       for r in rows if r.get("completion_tokens")]
                ttft_s = f"{np.median(ttft):.0f}"   if ttft   else "--"
                tps_s  = f"{np.median(tps):.1f}"    if tps    else "--"
                tot_s  = f"{np.median(totals):.2f}" if totals else "--"
                ct_s   = f"{np.median(cts):.0f}"    if cts    else "--"
            label_m = MODEL_LABEL[m] if m != prev_model else ""
            if label_m and prev_model is not None:
                lines.append(r"\midrule")
            lines.append(f"{label_m} & {BACKEND_LABEL[b].split(' (')[0]} & {ttft_s} & {tps_s} & {tot_s} & {ct_s} \\\\")
            prev_model = m
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines))


def main() -> None:
    data = _collect()
    covered = []
    for metric in ["ttft", "tps"]:
        for b in BACKEND_ORDER:
            for m in MODEL_ORDER:
                if m in data[metric][b]:
                    covered.append((metric, b, m, data[metric][b][m][3]))
    print(f"[fig:efficiency] loaded {len(covered)} (metric, backend, model) cells from MemArena-L SPARK run")
    for tup in covered[:6]:
        print(" ", tup)
    _render(data, OUT_PDF)
    print(f"[fig:efficiency] wrote {OUT_PDF}")
    print(f"[fig:efficiency] wrote {OUT_PDF.with_suffix('.png')}")
    _write_table(OUT_TEX)
    print(f"[tab:efficiency] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
