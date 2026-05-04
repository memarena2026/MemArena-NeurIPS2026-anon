"""D6 ablation: stacked bar chart of Leak / NO_ACCESS / DONT_KNOW / OTHER per cell.

For each (backend, reader) cell pooled across seeds {s2,s3,s4}, render a
4-segment vertical bar (sums to 100% of N_deny):
  Leak       — deterministic fact-leak (red)
  NO_ACCESS  — explicit policy refusal (green)
  DONT_KNOW  — epistemic absence (light grey)
  OTHER      — off-topic / parse / generic (dark grey)

Output:
  memarena/figures/figures/fig_d6_ablation_buckets.pdf  (and .png)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import figure_path
from paper_data import load_all_cells, MODEL_ORDER, MODEL_TEX

OUT_PDF = figure_path("fig_d6_ablation_buckets.pdf")
OUT_PNG = OUT_PDF.with_suffix(".png")

BACKENDS = ["vanilla", "rag", "oracle", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   "Vanilla",
    "rag":       "RAG",
    "oracle":    "Oracle",
    "memobase":  "Memobase",
    "memsearch": "MemSearch",
}

SEG_COLOR = {
    "Leak":      "#d62728",
    "NO_ACCESS": "#2ca02c",
    "DONT_KNOW": "#bdbdbd",
    "OTHER":     "#777777",
}


def _per_cell_buckets():
    grid = load_all_cells(include_ablation=False)
    out = defaultdict(lambda: {
        "deny_total": 0, "Leak": 0,
        "NO_ACCESS": 0, "DONT_KNOW": 0, "OTHER": 0,
    })
    for (seed, backend, reader), cell in grid.items():
        if backend not in BACKENDS or reader not in MODEL_ORDER:
            continue
        try:
            data = json.loads(Path(cell.source_path).read_text())
        except Exception:
            continue
        s = out[(backend, reader)]
        for d in data.get("details", []):
            qid = str(d.get("question_id", ""))
            if not qid.startswith("d4_perm"):
                continue
            mode = str(d.get("expected_answer_mode") or "").lower()
            if mode not in ("deny", "abstain"):
                continue
            s["deny_total"] += 1
            if bool(d.get("leaked_fact_in_output", False)):
                s["Leak"] += 1
                continue
            cat = str(d.get("access_category_v3") or "").upper()
            if cat in ("NO_ACCESS", "DONT_KNOW", "OTHER"):
                s[cat] += 1
            else:
                s["OTHER"] += 1  # treat unlabeled as OTHER for visualization
    return out


def main():
    stats = _per_cell_buckets()

    # Layout: 5 backends side-by-side, 5 bars per backend (one per reader).
    fig, ax = plt.subplots(figsize=(13.0, 4.6))

    bar_w = 0.7
    group_gap = 1.2
    n_readers = len(MODEL_ORDER)
    n_backends = len(BACKENDS)

    xs = []
    xticks_minor = []
    xticks_major = []
    xticks_major_labels = []
    cell_keys = []
    for bi, b in enumerate(BACKENDS):
        for ri, r in enumerate(MODEL_ORDER):
            x = bi * (n_readers + group_gap) + ri
            xs.append(x)
            xticks_minor.append(x)
            cell_keys.append((b, r))
        # Group label centred over the 5 bars
        xticks_major.append(bi * (n_readers + group_gap) + (n_readers - 1) / 2)
        xticks_major_labels.append(BACKEND_LABEL[b])

    # Compute segment heights as % of N_deny
    segs = ["Leak", "NO_ACCESS", "DONT_KNOW", "OTHER"]
    seg_arrays = {seg: [] for seg in segs}
    for (b, r) in cell_keys:
        s = stats.get((b, r))
        if s is None or s["deny_total"] == 0:
            for seg in segs:
                seg_arrays[seg].append(0.0)
            continue
        n = s["deny_total"]
        for seg in segs:
            seg_arrays[seg].append(s[seg] / n * 100)

    # Plot stacks
    bottoms = np.zeros(len(xs))
    for seg in segs:
        heights = np.array(seg_arrays[seg])
        ax.bar(xs, heights, width=bar_w, bottom=bottoms,
               color=SEG_COLOR[seg], edgecolor="white", lw=0.4,
               label=seg.replace("_", " "))
        bottoms = bottoms + heights

    # Annotate NO_ACCESS percentage above each bar (only when >= 3% to avoid clutter)
    for i, (b, r) in enumerate(cell_keys):
        v = seg_arrays["NO_ACCESS"][i]
        if v >= 3.0:
            ax.text(xs[i], 102, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#1a5c1a", fontweight="bold")

    # Reader labels under each bar
    ax.set_xticks(xticks_minor)
    ax.set_xticklabels([MODEL_TEX[r] for (_, r) in cell_keys],
                       rotation=40, ha="right", fontsize=7.5)
    # Backend group labels at the bottom
    sec = ax.secondary_xaxis("bottom")
    sec.set_xticks(xticks_major)
    sec.set_xticklabels(xticks_major_labels, fontsize=10, fontweight="bold")
    sec.tick_params(axis="x", which="major", pad=42, length=0)

    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("\\% of DENY items", fontsize=10)
    ax.set_title("D6 ablation: outcome composition per cell  "
                 "(green = real policy refusal; grey = amnesia/noise; red = leak)",
                 fontsize=11)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
              fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
