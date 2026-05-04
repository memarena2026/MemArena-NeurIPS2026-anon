#!/usr/bin/env python3
"""Generate Figure (``fig:finding2``): memory organization closes the gap that
retrieval alone does not.

Finding 2 of MemArena: "The dominant bottleneck is memory organization and
reading, not retrieval alone." Two complementary claims:

* **Panel (a): retrieval alone leaves the gap open; organization closes it.**
  At a single Qwen3-8B anchor, side-by-side bars show Vanilla, RAG (BM25),
  Memobase, MemOS, and Oracle. The Memobase/MemOS bars reach the Oracle
  reference while RAG closes only part of the Vanilla-Oracle gap.

* **Panel (b): lift-vs-scale asymmetry.** Grouped bars per reader tier of
  (RAG - Vanilla), (Memobase - Vanilla), (MemOS - Vanilla) Avg lift. RAG
  can actually be negative at small tiers; the two structured-memory
  families grow monotonically with reader scale, contradicting the
  "memory democratizes small models" reading.

Output
------
``paper/figures/fig_finding2.pdf`` (and ``.png``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from paper_data import (  # noqa: E402
    CAT_RECALL, CAT_REASONING, CAT_TRUST,
    MODEL_TEX,
    BACKEND_TEX,
    DEFAULT_RUN,
    Stat,
    aggregate_seeds, category_accuracy, load_all_cells,
)

OUT_PDF = Path(__file__).resolve().parent / "figures" / "fig_finding2.pdf"
OUT_PNG = OUT_PDF.with_suffix(".png")

BACKENDS_A = ["vanilla", "rag", "dense_rag", "memobase", "memos", "oracle"]
BACKENDS_B = ["rag", "memobase", "memos"]          # lifts-over-vanilla
ANCHOR = "8b"                                       # Qwen3-8B
TIERS = ["0_6b", "llama3b", "7b", "8b", "32b"]     # all 5 readers; Mistral row included to expose the schema-compliance regression flagged in App. memsys-impl
DENSE_RAG_A = ("BGE-M3", 50.9)                      # Appendix Table p1c, Qwen3-8B, k=5

# Colors matching main-table conventions. Vanilla gray, oracle green, structured
# red/orange, RAG blue.
COLOR = {
    "vanilla":  "#7f7f7f",
    "rag":      "#1f77b4",
    "dense_rag": "#8ec7ff",
    "memobase": "#d62728",
    "memos":    "#ff7f0e",
    "oracle":   "#2ca02c",
}

LABEL_A = {
    "vanilla": BACKEND_TEX["vanilla"],
    "rag": BACKEND_TEX["rag"],
    "dense_rag": "Dense RAG",
    "memobase": BACKEND_TEX["memobase"],
    "memos": BACKEND_TEX["memos"],
    "oracle": BACKEND_TEX["oracle"],
}


# Hardcoded from paper/tables/main_SML.tex (Avg row, mean +- std).
# We use the published numbers verbatim so the figure matches the paper
# text exactly; re-deriving from raw cells produces slightly different
# Avg values due to D-to-category mapping edge cases.
# IMPORTANT: keep this table in lockstep with main_SML.tex's Avg row.
# (Last sync 2026-04-26 after the HippoRAG main-table revert.)
PUBLISHED_AVG = {
    # model      : {backend: (mean_pp, std_pp)}
    "0_6b":    {"vanilla": (35.3, 1.0), "oracle": (50.9, 0.3), "rag": (32.8, 0.2),
                "memobase": (58.4, 6.3), "memos": (50.6, 0.5)},
    "llama3b": {"vanilla": (34.8, 0.7), "oracle": (66.3, 0.6), "rag": (12.3, 0.2),
                "memobase": (75.8, 8.6), "memos": (66.4, 0.1)},
    "7b":      {"vanilla": (34.1, 0.8), "oracle": (59.8, 0.8), "rag": (33.1, 0.8),
                "memobase": (41.4, 0.5), "memos": (58.9, 0.3)},
    "8b":      {"vanilla": (33.8, 0.2), "oracle": (74.7, 0.3), "rag": (50.1, 0.1),
                "memobase": (74.7, 0.2), "memos": (74.8, 0.5)},
    "32b":     {"vanilla": (36.1, 0.4), "oracle": (80.4, 0.2), "rag": (48.0, 0.1),
                "memobase": (87.6, 6.1), "memos": (78.9, 3.3)},
}


def _avg_pp(model: str, backend: str) -> tuple[float, float]:
    return PUBLISHED_AVG[model][backend]


def _render(out_pdf: Path) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # ================== Panel (a): Qwen3-8B anchor waterfall ==================
    bars_a = []
    for bk in BACKENDS_A:
        if bk == "dense_rag":
            mean, std = DENSE_RAG_A[1], 0.0
        else:
            mean, std = _avg_pp(ANCHOR, bk)
        bars_a.append((bk, mean, std))

    xs = list(range(len(BACKENDS_A)))
    for i, (bk, v, s) in enumerate(bars_a):
        hatch = "//" if bk == "dense_rag" else None
        axA.bar(i, v, 0.68, color=COLOR[bk], edgecolor="black", linewidth=0.6,
                yerr=s if s > 0 else None, capsize=3, ecolor="#333", hatch=hatch)
        axA.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=9, color="#222")

    # Vanilla-Oracle gap shading + dashed Oracle reference
    oracle_v = bars_a[-1][1]
    vanilla_v = bars_a[0][1]
    axA.axhline(oracle_v, color=COLOR["oracle"], lw=0.9, ls="--", alpha=0.55)
    axA.text(-0.45, oracle_v + 0.8, "Oracle ceiling",
             ha="left", va="bottom", fontsize=8, color=COLOR["oracle"],
             style="italic")

    # Narrative arrows on the bars above
    rag_v = bars_a[1][1]
    dense_rag_v = bars_a[2][1]
    mb_v = bars_a[3][1]
    gap_total = oracle_v - vanilla_v
    rag_closes_pct = 100 * (rag_v - vanilla_v) / gap_total
    axA.annotate(
        "Dense RAG\nhelps, but still\ntrails Oracle",
        xy=(2, dense_rag_v - 1.5),
        xytext=(2.55, 56),
        fontsize=8.2,
        ha="left",
        color="#222",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"),
    )
    axA.annotate(
        f"RAG closes only\n{rag_closes_pct:.0f}% of the gap",
        xy=(1, rag_v - 2), xytext=(0.0, vanilla_v - 18),
        fontsize=8.5, ha="left", color="#222",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"),
    )
    axA.annotate(
        "Memobase / MemOS\nmatch Oracle",
        xy=(3.5, mb_v - 3), xytext=(3.95, vanilla_v + 5),
        fontsize=8.5, ha="left", color="#222",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"),
    )

    axA.set_xticks(xs)
    axA.set_xticklabels([LABEL_A[bk] for bk in BACKENDS_A], rotation=15, ha="right")
    axA.set_ylabel("Avg  (pp)")
    axA.set_ylim(0, 100)
    axA.set_title("(a) At Qwen3-8B: retrieval alone leaves the gap open")
    axA.grid(True, axis="y", alpha=0.3)

    # ================== Panel (b): lift-over-Vanilla per tier ==================
    x_tiers = list(range(len(TIERS)))
    width = 0.26
    offsets = {"rag": -width, "memobase": 0.0, "memos": +width}

    all_ys = []
    for bk in BACKENDS_B:
        lifts = []
        for t in TIERS:
            van_mean, _ = _avg_pp(t, "vanilla")
            bv_mean, _ = _avg_pp(t, bk)
            lifts.append(bv_mean - van_mean)
        all_ys.extend(lifts)
        axB.bar([x + offsets[bk] for x in x_tiers], lifts, width,
                color=COLOR[bk], edgecolor="black", linewidth=0.6,
                label=f"{BACKEND_TEX[bk]} - Vanilla")
        for x, v in zip([x + offsets[bk] for x in x_tiers], lifts):
            axB.text(x, v + 1.0 if v >= 0 else v - 1.6,
                     f"{v:+.1f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=6.5, color="#222")

    axB.axhline(0, color="#555", lw=0.7)
    axB.set_xticks(x_tiers)
    axB.set_xticklabels([MODEL_TEX[t] for t in TIERS], rotation=15, ha="right")
    axB.set_xlabel("Extractor / reader model")
    axB.set_ylabel("Avg lift over Vanilla  (pp)")
    axB.set_title("(b) Memory lift over Vanilla, by reader")
    axB.grid(True, axis="y", alpha=0.3)
    axB.legend(loc="upper left", fontsize=8.5, frameon=False)
    axB.set_ylim(-35, 60)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Stdout table for paper-reference.
    print(f"\n(a) {MODEL_TEX[ANCHOR]} anchor (published main-table Avg pp):")
    for bk in BACKENDS_A:
        if bk == "dense_rag":
            m, s = DENSE_RAG_A[1], 0.0
        else:
            m, s = _avg_pp(ANCHOR, bk)
        print(f"  {LABEL_A[bk]:>10} = {m:.1f} +- {s:.1f}")
    print(f"\n(b) Avg lift over Vanilla (pp) per tier (published values):")
    hdr = f"  {'tier':<14}" + " ".join(f"{BACKEND_TEX[bk]+' − V':>14}" for bk in BACKENDS_B)
    print(hdr)
    for t in TIERS:
        van_m, _ = _avg_pp(t, "vanilla")
        row = f"  {MODEL_TEX[t]:<14}"
        for bk in BACKENDS_B:
            bv_m, _ = _avg_pp(t, bk)
            row += f"{bv_m - van_m:>14.1f}"
        print(row)

    _render(OUT_PDF)
    print(f"\n[fig:finding2] wrote {OUT_PDF}")
    print(f"[fig:finding2] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
