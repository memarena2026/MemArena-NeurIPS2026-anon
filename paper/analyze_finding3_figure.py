#!/usr/bin/env python3
"""Generate the merged 2-panel figure for Finding 3 (``fig:finding3``).

Combines two figures previously kept separate:

  Panel (a): On-device Pareto scatter --- Avg accuracy vs.\\ log
             answer-time energy on DGX Spark, $4$ readers $\\times$ $3$
             backends (Mistral-7B not on Spark), peak-temp colormap,
             within-backend curves, dominance arrow. Was
             ``fig_pareto_dominance``.
  Panel (b): Memobase / MemOS / RAG lift over Vanilla per reader tier.
             Was ``fig_finding2`` panel (b).

Output: ``paper/figures/fig_finding3.{pdf,png}``.

Source data:
- Avg-accuracy values (panels a/b/c): ``analyze_finding2.PUBLISHED_AVG``,
  the single-source-of-truth table mirroring ``main_SML.tex``.
- Energy + peak temp (panel a): live read from
  ``MASim/runs/l_20260408_111046/spark_results_s1/hw/hw_agg_*.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analyze_finding2 import PUBLISHED_AVG  # noqa: E402
from paper_data import BACKEND_TEX, MODEL_TEX  # noqa: E402


SPARK_HW_DIR = (
    Path(__file__).resolve().parent.parent
    / "MASim/runs/l_20260408_111046/spark_results_s1/hw"
)
OUT_PDF = Path(__file__).resolve().parent / "figures" / "fig_finding3.pdf"

# ---------------------------------------------------------------------------
# Shared panel-style constants
# ---------------------------------------------------------------------------

COLOR = {
    "vanilla":  "#7f7f7f",
    "rag":      "#1f77b4",
    "dense_rag": "#8ec7ff",
    "memobase": "#d62728",
    "memos":    "#ff7f0e",
    "oracle":   "#2ca02c",
}

LABEL_A_FULL = {
    "vanilla":   BACKEND_TEX["vanilla"],
    "rag":       BACKEND_TEX["rag"],
    "dense_rag": "Dense RAG",
    "memobase":  BACKEND_TEX["memobase"],
    "memos":     BACKEND_TEX["memos"],
    "oracle":    BACKEND_TEX["oracle"],
}

# ---------------------------------------------------------------------------
# Panel (a): on-device Pareto scatter
# ---------------------------------------------------------------------------

PARETO_MODELS = ["0_6b", "llama3b", "8b", "32b"]   # Mistral not on Spark
PARETO_BACKENDS = ["vanilla", "rag", "memobase"]
HW_STUB = {
    "0_6b":    {"vanilla": "qwen3_0.6b_vanilla",     "rag": "qwen3_0.6b_inmem",      "memobase": "qwen3_0.6b_memobase_answerB"},
    "llama3b": {"vanilla": "llama3_3b_vanilla",      "rag": "llama3_3b_inmem",       "memobase": "llama3_3b_memobase_answerB"},
    "8b":      {"vanilla": "qwen3_8b_vanilla",       "rag": "qwen3_8b_inmem",        "memobase": "qwen3_8b_memobase_answerB"},
    "32b":     {"vanilla": "qwen3_32b_awq_vanilla",  "rag": "qwen3_32b_awq_inmem",   "memobase": "qwen3_32b_awq_memobase_answerB"},
}
BACKEND_MARKER = {"vanilla": "o", "rag": "s", "memobase": "D"}
BACKEND_LINESTYLE = {
    "vanilla":  (0, (4, 2)),
    "rag":      (0, (1, 2)),
    "memobase": "-",
}


def _hw(stub: str) -> tuple[float, float]:
    path = SPARK_HW_DIR / f"hw_agg_{stub}.json"
    with path.open() as fh:
        per_cell = json.load(fh)["per_cell"]
    return per_cell["mean_energy_j_per_query"], per_cell["peak_temp_c"]


def _plot_pareto(ax, fig) -> None:
    cells = []
    for model in PARETO_MODELS:
        for backend in PARETO_BACKENDS:
            avg_pp = PUBLISHED_AVG[model][backend][0]
            j_q, t_c = _hw(HW_STUB[model][backend])
            cells.append((model, backend, avg_pp, j_q, t_c))

    temps = [c[4] for c in cells]
    norm = Normalize(vmin=min(temps), vmax=max(temps))
    cmap = plt.get_cmap("coolwarm")

    for bk in PARETO_BACKENDS:
        bk_cells = sorted([c for c in cells if c[1] == bk],
                          key=lambda c: PARETO_MODELS.index(c[0]))
        ax.plot([c[3] for c in bk_cells], [c[2] for c in bk_cells],
                color="#888", linewidth=1.0, linestyle=BACKEND_LINESTYLE[bk],
                alpha=0.6, zorder=1)

    for _, bk, avg, jq, tc in cells:
        ax.scatter(jq, avg, c=[cmap(norm(tc))], marker=BACKEND_MARKER[bk],
                   s=100, edgecolor="black", linewidth=0.6, zorder=3)

    p_dom = next(c for c in cells if c[0] == "0_6b" and c[1] == "memobase")
    p_van = next(c for c in cells if c[0] == "32b"  and c[1] == "vanilla")
    ax.annotate(
        "", xy=(p_dom[3], p_dom[2]), xytext=(p_van[3], p_van[2]),
        arrowprops=dict(arrowstyle="->", color="#222", lw=1.3, alpha=0.85),
        zorder=2,
    )

    # Endpoint labels for the dominance comparison.
    ax.text(p_dom[3] * 1.45, p_dom[2] + 5.5,
            "Memobase $\\times$ Q3-0.6B",
            fontsize=7.5, color="#222", ha="left", va="bottom", zorder=4)
    ax.text(p_van[3] * 0.85, p_van[2] - 6,
            "Vanilla $\\times$ Q3-32B-AWQ",
            fontsize=7.5, color="#222", ha="right", va="top", zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.4, 4000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Answer energy  (J / query, log)")
    ax.set_ylabel("Avg accuracy  (pp)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", which="both", alpha=0.12)
    ax.set_title("(a) Pareto: Avg vs.\\ energy on DGX Spark", fontsize=10)

    legend_elems = [
        Line2D([0], [0], marker=BACKEND_MARKER[bk], color="w",
               markerfacecolor="#bbb", markeredgecolor="black",
               markersize=9, label=BACKEND_TEX[bk] if bk != "memobase" else "Memobase")
        for bk in PARETO_BACKENDS
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.045, shrink=0.85)
    cbar.set_label("Peak temp ($^\\circ$C)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Panel (b): Qwen3-8B anchor bar
# ---------------------------------------------------------------------------

ANCHOR = "8b"
BACKENDS_B = ["vanilla", "rag", "dense_rag", "memobase", "memos", "oracle"]
DENSE_RAG_8B = ("BGE-M3", 50.9)


def _plot_anchor(ax) -> None:
    bars = []
    for bk in BACKENDS_B:
        if bk == "dense_rag":
            mean, std = DENSE_RAG_8B[1], 0.0
        else:
            mean, std = PUBLISHED_AVG[ANCHOR][bk]
        bars.append((bk, mean, std))

    xs = list(range(len(BACKENDS_B)))
    for i, (bk, v, s) in enumerate(bars):
        hatch = "//" if bk == "dense_rag" else None
        ax.bar(i, v, 0.68, color=COLOR[bk], edgecolor="black", linewidth=0.5,
               yerr=s if s > 0 else None, capsize=2.5, ecolor="#333", hatch=hatch)
        ax.text(i, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, color="#222")

    oracle_v = bars[-1][1]
    ax.axhline(oracle_v, color=COLOR["oracle"], lw=0.8, ls="--", alpha=0.55)

    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL_A_FULL[bk] for bk in BACKENDS_B],
                       rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Avg  (pp)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Q3-8B anchor: backend $\\times$ Avg", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)


# ---------------------------------------------------------------------------
# Panel (b): lift over Vanilla per reader tier
# ---------------------------------------------------------------------------

LIFT_TIERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
LIFT_BACKENDS = ["rag", "memobase", "memos"]


def _plot_lifts(ax) -> None:
    width = 0.26
    offsets = {"rag": -width, "memobase": 0.0, "memos": +width}
    x_tiers = list(range(len(LIFT_TIERS)))

    for bk in LIFT_BACKENDS:
        lifts = []
        for t in LIFT_TIERS:
            van_mean = PUBLISHED_AVG[t]["vanilla"][0]
            bv_mean  = PUBLISHED_AVG[t][bk][0]
            lifts.append(bv_mean - van_mean)
        ax.bar([x + offsets[bk] for x in x_tiers], lifts, width,
               color=COLOR[bk], edgecolor="black", linewidth=0.5,
               label=BACKEND_TEX[bk])
        for x, v in zip([x + offsets[bk] for x in x_tiers], lifts):
            ax.text(x, v + 1.0 if v >= 0 else v - 1.4,
                    f"{v:+.0f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=6, color="#222")

    ax.axhline(0, color="#555", lw=0.6)
    ax.set_xticks(x_tiers)
    ax.set_xticklabels([MODEL_TEX[t] for t in LIFT_TIERS],
                       rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Lift over Vanilla  (pp)")
    ax.set_ylim(-35, 60)
    ax.set_title("(b) Memory lift, by reader", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3,
              bbox_to_anchor=(0, 1.0))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _render(out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    _plot_pareto(axes[0], fig)
    _plot_lifts(axes[1])
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _render(OUT_PDF)
    print(f"[fig:finding3] wrote {OUT_PDF}")
    print(f"[fig:finding3] wrote {OUT_PDF.with_suffix('.png')}")


if __name__ == "__main__":
    main()
