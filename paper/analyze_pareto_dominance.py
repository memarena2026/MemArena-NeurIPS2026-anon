#!/usr/bin/env python3
"""Generate Figure (``fig:pareto-dominance``): structured memory's
Pareto-dominance over reader scaling on DGX Spark.

Plots Avg accuracy vs.\ log answer-time energy for every (reader x backend)
cell in the on-device matrix where Spark hardware probes ran:

  readers  = {Qwen3-0.6B, Llama-3.2-3B, Qwen3-8B, Qwen3-32B-AWQ}
             (Mistral-7B is NOT measured on Spark; omitted)
  backends = {Vanilla, RAG, Memobase}

Each point is colored by peak GPU temperature (cool->hot).  Within-backend
curves connect cells in reader-scale order, exposing that the Memobase
curve sits up-and-left of Vanilla and RAG curves at every matched scale.
A dominance arrow runs from (Vanilla x Q3-32B-AWQ) to (Memobase x Q3-0.6B).

Output
------
``paper/figures/fig_pareto_dominance.{pdf,png}``.

Energy and peak-temperature numbers are loaded from
``MASim/runs/l_20260408_111046/spark_results_s1/hw/hw_agg_*.json``
(``per_cell.mean_energy_j_per_query`` and ``per_cell.peak_temp_c``).
Avg-accuracy numbers are pulled from the published main-table values
in ``analyze_finding2.PUBLISHED_AVG`` (kept in lockstep with main_SML.tex).
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


SPARK_HW_DIR = (
    Path(__file__).resolve().parent.parent
    / "MASim/runs/l_20260408_111046/spark_results_s1/hw"
)

OUT_PDF = Path(__file__).resolve().parent / "figures" / "fig_pareto_dominance.pdf"

# Reader-scale order (small -> large).  Mistral-7B is dropped because Spark
# never ran it; the Pareto figure is restricted to cells where a hardware
# probe exists.
MODEL_ORDER = ["0_6b", "llama3b", "8b", "32b"]
MODEL_LABEL = {
    "0_6b": "Qwen3-0.6B",
    "llama3b": "Llama-3.2-3B",
    "8b": "Qwen3-8B",
    "32b": "Qwen3-32B-AWQ",
}

# (model_key -> hw_agg file stem) for the THREE backends in the figure.
HW_STUB = {
    "0_6b":    {"vanilla": "qwen3_0.6b_vanilla",     "rag": "qwen3_0.6b_inmem",      "memobase": "qwen3_0.6b_memobase_answerB"},
    "llama3b": {"vanilla": "llama3_3b_vanilla",      "rag": "llama3_3b_inmem",       "memobase": "llama3_3b_memobase_answerB"},
    "8b":      {"vanilla": "qwen3_8b_vanilla",       "rag": "qwen3_8b_inmem",        "memobase": "qwen3_8b_memobase_answerB"},
    "32b":     {"vanilla": "qwen3_32b_awq_vanilla",  "rag": "qwen3_32b_awq_inmem",   "memobase": "qwen3_32b_awq_memobase_answerB"},
}

BACKEND_ORDER = ["vanilla", "rag", "memobase"]
BACKEND_LABEL = {"vanilla": "Vanilla", "rag": "RAG", "memobase": "Memobase"}
BACKEND_MARKER = {"vanilla": "o", "rag": "s", "memobase": "D"}
BACKEND_LINESTYLE = {
    "vanilla":  (0, (4, 2)),    # dashed
    "rag":      (0, (1, 2)),    # dotted
    "memobase": "-",            # solid
}


def _load_hw(stub: str) -> tuple[float, float]:
    """Return ``(mean_energy_j_per_query, peak_temp_c)`` for a hw_agg cell."""
    path = SPARK_HW_DIR / f"hw_agg_{stub}.json"
    with path.open() as fh:
        per_cell = json.load(fh)["per_cell"]
    return per_cell["mean_energy_j_per_query"], per_cell["peak_temp_c"]


def _build_cells() -> list[tuple[str, str, float, float, float]]:
    """Return rows of (model_key, backend, avg_pp, j_per_q, peak_temp_c)."""
    rows: list[tuple[str, str, float, float, float]] = []
    for model in MODEL_ORDER:
        for backend in BACKEND_ORDER:
            avg_pp, _ = PUBLISHED_AVG[model][backend]
            j_q, t_c = _load_hw(HW_STUB[model][backend])
            rows.append((model, backend, avg_pp, j_q, t_c))
    return rows


def _render(out_pdf: Path) -> None:
    cells = _build_cells()

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    temps = [c[4] for c in cells]
    norm = Normalize(vmin=min(temps), vmax=max(temps))
    cmap = plt.get_cmap("coolwarm")  # blue -> red

    # Connecting curves per backend (rendered behind scatter).
    for bk in BACKEND_ORDER:
        bk_cells = [c for c in cells if c[1] == bk]
        bk_cells.sort(key=lambda c: MODEL_ORDER.index(c[0]))
        xs = [c[3] for c in bk_cells]
        ys = [c[2] for c in bk_cells]
        ax.plot(
            xs, ys,
            color="#888", linewidth=1.0, linestyle=BACKEND_LINESTYLE[bk],
            alpha=0.6, zorder=1,
        )

    # Scatter on top, colored by peak temperature.
    for model, bk, avg, jq, tc in cells:
        ax.scatter(
            jq, avg,
            c=[cmap(norm(tc))],
            marker=BACKEND_MARKER[bk],
            s=130,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.4, 4000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Answer-time energy  (J / query, log scale)")
    ax.set_ylabel("Avg accuracy  (pp)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", which="both", alpha=0.15)

    # Dominance arrow: Vanilla x Q3-32B  -->  Memobase x Q3-0.6B.
    p_dom = next(c for c in cells if c[0] == "0_6b" and c[1] == "memobase")
    p_van = next(c for c in cells if c[0] == "32b"  and c[1] == "vanilla")
    ax.annotate(
        "",
        xy=(p_dom[3], p_dom[2]),
        xytext=(p_van[3], p_van[2]),
        arrowprops=dict(arrowstyle="->", color="#222", lw=1.4, alpha=0.85),
        zorder=2,
    )

    # Reader labels on the Memobase curve (top curve, least clutter).
    for c in cells:
        if c[1] != "memobase":
            continue
        offset_x = 1.15 if c[0] != "32b" else 0.88
        ha = "left" if c[0] != "32b" else "right"
        ax.text(c[3] * offset_x, c[2] + 1.0, MODEL_LABEL[c[0]],
                fontsize=8, color="#222", ha=ha)

    # Marker-shape legend (backend identity).
    legend_elems = [
        Line2D(
            [0], [0],
            marker=BACKEND_MARKER[bk], color="w",
            markerfacecolor="#bbb", markeredgecolor="black",
            markersize=10, label=BACKEND_LABEL[bk],
        )
        for bk in BACKEND_ORDER
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9, frameon=False,
              title="Backend", title_fontsize=9)

    # Color bar for peak temperature.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.82)
    cbar.set_label("Peak GPU temperature  ($^\\circ$C)")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cells = _build_cells()
    print(f"\n{'model':<14} {'backend':<10} {'Avg':>6} {'J/q':>10} {'temp':>7}")
    for model, bk, avg, jq, tc in cells:
        print(f"  {MODEL_LABEL[model]:<14} {BACKEND_LABEL[bk]:<10} "
              f"{avg:>6.1f} {jq:>10.3f} {tc:>7.1f}")
    _render(OUT_PDF)
    print(f"\n[fig:pareto-dominance] wrote {OUT_PDF}")
    print(f"[fig:pareto-dominance] wrote {OUT_PDF.with_suffix('.png')}")


if __name__ == "__main__":
    main()
