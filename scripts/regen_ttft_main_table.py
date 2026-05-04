"""Regenerate paper/tables/ttft_main.tex from the latency fit artefact.

Source of truth: ``out/latency_selection/llm_fit_models.json`` (produced by
``scripts/fit_latency_model.py``). Each (backend, reader) cell uses the
``predicted_ttft_cells`` block; roman vs italic-with-$\\dagger$ formatting is
decided by ``measured_ttft_cells`` membership. The bottom ``$N_p$`` row
takes ``n_p_p50`` from the ``0_6b`` row of each backend, since prompt
lengths are reader-independent for the five backends.

Usage:
    python3 scripts/regen_ttft_main_table.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIT  = ROOT / "out" / "latency_selection" / "llm_fit_models.json"
OUT  = ROOT / "paper" / "tables" / "ttft_main.tex"

BACKENDS = ["vanilla", "oracle", "inmem", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   "Vanilla",
    "oracle":    "Oracle",
    "inmem":     "InMem",
    "memobase":  "Memobase",
    "memsearch": "MemSearch",
}
READERS = ["0_6b", "llama3b", "8b", "32b"]
READER_LABEL = {
    "0_6b":    "Qwen3-0.6B",
    "llama3b": "Llama-3.2-3B",
    "8b":      "Qwen3-8B",
    "32b":     "Qwen3-32B-AWQ",
}
NA_CELLS = {("inmem", "32b")}


def fmt_ms(t):
    return f"{t:.0f}" if t is not None else "--"


def main() -> int:
    data = json.loads(FIT.read_text())
    pred = data["predicted_ttft_cells"]
    meas = set(data.get("measured_ttft_cells", []))

    rows = []
    for r in READERS:
        cells = []
        for b in BACKENDS:
            key = f"{b}_{r}"
            if (b, r) in NA_CELLS:
                cells.append(r"\texttt{n/a}")
                continue
            cell = pred.get(key)
            if cell is None:
                cells.append("--")
                continue
            total = cell["t_search_meas_ms"] + cell["t_ttft_llm_ms"]
            txt = fmt_ms(total)
            if key not in meas:
                txt = rf"\textit{{{txt}}}$^\dagger$"
            cells.append(txt)
        rows.append(rf"{READER_LABEL[r]} & " + " & ".join(cells) + r" \\")

    # Bottom rows: T_search and N_p (taken from 0_6b cells of each backend).
    t_search = []
    n_p = []
    for b in BACKENDS:
        key = f"{b}_0_6b"
        cell = pred.get(key, {})
        t_search.append(fmt_ms(cell.get("t_search_meas_ms", 0)))
        v = cell.get("n_p_p50", 0)
        n_p.append(f"{v:,}".replace(",", "{,}"))
    rows.append(r"\midrule")
    rows.append(r"$T_{\text{search}}$ (ms)        & " + " & ".join(t_search) + r" \\")
    rows.append(r"$N_p$ (p50, prompt tokens) & " + " & ".join(n_p) + r" \\")

    body = "\n".join(rows)

    out = rf"""\begin{{table}}[!htbp]
\centering
\small
\caption{{End-to-end TTFT (ms, p50) on Spark GB10 at \texttt{{concurrency=1}}, decomposed as $T_{{\text{{search}}}} + T_{{\text{{TTFT,LLM}}}}$. Italic entries ($\dagger$) and the \texttt{{n/a}} cell are explained in Appendix~\ref{{app:lat:extrapolation}}.}}
\label{{tab:ttft-main}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Reader & {' & '.join(BACKEND_LABEL[b] for b in BACKENDS)} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    OUT.write_text(out)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
