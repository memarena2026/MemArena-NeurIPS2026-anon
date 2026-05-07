"""Emit two LaTeX tables for the latency-methodology appendix:

  paper/tables/appendix_latency_fit.tex      — per-reader α, β, γ, R²
  paper/tables/appendix_latency_predicted.tex — predicted T_total per cell
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC  = REPO / "out" / "latency_selection" / "llm_fit_models.json"
OUT_FIT  = REPO / "paper" / "tables" / "appendix_latency_fit.tex"
OUT_PRED = REPO / "paper" / "tables" / "appendix_latency_predicted.tex"

READERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
READER_LABEL = {
    "0_6b":    r"Qwen3-0.6B",
    "llama3b": r"Llama-3.2-3B",
    "7b":      r"Mistral-7B-Inst.\ v0.3",
    "8b":      r"Qwen3-8B",
    "32b":     r"Qwen3-32B-AWQ",
}
BACKENDS = ["vanilla", "oracle", "inmem", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   r"Vanilla",
    "oracle":    r"Oracle",
    "inmem":     r"InMem",
    "memobase":  r"Memobase",
    "memsearch": r"MemSearch",
}


def fmt_int(x): return "--" if x is None else f"{int(round(x))}"
def fmt_pct(x): return "--" if x is None else f"{100*x:.1f}\\%"


def write_fit_table(d: dict) -> None:
    fits = d["fits"]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Per-reader linear LLM-generation latency fit on Spark GB10 (\texttt{concurrency=1}). The model is $T_{\text{LLM}} = \alpha + \beta N_p + \gamma N_c$. Fit data is the union of vanilla / oracle / inmem cells (\S\ref{app:lat:fit}). $|r|_{\text{p95}}$ is the absolute residual at the 95th percentile.}",
        r"\label{tab:appendix-latency-fit}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Reader & $\alpha$ (ms) & $\beta$ (ms\,/\,1k prompt tok) & $\gamma$ (ms\,/\,compl tok) & $R^2$ & $n_{\text{obs}}$ & $|r|_{\text{p95}}$ (ms) \\",
        r"\midrule",
    ]
    for r in READERS:
        f = fits.get(r)
        if not f:
            lines.append(f"{READER_LABEL[r]} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        lines.append(
            f"{READER_LABEL[r]} & {f['alpha_ms']:.1f} & {f['beta_ms_per_prompt_tok']*1000:.2f} & {f['gamma_ms_per_completion_tok']:.2f} & {f['R2']:.3f} & {f['n_obs']} & {f['resid_p95_abs_ms']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_FIT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_FIT.relative_to(REPO)}")


def write_predicted_table(d: dict) -> None:
    pred = d["predicted_cells"]
    val  = d.get("validation", {})
    search_oh = d["search_overhead_ms"]

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Single-stream end-to-end latency $T_{\text{total}}$ on Spark GB10 (ms, p50, \texttt{concurrency=1}). \emph{Roman}: measured cell median (answer-time median plus $T_{\text{search}}$ from Table~\ref{tab:appendix-latency-fit}'s search row). \emph{Italic with $\dagger$}: predicted by composing the per-reader fit (Table~\ref{tab:appendix-latency-fit}) with the \texttt{0\_6b}-measured prompt-length distribution and the backend's $T_{\text{search}}$, where we have not run the cell directly (\S\ref{app:lat:extrapolation}). \texttt{n/a} marks cells with no fit available (Mistral-7B; see \S\ref{app:lat:caveats}).}",
        r"\label{tab:appendix-latency-predicted}",
        r"\begin{tabular}{l" + "r" * len(BACKENDS) + r"}",
        r"\toprule",
        r"Reader & " + " & ".join(BACKEND_LABEL[b] for b in BACKENDS) + r" \\",
        r"\midrule",
    ]
    for r in READERS:
        cells = []
        for b in BACKENDS:
            key = f"{b}_{r}"
            p = pred.get(key)
            v = val.get(key)
            if not p:
                cells.append(r"\texttt{n/a}")
                continue
            t_search = p["t_search_meas_ms"]
            if v:  # have a measured cell
                cells.append(f"{v['meas_median_ms'] + t_search:.0f}")
            else:
                cells.append(rf"\textit{{{p['t_total_pred_ms']:.0f}}}$^\dagger$")
        lines.append(f"{READER_LABEL[r]} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    sh_row = ["$T_{\\text{search}}$ (ms, p50)"] + [
        f"{(search_oh.get(b) or {}).get('median', 0.0):.0f}" for b in BACKENDS
    ]
    lines.append(" & ".join(sh_row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_PRED.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PRED.relative_to(REPO)}")


def main():
    d = json.loads(SRC.read_text())
    write_fit_table(d)
    write_predicted_table(d)


if __name__ == "__main__":
    main()
